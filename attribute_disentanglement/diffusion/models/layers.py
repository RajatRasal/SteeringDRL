import math

import torch
from torch import nn
from torch.nn import functional as F

from attribute_disentanglement.diffusion import utils


# Residual blocks

class ResidualBlock(nn.Module):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


# Noise level (and other) conditioning

class ConditionedModule(nn.Module):
    pass


class UnconditionedModule(ConditionedModule):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input, cond=None):
        return self.module(input)


class ConditionedSequential(nn.Sequential, ConditionedModule):
    def forward(self, input, cond):
        for module in self:
            if isinstance(module, ConditionedModule):
                input = module(input, cond)
            else:
                input = module(input)
        return input


class ConditionedResidualBlock(ConditionedModule):
    def __init__(self, *main, skip=None):
        super().__init__()
        self.main = ConditionedSequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input, cond):
        skip = self.skip(input, cond) if isinstance(self.skip, ConditionedModule) else self.skip(input)
        return self.main(input, cond) + skip


class AdaGN(ConditionedModule):
    def __init__(self, feats_in, c_out, num_groups, eps=1e-5, cond_key='cond'):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.cond_key = cond_key
        self.mapper = nn.Linear(feats_in, c_out * 2)
        nn.init.zeros_(self.mapper.weight)
        nn.init.zeros_(self.mapper.bias)

    def forward(self, input, cond):
        weight, bias = self.mapper(cond[self.cond_key]).chunk(2, dim=-1)
        input = F.group_norm(input, self.num_groups, eps=self.eps)
        return torch.addcmul(utils.append_dims(bias, input.ndim), input, utils.append_dims(weight, input.ndim) + 1)


# Attention


class SelfAttention2d(ConditionedModule):
    def __init__(self, c_in, n_head, norm, dropout_rate=0.):
        super().__init__()
        assert c_in % n_head == 0
        self.norm_in = norm(c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, input, cond):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm_in(input, cond))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class CrossAttention2d(ConditionedModule):
    def __init__(self, c_dec, c_enc, n_head, norm_dec, dropout_rate=0.,
                 cond_key='cross', cond_key_padding='cross_padding'):
        super().__init__()
        assert c_dec % n_head == 0
        self.cond_key = cond_key
        self.cond_key_padding = cond_key_padding
        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = norm_dec(c_dec)
        self.n_head = n_head
        self.q_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.kv_proj = nn.Linear(c_enc, c_dec * 2)
        self.out_proj = nn.Conv2d(c_dec, c_dec, 1)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.q = None
        self.k = None
        self.v = None

    def forward(self, input, cond):
        n, c, h, w = input.shape
        q = self.q_proj(self.norm_dec(input, cond))
        q = q.view([n, self.n_head, c // self.n_head, h * w]).transpose(2, 3)
        kv = self.kv_proj(self.norm_enc(cond[self.cond_key]))
        kv = kv.view([n, -1, self.n_head * 2, c // self.n_head]).transpose(1, 2)
        k, v = kv.chunk(2, dim=1)
        attn_mask = (cond[self.cond_key_padding][:, None, None, :]) * -1e4
        self.k = k
        self.q = q
        self.v = v
        y = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout.p)
        y = y.transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.out_proj(y)


class CrossAttention2dReZero(ConditionedModule):
    def __init__(
        self,
        c_dec: int,
        c_enc: int,
        n_head: int,
        norm_dec,
        dropout_rate: float = 0.0,
        cond_key: str = "cross",
        cond_key_padding: str = "cross_padding",
        attn_init_std: float = 0.02,
        init_res_scale: float = 0.1,   # ReZero-style: start at 0, or 0.1 if you want non-zero
    ):
        super().__init__()
        assert c_dec % n_head == 0, "c_dec must be divisible by n_head"

        self.cond_key = cond_key
        self.cond_key_padding = cond_key_padding

        self.norm_enc = nn.LayerNorm(c_enc)
        self.norm_dec = norm_dec(c_dec)
        self.n_head = n_head

        self.q_proj = nn.Conv2d(c_dec, c_dec, kernel_size=1)
        self.kv_proj = nn.Linear(c_enc, 2 * c_dec)
        self.out_proj = nn.Conv2d(c_dec, c_dec, kernel_size=1)

        self.dropout = nn.Dropout(dropout_rate)

        # ReZero-style residual gate on the CA branch
        self.res_scale = nn.Parameter(torch.tensor(init_res_scale))

        # For analysis (q/k/v captured each forward)
        self.q = None
        self.k = None
        self.v = None

        # Initialise weights
        self._reset_parameters(attn_init_std)

    def _reset_parameters(self, attn_init_std: float):
        # q / k / v projections: small gaussian
        nn.init.normal_(self.q_proj.weight, mean=0.0, std=attn_init_std)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)

        nn.init.normal_(self.kv_proj.weight, mean=0.0, std=attn_init_std)
        if self.kv_proj.bias is not None:
            nn.init.zeros_(self.kv_proj.bias)

        # out_proj starts at zero so CA path is initially off (together with res_scale)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, cond: dict) -> torch.Tensor:
        """
        x:    [B, C_dec, H, W]
        cond: dict with
              cond[self.cond_key]           -> [B, L, C_enc]
              cond[self.cond_key_padding]   -> [B, L] (True = pad)
        """
        B, C, H, W = x.shape

        # ---- queries from decoder feature map ----
        # norm_dec is your FiLM/cond-aware norm: norm_dec(x, cond)
        q_in = self.norm_dec(x, cond)            # [B, C_dec, H, W]
        q = self.q_proj(q_in)                    # [B, C_dec, H, W]
        q = q.view(B, self.n_head, C // self.n_head, H * W)  # [B, H, d, Q]
        q = q.transpose(2, 3)                    # [B, H, Q, d]

        # ---- keys / values from encoder tokens ----
        enc = cond[self.cond_key]                # [B, L, C_enc]
        enc = self.norm_enc(enc)                 # [B, L, C_enc]

        kv = self.kv_proj(enc)                   # [B, L, 2 * C_dec]
        kv = kv.view(B, -1, 2 * self.n_head, C // self.n_head)  # [B, L, 2H, d]
        kv = kv.transpose(1, 2)                  # [B, 2H, L, d]
        k, v = kv.chunk(2, dim=1)                # each [B, H, L, d]

        # ---- attention mask ----
        # cond_key_padding: [B, L] with True on padded positions
        pad = cond[self.cond_key_padding].to(q.device)  # [B, L]
        # scaled_dot_product_attention expects additive mask or bool:
        # we build an additive mask broadcasted to [B,H,Q,L]
        attn_mask = pad[:, None, None, :].float() * -1e4  # [B,1,1,L]

        # Save for analysis
        self.q = q
        self.k = k
        self.v = v

        # ---- scaled dot-product attention ----
        # q: [B, H, Q, d], k/v: [B, H, L, d]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
        )                                        # [B, H, Q, d]

        y = y.transpose(2, 3).contiguous().view(B, C, H, W)  # [B, C_dec, H, W]
        y = self.out_proj(y)                                  # [B, C_dec, H, W]

        # ReZero-style residual: learn how much CA to use
        return x + self.res_scale * y


# Downsampling/upsampling

_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic': 
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
        0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3': 
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
        -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
        0.44638532400131226, 0.13550527393817902, -0.066637322306633,
        -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}
_kernels['bilinear'] = _kernels['linear']
_kernels['bicubic'] = _kernels['cubic']


class Downsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv2d(x, weight, stride=2)


class LearnedDownsample2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([_kernels[kernel]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer('kernel', kernel_1d.T @ kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


class LearnedUpsample2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# Embeddings

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# U-Nets

class UNet(ConditionedModule):
    def __init__(self, d_blocks, u_blocks, mid_block=None, skip_stages=0):
        super().__init__()
        self.d_blocks = nn.ModuleList(d_blocks)
        self.u_blocks = nn.ModuleList(u_blocks)
        self.mid_block = mid_block
        self.skip_stages = skip_stages

    def forward(self, input, cond):
        skips = []
        for block in self.d_blocks[self.skip_stages:]:
            input = block(input, cond)
            skips.append(input)
        if self.mid_block is not None:
            input = self.mid_block(input, cond)
        for i, (block, skip) in enumerate(zip(self.u_blocks, reversed(skips))):
            input = block(input, cond, skip if i > 0 else None)
        return input


# Activations

class Erf(nn.Module):

    def forward(self, x):
        return torch.special.erf(x)


class Atan(nn.Module):

    def forward(self, x):
        return torch.atan(x)
