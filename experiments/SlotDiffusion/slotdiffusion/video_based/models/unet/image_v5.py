from abc import abstractmethod
from inspect import isfunction
import math

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .utils import checkpoint, conv_nd, linear, avg_pool_nd, zero_module, normalization


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def append_dims(x, target_dims):
    while x.ndim < target_dims:
        x = x[..., None]
    return x


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer(
            'weight',
            torch.randn([out_features // 2, in_features]) * std,
        )

    def forward(self, input):
        freqs = 2 * math.pi * input @ self.weight.T
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


class Erf(nn.Module):
    def forward(self, x):
        return torch.special.erf(x)


class Atan(nn.Module):
    def forward(self, x):
        return torch.atan(x)


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layer = nn.Linear(feats_in if i == 0 else feats_out, feats_out)
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.GELU()])
        super().__init__(*layers)


def Normalize(in_channels):
    return nn.GroupNorm(
        num_groups=32,
        num_channels=in_channels,
        eps=1e-6,
        affine=True,
    )


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            zero_module(nn.Linear(inner_dim, dim_out)),
        )

    def forward(self, x):
        return self.net(x)


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, cond):
        x = self.ln(x)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=-1)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        norm_ctx=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.norm_ctx = nn.LayerNorm(context_dim) if norm_ctx else nn.Identity()
        self.to_out = nn.Sequential(
            zero_module(nn.Linear(inner_dim, query_dim)),
            nn.Dropout(dropout),
        )

    def forward(self, x, context=None, mask=None):
        del mask
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        context = self.norm_ctx(context)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h),
            (q, k, v),
        )

        sim = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bij,bjd->bid', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        use_checkpoint=False,
        norm_style=None,
        norm_ctx=False,
        adaln_ctx_dim=256,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            norm_ctx=norm_ctx,
        )
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            norm_ctx=norm_ctx,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm_style = norm_style
        self.use_checkpoint = use_checkpoint

        if self.norm_style == 'adaln':
            self.norm1 = AdaLN(dim, adaln_ctx_dim)
            self.norm2 = AdaLN(dim, adaln_ctx_dim)
            self.norm3 = AdaLN(dim, adaln_ctx_dim)
        elif self.norm_style == 'hybrid':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = AdaLN(dim, adaln_ctx_dim)
            self.norm3 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None, emb=None):
        return checkpoint(
            self._forward,
            (x, context, emb),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x, context=None, emb=None):
        if self.norm_style == 'adaln':
            x = self.attn1(self.norm1(x, emb)) + x
            x = self.attn2(self.norm2(x, emb), context=context) + x
            x = self.ff(self.norm3(x, emb)) + x
            return x
        if self.norm_style == 'hybrid':
            x = self.attn1(self.norm1(x)) + x
            x = self.attn2(self.norm2(x, emb), context=context) + x
            x = self.ff(self.norm3(x)) + x
            return x
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_checkpoint=False,
        norm_style=None,
        norm_ctx=False,
        adaln_ctx_dim=256,
        residual_init=None,
        residual_constraint='identity',
        residual_scale=1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.residual_init = residual_init
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv2d(
            in_channels,
            inner_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    norm_style=norm_style,
                    norm_ctx=norm_ctx,
                    adaln_ctx_dim=adaln_ctx_dim,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = zero_module(
            nn.Conv2d(
                inner_dim,
                in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        if residual_init is not None:
            self.residual = nn.Parameter(th.tensor(residual_init))
            self.residual_constraint = {
                'identity': nn.Identity(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'softsign': nn.Softsign(),
                'erf': Erf(),
                'arctan': Atan(),
                'softplus': nn.Softplus(),
            }[residual_constraint]
            self.residual_scale = residual_scale

    def forward(self, x, context=None, emb=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, emb=emb)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        if self.residual_init is None:
            residual = x_in
        else:
            scale = self.residual_constraint(self.residual / self.residual_scale)
            residual = scale * x_in
        return x + residual


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                padding=padding,
            )

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x,
                (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                mode='nearest',
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = nn.Identity()
            self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels
                if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward,
            (x, emb),
            self.parameters(),
            self.use_checkpoint,
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = append_dims(emb_out, h.ndim)
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModelV5(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        latent_unit,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        dims=2,
        use_checkpoint=False,
        num_heads=8,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        conv_resample=True,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        norm_style=None,
        norm_ctx=False,
        skip_residual_init=None,
        skip_residual_constraint='identity',
        skip_residual_scale=1.0,
        trans_residual_init=None,
        trans_residual_constraint='identity',
        trans_residual_scale=1.0,
    ):
        super().__init__()
        del n_embed
        assert context_dim is not None, 'Include the dimension of cross-attention conditioning'

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.context_dim = context_dim
        self.latent_unit = latent_unit
        self.skip_residual_init = skip_residual_init

        time_embed_dim = model_channels * 4
        self.timestep_embed = FourierFeatures(1, model_channels)
        self.time_embed = MappingNet(model_channels, time_embed_dim, n_layers=2)

        self.input_block = conv_nd(dims, in_channels, model_channels, 3, padding=1)

        self.enc_blocks = nn.ModuleDict()
        self.down_blocks = nn.ModuleDict()

        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            layers = []
            for _ in range(num_res_blocks):
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        heads = num_heads
                    else:
                        heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            heads,
                            dim_head,
                            depth=transformer_depth,
                            dropout=dropout,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            norm_style=norm_style,
                            norm_ctx=norm_ctx,
                            adaln_ctx_dim=time_embed_dim,
                            residual_init=trans_residual_init,
                            residual_constraint=trans_residual_constraint,
                            residual_scale=trans_residual_scale,
                        )
                    )
            self.enc_blocks[str(level)] = TimestepEmbedSequential(*layers)
            if level != len(channel_mult) - 1:
                out_ch = ch
                down_block = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                        down=True,
                    )
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                self.down_blocks[str(level)] = TimestepEmbedSequential(down_block)
                ch = out_ch
                ds *= 2
            else:
                self.down_blocks[str(level)] = TimestepEmbedSequential(nn.Identity())

        if num_head_channels == -1:
            dim_head = ch // num_heads
            heads = num_heads
        else:
            heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint=use_checkpoint,
            ),
            SpatialTransformer(
                ch,
                heads,
                dim_head,
                depth=transformer_depth,
                dropout=dropout,
                context_dim=context_dim,
                use_checkpoint=use_checkpoint,
                norm_style=norm_style,
                norm_ctx=norm_ctx,
                adaln_ctx_dim=time_embed_dim,
                residual_init=trans_residual_init,
                residual_constraint=trans_residual_constraint,
                residual_scale=trans_residual_scale,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                use_checkpoint=use_checkpoint,
            ),
        )

        self.dec_blocks = nn.ModuleDict()
        self.up_blocks = nn.ModuleDict()

        for level, mult in list(enumerate(channel_mult))[::-1]:
            layers = []
            for _ in range(num_res_blocks + 1):
                layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                    )
                )
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                        heads = num_heads
                    else:
                        heads = ch // num_head_channels
                        dim_head = num_head_channels
                    layers.append(
                        SpatialTransformer(
                            ch,
                            heads,
                            dim_head,
                            depth=transformer_depth,
                            dropout=dropout,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            norm_style=norm_style,
                            norm_ctx=norm_ctx,
                            adaln_ctx_dim=time_embed_dim,
                            residual_init=trans_residual_init,
                            residual_constraint=trans_residual_constraint,
                            residual_scale=trans_residual_scale,
                        )
                    )
            self.dec_blocks[str(level)] = TimestepEmbedSequential(*layers)
            if level:
                out_ch = ch
                up_block = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_checkpoint=use_checkpoint,
                        up=True,
                    )
                    if resblock_updown else
                    Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
                self.up_blocks[str(level)] = TimestepEmbedSequential(up_block)
                ds //= 2
            else:
                self.up_blocks[str(level)] = TimestepEmbedSequential(nn.Identity())

        self.output_block = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if skip_residual_init is not None:
            self.skip_residuals = nn.ParameterDict()
            for level in range(len(self.channel_mult)):
                self.skip_residuals[str(level)] = nn.Parameter(
                    th.tensor(skip_residual_init)
                )
            self.skip_residual_constraint = {
                'identity': nn.Identity(),
                'tanh': nn.Tanh(),
                'sigmoid': nn.Sigmoid(),
                'softsign': nn.Softsign(),
                'erf': Erf(),
                'arctan': Atan(),
                'softplus': nn.Softplus(),
            }[skip_residual_constraint]
            self.skip_residual_scale = skip_residual_scale

    def param_groups(self, base_lr=2e-4):
        decay, no_decay = [], []
        norm_keywords = ('norm', 'ln', 'layernorm', 'groupnorm')

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            name_l = name.lower()
            is_bias = name_l.endswith('.bias')
            is_norm = any(keyword in name_l for keyword in norm_keywords)
            is_vector = param.ndim < 2
            if is_bias or is_norm or is_vector:
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {'params': decay, 'lr': base_lr},
            {'params': no_decay, 'lr': base_lr, 'weight_decay': 0.0},
        ]

    def forward(self, x, timesteps=None, context=None, **kwargs):
        del kwargs
        batch_size = x.shape[0]
        t_emb = self.timestep_embed(append_dims(timesteps, 2))
        emb = self.time_embed(t_emb)
        context = None if context is None else context.reshape(batch_size, -1, self.context_dim)

        h = self.input_block(x)
        h = self._forward(0, h, emb, context)
        return self.output_block(h)

    def _forward(self, level, x, emb=None, context=None):
        level_key = str(level)
        h_down = self.enc_blocks[level_key](x, emb, context)
        down = self.down_blocks[level_key](h_down, emb, context)

        if level == len(self.channel_mult) - 1:
            z = self.middle_block(down, emb, context)
        else:
            z = self._forward(level + 1, down, emb, context)

        if self.skip_residual_init is not None:
            scale = self.skip_residual_constraint(
                self.skip_residuals[level_key] / self.skip_residual_scale
            )
            skip = scale * h_down
        else:
            skip = h_down

        h_up = self.dec_blocks[level_key](z, emb, context) + skip
        return self.up_blocks[level_key](h_up, emb, context)

    @property
    def device(self):
        return self.time_embed[0].weight.device

    @property
    def dtype(self):
        return self.time_embed[0].weight.dtype
