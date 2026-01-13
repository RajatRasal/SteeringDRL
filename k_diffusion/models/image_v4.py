from abc import abstractmethod
from inspect import isfunction
import math

import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from k_diffusion.models.image_v2 import MappingNet
from k_diffusion import utils
from k_diffusion.layers import FourierFeatures


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


class SkipDropout(nn.Module):

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if self.training:
            batch_size, channels, height, width = x.shape
            mask = (th.rand(batch_size, channels, 1, 1, device=x.device) > self.rate).float()
            return x * mask / (1 - self.rate)
        else:
            return x


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
        ) if not glu else GEGLU(dim, inner_dim)

        # Note: This is important to ensure FF starts at 0 in early training.
        # This is important to be able to identify optimisation basins.
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            zero_module(nn.Linear(inner_dim, dim_out)),
        )

    def forward(self, x):
        return self.net(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        # At step 0, AdaLN(x | cond) = LN(x)
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
        dropout=0.,
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
            # Note: This is important to ensure cross-attention starts at 0 in early training.
            # This is important to be able to identify optimisation basins.
            zero_module(nn.Linear(inner_dim, query_dim)),
            nn.Dropout(dropout)
        )

        self.q = None
        self.k = None
        self.v = None

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        context = self.norm_ctx(context)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        self.q = q
        self.k = k
        self.v = v

        sim = einsum(q, k, 'b i d, b j d -> b i j') * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum(attn, v, 'b i j, b j d -> b i d')
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.,
        context_dim=None,
        gated_ff=True,
        norm_style=None, # "adaln",
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
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            norm_ctx=norm_ctx,
        )  # is self-attn if context is none
        self.norm_style = norm_style
        if self.norm_style == "adaln":
            self.norm1 = AdaLN(dim, adaln_ctx_dim)
            self.norm2 = AdaLN(dim, adaln_ctx_dim)
            self.norm3 = AdaLN(dim, adaln_ctx_dim)
        elif self.norm_style == "hybrid":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = AdaLN(dim, adaln_ctx_dim)
            self.norm3 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)

        self.h1 = None
        self.h2 = None
        self.h3 = None

    def forward(self, x, context=None, emb=None):
        if self.norm_style == "adaln":
            self.h1 = self.norm1(x, emb)
            x = self.attn1(self.h1) + x
            self.h2 = self.norm2(x, emb)
            x = self.attn2(self.h2, context=context) + x
            self.h3 = self.norm3(x, emb)
            x = self.ff(self.h3) + x
        elif self.norm_style == "hybrid":
            self.h1 = self.norm1(x)
            x = self.attn1(self.h1) + x
            self.h2 = self.norm2(x, emb)
            # TODO: SHORTCUT
            x = self.attn2(self.h2, context=context) + x
            self.h3 = self.norm3(x)
            x = self.ff(self.h3) + x
        else:
            x = self.attn1(self.norm1(x)) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
        return x


def ramp_alpha(progress: float, warmup_frac: float, alpha_min: float = 0.1) -> float:
    t = min(max(progress / warmup_frac, 0.0), 1.0)  # 0..1 over warmup
    return alpha_min + (1.0 - alpha_min) * t


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.,
        context_dim=None,
        norm_style=None,
        norm_ctx=False,
        adaln_ctx_dim=256,
        learned_residual="rezero",
        residual_scale=1.,
        residual_min=None,
    ):
        super().__init__()
        self.in_channels = in_channels
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
                    norm_style=norm_style,
                    norm_ctx=norm_ctx,
                    adaln_ctx_dim=adaln_ctx_dim,
                )
                for d in range(depth)
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

        if learned_residual:
            self.alpha = nn.Parameter(torch.tensor(residual_scale))
        else:
            self.alpha = residual_scale
        
        self.residual_min = residual_min

    def forward(self, x, context=None, emb=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, emb=emb)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        # TODO: SHORTCUT
        if self.residual_min is None:
            alpha = self.alpha
        else:
            alpha = self.residual_min + (1 - self.residual_min) * torch.sigmoid(self.alpha)
        return x + alpha * x_in


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

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
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
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
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class ImageDenoiserModelV4(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        latent_unit,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        dims=2,
        num_heads=8,
        use_scale_shift_norm=False,
        resblock_updown=True,
        conv_resample=False,
        transformer_depth=1,
        context_dim=None,
        norm_style=None,
        norm_ctx=False,
        skip_scale="default",
        kappa=0.5,
        skip_dropout_rate=0,
        skip_drop_resolutions=(1, 2, 4),
        transformer_learned_residual=False,
        transformer_residual_scale=1.,
        transformer_residual_min=None,
    ):
        super().__init__()
        assert context_dim is not None, 'Include the dimension of cross-attention conditioning'
        assert input_size[0] == input_size[1]

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.skip_scale = skip_scale
        self.kappa = kappa
        self.skip_dropout_rate = skip_dropout_rate
        self.skip_drop_resolutions = skip_drop_resolutions

        time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )
        self.timestep_embed = FourierFeatures(1, model_channels)
        self.time_embed = MappingNet(model_channels, time_embed_dim, n_layers=2)
        self.context_dim = context_dim
        self.latent_unit = latent_unit

        self.skip_drop_sizes = set()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            norm_style=norm_style,
                            norm_ctx=norm_ctx,
                            adaln_ctx_dim=time_embed_dim,
                            learned_residual=transformer_learned_residual,
                            residual_scale=transformer_residual_scale,
                            residual_min=transformer_residual_min,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if ds in self.skip_drop_resolutions:
                self.skip_drop_sizes.add(input_size[0] // ds)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown else
                        Downsample(
                            ch,
                            conv_resample,
                            dims=dims, 
                            out_channels=out_ch,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        dim_head = ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                norm_style=norm_style,
                norm_ctx=norm_ctx,
                adaln_ctx_dim=time_embed_dim,
                learned_residual=transformer_learned_residual,
                residual_scale=transformer_residual_scale,
                residual_min=transformer_residual_min,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    dim_head = ch // num_heads
                    layers.append(
                        SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            norm_style=norm_style,
                            norm_ctx=norm_ctx,
                            adaln_ctx_dim=time_embed_dim,
                            learned_residual=transformer_learned_residual,
                            residual_scale=transformer_residual_scale,
                            residual_min=transformer_residual_min,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ) if resblock_updown else
                        Upsample(
                            ch,
                            conv_resample,
                            dims=dims, 
                            out_channels=out_ch,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if self.skip_dropout_rate > 0:
            self.skip_dropout = SkipDropout(self.skip_dropout_rate)

    def param_groups(self, base_lr=2e-4):
        """
        AdamW-style param groups:
          - decay: Conv/Linear "weights" (dim >= 2), excluding norms
          - no_decay: biases, norm weights, and any 1D params
        """
        decay, no_decay = [], []

        # strings that commonly indicate normalization layers in parameter names
        norm_keywords = ("norm", "ln", "layernorm", "groupnorm")

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue

            name_l = name.lower()

            is_bias = name_l.endswith(".bias")
            is_norm = any(k in name_l for k in norm_keywords)
            is_vector = (p.ndim < 2)

            if is_bias or is_norm or is_vector:
                no_decay.append(p)
            else:
                decay.append(p)

        return [
            {"params": decay, "lr": base_lr},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]

    def forward(self, x, timesteps=None, cross_cond=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cross_cond: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        b = x.shape[0]
        hs = []

        t_emb = self.timestep_embed(utils.append_dims(timesteps, 2))
        emb = self.time_embed(t_emb)
        
        context = cross_cond.reshape(b, -1, self.context_dim)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)

        L = len(self.output_blocks)
        for i, module in enumerate(self.output_blocks):
            skip = hs.pop()
            if self.skip_scale == 'default':
                scale = 1
            elif self.skip_scale == 'root_2_cs':
                scale = 2 ** -0.5
            elif self.skip_scale == 'cs':
                power = (L - 1 - i)
                scale = self.kappa ** power
            skip = skip * scale
            # if self.skip_dropout_rate > 0 and skip.shape[-1] in self.attention_resolutions:
            # if skip.shape[-1] in self.attention_resolutions:
            if skip.shape[-1] in self.skip_drop_sizes:
                skip = th.zeros_like(skip)
            # TODO: SHORTCUT
            h_in = th.cat([h, skip], dim=1)
            h = module(h_in, emb, context)

        out = self.out(h)
        return out
