import torch
from torch import nn
from torch.nn import functional as F

from attribute_disentanglement.diffusion.models import layers


def orthogonal_(module):
    nn.init.orthogonal_(module.weight)
    return module


class ResConvBlock(layers.ConditionedResidualBlock):
    def __init__(self, feats_in, c_in, c_mid, c_out, group_size=32, dropout_rate=0.):
        skip = None if c_in == c_out else orthogonal_(nn.Conv2d(c_in, c_out, 1, bias=False))
        super().__init__(
            layers.AdaGN(feats_in, c_in, max(1, c_in // group_size)),
            nn.GELU(),
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            layers.AdaGN(feats_in, c_mid, max(1, c_mid // group_size)),
            nn.GELU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(dropout_rate, inplace=True),
            skip=skip)
        nn.init.zeros_(self.main[-2].weight)
        nn.init.zeros_(self.main[-2].bias)


class DBlock(layers.ConditionedSequential):
    def __init__(
        self,
        n_layers,
        feats_in,
        c_in,
        c_mid,
        c_out,
        group_size=32,
        head_size=64,
        attn_heads=None,
        dropout_rate=0.,
        downsample=False,
        learned_downsample=False,
        self_attn=False,
        cross_attn=False,
        c_enc=0,
        rezero=False,
        init_res_scale=1.0,
    ):
        modules = [nn.Identity()]
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            n_heads = attn_heads if attn_heads is not None else max(1, my_c_out // head_size)
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, n_heads, norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                if rezero:
                    modules.append(
                        layers.CrossAttention2dReZero(
                            my_c_out,
                            c_enc,
                            n_heads,
                            norm,
                            dropout_rate,
                            init_res_scale=init_res_scale,
                        )
                    )
                else:
                    modules.append(
                        layers.CrossAttention2d(
                            my_c_out,
                            c_enc,
                            n_heads,
                            norm,
                            dropout_rate,
                        )
                    )
        super().__init__(*modules)
        self.set_downsample(downsample, learned_downsample, c_in)

    def set_downsample(self, downsample, learned_downsample, ch):
        if downsample:
            if learned_downsample:
                ds = layers.LearnedDownsample2d(ch)
            else:
                ds = layers.Downsample2d()
        else:
            ds = nn.Identity()
        self[0] = ds
        return self


class UBlock(layers.ConditionedSequential):
    def __init__(
        self,
        n_layers,
        feats_in,
        c_in,
        c_mid,
        c_out,
        group_size=32,
        head_size=64,
        attn_heads=None,
        dropout_rate=0.,
        upsample=False,
        learned_upsample=False,
        self_attn=False,
        cross_attn=False,
        c_enc=0,
        rezero=False,
        init_res_scale=1.0,
    ):
        modules = []
        for i in range(n_layers):
            my_c_in = c_in if i == 0 else c_mid
            my_c_out = c_mid if i < n_layers - 1 else c_out
            modules.append(ResConvBlock(feats_in, my_c_in, c_mid, my_c_out, group_size, dropout_rate))
            n_heads = attn_heads if attn_heads is not None else max(1, my_c_out // head_size)
            if self_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                modules.append(layers.SelfAttention2d(my_c_out, n_heads, norm, dropout_rate))
            if cross_attn:
                norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, my_c_out // group_size))
                if rezero:
                    modules.append(
                        layers.CrossAttention2dReZero(
                            my_c_out,
                            c_enc,
                            n_heads,
                            norm,
                            dropout_rate,
                            init_res_scale=init_res_scale,
                        )
                    )
                else:
                    modules.append(
                        layers.CrossAttention2d(
                            my_c_out,
                            c_enc,
                            n_heads,
                            norm,
                            dropout_rate,
                        )
                    )
        modules.append(nn.Identity())
        super().__init__(*modules)
        self.set_upsample(upsample, learned_upsample, c_out)

    def forward(self, input, cond, skip=None):
        if skip is not None:
            input = torch.cat([input, skip], dim=1)
        return super().forward(input, cond)

    def set_upsample(self, upsample, learned_upsample, ch):
        if upsample:
            if learned_upsample:
                us = layers.LearnedUpsample2d(ch)
            else:
                us = layers.Upsample2d()
        else:
            us = nn.Identity()
        self[-1] = us
        return self


class BottleneckBlock(layers.ConditionedSequential):
    def __init__(
        self,
        feats_in,
        channels,
        group_size=32,
        head_size=64,
        attn_heads=None,
        dropout_rate=0.,
        self_attn=False,
        cross_attn=False,
        c_enc=0,
        rezero=False,
        init_res_scale=1.0,
    ):
        modules = []
        modules.append(ResConvBlock(feats_in, channels, channels, channels, group_size, dropout_rate))
        norm = lambda c_in: layers.AdaGN(feats_in, c_in, max(1, channels // group_size))
        n_heads = attn_heads if attn_heads is not None else max(1, channels // head_size)
        if self_attn:
            modules.append(layers.SelfAttention2d(channels, n_heads, norm, dropout_rate))
        if cross_attn:
            if rezero:
                modules.append(
                    layers.CrossAttention2dReZero(
                        channels,
                        c_enc,
                        n_heads,
                        norm,
                        dropout_rate,
                        init_res_scale=init_res_scale,
                    )
                )
            else:
                modules.append(
                    layers.CrossAttention2d(
                        channels,
                        c_enc,
                        n_heads,
                        norm,
                        dropout_rate,
                    )
                )
        modules.append(ResConvBlock(feats_in, channels, channels, channels, group_size, dropout_rate))
        super().__init__(*modules)


class MappingNet(nn.Sequential):
    def __init__(self, feats_in, feats_out, n_layers=2):
        layers = []
        for i in range(n_layers):
            layers.append(orthogonal_(nn.Linear(feats_in if i == 0 else feats_out, feats_out)))
            layers.append(nn.GELU())
        super().__init__(*layers)
