import torch
from torch import nn
from torch.nn import functional as F

from .. import layers, utils


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


class ImageDenoiserModelV2(nn.Module):
    def __init__(
        self,
        c_in,
        feats_in,
        depths,
        channels,
        self_attn_depths,
        cross_attn_depths=None,
        mapping_cond_dim=0,
        unet_cond_dim=0,
        cross_cond_dim=0,
        dropout_rate=0.,
        patch_size=1,
        skip_stages=0,
        has_variance=False,
        n_heads=8,
        learned_subsampling=False,
        up_down_asymmetry=False,
        use_mid_block=True,
    ):
        super().__init__()
        self.c_in = c_in
        self.channels = channels
        self.unet_cond_dim = unet_cond_dim
        self.patch_size = patch_size
        self.has_variance = has_variance
        self.timestep_embed = layers.FourierFeatures(1, feats_in)
        if mapping_cond_dim > 0:
            self.mapping_cond = nn.Linear(mapping_cond_dim, feats_in, bias=False)
        self.mapping = MappingNet(feats_in, feats_in)
        self.proj_in = nn.Conv2d(
            (c_in + unet_cond_dim) * self.patch_size ** 2,
            channels[max(0, skip_stages - 1)],
            1,
        )
        self.proj_out = nn.Conv2d(
            channels[max(0, skip_stages - 1)],
            c_in * self.patch_size ** 2 + (1 if self.has_variance else 0),
            1,
        )
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        if cross_cond_dim == 0:
            cross_attn_depths = [False] * len(self_attn_depths)
        d_blocks, u_blocks = [], []
        for i in range(len(depths)):
            my_c_in = channels[max(0, i - 1)]
            d_blocks.append(
                DBlock(
                    depths[i],
                    feats_in,
                    c_in=my_c_in,
                    c_mid=channels[i],
                    c_out=channels[i],
                    downsample=i > skip_stages and i != len(depths) - 1,
                    learned_downsample=learned_subsampling,
                    self_attn=self_attn_depths[i],
                    cross_attn=cross_attn_depths[i],
                    c_enc=cross_cond_dim,
                    dropout_rate=dropout_rate,
                    attn_heads=n_heads,
                )
            )
        mid_block = None
        if use_mid_block:
            mid_block = BottleneckBlock(
                feats_in,
                channels[-1],
                self_attn=True,
                cross_attn=True,
                c_enc=cross_cond_dim,
                dropout_rate=dropout_rate,
                attn_heads=n_heads,
            )
        for i in range(len(depths)):
            my_c_in = channels[i] * 2 if i < len(depths) - 1 else channels[i]
            my_c_out = channels[max(0, i - 1)]
            u_blocks.append(
                UBlock(
                    depths[i] + 1 if up_down_asymmetry else depths[i],
                    feats_in,
                    c_in=my_c_in,
                    c_mid=channels[i],
                    c_out=my_c_out,
                    upsample=i > skip_stages and i != len(depths) - 1,
                    learned_upsample=learned_subsampling,
                    self_attn=self_attn_depths[i],
                    cross_attn=cross_attn_depths[i],
                    c_enc=cross_cond_dim,
                    dropout_rate=dropout_rate,
                    attn_heads=n_heads,
                )
            )
        self.u_net = layers.UNet(
            d_blocks,
            reversed(u_blocks),
            mid_block=mid_block,
            skip_stages=skip_stages,
        )

    def param_groups(self, base_lr=2e-4):
        wd_names = []
        for name, _ in self.named_parameters():
            if name.startswith("mapping") or name.startswith("u_net"):
                if name.endswith(".weight"):
                    wd_names.append(name)
        wd, no_wd = [], []
        for name, param in self.named_parameters():
            if name in wd_names:
                wd.append(param)
            else:
                no_wd.append(param)
        groups = [
            {"params": wd, "lr": base_lr},
            {"params": no_wd, "lr": base_lr, "weight_decay": 0.0},
        ]
        return groups

    def forward(self, input, sigma, mapping_cond=None, unet_cond=None, cross_cond=None, cross_cond_padding=None, return_variance=False):
        c_noise = sigma.log() / 4
        timestep_embed = self.timestep_embed(utils.append_dims(c_noise, 2))
        mapping_cond_embed = torch.zeros_like(timestep_embed) if mapping_cond is None else self.mapping_cond(mapping_cond)
        mapping_out = self.mapping(timestep_embed + mapping_cond_embed)
        cond = {'cond': mapping_out}
        if unet_cond is not None:
            input = torch.cat([input, unet_cond], dim=1)
        if cross_cond is not None:
            cond['cross'] = cross_cond
            cond['cross_padding'] = cross_cond_padding
        if self.patch_size > 1:
            input = F.pixel_unshuffle(input, self.patch_size)
        input = self.proj_in(input)
        input = self.u_net(input, cond)
        input = self.proj_out(input)
        if self.has_variance:
            input, logvar = input[:, :-1], input[:, -1].flatten(1).mean(1)
        if self.patch_size > 1:
            input = F.pixel_shuffle(input, self.patch_size)
        if self.has_variance and return_variance:
            return input, logvar
        return input

    def set_skip_stages(self, skip_stages):
        self.proj_in = nn.Conv2d(self.proj_in.in_channels, self.channels[max(0, skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(self.channels[max(0, skip_stages - 1)], self.proj_out.out_channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.u_net.skip_stages = skip_stages
        for i, block in enumerate(self.u_net.d_blocks):
            block.set_downsample(i > skip_stages)
        for i, block in enumerate(reversed(self.u_net.u_blocks)):
            block.set_upsample(i > skip_stages)
        return self

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        self.proj_in = nn.Conv2d((self.c_in + self.unet_cond_dim) * self.patch_size ** 2, self.channels[max(0, self.u_net.skip_stages - 1)], 1)
        self.proj_out = nn.Conv2d(self.channels[max(0, self.u_net.skip_stages - 1)], self.c_in * self.patch_size ** 2 + (1 if self.has_variance else 0), 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
