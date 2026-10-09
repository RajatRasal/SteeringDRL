import torch
from torch import nn
from torch.nn import functional as F

from . import layers
from .. import utils
from .blocks import DBlock, UBlock, BottleneckBlock, MappingNet


class ImageDenoiserModelV3(nn.Module):
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
        rezero=False,
        init_res_scale=1.0,
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
                    rezero=rezero,
                    init_res_scale=init_res_scale,
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
                rezero=rezero,
                init_res_scale=init_res_scale,
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
                    rezero=rezero,
                    init_res_scale=init_res_scale,
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

    def forward(self, input, logsnr, mapping_cond=None, unet_cond=None, cross_cond=None, cross_cond_padding=None, return_variance=False):
        timestep_embed = self.timestep_embed(utils.append_dims(logsnr, 2))
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
