import math
from typing import List, Literal

import torch
from torch import nn

from attribute_disentanglement.diffusion.models.image_v4 import normalization, conv_nd, Downsample, zero_module


##################
# Encoder from EncDiff

class LegacyEncResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        d,
        context_dim,
        latent_unit,
        num_channels=3,
        legacy_resblocks=False,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.latent_unit = latent_unit
        if legacy_resblocks:
            residual_blocks = [
                LegacyEncResBlock(d, d, bn=True),
                LegacyEncResBlock(d, d, bn=True),
            ]
        else:
            residual_blocks = [
                EncResBlock(d, d, norm="bn"),
                EncResBlock(d, d, norm="bn"),
            ]

        self.encoder = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            # 32 -> 16
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            # 16 -> 8
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            # 8 -> 4
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),

            residual_blocks[0],
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            residual_blocks[1],

            nn.Flatten(start_dim=1),                # keeps B (e.g., 512)
            nn.Linear(d * 4 * 4, latent_unit),      # no hardcoding
        )

        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64), nn.ELU(inplace=True),
                nn.Linear(64, 128), nn.ELU(inplace=True),
                nn.Linear(128, context_dim),
                nn.LayerNorm(context_dim),
            ) for _ in range(latent_unit)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Default: orthogonal
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            # Make latent and context outputs slightly smaller scale
            if m.out_features == self.latent_unit:  # final encoder linear
                nn.init.orthogonal_(m.weight, gain=0.1)
            if m.out_features == self.context_dim:  # final per-dim MLP linear
                nn.init.orthogonal_(m.weight, gain=0.1)

    def forward(self, x):
        embed = self.encoding(x)
        return self.split_embedding(embed)

    def warp(self, x):
        outs = [self.net[i](x[:, i].unsqueeze(1)) for i in range(self.latent_unit)]
        return torch.stack(outs, dim=1)   

    def encoding(self, x):
        return self.encoder(x)

    def split_embedding(self, embed):
        outs = [self.net[i](embed[:, i].unsqueeze(1))   # each: [B, context_dim] e.g. [512, 16]
                for i in range(self.latent_unit)]
        return torch.stack(outs, dim=1)         # [B, latent_unit, context_dim] -> [512, 10, 16]


#######################

def norm2d(ch, kind="gn", groups=32):
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    elif kind == "gn":
        g = min(groups, ch)
        while ch % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, ch)
    else:
        raise ValueError(kind)


class EncResBlock(nn.Module):
    def __init__(self, channels, mid_channels=None, norm="gn", rezero=True, rezero_init=0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = channels

        self.norm1 = norm2d(channels, norm)
        self.norm2 = norm2d(mid_channels, norm)
        self.act = nn.SiLU(inplace=True)

        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, channels, kernel_size=3, stride=1, padding=1)

        self.rezero = rezero
        if rezero:
            self.alpha = nn.Parameter(torch.tensor(rezero_init))

        # Identity init: start residual branch at ~0
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        if self.rezero:
            return x + self.alpha * h
        return x + h


class ImprovedImageEncoder(nn.Module):
    def __init__(
        self,
        d,
        context_dim,
        latent_unit,
        norm="gn",
        num_channels=3,
        resnet_updown=False,
        res_blocks=2,
        rezero=False,
        rezero_init=0.0,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.latent_unit = latent_unit

        components = [
            # 64 -> 32
            nn.Identity() if not resnet_updown else EncResBlock(num_channels, num_channels, norm, rezero, rezero_init),
            nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1),
            norm2d(d, norm), nn.SiLU(inplace=True),
            # 32 -> 16
            nn.Identity() if not resnet_updown else EncResBlock(d, d, norm, rezero, rezero_init),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            norm2d(d, norm), nn.SiLU(inplace=True),
            # 16 -> 8
            nn.Identity() if not resnet_updown else EncResBlock(d, d, norm, rezero, rezero_init),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            norm2d(d, norm), nn.SiLU(inplace=True),
            # 8 -> 4
            nn.Identity() if not resnet_updown else EncResBlock(d, d, norm, rezero, rezero_init),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            norm2d(d, norm), nn.SiLU(inplace=True),
        ]

        for _ in range(res_blocks):
            components.append(EncResBlock(d, d, norm=norm, rezero=rezero, rezero_init=rezero_init))

        components.append(nn.Flatten(start_dim=1))  # keeps B (e.g., 512)
        components.append(nn.Linear(d * 4 * 4, latent_unit))

        self.encoder = nn.Sequential(*components)

        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 64), nn.SiLU(inplace=True),
                nn.Linear(64, 128), nn.SiLU(inplace=True),
                nn.Linear(128, context_dim),
                # TODO: Change to RMSNorm
                nn.LayerNorm(context_dim),
            ) for _ in range(latent_unit)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Default: orthogonal
            nn.init.orthogonal_(m.weight)
            nn.init.zeros_(m.bias)
            # Make latent and context outputs slightly smaller scale
            if m.out_features == self.latent_unit:  # final encoder linear
                nn.init.orthogonal_(m.weight, gain=0.1)
            if m.out_features == self.context_dim:  # final per-dim MLP linear
                nn.init.orthogonal_(m.weight, gain=0.1)

    def forward(self, x):
        embed = self.encoding(x)
        return self.split_embedding(embed)

    def warp(self, x):
        outs = [self.net[i](x[:, i].unsqueeze(1)) for i in range(self.latent_unit)]
        return torch.stack(outs, dim=1)   

    def encoding(self, x):
        return self.encoder(x)

    def split_embedding(self, embed):
        outs = [self.net[i](embed[:, i].unsqueeze(1))   # each: [B, context_dim] e.g. [512, 16]
                for i in range(self.latent_unit)]
        return torch.stack(outs, dim=1)         # [B, latent_unit, context_dim] -> [512, 10, 16]


#######################

class EncResBlock2(nn.Module):
    def __init__(self, channels, out_channels=None, dims=2, down=False, preact=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.down = down

        self.act = nn.SiLU()
        self.pre = nn.Sequential(normalization(channels), self.act) if preact else nn.Identity()

        self.in_conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

        if down:
            self.h_down = Downsample(channels, False, dims)
            self.x_down = Downsample(channels, False, dims)
        else:
            self.h_down = self.x_down = nn.Identity()

        self.out_norm = normalization(self.out_channels)
        self.out_conv = zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        self.skip = nn.Identity() if channels == self.out_channels else conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        h = self.pre(x)
        h = self.h_down(h)
        x_d = self.x_down(x)
        h = self.in_conv(h)

        h = self.act(self.out_norm(h))
        h = self.out_conv(h)

        return self.skip(x_d) + h


class ScalarToTokenSplitMLP(nn.Module):

    def __init__(self, latent_unit: int, context_dim: int, h1: int = 64, h2: int = 128):
        super().__init__()
        self.latent_unit = latent_unit

        self.net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, h1), nn.SiLU(inplace=True),
                nn.Linear(h1, h2), nn.SiLU(inplace=True),
                nn.Linear(h2, context_dim),
                nn.RMSNorm(context_dim),
            ) for _ in range(latent_unit)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outs = [self.net[i](z[:, i].unsqueeze(1)) for i in range(self.latent_unit)]
        return torch.stack(outs, dim=1)


class ScalarToTokenFiLM(nn.Module):
    """
    Simple FiLM: shared MLP backbone, per-latent (gamma,beta) modulate hidden layers.

    z:      [B, L]  latent scalars
    tokens: [B, L, D] context tokens (one per latent dim)

    Independence preserved: token_i depends only on z_i and (gamma_i,beta_i).
    """

    def __init__(self, latent_unit: int, context_dim: int, h1: int = 64, h2: int = 128):
        super().__init__()
        self.L = latent_unit
        self.D = context_dim

        # Shared backbone (same for all latent dims)
        self.fc1 = nn.Linear(1, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, context_dim)
        self.act = nn.SiLU(inplace=True)

        # Per-latent FiLM params for each hidden layer
        self.g1 = nn.Parameter(torch.ones(latent_unit, h1))
        self.b1 = nn.Parameter(torch.zeros(latent_unit, h1))
        self.g2 = nn.Parameter(torch.ones(latent_unit, h2))
        self.b2 = nn.Parameter(torch.zeros(latent_unit, h2))

        # Stabilisers
        # self.norm1 = nn.RMSNorm(h1)
        # self.norm2 = nn.RMSNorm(h2)
        self.out_norm = nn.RMSNorm(context_dim)

        # Final projection
        nn.init.orthogonal_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, L]
        returns: [B, L, D]
        """
        assert z.ndim == 2 and z.shape[1] == self.L, f"Expected [B,{self.L}] got {tuple(z.shape)}"

        outs = []
        for i in range(self.L):
            x = z[:, i].unsqueeze(1)  # [B,1]

            h = self.act(self.fc1(x))
            # h = self.norm1(h)
            h = h * self.g1[i] + self.b1[i]

            h = self.act(self.fc2(h))
            # h = self.norm2(h)
            h = h * self.g2[i] + self.b2[i]

            t = self.fc3(h)
            t = self.out_norm(t)

            outs.append(t)

        return torch.stack(outs, dim=1)  # [B,L,D]


class ScalarToTokenAdaFiLM(nn.Module):
    def __init__(self, latent_unit, context_dim, h1=64, h2=128):
        super().__init__()
        self.L = latent_unit

        self.fc1 = nn.Linear(1, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, context_dim)
        self.act = nn.SiLU(inplace=False)

        def make_mod(out_dim):
            m = nn.Sequential(
                nn.SiLU(inplace=False),
                zero_module(nn.Linear(1, 2 * out_dim)),
            )
            return m

        # self.mod1 = make_mod(h1)
        self.mod = make_mod(h2)

        self.out_norm = nn.RMSNorm(context_dim)
        # self.alpha = nn.Parameter(torch.full((latent_unit,), 0.2))

        nn.init.orthogonal_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, z):
        outs = []
        for i in range(self.L):
            z_i = z[:, i].unsqueeze(1)

            h = self.act(self.fc1(z_i))
            h = self.act(self.fc2(h))

            s, b = self.mod(z_i).chunk(2, dim=-1)
            h = h * (1 + s) + b

            t = self.out_norm(self.fc3(h))
            # t = self.alpha[i] * t
            outs.append(t)

        return torch.stack(outs, dim=1)

    
class ScalarToTokenAdaFiLM2(nn.Module):
    def __init__(self, latent_unit, context_dim, h1=64, h2=128):
        super().__init__()
        self.L = latent_unit

        self.fc1 = nn.Linear(1, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, context_dim)
        self.act = nn.SiLU(inplace=False)

        def make_mod(out_dim):
            m = nn.Sequential(
                nn.SiLU(inplace=False),
                zero_module(nn.Linear(1, 2 * out_dim)),
            )
            return m

        # self.mod1 = make_mod(h1)
        self.mod = make_mod(h2)

        self.out_norm = nn.RMSNorm(context_dim)
        # self.alpha = nn.Parameter(torch.full((latent_unit,), 0.2))

        nn.init.orthogonal_(self.fc3.weight, gain=0.1)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, z):
        outs = []
        for i in range(self.L):
            x = z[:, i].unsqueeze(1)

            h = self.act(self.fc1(x))
            h = self.act(self.fc2(h))
            s, b = self.mod(x).chunk(2, dim=-1)
            h = h * (1 + torch.tanh(s)) + torch.tanh(b)

            t = self.out_norm(self.fc3(h))
            # t = self.alpha[i] * t
            outs.append(t)

        return torch.stack(outs, dim=1)


class ImprovedImageEncoder2(nn.Module):
    def __init__(
        self,
        d,
        context_dim,
        latent_unit,
        num_channels=3,
        res_blocks=2,
        tokeniser="mlp",
        input_layer="resblock",   # "conv" or "resblock"
        flatten="linear",         # "linear" or "conv"
    ):
        super().__init__()
        self.context_dim = context_dim
        self.latent_unit = latent_unit

        components = []

        # 64 -> 32
        if input_layer == "conv":
            stem = nn.Sequential(
                conv_nd(2, num_channels, d, kernel_size=3, stride=2, padding=1),
                normalization(d),
                nn.SiLU(),
            )
        elif input_layer == "resblock":
            stem = EncResBlock2(num_channels, d, down=True, preact=False)
        else:
            raise NotImplementedError(input_layer)

        components.append(stem)

        components += [
            EncResBlock2(d, d, down=True),  # 32 -> 16
            EncResBlock2(d, d, down=True),  # 16 -> 8
            EncResBlock2(d, d, down=True),  # 8 -> 4
        ]

        for _ in range(res_blocks):
            components.append(EncResBlock2(d, d, down=False))

        # Head: [B,d,4,4] -> [B,latent_unit]
        if flatten == "linear":
            components.append(nn.Flatten(start_dim=1))
            components.append(nn.Linear(d * 4 * 4, latent_unit))
        elif flatten == "conv":
            components.append(conv_nd(2, d, latent_unit, kernel_size=1, stride=1, padding=0))
            components.append(nn.AdaptiveAvgPool2d(1))  # [B, latent_unit, 1, 1]
            components.append(nn.Flatten(1))            # [B, latent_unit]
        else:
            raise NotImplementedError(flatten)

        self.encoder = nn.Sequential(*components)

        tokenisers = {
            "mlp": ScalarToTokenSplitMLP,
            "film": ScalarToTokenFiLM,
            "adafilm": ScalarToTokenAdaFiLM,
        }
        self.tokeniser = tokenisers[tokeniser](latent_unit, context_dim, 64, 128)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # standard stable init
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            # make the 1x1 latent head gentle at init
            if m.kernel_size == (1, 1) and m.out_channels == self.latent_unit:
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

            if m.out_features == self.latent_unit:
                nn.init.orthogonal_(m.weight, gain=0.1)

            if m.out_features == self.context_dim:
                nn.init.orthogonal_(m.weight, gain=0.1)

    def forward(self, x):
        embed = self.encoding(x)
        return self.split_embedding(embed)

    def encoding(self, x):
        return self.encoder(x)

    def split_embedding(self, embed):
        return self.tokeniser(embed)

######################

class LabelEncoder(nn.Module):
    def __init__(
        self,
        d: int,
        context_dim: int,
        factors_num_values: List[int],
        factor_types_values: List[Literal["categorical", "ordinal", "circular"]],
    ):
        super().__init__()

        assert len(factors_num_values) == len(factor_types_values)
        self.factors_num_values = factors_num_values
        self.factor_types_values = factor_types_values

        # Build per-factor encoders
        self.encoders = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for ftype, nvals in zip(self.factor_types_values, self.factors_num_values):
            if ftype == "categorical":
                self.encoders.append(nn.Embedding(nvals, context_dim))
            elif ftype == "ordinal":
                # map k in [0..n-1] -> scalar in [0,1] -> MLP to d
                self.encoders.append(
                    nn.Sequential(
                        nn.Linear(1, d),
                        nn.SiLU(),
                        nn.Linear(d, context_dim),
                    )
                )
            elif ftype == "circular":
                # map k -> angle θ=2πk/n, use [sin θ, cos θ] -> MLP to d
                self.encoders.append(
                    nn.Sequential(
                        nn.Linear(2, d),
                        nn.SiLU(),
                        nn.Linear(d, context_dim),
                    )
                )
            else:
                raise ValueError(f"Unknown factor type: {ftype}")
            self.layer_norms.append(nn.LayerNorm(context_dim))

    def forward(self, factors):
        assert factors.dim() == 2 and factors.size(1) == len(self.factors_num_values)

        tokens = []
        for idx, (ftype, nvals) in enumerate(zip(self.factor_types_values, self.factors_num_values)):
            k = factors[:, idx]
            if ftype == "categorical":
                h = self.encoders[idx](k.long())  # (B, d)
            elif ftype == "ordinal":
                x = (k.float() / max(nvals - 1, 1)).unsqueeze(-1)  # (B,1)
                h = self.encoders[idx](x)  # (B, d)
            elif ftype == "circular":
                theta = (2 * math.pi * k.float()) / nvals
                sc = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)  # (B,2)
                h = self.encoders[idx](sc)  # (B, d)
            h = self.layer_norms[idx](h)
            tokens.append(h)

        H = torch.stack(tokens, dim=1)  # (B, F, d)
        return H  # feed as context sequence into cross-attn

######################
