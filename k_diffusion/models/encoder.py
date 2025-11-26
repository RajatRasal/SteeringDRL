import math
from typing import List, Literal

import torch
from torch import nn


class EncResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False, init_mode="kaiming_normal"):
        super(EncResBlock, self).__init__()

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
    def __init__(self, d, context_dim, latent_unit, bn=True, num_channels=3, init_mode="kaiming_normal"):
        super().__init__()
        self.context_dim = context_dim
        self.latent_unit = latent_unit

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

            EncResBlock(d, d, bn=bn),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            EncResBlock(d, d, bn=bn),

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
