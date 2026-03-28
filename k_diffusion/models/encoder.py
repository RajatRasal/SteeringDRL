import math
from typing import List, Literal

import torch
from torch import nn
from torch.nn import functional as F

from k_diffusion.models.image_v4 import normalization, conv_nd, Downsample, zero_module
from k_diffusion.dino import dino
from k_diffusion.resnet import resnet18, resnet34


##################
# Encoder from EncDiff

class EncResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
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
    def __init__(self, d, context_dim, latent_unit, num_channels=3):
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

            EncResBlock(d, d, norm="bn"),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            EncResBlock(d, d, norm="bn"),

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

def get_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim='2d',
):
    """Get Conv layer."""
    return eval(f'nn.Conv{dim}')(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_normalizer(norm, channels, groups=16, dim='2d'):
    """Get normalization layer."""
    if norm == '':
        return nn.Identity()
    elif norm == 'bn':
        return eval(f'nn.BatchNorm{dim}')(channels)
    elif norm == 'gn':
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == 'in':
        return eval(f'nn.InstanceNorm{dim}')(channels)
    elif norm == 'ln':
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f'Normalizer {norm} not supported!')


def get_act_func(act):
    """Get activation function."""
    if act == '':
        return nn.Identity()
    if act == 'relu':
        return nn.ReLU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'sigmoid':
        return nn.Sigmoid()
    elif act == 'swish':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'softplus':
        return nn.Softplus()
    elif act == 'mish':
        return nn.Mish()
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f'Activation function {act} not supported!')


def conv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm='bn',
    act='relu',
    dim='2d',
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ['bn', 'in'],
        dim=dim,
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def assert_shape(actual, expected, message=""):
    assert list(actual) == list(expected), \
        f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer('grid', build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention.

    We return the last attention map from SA as the segmentation mask.
    """
    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
        truncate=None,
        drop_path=0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.truncate = truncate
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.attn_scale = self.slot_size ** -0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            slots (torch.Tensor): [B, num_slots, C] slot inits.
            masks (torch.Tensor): [B, num_slots, N] segmentation mask.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        slots_init = slots

        # Multiple rounds of attention.
        for attn_iter in range(self.num_iterations):
            # BO-QSA paper: https://arxiv.org/abs/2210.08990
            if attn_iter == self.num_iterations - 1:
                if self.truncate == "bi-level":
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == "fixed-point":
                    slots = slots.detach()

            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

            # attn_map normalized along slot-dim is treated as seg_mask
            if attn_iter == self.num_iterations - 1:
                seg_mask = attn.detach().clone().permute(0, 2, 1)

            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.drop_path(self.mlp(slots))

        # [B, num_slots, slot_size], [B, num_slots, num_inputs (H*W)]
        return slots, seg_mask
    
    # @property
    # def dtype(self):
    #     return self.project_k.weight.dtype

    # @property
    # def device(self):
    #     return self.project_k.weight.device


class SlotEncoder(nn.Module):

    def __init__(
            self,
            resolution,
            slot_dict=dict(
                num_slots=7,
                slot_size=128,
                slot_mlp_size=256,
                num_iterations=2,
                eps=1e-6,
                truncate=None,
                drop_path=0,
            ),
            enc_dict=dict(
                enc_channels=(3, 64, 64, 64, 64),
                enc_ks=5,
                enc_out_channels=128,
                enc_norm='',
            ),
    ):
        super().__init__()
        self.resolution = resolution
        self.slot_dict = slot_dict
        self.enc_dict = enc_dict

        self._build_slot_attention()
        self._build_encoder()

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict['enc_out_channels']
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']
        self.slot_mlp_size = self.slot_dict['slot_mlp_size']
        self.num_iterations = self.slot_dict['num_iterations']
        self.eps = self.slot_dict['eps']
        self.truncate = self.slot_dict['truncate']
        self.drop_path = self.slot_dict['drop_path']

        # directly use learnable embeddings for each slot
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size))
        )

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
            truncate=self.truncate,
            drop_path=self.drop_path,
        )

    def _build_encoder(self):
        # Build Encoder
        # ResNet with less downsampling
        if self.enc_dict.get('resnet', False):
            use_layer4 = self.enc_dict['use_layer4']
            self.encoder = eval(self.enc_dict['resnet'])(
                small_inputs=True, use_layer4=use_layer4)
            if use_layer4:  # downsample by 8
                self.visual_resolution = tuple(i // 8 for i in self.resolution)
                self.visual_channels = 512
            else:  # downsample by 4
                self.visual_resolution = tuple(i // 4 for i in self.resolution)
                self.visual_channels = 256
        # using SSL pre-trained DINO encoder
        elif self.enc_dict.get('dino', False):
            print('Using DINO encoder!')
            patch_size = self.enc_dict['patch_size']
            small_size = self.enc_dict['small_size']

            if patch_size == 8:
                self.visual_resolution = tuple(i // 8 for i in self.resolution)
            elif patch_size == 16:
                self.visual_resolution = tuple(i // 16 for i in self.resolution)
            else:
                raise NotImplementedError('Patch size not supported! (8, 16)')
            if small_size:
                self.visual_channels = 384
            else:
                self.visual_channels = 768
            self.encoder = dino(patch_size=patch_size, small_size=small_size)
        # Conv CNN --> PosEnc --> MLP
        else:
            # CNN out visual resolution
            if self.resolution[0] > 64:
                self.visual_resolution = tuple(i // 2 for i in self.resolution)
                downsample = True
            else:
                self.visual_resolution = self.resolution
                downsample = False
            enc_channels = list(self.enc_dict['enc_channels'])  # CNN channels
            self.visual_channels = enc_channels[-1]  # CNN out visual channels

            enc_layers = len(enc_channels) - 1
            self.encoder = nn.Sequential(*[
                conv_norm_act(
                    enc_channels[i],
                    enc_channels[i + 1],
                    kernel_size=self.enc_dict['enc_ks'],
                    # 2x downsampling for 128x128 image
                    stride=2 if (i == 0 and downsample) else 1,
                    norm=self.enc_dict['enc_norm'],
                    act='relu' if i != (enc_layers - 1) else '',
                ) for i in range(enc_layers)
            ])  # relu except for the last layer

        # Build Encoder related modules
        self.encoder_pos_embedding = SoftPositionEmbed(
            self.visual_channels,
            self.visual_resolution,
        )
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.enc_out_channels),
            nn.ReLU(),
            nn.Linear(self.enc_out_channels, self.enc_out_channels),
        )

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        encoder_out = self.encoder(img) # .type(self.dtype)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [B, C, H, W]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [B, C, H*W]
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]
        return encoder_out

    def encode(self, img):
        """Encode from img to slots."""
        B, C, H, W = img.shape

        encoder_out = self._get_encoder_out(img)
        # `encoder_out` has shape: [B, H*W, out_features]

        # init slots
        init_slots = self.init_latents.repeat(B, 1, 1)

        # perform slot attention operation
        slots, masks = self.slot_attention(encoder_out, init_slots)
        masks = masks.unflatten(-1, self.visual_resolution)
        # [B, N, C], [B, N, h, w]

        # resize masks to the original resolution
        if not self.training:  # and self.visual_resolution != self.resolution:
            with torch.no_grad():
                masks = masks.flatten(0, 1).unsqueeze(1)  # [BN, 1, h, w]
                masks = F.interpolate(
                    masks,
                    self.resolution,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1).unflatten(0, (B, self.num_slots))  # [B, N, H, W]

        # [B, N, C], [B, N, H, W]
        return slots, masks

    def forward(self, img):
        return self.encode(img)[0]


if __name__ == "__main__":
    resolution = (224, 224)
    img_ch = 3
    # Slot Attention
    slot_size = 256
    slot_dict = dict(
        # DINOSAUR uses 7 slots on COCO
        num_slots=7,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 2,
        num_iterations=3,
        eps=1e-6,
        truncate=None,
        drop_path=0,
    )
    # DINO encoder
    enc_dict = dict(
        dino='dino-vits8',
        enc_out_channels=slot_size,
        patch_size=8,  # downsample img by 8 or 16
        # True uses small sized dino (channel = 384)
        # False uses large sized dino (channel = 768)
        small_size=True,
        resolution=resolution,
    )

    encoder = SlotEncoder(
        resolution=resolution,
        enc_dict=enc_dict,
        slot_dict=slot_dict,
    )
    slots, masks = encoder.encode(torch.ones(10, 3, 224, 224))
    print(slots.shape, masks.shape)

    encoder.eval()
    slots, masks = encoder.encode(torch.ones(10, 3, 224, 224))
    print(slots.shape, masks.shape)