import torch as th
import torch.nn as nn

from k_diffusion.models.image_v2 import MappingNet
from k_diffusion import utils
from k_diffusion.layers import FourierFeatures
from k_diffusion.models.image_v4 import (
    SpatialTransformer,
    ResBlock,
    TimestepEmbedSequential,
    normalization,
    zero_module,
    conv_nd,
)


class Erf(nn.Module):

    def forward(self, x):
        return th.special.erf(x)


class Atan(nn.Module):

    def forward(self, x):
        return th.atan(x)


class ImageDenoiserModelV5(nn.Module):
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
        transformer_depth=1,
        context_dim=None,
        norm_style=None,
        norm_ctx=False,
        skip_scale="default",
        kappa=0.5,
        skip_residual_init=None,
        skip_residual_constraint="identity",
        skip_residual_scale=1.,
        transformer_learned_residual=False,
        transformer_residual_scale=1.,
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
        self.skip_residual_init = skip_residual_init

        time_embed_dim = model_channels * 4
        self.timestep_embed = FourierFeatures(1, model_channels)
        self.time_embed = MappingNet(model_channels, time_embed_dim, n_layers=2)
        self.context_dim = context_dim
        self.latent_unit = latent_unit

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
                    )
                )
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
                            learned_residual=transformer_learned_residual,
                            residual_scale=transformer_residual_scale,
                        )
                    )
            self.enc_blocks[str(level)] = TimestepEmbedSequential(*layers)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.down_blocks[str(level)] = TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                )
                ch = out_ch
                ds *= 2
            else:
                self.down_blocks[str(level)] = TimestepEmbedSequential(nn.Identity())

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
                learned_residual=transformer_learned_residual,
                residual_scale=transformer_residual_scale,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
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
                    )
                )
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
                            learned_residual=transformer_learned_residual,
                            residual_scale=transformer_residual_scale,
                        )
                    )
            self.dec_blocks[str(level)] = TimestepEmbedSequential(*layers)
            if level:
                out_ch = ch
                self.up_blocks[str(level)] = TimestepEmbedSequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                        up=True,
                    )
                )
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
                self.skip_residuals[str(level)] = nn.Parameter(th.tensor(skip_residual_init))
            self.skip_residual_constraint = {
                "identity": nn.Identity(),
                "tanh": nn.Tanh(),
                "sigmoid": nn.Sigmoid(),
                "softsign": nn.Softsign(),
                "erf": Erf(), 
                "arctan": Atan(),
            }[skip_residual_constraint]
            self.skip_residual_scale = skip_residual_scale

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

        h = self.input_block(h)
        h = self._forward(0, h, emb, context, **kwargs)
        out = self.output_block(h)
        return out

    def _forward(self, lvl, x, emb=None, context=None, **kwargs):
        lvl_key = str(lvl)

        h_d = self.enc_blocks[lvl_key](x, emb, context)
        d = self.down_blocks[lvl_key](h_d, emb, context)

        if lvl == len(self.channel_mult) - 1:
            z = self.middle_block(d, emb, context)
        else:
            z = self._forward(lvl + 1, d, emb, context)

        if self.skip_residual_init is not None:
            skip = self.skip_residual_constraint(self.skip_residuals[lvl_key] / self.skip_residual_scale) * h_d
        else:
            skip = h_d

        h_u = self.dec_blocks[lvl_key](z, emb, context) + skip
        u = self.up_blocks[lvl_key](h_u, emb, context)

        return u


if __name__ == "__main__":
    import k_diffusion as K
    file = "./configs_logsnr2/shapes3d/v_cosine_sigmoid_improved9_skip_res12_resunet.json"
    device = th.device("cuda")
    config = K.config.load_config(file)
    model_config = config["model"]
    size = model_config["input_size"]
    model = K.config.make_model(config).to(device)

    x = th.zeros([1, model_config['input_channels'], size[0] * 4, size[1] * 4], device=device)
    conds = th.ones([1, 10, 16], device=device)
    t = th.ones([1], device=device)
    model(x, t, conds)
        
