from functools import partial
import os
import json
import math
from pathlib import Path
from typing import Callable

from jsonmerge import merge
import torch
from torch.distributions import Normal

from . import models, utils, external2
from attribute_disentanglement.datasets import named_data
from attribute_disentanglement.vqvae.VQVAE import VQVAE, VQVAEWrapper


def load_config(path_or_dict):
    defaults_image_v3 = {
        'model': {
            'patch_size': 1,
            'mapping_cond_dim': 0,
            'unet_cond_dim': 0,
            'cross_cond_dim': 0,
            'cross_attn_depths': None,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.95, 0.999],
            'eps': 1e-6,
            'weight_decay': 1e-3,
        },
    }
    defaults = {
        'model': {
            'sigma_data': 1.,
            'dropout_rate': 0.,
            'loss_config': 'vdmpp_v',
        },
        'dataset': {
            'type': 'shapes3d',
            'num_classes': 0,
            'cond_dropout_rate': 0.1,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 1e-4,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
        'lr_sched': {
            'type': 'constant',
            'warmup': 0.,
        },
        'ema_sched': {
            'type': 'inverse',
            'power': 0.6667,
            'max_value': 0.9999
        },
    }
    if not isinstance(path_or_dict, dict):
        file = Path(path_or_dict)
        if file.suffix == '.safetensors':
            metadata = utils.get_safetensors_metadata(file)
            config = json.loads(metadata['config'])
        else:
            config = json.loads(file.read_text())
    else:
        config = path_or_dict
    if config['model']['type'] == 'image_v3':
        config = merge(defaults_image_v3, config)
    config = merge(defaults, config)
    supported_models = {'image_v3', 'image_v4', 'image_v5'}
    if config['model']['type'] not in supported_models:
        raise ValueError(
            f"Unsupported model {config['model']['type']!r}; "
            f"choose one of {sorted(supported_models)}"
        )
    supported_datasets = {'shapes3d', 'cars3d', 'mpi3d'}
    if config['dataset']['type'] not in supported_datasets:
        raise ValueError(
            f"Unsupported dataset {config['dataset']['type']!r}; "
            f"choose one of {sorted(supported_datasets)}"
        )
    return config


def make_model(config):
    config = config['model']
    if config['type'] == 'image_v3':
        model = models.ImageDenoiserModelV3(
            config['input_channels'],
            config['mapping_out'],
            config['depths'],
            config['channels'],
            config['self_attn_depths'],
            config['cross_attn_depths'],
            patch_size=config['patch_size'],
            dropout_rate=config['dropout_rate'],
            mapping_cond_dim=config['mapping_cond_dim'],
            unet_cond_dim=config['unet_cond_dim'],
            cross_cond_dim=config['cross_cond_dim'],
            skip_stages=config['skip_stages'],
            has_variance=config['has_variance'],
            learned_subsampling=config['learned_subsampling'],
            up_down_asymmetry=config['up_down_asymmetry'],
            use_mid_block=config['use_mid_block'],
            rezero=config.get('rezero', False),
            init_res_scale=config.get('init_res_scale', 1.0),
        )
    elif config['type'] == 'image_v4':
        model = models.ImageDenoiserModelV4(
            input_size=config['input_size'],
            in_channels=config['input_channels'],
            out_channels=config['out_channels'],
            model_channels=config['model_channels'],
            num_res_blocks=config['num_res_blocks'],
            attention_resolutions=config['attention_resolutions'],
            channel_mult=config['channel_mult'],
            dropout=config['dropout'],
            num_heads=config.get('num_heads', None),
            num_head_channels=config.get('num_head_channels', None),
            use_scale_shift_norm=config['use_scale_shift_norm'],
            resblock_updown=config.get('resblock_updown', True),
            conv_resample=config.get('conv_resample', False),
            transformer_depth=config['transformer_depth'],
            latent_unit=config['latent_units'],
            context_dim=config['context_dim'],
            norm_style=config.get('norm_style', False),
            norm_ctx=config.get('norm_ctx', False),
            skip_scale=config.get('skip_scale', 'default'),
            kappa=config.get('kappa', 1.0),
            skip_dropout_rate=config.get('skip_dropout_rate', 0),
            skip_drop_resolutions=config.get('skip_drop_resolutions', []),
            trans_residual_init=config.get('trans_residual_init', None),
            trans_residual_constraint=config.get('trans_residual_constraint', 'identity'),
            trans_residual_scale=config.get('trans_residual_scale', 1.),
        )
    elif config['type'] == 'image_v5':
        model = models.ImageDenoiserModelV5(
            in_channels=config['input_channels'],
            out_channels=config['out_channels'],
            model_channels=config['model_channels'],
            num_res_blocks=config['num_res_blocks'],
            attention_resolutions=config['attention_resolutions'],
            channel_mult=config['channel_mult'],
            dropout=config['dropout'],
            num_heads=config.get('num_heads', None),
            num_head_channels=config.get('num_head_channels', None),
            use_scale_shift_norm=config['use_scale_shift_norm'],
            resblock_updown=config.get('resblock_updown', True),
            conv_resample=config.get('conv_resample', False),
            transformer_depth=config['transformer_depth'],
            latent_unit=config['latent_units'],
            context_dim=config['context_dim'],
            norm_style=config.get('norm_style', False),
            norm_ctx=config.get('norm_ctx', False),
            skip_residual_init=config.get('skip_residual_init', None),
            skip_residual_constraint=config.get('skip_residual_constraint', 'identity'),
            skip_residual_scale=config.get('skip_residual_scale', 1.),
            trans_residual_init=config.get('trans_residual_init', None),
            trans_residual_constraint=config.get('trans_residual_constraint', 'identity'),
            trans_residual_scale=config.get('trans_residual_scale', 1.),
        )
    else:
        raise ValueError(f'unsupported model type {config["type"]}')
    return model


def make_encoder(config):
    internal_dim = config['encoder']['internal_dim']
    context_dim = config['encoder']['context_dim']

    if config['encoder']['type'] == 'image':
        latent_units = config['encoder']['latent_units']
        return models.ImageEncoder(
            d=internal_dim,
            context_dim=context_dim,
            latent_unit=latent_units,
            legacy_resblocks=config['encoder'].get('legacy_resblocks', False),
        )
    elif config['encoder']['type'] == 'image_improved':
        latent_units = config['encoder']['latent_units']
        return models.ImprovedImageEncoder(
            d=internal_dim,
            context_dim=context_dim,
            latent_unit=latent_units,
            norm=config['encoder'].get('norm', 'bn'),
            resnet_updown=config['encoder'].get('resnet_updown', False),
            res_blocks=config['encoder'].get('res_blocks', 2),
            rezero=config['encoder'].get('rezero', False),
            rezero_init=config['encoder'].get('rezero_init', 0.0)
        )
    elif config['encoder']['type'] == 'image_improved2':
        latent_units = config['encoder']['latent_units']
        return models.ImprovedImageEncoder2(
            d=internal_dim,
            context_dim=context_dim,
            latent_unit=latent_units,
            res_blocks=config['encoder'].get('res_blocks', 2),
            tokeniser=config['encoder'].get('tokeniser', 'mlp'),
            input_layer=config['encoder'].get('input_layer', 'resblock'),
            flatten=config['encoder'].get('flatten', 'linear'),
        )
    elif config['encoder']['type'] == 'label':
        dataset_obj = named_data.get_named_ground_truth_data(config['dataset']['type'])
        return models.LabelEncoder(
            d=internal_dim,
            context_dim=context_dim,
            factors_num_values=dataset_obj.factors_num_values,
            factor_types_values=dataset_obj.factor_types_values,
        )
    else:
        raise ValueError(f'unsupported encoder type {config["type"]}')


VQVAE_CKPT_PATHS = {
    'shapes3d': 'ckpts/vqvae/0_shapes3d_True_32_128_5e-05_new/training-state-150000.pth',
    'mpi3d': 'ckpts/vqvae/0_mpi3d_True_32_128_5e-05_new/training-state-150000.pth',
    'cars3d': 'ckpts/vqvae/0_cars3d_True_32_128_5e-05_new/training-state-150000.pth',
}

def make_vqvae(config):
    dataset = config['dataset']['type']
    if dataset not in VQVAE_CKPT_PATHS:
        raise ValueError(f'Unsupported dataset for attribute disentanglement: {dataset}')
    # Allow experiments to pin their VQ-VAE in the copied JSON config.  Falling
    # back to the global mapping preserves existing configurations.
    path = config['dataset'].get('vqvae_checkpoint')
    if path is None:
        path = VQVAE_CKPT_PATHS[dataset]
    print(f"VQ-VAE checkpoint: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    vqvae = VQVAE(
        enc_dec_dict=dict(
            resolution=64,
            in_channels=3,
            z_channels=3,
            ch=32,
            ch_mult=[1, 2, 4],
            num_res_blocks=2,
            attn_resolutions=[],
            out_ch=3,
            dropout=0.0,
        ),
        vq_dict=dict(n_embed=2048, embed_dim=3),
    )
    vqvae.load_state_dict(ckpt, strict=True)
    vqvae.eval().requires_grad_(False)
    if 'vqvae_scale_factor' in config['dataset']:
        if config['dataset']['vqvae_scale_factor']:
            scale_factor_path = config['dataset'].get(
                'vqvae_scale_factor_path',
                os.path.join(os.path.dirname(path), "std.pt"),
            )
            scale_factor = torch.load(scale_factor_path)
        else:
            scale_factor = torch.tensor(1.0)
    else:
        scale_factor = torch.tensor(1.0)
    print(f"Scale Factor: {scale_factor}")
    vqvae = VQVAEWrapper(vqvae, scale_factor)
    return vqvae

def make_denoiser_wrapper(config):
    config = config['model']
    has_variance = config.get('has_variance', False)
    loss_config = config.get('loss_config', 'vdmpp_v')
    if loss_config.startswith('vdmpp_'):
        if has_variance:
            raise ValueError('VDMpp config does not support a variance output')
        if loss_config == 'vdmpp_eps':
            return external2.EpsPredictionDenoiser
        elif loss_config == 'vdmpp_x':
            return external2.XPredictionDenoiser
        elif loss_config == 'vdmpp_v':
            return external2.VPredictionDenoiser
    raise ValueError('Unknown loss config type')


def make_sample_density(config):
    sd_config = config['sigma_sample_density']
    sigma_data = config['sigma_data']
    if sd_config['type'] == 'lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale = sd_config['std'] if 'std' in sd_config else sd_config['scale']
        return partial(utils.rand_log_normal, loc=loc, scale=scale)
    if sd_config['type'] == 'loglogistic':
        loc = sd_config['loc'] if 'loc' in sd_config else math.log(sigma_data)
        scale = sd_config['scale'] if 'scale' in sd_config else 0.5
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 0.
        max_value = sd_config['max_value'] if 'max_value' in sd_config else float('inf')
        return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'loguniform':
        min_value = sd_config['min_value'] if 'min_value' in sd_config else config['sigma_min']
        max_value = sd_config['max_value'] if 'max_value' in sd_config else config['sigma_max']
        return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)
    if sd_config['type'] in {'v-diffusion', 'cosine'}:
        min_value = sd_config['min_value'] if 'min_value' in sd_config else 1e-3
        max_value = sd_config['max_value'] if 'max_value' in sd_config else 1e3
        return partial(utils.rand_v_diffusion, sigma_data=sigma_data, min_value=min_value, max_value=max_value)
    if sd_config['type'] == 'split-lognormal':
        loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
        scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
        scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
        return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
    if sd_config['type'] == 'cosine-interpolated':
        min_value = sd_config.get('min_value', min(config['sigma_min'], 1e-3))
        max_value = sd_config.get('max_value', max(config['sigma_max'], 1e3))
        image_d = sd_config.get('image_d', max(config['input_size']))
        noise_d_low = sd_config.get('noise_d_low', 32)
        noise_d_high = sd_config.get('noise_d_high', max(config['input_size']))
        return partial(
            utils.rand_cosine_interpolated,
            image_d=image_d,
            noise_d_low=noise_d_low,
            noise_d_high=noise_d_high,
            sigma_data=sigma_data,
            min_value=min_value,
            max_value=max_value,
        )

    raise ValueError('Unknown sample density type')


def make_lambda_noise_schedule(config):
    ns_config = config['lambda_noise_schedule']
    shift = ns_config.get('shift', 0)
    lambda_min = config.get('lambda_min', -15)
    lambda_max = config.get('lambda_max', 15)
    lambda_min_start = config.get('lambda_min_start', -15)
    lambda_max_start = config.get('lambda_max_start', 15)
    widening_steps = config.get('widening_steps', 0)
    mu = ns_config.get('mu', 2.4)
    sigma = ns_config.get('sigma', 2.4)

    schedules = {
        'cosine': external2.CosineSchedule,
        'edm': partial(external2.EDMTrainingSchedule, mu=mu, sigma=sigma),
    }
    return schedules[ns_config['type']](
        shift=shift,
        logsnr_min=lambda_min,
        logsnr_max=lambda_max,
        logsnr_min_start=lambda_min_start,
        logsnr_max_start=lambda_max_start,
        widening_steps=widening_steps,
    )


class DynamicBandWeighting:
    """
    Wraps a base weighting function w_base(lam) and applies
    a time-dependent band gate in λ-space for the first `max_steps` calls.

    After `max_steps` calls, it returns w_base(lam) unchanged.

    The band starts as [dynamic_min, dynamic_max] and expands
    linearly to cover [lam_min, lam_max].
    """

    def __init__(
        self,
        base_w_fn: Callable[[torch.Tensor], torch.Tensor],
        lam_min: float,
        lam_max: float,
        dynamic_min: float,
        dynamic_max: float,
        max_steps: int,
        sharpness: float = 10.0,
    ):
        self.base_w_fn = base_w_fn
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.dynamic_min = float(dynamic_min)
        self.dynamic_max = float(dynamic_max)
        self.max_steps = int(max_steps)
        self.sharpness = float(sharpness)

        # initial band centre + radius
        self.mu = 0.5 * (self.dynamic_min + self.dynamic_max)
        self.r_min = 0.5 * (self.dynamic_max - self.dynamic_min)
        self.r_max = 0.5 * (self.lam_max - self.lam_min)

        # internal step counter
        self.step = 0

    def _progress(self) -> float:
        if self.max_steps <= 0:
            return 1.0
        return min(self.step / float(self.max_steps), 1.0)

    def _current_radius(self) -> float:
        p = self._progress()
        return self.r_min + p * (self.r_max - self.r_min)

    def _soft_band_gate(self, lam: torch.Tensor) -> torch.Tensor:
        """
        Soft gate in λ-space. ≈1 in current band, ≈0 outside.
        """
        radius = self._current_radius()
        if radius <= 0:
            # degenerate initial case: very tiny band
            radius = 1e-6

        # keep everything in lam's dtype/device
        mu = lam.new_tensor(self.mu)
        left = mu - radius
        right = mu + radius
        sharp = self.sharpness

        left_gate = torch.sigmoid(sharp * (lam - left))
        right_gate = torch.sigmoid(sharp * (right - lam))
        return left_gate * right_gate

    def __call__(self, lam: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic band weighting to lam.

        - For step < max_steps: w(lam) = gate(λ; step) * w_base(lam)
        - For step >= max_steps: w(lam) = w_base(lam)
        """
        w_base = self.base_w_fn(lam)

        # After curriculum is over, return base weighting
        if self.step >= self.max_steps:
            return w_base

        gate = self._soft_band_gate(lam)
        # increment step *after* using current gate
        self.step += 1

        return gate * w_base


def make_weighting_function(config):
    w_config = config['weighting']
    w_type = w_config['type']
    shift = float(w_config.get('shift', 0.0))
    gamma = float(w_config.get('gamma', 0.5))

    # Optional lambda range for grid-based peak finding
    lam_min = float(w_config.get('lam_min', -15.0))
    lam_max = float(w_config.get('lam_max', 15.0))

    sigma_data = float(w_config.get('sigma_data', 0.5))
    edm_mu = float(w_config.get('edm_mu', 2.4))
    edm_sigma = float(w_config.get('edm_sigma', 2.4))

    dynamic = bool(w_config.get('dynamic', False))
    dynamic_min = float(w_config.get('dynamic_min', -3.0))
    dynamic_max = float(w_config.get('dynamic_max', 2.0))
    dynamic_steps = int(w_config.get('dynamic_steps', 80000))
    dynamic_sharpness = float(w_config.get('dynamic_sharpness', 10.0))

    def w_edm_fn(lam: torch.Tensor) -> torch.Tensor:
        """
        EDM-style weighting in λ-space:
          w(λ) = N(λ; μ, σ^2) * (exp(-λ) + σ_data^2)
        """
        lam = lam.to(torch.float32)
        mu = torch.tensor(edm_mu, device=lam.device, dtype=lam.dtype)
        sigma = torch.tensor(edm_sigma, device=lam.device, dtype=lam.dtype)
        dist = Normal(mu, sigma)
        p_lam = dist.log_prob(lam).exp()
        return p_lam * (torch.exp(-lam) + sigma_data ** 2)

    # ----------------- pick base weighting -----------------

    if w_type == 'elbo':
        def w_fn(lam):
            return torch.ones_like(lam)

    elif w_type == 'exp':
        def w_fn(lam):
            lam_shifted = lam + shift
            return torch.exp(-lam_shifted / 2.0)

    elif w_type == 'sech':
        def w_fn(lam):
            lam_shifted = lam + shift
            return 1.0 / torch.cosh(lam_shifted / 2.0)

    elif w_type == 'min_snr':
        def w_fn(lam):
            lam_shifted = lam + shift
            gate = torch.clamp(gamma * torch.exp(-lam_shifted), max=1.0)
            return (1.0 / torch.cosh(lam_shifted / 2.0)) * gate

    elif w_type == 'edm':
        def w_fn(lam):
            lam_shifted = lam + shift
            return w_edm_fn(lam_shifted)

    elif w_type == 'edm-monotonic':
        # Precompute peak of shifted EDM weighting on a grid
        grid = torch.linspace(lam_min, lam_max, 50000)
        with torch.no_grad():
            vals = w_edm_fn(grid + shift)
            idx = torch.argmax(vals)
            lam_peak = grid[idx]                  # λ* in *unshifted* lam-space
            w_peak = vals[idx]                    # w(λ*)

        def w_fn(lam):
            lam_shifted = lam + shift
            w = w_edm_fn(lam_shifted)
            # Clamp to w_peak for lam_shifted < lam_peak
            return torch.where(
                lam_shifted < lam_peak.to(lam.device, lam.dtype),
                w_peak.to(lam.device, lam.dtype),
                w,
            )

    elif w_type == 'sigmoid':
        # w(λ) = sigmoid(−λ + shift)
        def w_fn(lam):
            lam_shifted = -lam + shift
            return torch.sigmoid(lam_shifted)

    elif w_type in ('laplace', 'laplace-monotonic'):
        raise NotImplementedError

    else:
        raise ValueError(f'Unknown weighting function type: {w_type}')

    # ----------------- optional dynamic wrapper -----------------

    if dynamic:
        w_fn = DynamicBandWeighting(
            base_w_fn=w_fn,
            lam_min=lam_min,
            lam_max=lam_max,
            dynamic_min=dynamic_min,
            dynamic_max=dynamic_max,
            max_steps=dynamic_steps,
            sharpness=dynamic_sharpness,
        )

    return w_fn
