from functools import partial
import json
import math
import pickle
from pathlib import Path

from jsonmerge import merge
import torch
from torch.distributions import Normal

from . import augmentation, layers, models, utils, external, external2
from data.ground_truth import named_data


def round_to_power_of_two(x, tol):
    approxs = []
    for i in range(math.ceil(math.log2(x))):
        mult = 2**i
        approxs.append(round(x / mult) * mult)
    for approx in reversed(approxs):
        error = abs((approx - x) / x)
        if error <= tol:
            return approx
    return approxs[0]


def load_config(path_or_dict):
    defaults_image_v1 = {
        'model': {
            'patch_size': 1,
            'augment_wrapper': True,
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
    defaults_image_transformer_v1 = {
        'model': {
            'd_ff': 0,
            'augment_wrapper': False,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-4,
            'betas': [0.9, 0.99],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
    }
    defaults_image_transformer_v2 = {
        'model': {
            'mapping_width': 256,
            'mapping_depth': 2,
            'mapping_d_ff': None,
            'mapping_cond_dim': 0,
            'mapping_dropout_rate': 0.,
            'd_ffs': None,
            'self_attns': None,
            'dropout_rate': None,
            'augment_wrapper': False,
            'skip_stages': 0,
            'has_variance': False,
        },
        'optimizer': {
            'type': 'adamw',
            'lr': 5e-4,
            'betas': [0.9, 0.99],
            'eps': 1e-8,
            'weight_decay': 1e-4,
        },
    }
    defaults = {
        'model': {
            'sigma_data': 1.,
            'dropout_rate': 0.,
            'augment_prob': 0.,
            'loss_config': 'karras',
            'loss_weighting': 'karras',
            'loss_scales': 1,
        },
        'dataset': {
            'type': 'imagefolder',
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
    if config['model']['type'] in ['image_v1', 'image_v2', 'image_v3']:
        config = merge(defaults_image_v1, config)
    elif config['model']['type'] == 'image_transformer_v1':
        config = merge(defaults_image_transformer_v1, config)
        if not config['model']['d_ff']:
            config['model']['d_ff'] = round_to_power_of_two(config['model']['width'] * 8 / 3, tol=0.05)
    elif config['model']['type'] == 'image_transformer_v2':
        config = merge(defaults_image_transformer_v2, config)
        if not config['model']['mapping_d_ff']:
            config['model']['mapping_d_ff'] = config['model']['mapping_width'] * 3
        if not config['model']['d_ffs']:
            d_ffs = []
            for width in config['model']['widths']:
                d_ffs.append(width * 3)
            config['model']['d_ffs'] = d_ffs
        if not config['model']['self_attns']:
            self_attns = []
            default_neighborhood = {"type": "neighborhood", "d_head": 64, "kernel_size": 7}
            default_global = {"type": "global", "d_head": 64}
            for i in range(len(config['model']['widths'])):
                self_attns.append(default_neighborhood if i < len(config['model']['widths']) - 1 else default_global)
            config['model']['self_attns'] = self_attns
        if config['model']['dropout_rate'] is None:
            config['model']['dropout_rate'] = [0.0] * len(config['model']['widths'])
        elif isinstance(config['model']['dropout_rate'], float):
            config['model']['dropout_rate'] = [config['model']['dropout_rate']] * len(config['model']['widths'])
    return merge(defaults, config)


def make_model(config):
    dataset_config = config['dataset']
    num_classes = dataset_config['num_classes']
    config = config['model']
    if config['type'] == 'image_v1':
        model = models.ImageDenoiserModelV1(
            config['input_channels'],
            config['mapping_out'],
            config['depths'],
            config['channels'],
            config['self_attn_depths'],
            config['cross_attn_depths'],
            patch_size=config['patch_size'],
            dropout_rate=config['dropout_rate'],
            mapping_cond_dim=config['mapping_cond_dim'] + (9 if config['augment_wrapper'] else 0),
            unet_cond_dim=config['unet_cond_dim'],
            cross_cond_dim=config['cross_cond_dim'],
            skip_stages=config['skip_stages'],
            has_variance=config['has_variance'],
        )
        if config['augment_wrapper']:
            model = augmentation.KarrasAugmentWrapper(model)
    elif config['type'] == 'image_v2':
        model = models.ImageDenoiserModelV2(
            config['input_channels'],
            config['mapping_out'],
            config['depths'],
            config['channels'],
            config['self_attn_depths'],
            config['cross_attn_depths'],
            patch_size=config['patch_size'],
            dropout_rate=config['dropout_rate'],
            mapping_cond_dim=config['mapping_cond_dim'] + (9 if config['augment_wrapper'] else 0),
            unet_cond_dim=config['unet_cond_dim'],
            cross_cond_dim=config['cross_cond_dim'],
            skip_stages=config['skip_stages'],
            has_variance=config['has_variance'],
            learned_subsampling=config['learned_subsampling'],
            up_down_asymmetry=config['up_down_asymmetry'],
            use_mid_block=config['use_mid_block'],
        )
        if config['augment_wrapper']:
            model = augmentation.KarrasAugmentWrapper(model)
    elif config['type'] == 'image_v3':
        model = models.ImageDenoiserModelV3(
            config['input_channels'],
            config['mapping_out'],
            config['depths'],
            config['channels'],
            config['self_attn_depths'],
            config['cross_attn_depths'],
            patch_size=config['patch_size'],
            dropout_rate=config['dropout_rate'],
            mapping_cond_dim=config['mapping_cond_dim'] + (9 if config['augment_wrapper'] else 0),
            unet_cond_dim=config['unet_cond_dim'],
            cross_cond_dim=config['cross_cond_dim'],
            skip_stages=config['skip_stages'],
            has_variance=config['has_variance'],
            learned_subsampling=config['learned_subsampling'],
            up_down_asymmetry=config['up_down_asymmetry'],
            use_mid_block=config['use_mid_block'],
        )
        if config['augment_wrapper']:
            model = augmentation.KarrasAugmentWrapper(model)
    elif config['type'] == 'image_transformer_v1':
        model = models.ImageTransformerDenoiserModelV1(
            n_layers=config['depth'],
            d_model=config['width'],
            d_ff=config['d_ff'],
            in_features=config['input_channels'],
            out_features=config['input_channels'],
            patch_size=config['patch_size'],
            num_classes=num_classes + 1 if num_classes else 0,
            dropout=config['dropout_rate'],
            sigma_data=config['sigma_data'],
        )
    elif config['type'] == 'image_transformer_v2':
        assert len(config['widths']) == len(config['depths'])
        assert len(config['widths']) == len(config['d_ffs'])
        assert len(config['widths']) == len(config['self_attns'])
        assert len(config['widths']) == len(config['dropout_rate'])
        levels = []
        for depth, width, d_ff, self_attn, dropout in zip(config['depths'], config['widths'], config['d_ffs'], config['self_attns'], config['dropout_rate']):
            if self_attn['type'] == 'global':
                self_attn = models.image_transformer_v2.GlobalAttentionSpec(self_attn.get('d_head', 64))
            elif self_attn['type'] == 'neighborhood':
                self_attn = models.image_transformer_v2.NeighborhoodAttentionSpec(self_attn.get('d_head', 64), self_attn.get('kernel_size', 7))
            elif self_attn['type'] == 'shifted-window':
                self_attn = models.image_transformer_v2.ShiftedWindowAttentionSpec(self_attn.get('d_head', 64), self_attn['window_size'])
            elif self_attn['type'] == 'none':
                self_attn = models.image_transformer_v2.NoAttentionSpec()
            else:
                raise ValueError(f'unsupported self attention type {self_attn["type"]}')
            levels.append(models.image_transformer_v2.LevelSpec(depth, width, d_ff, self_attn, dropout))
        mapping = models.image_transformer_v2.MappingSpec(config['mapping_depth'], config['mapping_width'], config['mapping_d_ff'], config['mapping_dropout_rate'])
        model = models.ImageTransformerDenoiserModelV2(
            levels=levels,
            mapping=mapping,
            in_channels=config['input_channels'],
            out_channels=config['input_channels'],
            patch_size=config['patch_size'],
            num_classes=num_classes + 1 if num_classes else 0,
            mapping_cond_dim=config['mapping_cond_dim'],
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
    'shapes3d': 'ckpts/vqvae/2_shapes3d_True_32_128_5e-05/training-state-030000.pkl',
    'mpi3d': 'ckpts/vqvae/0_mpi3d_True_32_128_5e-05/training-state-030000.pkl',
    'cars3d': 'ckpts/vqvae/0_cars3d_True_32_128_5e-05/training-state-030000.pkl',
}

def make_vqvae(config):
    dataset = config['dataset']['type']
    vqvae_ckpt_path = VQVAE_CKPT_PATHS[dataset]
    with open(vqvae_ckpt_path, "rb") as f:
        data = pickle.load(f)
    vqvae = data["net"].eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    return vqvae


def make_denoiser_wrapper(config):
    config = config['model']
    sigma_data = config.get('sigma_data', 1.)
    has_variance = config.get('has_variance', False)
    loss_config = config.get('loss_config', 'karras')
    if loss_config.startswith('karras_'):
        weighting = config.get('loss_weighting', 'karras')
        scales = config.get('loss_scales', 1)
        if has_variance:
            return partial(layers.DenoiserWithVariance, sigma_data=sigma_data, weighting=weighting)
        if loss_config == 'karras_x':
            return partial(layers.SimpleLossXDenoiser, sigma_data=sigma_data, weighting=weighting, scales=scales)
        elif loss_config == 'karras_eps':
            return partial(layers.SimpleLossDenoiser, sigma_data=sigma_data, weighting=weighting, scales=scales)
        elif loss_config == 'karras_f':
            return partial(layers.Denoiser, sigma_data=sigma_data, weighting=weighting, scales=scales)
    if loss_config.startswith('vp_'):
        if has_variance:
            raise ValueError('VP config does not support a variance output')
        if loss_config == 'vp_eps':
            return external.ContinuousEpsDDPMDenoiser
        elif loss_config == 'vp_v':
            return external.ContinuousVDDPMDenoiser
        elif loss_config == 'vp_x':
            return external.ContinuousXDDPMDenoiser
    if loss_config.startswith('vdmpp_'):
        if has_variance:
            raise ValueError('VP config does not support a variance output')
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
    if ns_config['type'] == 'cosine':
        ns = external2.CosineSchedule(shift, lambda_min, lambda_max)
    elif ns_config['type'] == 'edm':
        ns = external2.EDMTrainingSchedule(shift, lambda_min, lambda_max)
    else:
        raise ValueError('Unknown lambda noise schedule')
    return ns


def make_weighting_function(config):
    w_config = config['weighting']
    w_type = w_config['type']
    shift = float(w_config.get('shift', 0.0))

    # Optional lambda range for grid-based peak finding
    lam_min = w_config.get('lam_min', -15.0)
    lam_max = w_config.get('lam_max', 15.0)

    sigma_data = float(w_config.get('sigma_data', 0.5))
    edm_mu = float(w_config.get('edm_mu', 2.4))
    edm_sigma = float(w_config.get('edm_sigma', 2.4))

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
            lam_shifted = -lam + shift  # optional; you could also treat 'shift' as the k directly
            return torch.sigmoid(lam_shifted)

    elif w_type == 'laplace':
        raise NotImplementedError

    elif w_type == 'laplace-monotonic':
        raise NotImplementedError

    else:
        raise ValueError(f'Unknown weighting function type: {w_type}')

    return w_fn

