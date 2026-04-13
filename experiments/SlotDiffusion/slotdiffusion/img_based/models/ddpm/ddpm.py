"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import numpy as np
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from einops import rearrange, repeat

from .continuous_ddim import ContinuousDDIMSampler
from .ddim import DDIMSampler
from .ema import LitEma
from .utils import default, is_list_tuple, make_beta_schedule, extract_to, \
    noise_like, append_dims, logsnr_schedule_cosine, \
    inverse_logsnr_schedule_cosine, sigmoid_loss_weight
from ..unet import build_unet


class DDPM(nn.Module):
    """Classic DDPM with Gaussian diffusion, in image space."""

    def __init__(
            self,
            resolution,
            unet_dict,
            use_ema=True,
            diffusion_dict=dict(
                pred_target='eps',  # 'eps' or 'x0', predict noise or direct x0
                timesteps=1000,
                beta_schedule="linear",
                linear_start=1e-4,
                linear_end=2e-2,
                cosine_s=8e-3,
                log_every_t=100,  # log every t steps in denoising sampling
                logvar_init=0.,
            ),
            conditioning_key=None,  # 'concat', 'crossattn'
    ):
        super().__init__()

        self.channels = unet_dict.get('in_channels', 3)  # RGB image
        self.resolution = resolution

        self.clip_denoised = True
        self.log_every_t = diffusion_dict.pop('log_every_t')

        self.model = DiffusionWrapper(unet_dict, conditioning_key)
        # count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # sampling schedules
        self.register_schedule(**diffusion_dict)

        if not conditioning_key and not self.continuous_time:
            assert self.pred_target == 'eps', \
                'should predict noise in unconditional DDPM'

    def register_schedule(
        self,
        pred_target='eps',
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        logvar_init=0.,
        continuous_time=False,
        continuous_schedule='cosine',
        continuous_loss='sigmoid',
        logsnr_min=-15.0,
        logsnr_max=15.0,
        logsnr_min_start=None,
        logsnr_max_start=None,
        widening_steps=0,
        sigmoid_bias=-3.0,
        sampling_timesteps=None,
        continuous_dpm_fallback_steps=50,
    ):
        self.continuous_time = continuous_time
        self.continuous_schedule = continuous_schedule
        self.continuous_loss = continuous_loss
        self.logsnr_min = float(logsnr_min)
        self.logsnr_max = float(logsnr_max)
        self.logsnr_min_start = float(
            logsnr_min if logsnr_min_start is None else logsnr_min_start)
        self.logsnr_max_start = float(
            logsnr_max if logsnr_max_start is None else logsnr_max_start)
        self.widening_steps = int(widening_steps)
        self.sigmoid_bias = float(sigmoid_bias)
        self.sampling_timesteps = sampling_timesteps
        self.continuous_dpm_fallback_steps = int(continuous_dpm_fallback_steps)
        if hasattr(self, '_continuous_step'):
            self._continuous_step.zero_()
        else:
            self.register_buffer(
                '_continuous_step',
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )

        if self.continuous_time:
            assert pred_target in ['eps', 'x0', 'v']
            if self.continuous_schedule != 'cosine':
                raise NotImplementedError(
                    f'Unknown continuous schedule: {self.continuous_schedule}'
                )
            assert self.logsnr_min <= self.logsnr_min_start <= self.logsnr_max_start <= self.logsnr_max
            self.pred_target = pred_target
            self.beta_schedule = beta_schedule
            self.num_timesteps = int(timesteps)
            self.logvar = torch.full(
                fill_value=logvar_init, size=(self.num_timesteps, ))
            self._warned_continuous_dpm_fallback = False
            return

        assert pred_target in ['eps', 'x0']
        self.pred_target = pred_target

        self.beta_schedule = beta_schedule
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_prev = np.append(1., alphas_bar[:-1])
        self.num_timesteps = betas.shape[0]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_bar', to_torch(alphas_bar))
        self.register_buffer('alphas_bar_prev', to_torch(alphas_bar_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', to_torch(np.sqrt(alphas_bar)))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             to_torch(np.sqrt(1. - alphas_bar)))
        self.register_buffer('log_one_minus_alphas_bar',
                             to_torch(np.log(1. - alphas_bar)))
        self.register_buffer('sqrt_recip_alphas_bar',
                             to_torch(np.sqrt(1. / alphas_bar)))
        self.register_buffer('sqrt_recipm1_alphas_bar',
                             to_torch(np.sqrt(1. / alphas_bar - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_bar_prev) / \
            (1. - alphas_bar)
        # above: equal to 1. / (1. / (1. - alpha_bar_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_bar_prev) / (1. - alphas_bar)))
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch(
                (1. - alphas_bar_prev) * np.sqrt(alphas) / (1. - alphas_bar)))

        self.logvar = torch.full(
            fill_value=logvar_init, size=(self.num_timesteps, ))

    def _continuous_logsnr(self, t):
        if self.continuous_schedule == 'cosine':
            return logsnr_schedule_cosine(
                t,
                logsnr_min=self.logsnr_min,
                logsnr_max=self.logsnr_max,
            )
        raise NotImplementedError(
            f'Unknown continuous schedule: {self.continuous_schedule}')

    def _continuous_alpha_sigma(self, logsnr):
        alpha = torch.sqrt(torch.sigmoid(logsnr))
        sigma = torch.sqrt(torch.sigmoid(-logsnr))
        return alpha, sigma

    def _continuous_band_bounds(self):
        if self.widening_steps <= 0:
            return self.logsnr_min, self.logsnr_max

        progress = min(
            float(self._continuous_step.item()) / max(self.widening_steps, 1),
            1.0,
        )
        band_min = self.logsnr_min_start + progress * (
            self.logsnr_min - self.logsnr_min_start)
        band_max = self.logsnr_max_start + progress * (
            self.logsnr_max - self.logsnr_max_start)
        return band_min, band_max

    def _continuous_band_t_bounds(self, device, dtype):
        band_min, band_max = self._continuous_band_bounds()
        band_min = torch.tensor(band_min, device=device, dtype=dtype)
        band_max = torch.tensor(band_max, device=device, dtype=dtype)
        t_lower = inverse_logsnr_schedule_cosine(
            band_max,
            logsnr_min=self.logsnr_min,
            logsnr_max=self.logsnr_max,
        )
        t_upper = inverse_logsnr_schedule_cosine(
            band_min,
            logsnr_min=self.logsnr_min,
            logsnr_max=self.logsnr_max,
        )
        return t_lower, t_upper

    def _continuous_band_mass(self, device, dtype):
        t_lower, t_upper = self._continuous_band_t_bounds(device, dtype)
        return (t_upper - t_lower).clamp(min=1e-8)

    def _continuous_sample_t(self, batch_size, device, dtype):
        t_lower, t_upper = self._continuous_band_t_bounds(device, dtype)
        u = torch.rand(batch_size, device=device, dtype=dtype)
        # Low-discrepancy sampling inside the current band.
        u = (u + torch.arange(batch_size, device=device, dtype=dtype) / batch_size) % 1.0
        t = t_lower + (t_upper - t_lower) * u
        return t.clamp(min=1e-6, max=1 - 1e-6)

    def _continuous_sample_xt_from_x0(self, x0, t=None, logsnr=None, noise=None):
        """Sample x_t from q(x_t | x_0) using a continuous logSNR schedule."""
        if logsnr is None:
            assert t is not None
            logsnr = self._continuous_logsnr(t)
        noise = default(noise, lambda: torch.randn_like(x0))
        alpha, sigma = self._continuous_alpha_sigma(logsnr)
        alpha = append_dims(alpha, x0.ndim)
        sigma = append_dims(sigma, x0.ndim)
        return alpha * x0 + sigma * noise

    def _continuous_model_to_x0_eps(self, x_t, model_out, logsnr):
        alpha, sigma = self._continuous_alpha_sigma(logsnr)
        alpha = append_dims(alpha, x_t.ndim)
        sigma = append_dims(sigma, x_t.ndim)
        alpha_sq = alpha**2
        sigma_sq = sigma**2
        denom = (alpha_sq + sigma_sq).clamp(min=1e-12)

        if self.pred_target == 'x0':
            x0 = model_out
            eps = (x_t - alpha * x0) / sigma.clamp(min=1e-12)
        elif self.pred_target == 'eps':
            eps = model_out
            x0 = (x_t - sigma * eps) / alpha.clamp(min=1e-12)
        elif self.pred_target == 'v':
            v = model_out
            eps = (sigma * x_t + alpha * v) / denom
            x0 = (alpha * x_t - sigma * v) / denom
        else:
            raise NotImplementedError(
                f'Unknown pred_target: {self.pred_target}')
        return x0, eps

    def _continuous_loss_weight(self, t, logsnr):
        if self.continuous_loss == 'sigmoid':
            return sigmoid_loss_weight(
                t,
                logsnr,
                bias=self.sigmoid_bias,
                logsnr_min=self.logsnr_min,
                logsnr_max=self.logsnr_max,
            )
        if self.continuous_loss in ['mse', 'x0_mse']:
            return torch.ones_like(logsnr)
        raise NotImplementedError(
            f'Unknown continuous loss: {self.continuous_loss}')

    def _continuous_loss_function(self, x0, **model_kwargs):
        B = x0.shape[0]
        t = self._continuous_sample_t(B, x0.device, torch.float32)
        logsnr = self._continuous_logsnr(t)
        noise = torch.randn_like(x0)
        x_noisy = self._continuous_sample_xt_from_x0(
            x0=x0, logsnr=logsnr, noise=noise)
        model_out = self.forward(x_noisy, logsnr, **model_kwargs)
        pred_x0, _ = self._continuous_model_to_x0_eps(
            x_noisy, model_out, logsnr)
        weight = self._continuous_loss_weight(t, logsnr)
        weight = weight * self._continuous_band_mass(x0.device, logsnr.dtype)
        weight = append_dims(weight, x0.ndim)
        self._continuous_step.add_(1)
        return {'denoise_loss': (weight * (pred_x0 - x0)**2).mean()}

    def _continuous_default_ddim_steps(self):
        if self.sampling_timesteps is not None:
            return int(self.sampling_timesteps)
        return max(200, self.num_timesteps // 5)

    def _default_ddim_steps(self):
        if self.sampling_timesteps is not None:
            return int(self.sampling_timesteps)
        return max(200, self.num_timesteps // 5)

    def _continuous_diffusion_row(self, x0):
        diffusion_row = [x0.detach().clone()]
        B = x0.shape[0]
        for step in range(self.num_timesteps):
            if (step + 1) % self.log_every_t == 0 or step == self.num_timesteps - 1:
                t = repeat(
                    torch.tensor([(step + 1) / self.num_timesteps],
                                 device=x0.device),
                    '1 -> b',
                    b=B,
                )
                noise = torch.randn_like(x0)
                x_noisy = self._continuous_sample_xt_from_x0(
                    x0=x0, t=t, noise=noise)
                diffusion_row.append(x_noisy)
        return torch.stack(diffusion_row, dim=0)

    @contextmanager
    def ema_scope(self, context=None):
        """Switch model to EMA weight, do something, then switch back."""
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def _q_mean_variance(self, x0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x0: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x0's shape.
        """
        mean = extract_to(self.sqrt_alphas_bar, t, x0.shape) * x0
        variance = extract_to(1. - self.alphas_bar, t, x0.shape)
        log_variance = extract_to(self.log_one_minus_alphas_bar, t, x0.shape)
        return mean, variance, log_variance

    def _sample_xt_from_x0(self, x0, t, noise=None):
        """Sample x_t from q(x_t | x_0)."""
        noise = default(noise, lambda: torch.randn_like(x0))
        return extract_to(self.sqrt_alphas_bar, t, x0.shape) * x0 + \
            extract_to(self.sqrt_one_minus_alphas_bar, t, x0.shape) * noise

    def _predict_x0_from_xt(self, x_t, t, noise):
        """Predict x_0 given x_t and eps(x_t)."""
        return extract_to(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - \
            extract_to(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * noise

    def _q_posterior(self, x0, x_t, t):
        """Estimate x_{t-1} given x_0 and x_t."""
        posterior_mean = (
            extract_to(self.posterior_mean_coef1, t, x_t.shape) * x0 +
            extract_to(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract_to(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_to(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _p_mean_variance(self, x, t, clip_denoised: bool):
        """
        Get the distribution p(x_{t-1} | x_t).
        First estimate x0, then use x0 and x_t to compute x_{t-1}.
        """
        eps = self.forward(x, t)
        x0_recon = self._predict_x0_from_xt(x, t=t, noise=eps)
        if clip_denoised:
            x0_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = \
            self._q_posterior(x0=x0_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def _p_sample(self, x, t, clip_denoised=True):
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self._p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device)
        # no noise when t == 0
        mask = (1 - (t == 0).float()).reshape(b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def _sample_x0_from_noise(self, shape, ret_intermed=False, verbose=True):
        """Generate x0 from random noise by iterative sampling."""
        if self.continuous_time:
            sampler = ContinuousDDIMSampler(self)
            return sampler.generate_imgs(
                steps=self.num_timesteps,
                shape=shape,
                conditioning=None,
                eta=1.0,
                verbose=verbose,
                log_every_t=self.log_every_t,
                ret_intermed=ret_intermed,
            )
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc='Sampling t',
                total=self.num_timesteps,
                disable=not verbose):
            img = self._p_sample(
                img,
                torch.full((b, ), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if ret_intermed:
            intermediates = torch.stack(intermediates, dim=0)  # [N, B, ...]
            return img, intermediates
        return img

    @torch.no_grad()
    def generate_imgs(
        self,
        batch_size=16,
        ret_intermed=False,
        verbose=True,
        use_ddim=False,
        use_dpm=False,
        eta=None,
    ):
        """Generate `batch_size` images."""
        shape = (batch_size, self.channels, *self.resolution)
        if self.continuous_time:
            sampler = ContinuousDDIMSampler(self)
            if use_dpm:
                if not self._warned_continuous_dpm_fallback:
                    print(
                        'Continuous-time diffusion does not support the '
                        'discrete DPM-Solver path; falling back to the '
                        'continuous DDIM sampler.'
                    )
                    self._warned_continuous_dpm_fallback = True
                return sampler.generate_imgs(
                    self.continuous_dpm_fallback_steps,
                    shape,
                    conditioning=None,
                    eta=0.0 if eta is None else eta,
                    verbose=verbose,
                    log_every_t=self.log_every_t,
                    ret_intermed=ret_intermed,
                )
            if use_ddim:
                return sampler.generate_imgs(
                    self._continuous_default_ddim_steps(),
                    shape,
                    conditioning=None,
                    eta=0.0 if eta is None else eta,
                    verbose=verbose,
                    log_every_t=self.log_every_t,
                    ret_intermed=ret_intermed,
                )
            return sampler.generate_imgs(
                self.num_timesteps,
                shape,
                conditioning=None,
                eta=1.0 if eta is None else eta,
                verbose=verbose,
                log_every_t=self.log_every_t,
                ret_intermed=ret_intermed,
            )
        # faster sampling with DDIM
        if use_ddim:
            ddim_sampler = DDIMSampler(self, schedule=self.beta_schedule)
            return ddim_sampler.generate_imgs(
                self._default_ddim_steps(),
                shape,
                conditioning=None,
                verbose=verbose,
                ret_intermed=ret_intermed,
            )
        return self._sample_x0_from_noise(
            shape,
            ret_intermed=ret_intermed,
            verbose=verbose,
        )

    def loss_function(self, data_dict):
        """Make a noisy sample, then use model to predict the noise."""
        x0 = data_dict['img']
        if self.continuous_time:
            return self._continuous_loss_function(x0)
        t = torch.randint(
            0, self.num_timesteps, (x0.shape[0], ), device=self.device).long()
        noise = torch.randn_like(x0)
        x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
        # conditional data
        eps = self.forward(x_noisy, t)
        loss_dict = {'denoise_loss': F.mse_loss(eps, noise)}
        return loss_dict

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

    def _training_step_end(self, *args, **kwargs):
        # accumulate the moving average of model weights
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        """Put the input samples NxBxcxhxw into a grid of B x N grid."""
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(
            denoise_grid, nrow=n_imgs_per_row, pad_value=-1.)
        return denoise_grid

    @torch.no_grad()
    def log_images(
        self,
        batch,
        **kwargs,
    ):
        log = {}
        x = batch['img']
        B = x.shape[0]

        # get diffusion row
        if self.continuous_time:
            diffusion_row = self._continuous_diffusion_row(x)
        else:
            diffusion_row = [x.detach().clone()]
            x0 = x

            for t in range(self.num_timesteps):
                if (t + 1) % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=B)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(x0)
                    x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
                    diffusion_row.append(x_noisy)
            diffusion_row = torch.stack(
                diffusion_row, dim=0)  # [N, B, C, H, W]
        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # get denoise row
        with self.ema_scope("Plotting"):
            samples, denoise_row = self.generate_imgs(
                batch_size=B, ret_intermed=True, **kwargs)

        log["samples"] = samples
        log["denoise_row"] = self._get_rows_from_list(denoise_row)

        return log

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype


class DiffusionWrapper(nn.Module):
    """A wrapper for forwarding the diffusion model. Cond/Unconditional one."""

    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()

        self.diffusion_model = build_unet(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn']

    def forward(self, x, t, context=None):
        if self.conditioning_key is None:
            return self.diffusion_model(x, t)

        assert context is not None, 'Please provide conditioning data'
        if self.conditioning_key == 'concat':
            if not is_list_tuple(context):
                context = [context]
            xc = torch.cat([x] + context, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if is_list_tuple(context):
                context = torch.cat(context, 1)
            out = self.diffusion_model(x, t, context=context)
        else:
            raise NotImplementedError

        return out

    @property
    def device(self):
        return self.diffusion_model.device

    @property
    def dtype(self):
        return self.diffusion_model.dtype
