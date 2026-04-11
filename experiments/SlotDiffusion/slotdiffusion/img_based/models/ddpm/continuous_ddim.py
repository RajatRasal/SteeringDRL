"""Continuous-time DDIM/DDPM-style sampling in logSNR space."""

from tqdm import tqdm

import torch

from .utils import append_dims, noise_like


class ContinuousDDIMSampler(object):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pred_target = model.pred_target
        assert self.pred_target in ['eps', 'x0', 'v'], \
            f'pred_target {self.pred_target} not supported by continuous DDIM'

        self.clip_denoised = self.model.clip_denoised
        self.vq_denoised = getattr(self.model, 'vq_denoised', False)
        assert not (self.clip_denoised and self.vq_denoised), \
            'clip_denoised is image space DM, vq_denoised is LDM'

    @torch.no_grad()
    def generate_imgs(
        self,
        steps,
        shape,
        conditioning=None,
        eta=0.,
        verbose=True,
        log_every_t=100,
        ret_intermed=True,
        same_noise=False,
        **kwargs,
    ):
        if conditioning is not None:
            assert conditioning.shape[0] == shape[0]

        device = self.model.device
        b = shape[0]
        img = noise_like(shape, device, repeat=same_noise)
        intermediates = [img]
        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        if verbose:
            print(
                f'Running continuous DDIM sampling with {steps} steps, eta={eta}'
            )
        iterator = tqdm(
            range(steps),
            desc='Continuous DDIM Sampler',
            total=steps,
            disable=not verbose,
        )

        for i in iterator:
            t = torch.full((b, ), time_grid[i], device=device, dtype=torch.float32)
            s = torch.full(
                (b, ), time_grid[i + 1], device=device, dtype=torch.float32)

            logsnr_t = self.model._continuous_logsnr(t)
            logsnr_s = self.model._continuous_logsnr(s)

            if conditioning is None:
                model_out = self.model(img, logsnr_t)
            else:
                model_out = self.model(img, logsnr_t, context=conditioning)

            pred_x0, pred_eps = self.model._continuous_model_to_x0_eps(
                img, model_out, logsnr_t)

            if self.clip_denoised:
                pred_x0.clamp_(-1., 1.)
            if self.vq_denoised:
                pred_x0 = self.model.vae.quantize(pred_x0)

            alpha_t, sigma_t = self.model._continuous_alpha_sigma(logsnr_t)
            alpha_s, sigma_s = self.model._continuous_alpha_sigma(logsnr_s)
            alpha_t = append_dims(alpha_t, img.ndim)
            sigma_t = append_dims(sigma_t, img.ndim)
            alpha_s = append_dims(alpha_s, img.ndim)
            sigma_s = append_dims(sigma_s, img.ndim)

            if eta == 0.0:
                img = alpha_s * pred_x0 + sigma_s * pred_eps
            else:
                a_t = alpha_t**2
                a_s = alpha_s**2
                sigma_eta = eta * torch.sqrt((1 - a_s) / (1 - a_t + 1e-12)) \
                    * torch.sqrt(torch.clamp(1 - (a_t / (a_s + 1e-12)), min=0.0))
                eps_coeff = torch.sqrt(
                    torch.clamp(1 - a_s - sigma_eta**2, min=0.0))
                noise = sigma_eta * noise_like(
                    img.shape, device, repeat=same_noise)
                img = alpha_s * pred_x0 + eps_coeff * pred_eps + noise

            if i % log_every_t == 0 or i == steps - 1:
                intermediates.append(img)

        if ret_intermed:
            return img, torch.stack(intermediates, dim=0)
        return img
