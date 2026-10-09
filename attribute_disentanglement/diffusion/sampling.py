import torch

from attribute_disentanglement.diffusion import utils


def sample_ddim_lambda(model, noise, noise_schedule, steps, eta=1.0, clamp=False, extra_args={}):
    """
    If eta=1.0: stochastic (roughly DDPM-like)
    If eta=0.0: deterministic (DDIM / probability-flow ODE style)
    For VQ latents: keep clamp=False (or clamp to latent-appropriate bounds, not [-1,1]).
    """
    ts = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)
    x_t = noise
    B = noise.size(0)

    def step(model_out, x_t, logsnr_t, logsnr_s):
        # alpha, sigma at t and s
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        # broadcast
        alpha_t = utils.append_dims(alpha_t, x_t.ndim)
        sigma_t = utils.append_dims(sigma_t, x_t.ndim)
        alpha_s = utils.append_dims(alpha_s, x_t.ndim)
        sigma_s = utils.append_dims(sigma_s, x_t.ndim)

        # x0, eps prediction
        x0 = model.to_x(model_out, x_t, sigma_t, alpha_t)
        eps = model.to_eps(model_out, x_t, sigma_t, alpha_t)
        if clamp:
            x0 = x0.clamp(-1, 1)

        # Note: Below is an exact implementation of the DDIM paper (Equation 12).
        if eta == 0.0:
            # DDIM (deterministic): x_s = alpha_s x0 + sigma_s eps
            return alpha_s * x0 + sigma_s * eps

        # DDIM(eta) stochasticity:
        a_t = alpha_t ** 2   # = bar_alpha_t
        a_s = alpha_s ** 2

        # sigma_eta = eta * sqrt((1-a_s)/(1-a_t)) * sqrt(1 - a_t/a_s)
        sigma_eta = eta * torch.sqrt((1 - a_s) / (1 - a_t + 1e-12)) \
            * torch.sqrt(torch.clamp(1 - (a_t / (a_s + 1e-12)), min=0.0))

        # eps coeff so that total variance matches: 1 - a_s = eps_coeff^2 + sigma_eta^2
        eps_coeff = torch.sqrt(torch.clamp(1 - a_s - sigma_eta ** 2, min=0.0))

        z = torch.randn_like(x_t)
        return alpha_s * x0 + eps_coeff * eps + sigma_eta * z

    for i in range(len(ts) - 1):
        t = ts[i].repeat(B)
        s = ts[i + 1].repeat(B)
        logsnr_t = noise_schedule.schedule(t)
        logsnr_s = noise_schedule.schedule(s)

        model_out = model(x_t, logsnr_t, **extra_args)
        x_t = step(model_out, x_t, logsnr_t, logsnr_s)

    return x_t
