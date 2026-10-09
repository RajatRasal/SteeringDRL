from abc import ABC
from typing import Callable

import torch
from torch import nn

from . import utils
from .noise_schedules import BandWideningNoiseSchedule


class BaseLogSNRDenoiser(ABC, nn.Module):
    """
    Generic log-SNR denoiser wrapper.

    - Wraps an underlying `model(x_t, logsnr, **kwargs)`.
    - Uses a NoiseSchedule to map t -> λ and give p(λ).
    - Training loss is defined in ε-space with weighted-ELBO style weights.
    - forward(x_t, logsnr, **kwargs) always returns x̂ (denoised x0 estimate),
      regardless of prediction parametrisation (ε / x / v).
    """

    def __init__(self, inner_model: nn.Module):
        super().__init__()
        self.inner_model = inner_model

    # ---------- abstract: how to interpret model_out ----------

    def to_eps(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map model_out to ε̂.
        Implemented by subclasses.
        """
        raise NotImplementedError()

    def to_x(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map model_out to x̂ (denoised x0).
        Implemented by subclasses.
        """
        raise NotImplementedError()

    # ---------- helper: ensure logsnr shape ----------

    def _prepare_lambda(self, logsnr: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Ensure logsnr is shape (B,), broadcasting a scalar or (1,) if needed.
        """
        if logsnr.dim() == 0:
            logsnr = logsnr.view(1).expand(batch_size)
        elif logsnr.shape[0] == 1 and batch_size > 1:
            logsnr = logsnr.expand(batch_size)
        return logsnr

    # ---------- forward: always return x̂ for sampling ----------

    def forward(self, x_t, logsnr, **kwargs):
        return self.inner_model(x_t, logsnr, **kwargs)

    # ---------- training loss: weighted ELBO in λ ----------

    def loss(
        self,
        x0: torch.Tensor,
        weight_fn: Callable,
        noise_schedule: BandWideningNoiseSchedule,
        generator: torch.Generator,
        **kwargs,
    ) -> torch.Tensor:
        """
        x0       : clean data (B,C,H,W)
        logsnr        : t ~ U[0,1], f(t) = logsnr (B,)
        weight_fn: function w(λ) returning per-sample weights (B,)

        Implements:
            L ≈ 0.5 E_{t,ε}[ w(λ)/p(λ) * ||ε̂ - ε||^2 ]
        where λ = schedule.schedule(t), p(λ) = schedule.density(λ).
        """
        device = x0.device
        B = x0.shape[0]

        # λ, α, σ
        logsnr = noise_schedule.sample_lambda(B, generator, device)  # (B,)
        alpha_lam = utils.append_dims(torch.sqrt(torch.sigmoid(logsnr)), 4)
        sigma_lam = utils.append_dims(torch.sqrt(torch.sigmoid(-logsnr)), 4)

        # forward noising x0 -> x_lam
        eps = torch.randn(x0.size(), generator=generator).to(device=device)
        x_lam = alpha_lam * x0 + sigma_lam * eps

        # model prediction
        model_out = self(x_lam, logsnr, **kwargs)
        eps_pred = self.to_eps(model_out, x_lam, sigma_lam, alpha_lam)

        # importance weight: w(λ) / p(λ)
        w_lambda = weight_fn(logsnr)                        # (B,)
        p_lambda = noise_schedule.density(logsnr)            # (B,)
        importance = w_lambda / (p_lambda + 1e-12)
        importance = importance * noise_schedule.band_mass(logsnr.device, logsnr.dtype)

        importance = utils.append_dims(importance, 4)
        loss = 0.5 * (importance * (eps_pred - eps) ** 2).mean()

        noise_schedule.increment_step()

        return loss


# ---------- ε-prediction ----------

class EpsPredictionDenoiser(BaseLogSNRDenoiser):
    """
    Model predicts ε directly:
        model_out = ε̂.
    """

    def to_eps(self, model_out, x_t, sigma_t, alpha_t):
        return model_out

    def to_x(self, model_out, x_t, sigma_t, alpha_t):
        eps_hat = model_out
        return (x_t - sigma_t * eps_hat) / alpha_t


# ---------- x-prediction ----------

class XPredictionDenoiser(BaseLogSNRDenoiser):
    """
    Model predicts x0 directly:
        model_out = x̂.
    """

    def to_eps(self, model_out, x_t, sigma_t, alpha_t):
        x_hat = model_out
        return (x_t - alpha_t * x_hat) / sigma_t

    def to_x(self, model_out, x_t, sigma_t, alpha_t):
        return model_out


# ---------- v-prediction ----------

class VPredictionDenoiser(BaseLogSNRDenoiser):
    """
    Model predicts v = α ε - σ x0 (VP-style).
        model_out = v̂.
    """

    def to_eps(self, model_out, x_t, sigma_t, alpha_t):
        v_hat = model_out
        # Note: In VP models, α^2 + σ^2 = 1 => demon = 1
        denom = (alpha_t ** 2 + sigma_t ** 2)
        # ε = (σ x_t + α v) / (α^2 + σ^2)
        return (sigma_t * x_t + alpha_t * v_hat) / denom

    def to_x(self, model_out, x_t, sigma_t, alpha_t):
        v_hat = model_out
        denom = (alpha_t ** 2 + sigma_t ** 2)
        # x0 = (α x_t - σ v) / (α^2 + σ^2)
        return (alpha_t * x_t - sigma_t * v_hat) / denom
