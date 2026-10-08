from abc import ABC, abstractmethod
from typing import Callable
import torch
from torch import nn
from torch.distributions import Normal
from k_diffusion import utils


PI = torch.pi


class NoiseSchedule(ABC):
    """
    Base class for log-SNR schedules λ = f_λ(t).

    Subclasses implement the *pure* base schedule (no shift, no truncation):
        schedule_base(t) : f_λ(t)
        inverse_base(λ)  : f_λ^{-1}(λ)
        density_base(λ)  : p(λ) = - d/dλ f_λ^{-1}(λ)

    The base class then applies:
        - global shift
        - truncation (Eqs. 126–131)
    """

    def __init__(self, shift=0.0, logsnr_min=None, logsnr_max=None):
        self.shift = shift                  # additive λ-shift
        self.logsnr_min = logsnr_min        # truncation endpoints (in λ-space)
        self.logsnr_max = logsnr_max

    # ============================================================
    # =============== PURE BASE FUNCTIONS (unshifted) =============
    # ============================================================

    @abstractmethod
    def schedule_base(self, t: torch.Tensor) -> torch.Tensor:
        """Unshifted λ = f_λ(t)."""
        raise NotImplementedError()

    @abstractmethod
    def inverse_base(self, lam: torch.Tensor) -> torch.Tensor:
        """Unshifted inverse: t = f_λ^{-1}(λ)."""
        raise NotImplementedError()

    @abstractmethod
    def density_base(self, lam: torch.Tensor) -> torch.Tensor:
        """Unshifted density p(λ)."""
        raise NotImplementedError()

    # ============================================================
    # ================= SHIFTED (but untruncated) =================
    # ============================================================

    # --- λ(t) with shift ---
    def schedule_untruncated(self, t):
        """Shifted schedule: f_λ(t) + shift."""
        return self.schedule_base(t) + self.shift

    # --- inverse with shift ---
    def inverse_untruncated(self, lam):
        """
        Shifted inverse: f_λ^{-1}(λ - shift)
        because shifting λ by +c shifts back the inverse by -c.
        """
        return self.inverse_base(lam - self.shift)

    # --- density with shift ---
    def density_untruncated(self, lam):
        """
        Shifted density: p(λ - shift)
        because additive shifts do *not* change d/dλ f^{-1}(λ).
        """
        return self.density_base(lam - self.shift)

    # ============================================================
    # ===================== TRUNCATED VERSIONS ====================
    # ============================================================

    @property
    def t0(self):
        """t0 = f^{-1}(λ_max) (unshifted version automatically handled)."""
        if self.logsnr_max is None:
            return None
        return self.inverse_untruncated(torch.tensor(self.logsnr_max))

    @property
    def t1(self):
        """t1 = f^{-1}(λ_min)."""
        if self.logsnr_min is None:
            return None
        return self.inverse_untruncated(torch.tensor(self.logsnr_min))

    # --- truncated λ(t) ---
    def schedule(self, t):
        """
        Implements Eq. (126):
            \tilde f_λ(t) = f_λ( t0 + (t1 - t0) * t )
        applied to the *shifted* untruncated schedule.
        """
        if self.logsnr_min is None or self.logsnr_max is None:
            return self.schedule_untruncated(t)

        t0 = self.t0.to(t.device, t.dtype)
        t1 = self.t1.to(t.device, t.dtype)
        t_inner = t0 + (t1 - t0) * t
        return self.schedule_untruncated(t_inner)

    # --- truncated inverse ---
    def inverse_schedule(self, lam):
        """
        Eq. (129):
            \tilde f^{-1}(λ) = ( f^{-1}(λ) - t0 ) / (t1 - t0)
        """
        if self.logsnr_min is None or self.logsnr_max is None:
            return self.inverse_untruncated(lam)

        t0 = self.t0.to(lam.device, lam.dtype)
        t1 = self.t1.to(lam.device, lam.dtype)
        return (self.inverse_untruncated(lam) - t0) / (t1 - t0)

    # --- truncated density ---
    def density(self, lam):
        """
        Eq. (130–131):
            \tilde p(λ) = p(λ) / (t1 - t0),   if λ ∈ [λ_min, λ_max]
                        = 0                 otherwise
        """
        if self.logsnr_min is None or self.logsnr_max is None:
            return self.density_untruncated(lam)

        base = self.density_untruncated(lam)
        t0 = self.t0.to(lam.device, lam.dtype)
        t1 = self.t1.to(lam.device, lam.dtype)
        scale = (t1 - t0)

        mask = (lam >= self.logsnr_min) & (lam <= self.logsnr_max)
        return torch.where(mask, base / scale, torch.zeros_like(lam))

    # ============================================================
    # ============= Sampling interface (λ, t, p(λ)) ===============
    # ============================================================

    def sample_t(self, B, generator, device):
        u = torch.rand(B, generator=generator).to(device)
        # Low-discrepency sampler (VDM paper Appendix I.1)
        t = (u + torch.arange(B, device=device) / B) % 1
        return t.clamp(min=1e-6, max=1 - 1e-6)

    def sample_lambda(self, batch_size, generator, device):
        """Sample t ~ U[0,1], return λ(t) and its density p(λ)."""
        t = self.sample_t(batch_size, generator, device)
        lam = self.schedule(t)
        return lam


class BandWideningNoiseSchedule(NoiseSchedule):
    """
    Method A: sampling-only λ-band curriculum.
    Objective (schedule + density) is unchanged.
    """

    def __init__(
        self,
        shift,
        logsnr_min,
        logsnr_max,
        logsnr_min_start,
        logsnr_max_start,
        widening_steps,
    ):
        assert logsnr_min <= logsnr_min_start <= logsnr_max_start <= logsnr_max

        super().__init__(
            shift=shift,
            logsnr_min=logsnr_min,
            logsnr_max=logsnr_max,
        )

        self.lam_min_full = logsnr_min
        self.lam_max_full = logsnr_max

        self.lam_min_start = logsnr_min_start
        self.lam_max_start = logsnr_max_start

        self.max_steps = widening_steps
        self.cur_steps = 0

    # ------------------------------------------------------------
    # Band endpoints in λ-space
    # ------------------------------------------------------------
    def _band_lams(self):
        # if self.cur_steps >= self.max_steps:
        #     return self.lam_min_full, self.lam_max_full
        r = self.cur_steps / self.max_steps
        lam_L = self.lam_min_start + r * (self.lam_min_full - self.lam_min_start)
        lam_U = self.lam_max_start + r * (self.lam_max_full - self.lam_max_start)
        return lam_L, lam_U

    # ------------------------------------------------------------
    # Z(s) = probability mass of band under base sampler
    # ------------------------------------------------------------
    def band_mass(self, device, dtype):
        if self.cur_steps >= self.max_steps:
            t0 = self.t0.to(device, dtype=dtype)
            t1 = self.t1.to(device, dtype=dtype)
            return (t1 - t0)

        lam_L, lam_U = self._band_lams()
        lam_L = torch.tensor(lam_L, device=device, dtype=dtype)
        lam_U = torch.tensor(lam_U, device=device, dtype=dtype)

        t_L = self.inverse_schedule(lam_L)
        t_U = self.inverse_schedule(lam_U)

        return (t_L - t_U).clamp(min=1e-8)

    # ------------------------------------------------------------
    # Sampling-only modification
    # ------------------------------------------------------------
    def sample_t(self, B, generator, device):
        if self.cur_steps >= self.max_steps:
            return super().sample_t(B, generator, device)

        lam_L, lam_U = self._band_lams()
        lam_L = torch.tensor(lam_L, device=device)
        lam_U = torch.tensor(lam_U, device=device)

        t_L = self.inverse_schedule(lam_L)
        t_U = self.inverse_schedule(lam_U)

        # LD sampler
        u = torch.rand(B, generator=generator).to(device=device)
        t = (u + torch.arange(B, device=device) / B) % 1.0

        t = t_L + (t_U - t_L) * t
        return t.clamp(min=1e-6, max=1 - 1e-6)

    # ------------------------------------------------------------
    def increment_step(self):
        self.cur_steps += 1


# -------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal, 2021)
# λ = -2 log tan(π t / 2)
# t = (2/π) arctan(e^{-λ/2})
# p(λ) = sech(λ/2) / (2π)
# -------------------------------------------------
class CosineSchedule(BandWideningNoiseSchedule):

    def schedule_base(self, t: torch.Tensor) -> torch.Tensor:
        # λ(t) = -2 log tan(π t / 2)
        return -2.0 * torch.log(torch.tan(PI * t / 2.0))

    def inverse_base(self, lam: torch.Tensor) -> torch.Tensor:
        # t(λ) = (2/π) arctan(e^{-λ/2})
        return (2.0 / PI) * torch.atan(torch.exp(-0.5 * lam))

    def density_base(self, lam: torch.Tensor) -> torch.Tensor:
        # p(λ) = sech(λ/2) / (2π) = 1 / (2π cosh(λ/2))
        return 1.0 / (2.0 * PI * torch.cosh(lam / 2.0))


# -------------------------------------------------
# EDM training schedule (log-normal in σ, Normal in λ)
# Table 5 row "EDM (training)"
#
# λ(t)   = -F_N^{-1}(t; μ, σ^2)
# t(λ)   = F_N(-λ; μ, σ^2)
# p(λ)   = N(λ; μ, σ^2)
# defaults: μ = 2.4, σ = 2.4
# -------------------------------------------------
class EDMTrainingSchedule(BandWideningNoiseSchedule):

    def __init__(
        self,
        mu: float = 2.4,
        sigma: float = 2.4,
        *args,
        **kwargs,
        # shift: float = 0.0,
        # logsnr_min=None, logsnr_max=None,
    ):
        super().__init__(*args, **kwargs) # shift=shift, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
        self.mu = float(mu)
        self.sigma = float(sigma)

    def _dist(self, device, dtype):
        mu = torch.tensor(self.mu, device=device, dtype=dtype)
        sigma = torch.tensor(self.sigma, device=device, dtype=dtype)
        return Normal(loc=mu, scale=sigma)

    def schedule_base(self, t: torch.Tensor) -> torch.Tensor:
        # λ(t) = -F_N^{-1}(t; μ, σ^2)
        t = t.clamp(1e-6, 1.0 - 1e-6)
        dist = self._dist(t.device, t.dtype)
        return -dist.icdf(t)

    def inverse_base(self, lam: torch.Tensor) -> torch.Tensor:
        # t(λ) = F_N(-λ; μ, σ^2)
        dist = self._dist(lam.device, lam.dtype)
        return dist.cdf(-lam)

    def density_base(self, lam: torch.Tensor) -> torch.Tensor:
        # p(λ) = N(λ; μ, σ^2)
        dist = self._dist(lam.device, lam.dtype)
        return dist.log_prob(lam).exp()


# -------------------------------------------------
# EDM sampling schedule (Karras σ_min/σ_max, ρ)
# Table 5 row "EDM (sampling)"
#
# λ(t) = -2ρ log( σ_max^{1/ρ} + (1-t)(σ_min^{1/ρ} - σ_max^{1/ρ}) )
# t(λ) = 1 - (e^{-λ/(2ρ)} - σ_max^{1/ρ}) / (σ_min^{1/ρ} - σ_max^{1/ρ})
# p(λ) = e^{-λ/(2ρ)} / [ 2ρ (σ_max^{1/ρ} - σ_min^{1/ρ}) ]
# -------------------------------------------------
class EDMSamplingSchedule(BandWideningNoiseSchedule):

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        shift: float = 0.0,
        logsnr_min=None, logsnr_max=None,
    ):
        super().__init__(shift=shift, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)

    def _constants(self, device, dtype):
        rho = torch.tensor(self.rho, device=device, dtype=dtype)
        c_min = torch.tensor(self.sigma_min, device=device, dtype=dtype) ** (1.0 / rho)
        c_max = torch.tensor(self.sigma_max, device=device, dtype=dtype) ** (1.0 / rho)
        return rho, c_min, c_max

    def schedule_base(self, t: torch.Tensor) -> torch.Tensor:
        # λ(t) from Table 5
        rho, c_min, c_max = self._constants(t.device, t.dtype)
        inner = c_max + (1.0 - t) * (c_min - c_max)
        return -2.0 * rho * torch.log(inner)

    def inverse_base(self, lam: torch.Tensor) -> torch.Tensor:
        # t(λ) from Table 5
        rho, c_min, c_max = self._constants(lam.device, lam.dtype)
        e_term = torch.exp(-lam / (2.0 * rho))
        return 1.0 - (e_term - c_max) / (c_min - c_max)

    def density_base(self, lam: torch.Tensor) -> torch.Tensor:
        # p(λ) from Table 5
        rho, c_min, c_max = self._constants(lam.device, lam.dtype)
        e_term = torch.exp(-lam / (2.0 * rho))
        denom = 2.0 * rho * (c_max - c_min)
        return e_term / denom


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
