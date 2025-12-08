from abc import ABC, abstractmethod
from typing import Callable, Optional
import torch
from torch import nn
from torch.distributions import Normal
from k_diffusion import utils
import matplotlib.pyplot as plt


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

    def sample_lambda(self, batch_size, device):
        """Sample t ~ U[0,1], return λ(t) and its density p(λ)."""
        t = torch.rand(batch_size, device=device)
        lam = self.schedule(t)
        p_lam = self.density(lam)
        return lam, t, p_lam


# -------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal, 2021)
# λ = -2 log tan(π t / 2)
# t = (2/π) arctan(e^{-λ/2})
# p(λ) = sech(λ/2) / (2π)
# -------------------------------------------------
class CosineSchedule(NoiseSchedule):

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
class EDMTrainingSchedule(NoiseSchedule):

    def __init__(
        self,
        mu: float = 2.4,
        sigma: float = 2.4,
        shift: float = 0.0,
        logsnr_min=None, logsnr_max=None,
    ):
        super().__init__(shift=shift, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
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
class EDMSamplingSchedule(NoiseSchedule):

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


class ImportanceWeightMonitor:
    """
    Tracks the empirical effective weighting over λ via accumulated
    importance weights per λ-bin, and compares/plots against a target
    weighting function w(λ).

    - You should pass:
        lambda_min, lambda_max: range of λ (logSNR) you care about
        num_bins: number of λ-bins
        weight_fn: callable w(λ): Tensor[B] -> Tensor[B]
        similarity_fn: callable (emp, target) -> float, e.g. MAE or R²

    - During training, call:
        monitor(logsnr, importance)

      where:
        logsnr: λ values (B,)
        importance: w(λ)/p(λ) per sample (B,)
    """

    def __init__(
        self,
        lambda_min: float,
        lambda_max: float,
        num_bins: int,
        weight_fn: Callable[[torch.Tensor], torch.Tensor],
        similarity_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: str = "cpu",
    ):
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_bins = num_bins
        self.device = torch.device(device)

        self.weight_fn = weight_fn          # theoretical w(λ)
        self.similarity_fn = similarity_fn  # e.g. MAE / TV / R²

        # ∑ importance per bin (≈ ∫_bin w(λ) dλ)
        self.weight_sum = torch.zeros(num_bins, device=self.device)

    # ---------- accumulation ----------

    def _bin_indices(self, logsnr: torch.Tensor) -> torch.Tensor:
        """
        Map λ=logsnr to integer bin indices in [0, num_bins-1].
        """
        lam = logsnr.clamp(self.lambda_min, self.lambda_max)
        rel = (lam - self.lambda_min) / (self.lambda_max - self.lambda_min)  # [0,1]
        bins = (rel * self.num_bins).long()
        return bins.clamp(0, self.num_bins - 1)

    def reset(self):
        self.weight_sum.zero_()

    @torch.no_grad()
    def __call__(
        self,
        logsnr: torch.Tensor,     # shape (B,)
        importance: torch.Tensor  # shape (B,), w(λ)/p(λ)
    ):
        """
        Accumulate ∑ importance per λ-bin (≈ ∫ w(λ) dλ in each bin).
        """
        logsnr = logsnr.detach().to(self.device).view(-1)
        importance = importance.detach().to(self.device).view(-1)

        bins = self._bin_indices(logsnr)
        self.weight_sum.index_add_(0, bins, importance)

    # ---------- densities / curves ----------

    def bin_centers(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns λ bin centers as a 1D tensor of shape [num_bins].
        """
        if device is None:
            device = self.device
        edges = torch.linspace(self.lambda_min, self.lambda_max, self.num_bins + 1, device=device)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers

    def empirical_density(self) -> torch.Tensor:
        """
        Sum-normalised empirical density over λ-bins based on accumulated
        importance. In expectation this approximates discretised w(λ).
        """
        eps = 1e-8
        dens = self.weight_sum / (self.weight_sum.sum() + eps)
        return dens

    def target_density(self) -> torch.Tensor:
        """
        Sum-normalised theoretical density over λ-bins by evaluating w(λ)
        on the bin centers.
        """
        lam = self.bin_centers(device=self.device)
        target = self.weight_fn(lam)
        target = target.clone()
        eps = 1e-8
        target = target / (target.sum() + eps)
        return target

    def similarity(self) -> float:
        """
        Compute similarity(empirical_density, target_density) on
        sum-normalised densities (good for MAE / TV / R²).
        """
        emp = self.empirical_density()
        target = self.target_density()
        return self.similarity_fn(emp, target)

    def curve(self, normalise_max: bool = True):
        """
        Returns (bin_centers, emp_curve, target_curve).

        - If normalise_max=True:
            both curves are individually normalised by their max (shape only).
        - If normalise_max=False:
            both curves are sum-normalised densities.
        """
        lam = self.bin_centers(device=self.device)

        emp = self.empirical_density()      # sum-normalised
        target = self.target_density()      # sum-normalised

        eps = 1e-8
        if normalise_max:
            emp = emp / (emp.max() + eps)
            target = target / (target.max() + eps)

        return lam, emp, target

    def plot(
        self,
        normalise_max: bool = True,
        label_emp: str = "empirical w_eff(λ)",
        label_target: str = "target w(λ)",
        filename: str = "importance_weight_monitor.pdf",
        title: str = "Importance Weight Monitor",
    ):
        """
        Plot empirical effective w(λ) and theoretical w(λ) on the same λ-grid.

        - normalise_max=True → both curves normalised by their own max (paper-style).
        - normalise_max=False → both curves are sum-normalised densities.

        If filename is provided, saves to that path, otherwise just returns the fig/ax.
        """
        lam, emp, target = self.curve(normalise_max=normalise_max)
        lam_cpu = lam.detach().cpu()
        emp_cpu = emp.detach().cpu()
        target_cpu = target.detach().cpu()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(lam_cpu, emp_cpu, label=label_emp, linewidth=2)
        ax.plot(
            lam_cpu,
            target_cpu,
            label=label_target,
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel("λ (log SNR)")
        ax.set_ylabel("w(λ) (normalised)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.2)

        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)


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
        noise_schedule: NoiseSchedule,
        generator: torch.Generator,
        importance_monitor: Optional[ImportanceWeightMonitor] = None,
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
        u = torch.rand(B, generator=generator).to(device=device)

        # Low-discrepency sampler (VDM paper Appendix I.1)
        t = (u + torch.arange(B, device=device) / B) % 1

        # λ, α, σ
        logsnr = noise_schedule.schedule(t)              # (B,)
        alpha_t = utils.append_dims(torch.sqrt(torch.sigmoid(logsnr)), 4)
        sigma_t = utils.append_dims(torch.sqrt(torch.sigmoid(-logsnr)), 4)

        # forward noising x0 -> x_t
        eps = torch.randn(x0.size(), generator=generator).to(device=device)
        x_t = alpha_t * x0 + sigma_t * eps

        # model prediction
        model_out = self(x_t, logsnr, **kwargs)
        eps_pred = self.to_eps(model_out, x_t, sigma_t, alpha_t)

        # importance weight: w(λ) / p(λ)
        w_lambda = weight_fn(logsnr)                        # (B,)
        p_lambda = noise_schedule.density(logsnr)            # (B,)
        importance = w_lambda / (p_lambda + 1e-12)

        importance = utils.append_dims(importance, 4)
        loss = 0.5 * (importance * (eps_pred - eps) ** 2).mean()

        if importance_monitor is not None:
            importance_monitor(
                logsnr=logsnr,
                importance=importance,
            ) 

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
        # Note: In VP models, α^2 - σ^2 = 1 => demon = 1
        denom = (alpha_t ** 2 + sigma_t ** 2)
        # ε = (σ x_t + α v) / (α^2 + σ^2)
        return (sigma_t * x_t + alpha_t * v_hat) / denom

    def to_x(self, model_out, x_t, sigma_t, alpha_t):
        v_hat = model_out
        denom = (alpha_t ** 2 + sigma_t ** 2)
        # x0 = (α x_t - σ v) / (α^2 + σ^2)
        return (alpha_t * x_t - sigma_t * v_hat) / denom

