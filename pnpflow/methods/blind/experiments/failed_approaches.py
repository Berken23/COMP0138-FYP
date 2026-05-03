"""Reproduces every failed sigma estimation approach reported in the
thesis with the specific numerical values referenced in the text and
tables.

Each approach has its own function and writes a JSON file under
results/blind/failed_approaches/<approach>/. Run one or all via the CLI.

Implemented approaches:

    a14            A.14 per step Adam (re-runs the existing baseline here for
                   convenience and produces the same metrics)
    sgd            Plain SGD, no momentum, identical drift to Adam
    time_lr        Time restricted updates (t<0.3) with exponential LR decay
    convergence    Convergence detection by freezing on small |delta_sigma|
    grid_warm      Decoupled grid search with warm start
    grid_fresh     Decoupled grid search with fresh start
    em_tweedie     EM with Tweedie estimates, no damping
    em_damped      EM with damping (eta=0.1)
    morozov        Morozov stopping rule on top of EM with damping
    whiteness      Residual whiteness sigma estimation
    type2          Type II marginal likelihood with operator mismatch
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import numpy as np
import torch

from .._helpers import (
    SEED,
    TimingTracker,
    find_repo_root,
    load_model,
    pnp_flow_trajectory_blind,
    save_json,
)
from ..data import load_celeba
from ..evaluation import evaluate
from ..forward import gaussian_blur_fft
from ..reconstruction import _denoiser, _interpolate, _lr_schedule, pnp_flow_reconstruct
from ..sigma_estimation import compute_blur_sure


SIGMA_TRUE = 1.5
SIGMA_INIT = 3.0
SIGMA_MAX = 5.0
SIGMA_MIN = 0.1
NOISE_STD = 0.05
PNP_STEPS = 100


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_observation(x_gt: torch.Tensor, sigma: float, noise_std: float, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return gaussian_blur_fft(x_gt, sigma) + torch.randn_like(x_gt) * noise_std


def _data_residual_sq(y: torch.Tensor, x: torch.Tensor, sigma: float) -> float:
    """Return ||H_sigma(x) - y||^2 / N (mean squared)."""
    return float(torch.mean((gaussian_blur_fft(x, sigma) - y) ** 2).item())


def _grid_search_sigma(
    x_hat: torch.Tensor,
    y: torch.Tensor,
    n_grid: int = 50,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
) -> float:
    """Find argmin_sigma ||H_sigma(x_hat) - y||^2 over a 1D grid with parabolic refinement."""
    grid = torch.linspace(sigma_min, sigma_max, n_grid)
    losses = [_data_residual_sq(y, x_hat, float(s)) for s in grid]
    j = int(torch.tensor(losses).argmin().item())
    if 0 < j < n_grid - 1:
        s0, s1, s2 = float(grid[j - 1]), float(grid[j]), float(grid[j + 1])
        l0, l1, l2 = losses[j - 1], losses[j], losses[j + 1]
        denom = (l0 - 2 * l1 + l2)
        if abs(denom) > 1e-12:
            return s1 - 0.5 * (s2 - s0) * (l2 - l0) / (2.0 * denom)
    return float(grid[j])


def _golden_section(
    f, lo: float, hi: float, tol: float = 1e-3, max_iter: int = 60,
) -> float:
    """Standard golden section search for unimodal min over [lo, hi]."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Approach 1: A.14 Adam baseline (alias, calls existing implementation)
# ---------------------------------------------------------------------------

def run_a14(model, images: List[torch.Tensor], outer_iters: int) -> Dict:
    """A.14 per step Adam — same as a14_baseline but runs alongside the others
    so the failed_approaches output is self contained."""
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        x = y.clone()
        sigma_history = []
        for _ in range(outer_iters):
            x, sigma_final, _ = pnp_flow_trajectory_blind(
                model, x_init=x, y=y,
                sigma_init=sigma_history[-1] if sigma_history else SIGMA_INIT,
                sigma_max=SIGMA_MAX, num_steps=PNP_STEPS,
            )
            sigma_history.append(sigma_final)
        m = evaluate(x, x_gt)
        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "sigma_final": sigma_final,
            "sigma_error": abs(sigma_final - SIGMA_TRUE),
            "psnr": m["psnr"],
        })
        print(f"  [A.14    img {idx}] sigma={sigma_final:.4f} err={records[-1]['sigma_error']:.4f}")
    return {
        "approach": "a14_adam",
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "psnr_mean": float(np.mean([r["psnr"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Approach 2: SGD variant (no momentum, no adaptive scaling)
# ---------------------------------------------------------------------------

def _blind_trajectory_with_optimiser(
    model, *, x_init, y, sigma_init, sigma_max, num_steps,
    optimiser_factory, only_when_t_below: float | None = None,
):
    """Generic blind PnP-Flow trajectory parameterised by optimiser factory.
    optimiser_factory takes the operator's parameters and returns a torch.optim.Optimizer.
    """
    from pnpflow.blind_degradations import LearnableGaussianBlur
    device = x_init.device
    op = LearnableGaussianBlur(
        init_sigma=sigma_init, sigma_max=sigma_max, padding="reflect",
    ).to(device=device, dtype=x_init.dtype)
    opt = optimiser_factory(op.parameters())

    x = x_init.clone()
    delta = 1.0 / num_steps
    sigma_history: List[float] = []

    for k in range(num_steps):
        t_scalar = delta * k
        t = torch.full((len(x),), t_scalar, device=device)
        with torch.no_grad():
            residual = op(x) - y
            grad = op(residual)
            lr_k = _lr_schedule(1.0, t_scalar, "1_minus_t")
            z = x - lr_k * grad
        with torch.no_grad():
            z_tilde = _interpolate(z, t)
            x = _denoiser(model, z_tilde, t, "ot")

        if only_when_t_below is None or t_scalar < only_when_t_below:
            opt.zero_grad(set_to_none=True)
            loss_kernel = torch.mean((y - op(x.detach())) ** 2)
            loss_kernel.backward()
            torch.nn.utils.clip_grad_norm_(op.parameters(), 1.0)
            opt.step()
            op.clamp_params_()
        sigma_history.append(op.sigma().item())

    return x.detach(), op.sigma().item(), sigma_history


def run_sgd(model, images: List[torch.Tensor], outer_iters: int) -> Dict:
    """Replace Adam with plain SGD. Same drift behaviour expected."""
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        x = y.clone()
        sigma_current = SIGMA_INIT
        sigma_history = []
        for _ in range(outer_iters):
            x, sigma_current, _ = _blind_trajectory_with_optimiser(
                model, x_init=x, y=y, sigma_init=sigma_current,
                sigma_max=SIGMA_MAX, num_steps=PNP_STEPS,
                optimiser_factory=lambda params: torch.optim.SGD(params, lr=1e-3),
            )
            sigma_history.append(sigma_current)
        m = evaluate(x, x_gt)
        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "sigma_final": sigma_current,
            "sigma_error": abs(sigma_current - SIGMA_TRUE),
            "psnr": m["psnr"],
        })
        print(f"  [SGD     img {idx}] sigma={sigma_current:.4f} err={records[-1]['sigma_error']:.4f}")
    return {
        "approach": "sgd",
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "psnr_mean": float(np.mean([r["psnr"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Approach 3: Time restricted updates with LR decay
# ---------------------------------------------------------------------------

def run_time_lr_decay(
    model, images: List[torch.Tensor], outer_iters: int,
    decay: float, t_threshold: float = 0.3,
) -> Dict:
    """Adam updates only for t<t_threshold and the operator LR decays
    exponentially across outer iterations.
    """
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        x = y.clone()
        sigma_current = SIGMA_INIT
        sigma_history = []
        for outer in range(outer_iters):
            outer_lr = 1e-3 * (decay ** outer)
            x, sigma_current, _ = _blind_trajectory_with_optimiser(
                model, x_init=x, y=y, sigma_init=sigma_current,
                sigma_max=SIGMA_MAX, num_steps=PNP_STEPS,
                optimiser_factory=lambda params: torch.optim.Adam(params, lr=outer_lr),
                only_when_t_below=t_threshold,
            )
            sigma_history.append(sigma_current)
        m = evaluate(x, x_gt)
        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "sigma_final": sigma_current,
            "sigma_error": abs(sigma_current - SIGMA_TRUE),
            "psnr": m["psnr"],
        })
        print(f"  [TimeLR  img {idx} decay={decay}] sigma={sigma_current:.4f} err={records[-1]['sigma_error']:.4f}")
    return {
        "approach": "time_lr",
        "decay": decay,
        "t_threshold": t_threshold,
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "psnr_mean": float(np.mean([r["psnr"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Approach 4: Convergence detection by freezing on small delta sigma
# ---------------------------------------------------------------------------

def run_convergence_detection(
    model, images: List[torch.Tensor], outer_iters: int,
    delta_threshold: float = 0.01, min_iters_before_freeze: int = 15,
    consecutive_required: int = 3,
) -> Dict:
    """A.14 with a stopping rule that freezes sigma when |delta sigma| stays
    below a threshold for a number of consecutive iterations.
    """
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        x = y.clone()
        sigma_history = []
        consecutive_small = 0
        frozen_at = None

        for outer in range(outer_iters):
            if frozen_at is not None:
                continue
            x, sigma_final, _ = pnp_flow_trajectory_blind(
                model, x_init=x, y=y,
                sigma_init=sigma_history[-1] if sigma_history else SIGMA_INIT,
                sigma_max=SIGMA_MAX, num_steps=PNP_STEPS,
            )
            sigma_history.append(sigma_final)
            if outer >= min_iters_before_freeze and len(sigma_history) >= 2:
                if abs(sigma_history[-1] - sigma_history[-2]) < delta_threshold:
                    consecutive_small += 1
                else:
                    consecutive_small = 0
                if consecutive_small >= consecutive_required:
                    frozen_at = sigma_history[-1]

        sigma_final_used = frozen_at if frozen_at is not None else sigma_history[-1]
        m = evaluate(x, x_gt)
        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "frozen_at": frozen_at,
            "sigma_final": sigma_final_used,
            "sigma_error": abs(sigma_final_used - SIGMA_TRUE),
            "psnr": m["psnr"],
        })
        print(f"  [Conv    img {idx}] frozen={frozen_at} sigma={sigma_final_used:.4f} err={records[-1]['sigma_error']:.4f}")
    return {
        "approach": "convergence_detection",
        "delta_threshold": delta_threshold,
        "min_iters_before_freeze": min_iters_before_freeze,
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Approach 5: Decoupled alternating minimisation with grid search
# ---------------------------------------------------------------------------

def run_decoupled_grid(
    model, images: List[torch.Tensor], outer_iters: int, warm_start: bool,
) -> Dict:
    """Alternating minimisation: phase 1 fixes sigma and runs full PnP-Flow,
    phase 2 fixes x_hat and grid searches over sigma.
    """
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        sigma_current = SIGMA_INIT
        sigma_history = []
        x = y.clone()

        for outer in range(outer_iters):
            x_init = x if warm_start else y.clone()
            x = pnp_flow_reconstruct(
                model, y=y, sigma_blur=sigma_current, num_steps=PNP_STEPS, x_init=x_init,
            )
            sigma_current = _grid_search_sigma(x.detach(), y, n_grid=50)
            sigma_history.append(sigma_current)

        m = evaluate(x, x_gt)
        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "sigma_final": sigma_current,
            "sigma_error": abs(sigma_current - SIGMA_TRUE),
            "psnr": m["psnr"],
        })
        tag = "warm" if warm_start else "fresh"
        print(f"  [Grid {tag} img {idx}] sigma_path={[f'{s:.2f}' for s in sigma_history[:5]]}... final={sigma_current:.4f}")
    return {
        "approach": f"grid_{'warm' if warm_start else 'fresh'}",
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Approach 6: EM with Tweedie estimates
# ---------------------------------------------------------------------------

def _collect_tweedie_estimates(
    model, *, x_init, y, sigma_blur, num_steps,
) -> List[torch.Tensor]:
    """Run a non-blind PnP-Flow trajectory and return all Tweedie estimates
    (denoiser outputs at each step)."""
    device = x_init.device
    x = x_init.clone()
    delta = 1.0 / num_steps
    estimates: List[torch.Tensor] = []
    with torch.no_grad():
        for k in range(num_steps):
            t_scalar = delta * k
            t = torch.full((len(x),), t_scalar, device=device)
            residual = gaussian_blur_fft(x, sigma_blur) - y
            grad = gaussian_blur_fft(residual, sigma_blur)
            lr_k = _lr_schedule(1.0, t_scalar, "1_minus_t")
            z = x - lr_k * grad
            z_tilde = _interpolate(z, t)
            x_hat_t = _denoiser(model, z_tilde, t, "ot")
            estimates.append(x_hat_t.detach().clone())
            x = x_hat_t
    return estimates


def _q_function(sigma: float, estimates: List[torch.Tensor], y: torch.Tensor) -> float:
    return float(np.mean([
        torch.mean((gaussian_blur_fft(x_hat, sigma) - y) ** 2).item()
        for x_hat in estimates
    ]))


def run_em_tweedie(
    model, images: List[torch.Tensor], em_iters: int, damping: float | None,
    morozov: bool = False,
) -> Dict:
    """EM with Tweedie estimates. damping=None means no damping. morozov=True
    enables the discrepancy stopping rule."""
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        sigma_current = SIGMA_INIT
        sigma_history = [sigma_current]
        residual_history: List[float] = []
        mstep_history: List[float] = []
        morozov_consecutive_increase = 0
        morozov_min_residual = float("inf")
        morozov_min_sigma = sigma_current
        morozov_stopped_at = None

        for outer in range(em_iters):
            # E-step: collect Tweedie estimates with current sigma
            estimates = _collect_tweedie_estimates(
                model, x_init=y.clone(), y=y, sigma_blur=sigma_current, num_steps=PNP_STEPS,
            )
            # M-step: minimise Q(sigma)
            sigma_mstep = _golden_section(
                lambda s: _q_function(s, estimates, y),
                lo=SIGMA_MIN, hi=SIGMA_MAX, tol=1e-3, max_iter=40,
            )
            mstep_history.append(sigma_mstep)

            if damping is None:
                sigma_current = sigma_mstep
            else:
                sigma_current = (1.0 - damping) * sigma_current + damping * sigma_mstep
            sigma_history.append(sigma_current)

            # Final reconstruction with current sigma for diagnostic residual
            x_hat = pnp_flow_reconstruct(model, y=y, sigma_blur=sigma_current, num_steps=PNP_STEPS)
            residual = _data_residual_sq(y, x_hat.detach(), sigma_current)
            residual_history.append(residual)

            if morozov:
                if outer >= 15:
                    if residual < morozov_min_residual:
                        morozov_min_residual = residual
                        morozov_min_sigma = sigma_current
                        morozov_consecutive_increase = 0
                    else:
                        morozov_consecutive_increase += 1
                    if morozov_consecutive_increase >= 3 and morozov_stopped_at is None:
                        morozov_stopped_at = outer

        if morozov:
            sigma_final = morozov_min_sigma if morozov_stopped_at is not None else sigma_current
        else:
            sigma_final = sigma_current

        x_hat = pnp_flow_reconstruct(model, y=y, sigma_blur=sigma_final, num_steps=PNP_STEPS)
        m = evaluate(x_hat, x_gt)

        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "mstep_history": mstep_history,
            "residual_history": residual_history,
            "sigma_final": sigma_final,
            "sigma_error": abs(sigma_final - SIGMA_TRUE),
            "psnr": m["psnr"],
            "morozov_stopped_at": morozov_stopped_at if morozov else None,
        })
        tag = "EM-D" if damping is not None else "EM-T"
        if morozov:
            tag = "EM-Mo"
        print(f"  [{tag:<5} img {idx}] sigma={sigma_final:.4f} err={records[-1]['sigma_error']:.4f}")

    name = "em_tweedie" if damping is None else ("morozov" if morozov else "em_damped")
    return {
        "approach": name,
        "damping": damping,
        "morozov": morozov,
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "psnr_mean": float(np.mean([r["psnr"] for r in records])),
    }


# ---------------------------------------------------------------------------
# Approach 9: Residual whiteness
# ---------------------------------------------------------------------------

def _whiteness_imbalance(residual: torch.Tensor) -> float:
    """Compute the signed low/high frequency power imbalance of a residual,
    expressed in decibels. Positive value indicates excess low frequency
    content (suggests sigma too small).
    """
    res = residual.squeeze(0) if residual.ndim == 4 else residual
    _, H, W = res.shape
    Y = torch.fft.fft2(res.float())
    P = (Y.abs() ** 2).mean(dim=0)
    fy = torch.fft.fftfreq(H, device=res.device)
    fx = torch.fft.fftfreq(W, device=res.device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(FY ** 2 + FX ** 2)
    cutoff = 0.25
    low = float(P[radius < cutoff].mean().item()) + 1e-12
    high = float(P[radius >= cutoff].mean().item()) + 1e-12
    return 10.0 * math.log10(low / high)


def run_whiteness(
    model, images: List[torch.Tensor], iters: int,
) -> Dict:
    """At each iteration, reconstruct with current sigma then compute the
    residual whiteness imbalance. The reported gradient signal is the imbalance.
    """
    records = []
    for idx, x_gt in enumerate(images):
        y = _make_observation(x_gt, SIGMA_TRUE, NOISE_STD, SEED + idx)
        sigma_current = SIGMA_INIT
        gradient_history = []
        sigma_history = [sigma_current]

        for _ in range(iters):
            x_hat = pnp_flow_reconstruct(model, y=y, sigma_blur=sigma_current, num_steps=PNP_STEPS)
            residual = y - gaussian_blur_fft(x_hat, sigma_current)
            grad_signal = _whiteness_imbalance(residual)
            gradient_history.append(grad_signal)
            sigma_current = max(SIGMA_MIN, min(SIGMA_MAX, sigma_current - 0.05 * grad_signal))
            sigma_history.append(sigma_current)

        records.append({
            "index": idx,
            "sigma_history": sigma_history,
            "gradient_history": gradient_history,
            "sigma_final": sigma_current,
            "sigma_error": abs(sigma_current - SIGMA_TRUE),
        })
        print(f"  [White   img {idx}] grad_range=[{min(gradient_history):.2f},{max(gradient_history):.2f}] sigma={sigma_current:.4f}")
    return {
        "approach": "whiteness",
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "gradient_overall_min": float(min(g for r in records for g in r["gradient_history"])),
        "gradient_overall_max": float(max(g for r in records for g in r["gradient_history"])),
    }


# ---------------------------------------------------------------------------
# Approach 10: Type II marginal likelihood with operator mismatch
# ---------------------------------------------------------------------------

def _type2_gradient(y: torch.Tensor, sigma: float, noise_std: float, prior_std: float) -> float:
    """Analytical gradient of log p(y | sigma) under a diagonal Gaussian
    image prior with std prior_std. Computed in the FFT domain.
    """
    y_sq = y.squeeze(0) if y.ndim == 4 else y
    _, H, W = y_sq.shape
    fy = torch.fft.fftfreq(H, device=y_sq.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(W, device=y_sq.device, dtype=torch.float32)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    freq_sq = (FY ** 2 + FX ** 2) * (2.0 * torch.pi) ** 2
    H_sq = torch.exp(-sigma ** 2 * freq_sq)
    var_y = prior_std ** 2 * H_sq + noise_std ** 2
    Y_f = torch.fft.fft2(y_sq.float())
    Y_pow = (Y_f.abs() ** 2).mean(dim=0)
    dH_dsigma_sq = -2.0 * sigma * freq_sq * H_sq
    dvar_dsigma = prior_std ** 2 * dH_dsigma_sq
    grad = -0.5 * dvar_dsigma / var_y + 0.5 * Y_pow * dvar_dsigma / (var_y ** 2)
    return float(grad.sum().item())


def run_type2_with_mismatch(images: List[torch.Tensor], iters: int) -> Dict:
    """Generate observations with LearnableGaussianBlur (reflect padding) but
    estimate using FFT model. Reproduces the operator mismatch failure where
    the gradient is enormous and uninformative."""
    from pnpflow.blind_degradations import LearnableGaussianBlur

    records = []
    for idx, x_gt in enumerate(images):
        torch.manual_seed(SEED + idx)
        op_gen = LearnableGaussianBlur(
            init_sigma=SIGMA_TRUE, sigma_max=SIGMA_MAX, padding="reflect",
        ).to(device=x_gt.device, dtype=x_gt.dtype)
        with torch.no_grad():
            y = op_gen(x_gt) + torch.randn_like(x_gt) * NOISE_STD

        sigma_current = SIGMA_INIT
        gradient_history = []
        sigma_history = [sigma_current]
        for _ in range(iters):
            grad = _type2_gradient(y, sigma_current, NOISE_STD, prior_std=0.5)
            gradient_history.append(grad)
            sigma_current = max(SIGMA_MIN, min(SIGMA_MAX, sigma_current - 1e-6 * grad))
            sigma_history.append(sigma_current)

        records.append({
            "index": idx,
            "gradient_history": gradient_history,
            "sigma_history": sigma_history,
            "sigma_final": sigma_current,
            "sigma_error": abs(sigma_current - SIGMA_TRUE),
        })
        print(f"  [Type2   img {idx}] grad~{gradient_history[0]:.0f} sigma={sigma_current:.4f}")
    return {
        "approach": "type2_marginal_likelihood",
        "per_image": records,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "gradient_overall_mean": float(np.mean([g for r in records for g in r["gradient_history"]])),
        "gradient_constancy_std": float(np.std([g for r in records for g in r["gradient_history"]])),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_APPROACHES = [
    "a14",
    "sgd",
    "time_lr",
    "convergence",
    "grid_warm",
    "grid_fresh",
    "em_tweedie",
    "em_damped",
    "morozov",
    "whiteness",
    "type2",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproduce every failed sigma estimation approach with numerical results"
    )
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--outer-iters", type=int, default=30,
                   help="Outer iterations for the alternating approaches")
    p.add_argument("--em-iters", type=int, default=40,
                   help="Outer iterations for the EM approaches")
    p.add_argument("--whiteness-iters", type=int, default=20)
    p.add_argument("--type2-iters", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--approaches", nargs="+", default=ALL_APPROACHES,
                   choices=ALL_APPROACHES + ["all"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    output_dir = args.output_dir or os.path.join(repo_root, "results", "blind", "failed_approaches")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model")
    model = load_model(repo_root, args.device)
    print(f"Loading {args.num_images} CelebA images")
    images = load_celeba(repo_root, args.device, num_images=args.num_images, seed=SEED)

    selected = ALL_APPROACHES if "all" in args.approaches else args.approaches
    tt = TimingTracker()

    summary: Dict[str, Dict] = {}

    if "a14" in selected:
        with tt.track("a14"):
            r = run_a14(model, images, outer_iters=args.outer_iters)
        save_json(os.path.join(output_dir, "a14.json"), r)
        summary["a14"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "sgd" in selected:
        with tt.track("sgd"):
            r = run_sgd(model, images, outer_iters=args.outer_iters)
        save_json(os.path.join(output_dir, "sgd.json"), r)
        summary["sgd"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "time_lr" in selected:
        with tt.track("time_lr_moderate"):
            r_mod = run_time_lr_decay(model, images, outer_iters=args.outer_iters, decay=0.95)
        with tt.track("time_lr_aggressive"):
            r_agg = run_time_lr_decay(model, images, outer_iters=args.outer_iters, decay=0.7)
        save_json(os.path.join(output_dir, "time_lr_moderate.json"), r_mod)
        save_json(os.path.join(output_dir, "time_lr_aggressive.json"), r_agg)
        summary["time_lr_moderate"] = {"sigma_error_mean": r_mod["sigma_error_mean"]}
        summary["time_lr_aggressive"] = {"sigma_error_mean": r_agg["sigma_error_mean"]}

    if "convergence" in selected:
        with tt.track("convergence"):
            r = run_convergence_detection(model, images, outer_iters=args.outer_iters)
        save_json(os.path.join(output_dir, "convergence.json"), r)
        summary["convergence"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "grid_warm" in selected:
        with tt.track("grid_warm"):
            r = run_decoupled_grid(model, images, outer_iters=args.outer_iters, warm_start=True)
        save_json(os.path.join(output_dir, "grid_warm.json"), r)
        summary["grid_warm"] = {"sigma_error_mean": r["sigma_error_mean"]}
    if "grid_fresh" in selected:
        with tt.track("grid_fresh"):
            r = run_decoupled_grid(model, images, outer_iters=args.outer_iters, warm_start=False)
        save_json(os.path.join(output_dir, "grid_fresh.json"), r)
        summary["grid_fresh"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "em_tweedie" in selected:
        with tt.track("em_tweedie"):
            r = run_em_tweedie(model, images, em_iters=args.em_iters, damping=None)
        save_json(os.path.join(output_dir, "em_tweedie.json"), r)
        summary["em_tweedie"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "em_damped" in selected:
        with tt.track("em_damped"):
            r = run_em_tweedie(model, images, em_iters=args.em_iters, damping=0.1)
        save_json(os.path.join(output_dir, "em_damped.json"), r)
        summary["em_damped"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "morozov" in selected:
        with tt.track("morozov"):
            r = run_em_tweedie(model, images, em_iters=args.em_iters, damping=0.1, morozov=True)
        save_json(os.path.join(output_dir, "morozov.json"), r)
        summary["morozov"] = {"sigma_error_mean": r["sigma_error_mean"]}

    if "whiteness" in selected:
        with tt.track("whiteness"):
            r = run_whiteness(model, images, iters=args.whiteness_iters)
        save_json(os.path.join(output_dir, "whiteness.json"), r)
        summary["whiteness"] = {
            "sigma_error_mean": r["sigma_error_mean"],
            "gradient_range": [r["gradient_overall_min"], r["gradient_overall_max"]],
        }

    if "type2" in selected:
        with tt.track("type2"):
            r = run_type2_with_mismatch(images, iters=args.type2_iters)
        save_json(os.path.join(output_dir, "type2.json"), r)
        summary["type2"] = {
            "sigma_error_mean": r["sigma_error_mean"],
            "gradient_overall_mean": r["gradient_overall_mean"],
        }

    save_json(os.path.join(output_dir, "summary.json"), summary)
    print()
    print("Summary:")
    for name, s in summary.items():
        print(f"  {name:<22}: {s}")
    print()
    print(tt.summary())


if __name__ == "__main__":
    main()
