"""Blur-SURE sigma estimation.

Recovers the blur parameter from the degraded observation alone, without
referring to any reconstruction. The estimator is an unbiased estimate of
the blur mean squared error and its minimum coincides with the true blur
parameter, satisfying the independence and operator consistency conditions
required for sound blind estimation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from scipy.optimize import minimize_scalar


def compute_blur_sure(
    y: torch.Tensor,
    sigma: float,
    noise_var: float,
    lam: Optional[float] = None,
) -> float:
    """Blur-SURE evaluated at a candidate sigma.

    Returns an unbiased estimate of the blur mean squared error using only
    the observation y, the candidate sigma, and the noise variance.
    """
    if sigma <= 0:
        return float("inf")
    if lam is None:
        lam = noise_var

    y_sq = y.squeeze(0) if y.ndim == 4 else y
    _, H, W = y_sq.shape

    fy = torch.fft.fftfreq(H, device=y_sq.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(W, device=y_sq.device, dtype=torch.float32)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    freq_sq = (FY ** 2 + FX ** 2) * (2.0 * torch.pi) ** 2

    H_sigma = torch.exp(-0.5 * sigma ** 2 * freq_sq)
    H_sq = H_sigma ** 2
    F_sigma = H_sq / (H_sq + lam)

    Y_f = torch.fft.fft2(y_sq.float())
    diff = torch.fft.ifft2(F_sigma.unsqueeze(0) * Y_f - Y_f).real

    term1 = (diff ** 2).mean().item()
    term2 = 2.0 * noise_var * F_sigma.mean().item()
    return term1 - noise_var + term2


def estimate_sigma_blur_sure(
    y: torch.Tensor,
    noise_var: float,
    sigma_min: float = 0.3,
    sigma_max: float = 8.0,
    n_grid: int = 80,
    lam: Optional[float] = None,
    refine: bool = True,
) -> Dict:
    """Estimate sigma by minimising blur-SURE over a 1D grid with optional
    bounded refinement. Uses y only.

    Returns a dict with keys sigma_star, sure_curve, sigma_grid, sigma_coarse.
    """
    if lam is None:
        lam = noise_var

    sigma_grid = torch.linspace(sigma_min, sigma_max, n_grid)
    sure_values: List[float] = []
    with torch.no_grad():
        for s in sigma_grid:
            sure_values.append(compute_blur_sure(y, float(s), noise_var, lam))
    sure_curve = torch.tensor(sure_values)

    idx_min = int(sure_curve.argmin().item())
    sigma_coarse = float(sigma_grid[idx_min])

    if refine:
        spacing = (sigma_max - sigma_min) / (n_grid - 1)
        lo = max(sigma_min, sigma_coarse - 2 * spacing)
        hi = min(sigma_max, sigma_coarse + 2 * spacing)
        res = minimize_scalar(
            lambda s: compute_blur_sure(y, float(s), noise_var, lam),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 50},
        )
        sigma_star = float(res.x)
    else:
        sigma_star = sigma_coarse

    return {
        "sigma_star": sigma_star,
        "sure_curve": sure_curve,
        "sigma_grid": sigma_grid,
        "sigma_coarse": sigma_coarse,
    }


def estimate_sigma_multi_sure(
    ys: List[torch.Tensor],
    noise_var: float,
    sigma_min: float = 0.3,
    sigma_max: float = 8.0,
    n_grid: int = 80,
    lam: Optional[float] = None,
    refine: bool = True,
) -> Dict:
    """Estimate sigma by averaging blur-SURE curves across B observations.
    Variance scales as 1/B.
    """
    if lam is None:
        lam = noise_var

    sigma_grid = torch.linspace(sigma_min, sigma_max, n_grid)
    all_curves: List[torch.Tensor] = []
    for y in ys:
        curve: List[float] = []
        with torch.no_grad():
            for s in sigma_grid:
                curve.append(compute_blur_sure(y, float(s), noise_var, lam))
        all_curves.append(torch.tensor(curve))
    avg_curve = torch.stack(all_curves).mean(dim=0)

    idx_min = int(avg_curve.argmin().item())
    sigma_coarse = float(sigma_grid[idx_min])

    if refine:
        spacing = (sigma_max - sigma_min) / (n_grid - 1)
        lo = max(sigma_min, sigma_coarse - 2 * spacing)
        hi = min(sigma_max, sigma_coarse + 2 * spacing)

        def avg_sure(s: float) -> float:
            return float(
                sum(compute_blur_sure(y, s, noise_var, lam) for y in ys) / len(ys)
            )

        res = minimize_scalar(
            avg_sure,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-4, "maxiter": 50},
        )
        sigma_star = float(res.x)
    else:
        sigma_star = sigma_coarse

    return {
        "sigma_star": sigma_star,
        "sure_curve": avg_curve,
        "sigma_grid": sigma_grid,
        "sigma_coarse": sigma_coarse,
        "batch_size": len(ys),
    }
