"""PnP-Flow reconstruction conditioned on a fixed blur parameter.

Implements the second stage of the decoupled blind deconvolution pipeline.
The estimated sigma from blur-SURE is supplied to a standard PnP-Flow
trajectory which then performs the reconstruction under a known operator.
"""
from __future__ import annotations

import torch

from .forward import gaussian_blur_fft


def _model_velocity(model, x: torch.Tensor, t: torch.Tensor, model_type: str = "ot") -> torch.Tensor:
    if model_type == "ot":
        return model(x, t)
    raise ValueError(f"Unknown model_type '{model_type}'.")


def _lr_schedule(lr: float, t: float, style: str, alpha: float = 1.0) -> float:
    if style == "1_minus_t":
        return lr * (1.0 - t)
    if style == "sqrt_1_minus_t":
        return lr * (1.0 - t) ** 0.5
    if style == "alpha_1_minus_t":
        return lr * (1.0 - t) ** alpha
    if style == "constant":
        return lr
    raise ValueError(f"Unknown gamma_style '{style}'.")


def _interpolate(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t_vec = t.view(-1, 1, 1, 1)
    return t_vec * x + (1.0 - t_vec) * torch.randn_like(x)


def _denoiser(model, x: torch.Tensor, t: torch.Tensor, model_type: str = "ot") -> torch.Tensor:
    t_vec = t.view(-1, 1, 1, 1)
    v = _model_velocity(model, x, t, model_type)
    return x + (1.0 - t_vec) * v


def pnp_flow_reconstruct(
    model,
    *,
    y: torch.Tensor,
    sigma_blur: float,
    num_steps: int = 100,
    lr: float = 1.0,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
    x_init: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a PnP-Flow trajectory with sigma_blur fixed and return the
    reconstructed image. y is the degraded observation in (B, C, H, W).
    """
    device = y.device
    x = y.clone() if x_init is None else x_init.clone()
    delta = 1.0 / num_steps

    with torch.no_grad():
        for k in range(num_steps):
            t_scalar = delta * k
            t = torch.full((len(x),), t_scalar, device=device)

            residual = gaussian_blur_fft(x, sigma_blur) - y
            grad = gaussian_blur_fft(residual, sigma_blur)
            lr_k = _lr_schedule(lr, t_scalar, gamma_style, alpha)
            z = x - lr_k * grad

            x_new = torch.zeros_like(x)
            for _ in range(num_samples):
                z_tilde = _interpolate(z, t)
                x_new = x_new + _denoiser(model, z_tilde, t, model_type)
            x = x_new / num_samples

    return x.detach()


def pnp_flow_reconstruct_tracked(
    model,
    *,
    y: torch.Tensor,
    sigma_blur: float,
    x_clean: torch.Tensor,
    num_steps: int = 100,
    lr: float = 1.0,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
    x_init: torch.Tensor | None = None,
):
    """Same trajectory as ``pnp_flow_reconstruct`` but records convergence
    diagnostics at every step:

        x_l2_error_history    : ||x_t - x_clean||_2
        objective_history     : ||H_sigma(x_t) - y||^2
        psnr_history          : PSNR(x_t, x_clean) on the [0, 1] postprocessed scale

    Used only for a small subset of images per (sigma, eta) cell to keep
    output size manageable. Returns (x_final, history).
    """
    import math
    device = y.device
    x = y.clone() if x_init is None else x_init.clone()
    delta = 1.0 / num_steps
    history = {
        "x_l2_error_history": [],
        "objective_history": [],
        "psnr_history": [],
    }

    def _record(x_tensor: torch.Tensor) -> None:
        x_pp = ((x_tensor.clamp(-1, 1) + 1) / 2)
        x_clean_pp = ((x_clean + 1) / 2)
        l2 = float(torch.linalg.vector_norm(x_tensor.clamp(-1, 1) - x_clean).item())
        mse = float(torch.mean((x_pp - x_clean_pp) ** 2).item())
        psnr = float("inf") if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)
        obj = float(torch.mean((gaussian_blur_fft(x_tensor, sigma_blur) - y) ** 2).item())
        history["x_l2_error_history"].append(l2)
        history["objective_history"].append(obj)
        history["psnr_history"].append(psnr)

    with torch.no_grad():
        _record(x)
        for k in range(num_steps):
            t_scalar = delta * k
            t = torch.full((len(x),), t_scalar, device=device)

            residual = gaussian_blur_fft(x, sigma_blur) - y
            grad = gaussian_blur_fft(residual, sigma_blur)
            lr_k = _lr_schedule(lr, t_scalar, gamma_style, alpha)
            z = x - lr_k * grad

            x_new = torch.zeros_like(x)
            for _ in range(num_samples):
                z_tilde = _interpolate(z, t)
                x_new = x_new + _denoiser(model, z_tilde, t, model_type)
            x = x_new / num_samples
            _record(x)

    return x.detach(), history
