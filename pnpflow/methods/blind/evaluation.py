"""Evaluation metrics for the blind deconvolution pipeline.

Implements PSNR, SSIM, and LPIPS for tensors in [-1, 1], plus a non blind
oracle wrapper that runs reconstruction with the true blur parameter for
the PSNR gap measurement reported throughout the thesis.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from skimage.metrics import structural_similarity as compare_ssim

from .reconstruction import pnp_flow_reconstruct


_LPIPS_FN = None


def _get_lpips(device):
    global _LPIPS_FN
    if _LPIPS_FN is None:
        import lpips
        _LPIPS_FN = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS_FN


def postprocess(x: torch.Tensor) -> torch.Tensor:
    """Map a tensor in [-1, 1] to [0, 1] with clipping."""
    return (x.clamp(-1, 1) + 1) / 2


def psnr_db(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [-1, 1]."""
    mse = torch.mean((postprocess(x_hat) - postprocess(x_gt)) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def compute_ssim(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """SSIM between two tensors in [-1, 1]."""
    img_hat = postprocess(x_hat).squeeze(0).cpu().permute(1, 2, 0).numpy()
    img_gt = postprocess(x_gt).squeeze(0).cpu().permute(1, 2, 0).numpy()
    return float(compare_ssim(img_gt, img_hat, data_range=1.0, channel_axis=2))


def compute_lpips(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """LPIPS between two tensors in [-1, 1]. Lower is better."""
    device = x_hat.device
    lpips_fn = _get_lpips(device)
    with torch.no_grad():
        x_h = x_hat.float()
        x_g = x_gt.float().to(device)
        if x_h.ndim == 3:
            x_h = x_h.unsqueeze(0)
        if x_g.ndim == 3:
            x_g = x_g.unsqueeze(0)
        return float(lpips_fn(x_h, x_g).item())


def evaluate(x_hat: torch.Tensor, x_gt: torch.Tensor) -> Dict[str, float]:
    """Compute PSNR, SSIM, and LPIPS in a single call."""
    return {
        "psnr": psnr_db(x_hat, x_gt),
        "ssim": compute_ssim(x_hat, x_gt),
        "lpips": compute_lpips(x_hat, x_gt),
    }


def non_blind_oracle(
    model,
    *,
    y: torch.Tensor,
    sigma_true: float,
    num_steps: int = 100,
    lr: float = 1.0,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
) -> torch.Tensor:
    """Run reconstruction with the true blur parameter. Output serves as the
    upper bound against which the blind estimator is compared via PSNR gap.
    """
    return pnp_flow_reconstruct(
        model,
        y=y,
        sigma_blur=sigma_true,
        num_steps=num_steps,
        lr=lr,
        gamma_style=gamma_style,
        alpha=alpha,
        num_samples=num_samples,
        model_type=model_type,
    )
