from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from pnpflow.blind_degradations import LearnableGaussianBlur


@dataclass(frozen=True)
class BlindGaussianBlurProblem:
    """
    Synthetic blind Gaussian deblurring measurement model.

    y = H_{sigma_true}(x_gt) + eps, eps ~ N(0, sigma_noise^2)

    - true_op is frozen (requires_grad=False)
    - learnable_op is trainable (created separately in Step 3)
    """
    x_gt: torch.Tensor
    y: torch.Tensor
    sigma_true: float
    sigma_noise: float
    true_op: LearnableGaussianBlur


def make_blind_gaussian_blur_problem(
    x_gt: torch.Tensor,
    sigma_true: float,
    sigma_noise: float,
    *,
    sigma_max: float,
    padding: str = "reflect",
    seed: Optional[int] = None,
) -> BlindGaussianBlurProblem:
    """
    Build a frozen true operator and generate measurement y.

    Requirements:
    - Uses LearnableGaussianBlur for truth (same family as learnable operator)
    - true_op is frozen (requires_grad_(False))
    - Deterministic noise when seed is provided
    - No numpy; all torch
    """
    if x_gt.ndim != 4:
        raise ValueError(f"x_gt must be (B,C,H,W), got {tuple(x_gt.shape)}")
    if sigma_true <= 0:
        raise ValueError("sigma_true must be > 0")
    if sigma_noise < 0:
        raise ValueError("sigma_noise must be >= 0")
    if sigma_max <= 0:
        raise ValueError("sigma_max must be > 0")
    if sigma_true > sigma_max:
        raise ValueError("sigma_true must be <= sigma_max to match fixed-kernel design")

    # --- Noise generation (portable, deterministic) ---
    if seed is not None:
        g = torch.Generator(device=x_gt.device)
        g.manual_seed(seed)
        noise = torch.randn(
            x_gt.shape,
            device=x_gt.device,
            dtype=x_gt.dtype,
            generator=g,
        )
    else:
        noise = torch.randn_like(x_gt)

    # --- True (frozen) operator ---
    true_op = LearnableGaussianBlur(
        init_sigma=sigma_true,
        sigma_max=sigma_max,
        padding=padding,
    ).to(device=x_gt.device, dtype=x_gt.dtype)
    true_op.requires_grad_(False)

    with torch.no_grad():
        y_clean = true_op(x_gt)
        y = y_clean + sigma_noise * noise

    return BlindGaussianBlurProblem(
        x_gt=x_gt,
        y=y,
        sigma_true=float(sigma_true),
        sigma_noise=float(sigma_noise),
        true_op=true_op,
    )


def operator_mse(op: LearnableGaussianBlur, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Convenience: MSE between op(x) and y.
    """
    pred = op(x)
    return torch.mean((pred - y) ** 2)
