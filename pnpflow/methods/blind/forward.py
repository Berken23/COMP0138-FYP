"""FFT domain Gaussian blur used as the forward operator throughout the
blind deconvolution pipeline. The operator is self adjoint for symmetric
zero phase Gaussian kernels so the same function serves as H and H^T.

The function accepts sigma either as a Python float or as a 0-D
``torch.Tensor``. When sigma is a tensor with ``requires_grad=True`` the
output is differentiable in sigma, enabling gradient based estimation of
the blur parameter via standard PyTorch optimisers.
"""
from __future__ import annotations

from typing import Union

import torch


def gaussian_blur_fft(
    x: torch.Tensor, sigma: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Apply isotropic Gaussian blur in the frequency domain.

    The transfer function is H(omega) = exp(-0.5 * sigma^2 * |omega|^2),
    which is real and non negative, so the operator equals its own adjoint.

    Accepts (B, C, H, W) or (C, H, W) tensors of any range. Sigma may be a
    Python float or a torch tensor. A small lower clamp keeps the kernel
    well defined when sigma is being optimised by gradient descent.
    """
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.tensor(float(sigma), device=x.device, dtype=torch.float32)
    sigma_eff = sigma.clamp(min=1e-6)

    squeeze = x.ndim == 3
    if squeeze:
        x = x.unsqueeze(0)

    _, _, H, W = x.shape
    fy = torch.fft.fftfreq(H, device=x.device, dtype=torch.float32)
    fx = torch.fft.fftfreq(W, device=x.device, dtype=torch.float32)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    freq_sq = (FY ** 2 + FX ** 2) * (2.0 * torch.pi) ** 2

    kernel = torch.exp(-0.5 * sigma_eff ** 2 * freq_sq).unsqueeze(0).unsqueeze(0)
    blurred = torch.fft.ifft2(kernel * torch.fft.fft2(x.float())).real

    return blurred.squeeze(0) if squeeze else blurred


def gaussian_blur_adjoint(
    x: torch.Tensor, sigma: Union[float, torch.Tensor]
) -> torch.Tensor:
    """Adjoint of the FFT Gaussian blur. Identical to the forward call
    because the kernel is real, symmetric, and zero phase."""
    return gaussian_blur_fft(x, sigma)
