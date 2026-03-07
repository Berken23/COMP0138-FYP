from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PaddingMode = Literal["reflect", "circular", "replicate", "constant"]


def _odd_ks_from_sigma_max(sigma_max: float, truncate: float = 3.0) -> int:
    if sigma_max <= 0:
        raise ValueError("sigma_max must be > 0")
    r = int(math.ceil(truncate * sigma_max))
    return 2 * r + 1


class LearnableGaussianBlur(nn.Module):
    """
    Learnable isotropic Gaussian blur (depthwise conv), with a single parameter sigma.

    Key properties (per your Step 1 spec):
    - Parameter is log_sigma (sigma = exp(log_sigma) > 0)
    - Kernel is built using torch ops only
    - Kernel is normalized (sum=1)
    - Depthwise convolution per channel
    - Padding is consistent
    - sigma is clamped to (eps, sigma_max)
    """

    def __init__(
        self,
        *,
        init_sigma: float,
        sigma_max: float,
        padding: str = "reflect",
        kernel_size: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        if init_sigma <= 0:
            raise ValueError("init_sigma must be > 0")
        if sigma_max <= 0:
            raise ValueError("sigma_max must be > 0")
        if init_sigma > sigma_max:
            raise ValueError("init_sigma must be <= sigma_max")
        if padding not in ("reflect", "circular", "replicate", "constant"):
            raise ValueError(f"Unsupported padding: {padding}")

        self.sigma_max = float(sigma_max)
        self.padding = padding
        self.eps = float(eps)

        # log_sigma is the only trainable parameter
        self.log_sigma = nn.Parameter(torch.tensor(float(math.log(init_sigma)), dtype=torch.float32))

        # If kernel_size not provided, we will infer from current sigma each forward.
        # If you want fixed kernel size, set kernel_size explicitly.
        self.kernel_size = kernel_size

    def sigma(self) -> torch.Tensor:
        # Clamp sigma in value-space (important for stable behaviour)
        s = torch.exp(self.log_sigma)
        return torch.clamp(s, min=self.eps, max=self.sigma_max)

    @torch.no_grad()
    def clamp_params_(self) -> None:
        # Clamp log_sigma so that sigma remains in (eps, sigma_max)
        lo = math.log(self.eps)
        hi = math.log(self.sigma_max)
        self.log_sigma.clamp_(min=lo, max=hi)

    @torch.no_grad()
    def set_sigma_(self, value: float) -> None:
        """Set sigma in-place by updating log_sigma."""
        if value <= 0:
            raise ValueError("sigma must be > 0")
        self.log_sigma.fill_(math.log(float(value)))

    def _make_kernel_2d(self, sigma: torch.Tensor, device, dtype) -> torch.Tensor:
        """
        Build a 2D Gaussian kernel using torch ops.
        """
        # Choose kernel size
        if self.kernel_size is None:
            # Rule of thumb: ~6*sigma coverage, force odd, minimum 3
            # Use float sigma but produce int kernel size
            ks = int(max(3, 2 * int(torch.ceil(3.0 * sigma).item()) + 1))
        else:
            ks = int(self.kernel_size)
            if ks % 2 == 0 or ks < 3:
                raise ValueError("kernel_size must be odd and >= 3")

        radius = ks // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(coords, coords, indexing="ij")

        # Gaussian
        # Avoid division by 0 via sigma already clamped by eps
        g = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * (sigma ** 2)))
        g = g / g.sum()

        return g  # (ks, ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")

        b, c, h, w = x.shape
        sigma = self.sigma().to(device=x.device, dtype=x.dtype)

        k2d = self._make_kernel_2d(sigma, device=x.device, dtype=x.dtype)  # (ks, ks)
        ks = k2d.shape[-1]

        # Depthwise conv weight: (C,1,ks,ks)
        weight = k2d.view(1, 1, ks, ks).repeat(c, 1, 1, 1)

        pad = ks // 2
        if self.padding == "reflect":
            x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        elif self.padding == "circular":
            x_pad = F.pad(x, (pad, pad, pad, pad), mode="circular")
        elif self.padding == "replicate":
            x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
        else:  # "constant"
            x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

        y = F.conv2d(x_pad, weight=weight, bias=None, stride=1, padding=0, groups=c)
        return y

@dataclass(frozen=True)
class DegradationAdapter:
    """
    Optional adapter if your solver expects an object with H / H_adj methods.
    For Gaussian blur with symmetric kernel, H_adj is typically the same operation
    under the same padding convention (approximate but often acceptable).
    If you need exact adjoint under your boundary conditions, implement it explicitly.
    """
    op: LearnableGaussianBlur

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    def H_adj(self, r: torch.Tensor) -> torch.Tensor:
        # For Gaussian symmetric kernel, using same op is a common practical choice.
        return self.op(r)
