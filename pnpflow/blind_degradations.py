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
    Depthwise Gaussian blur with learnable sigma (parameterised as log_sigma).

    Design goals:
    - Fixed kernel size derived from sigma_max (smooth optimisation w.r.t sigma)
    - Kernel built using pure torch ops (no numpy)
    - Depthwise conv per channel (groups=C)
    - Padding mode is explicit and consistent

    Notes:
    - This implements H(x). For Gaussian blur with symmetric kernel,
      the adjoint under common padding assumptions is approximately itself.
      (Still implement H_adj explicitly if your solver requires it.)
    """

    def __init__(
        self,
        init_sigma: float,
        sigma_max: float,
        padding: PaddingMode = "reflect",
        truncate: float = 3.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if init_sigma <= 0:
            raise ValueError("init_sigma must be > 0")
        if sigma_max <= 0:
            raise ValueError("sigma_max must be > 0")
        if init_sigma > sigma_max:
            # not fatal, but signals a likely config error
            raise ValueError("init_sigma must be <= sigma_max for this fixed-kernel design")

        self.padding: PaddingMode = padding
        self.truncate = float(truncate)
        self.eps = float(eps)

        ks = _odd_ks_from_sigma_max(sigma_max=sigma_max, truncate=truncate)
        self.ks = int(ks)
        self.radius = self.ks // 2

        # Only learnable parameter
        self.log_sigma = nn.Parameter(torch.tensor(math.log(init_sigma), dtype=torch.float32))

        # Precompute coordinate grid as buffers (device-agnostic; moved with module)
        coords = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("xx", xx, persistent=False)
        self.register_buffer("yy", yy, persistent=False)

    def sigma(self) -> torch.Tensor:
        # expose sigma as a tensor (useful for logging)
        return torch.exp(self.log_sigma)

    def _kernel_2d(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # Build continuous Gaussian kernel on the fly (depends on sigma)
        sigma = self.sigma().to(device=device, dtype=dtype)
        sigma2 = sigma * sigma + self.eps

        xx = self.xx.to(device=device, dtype=dtype)
        yy = self.yy.to(device=device, dtype=dtype)

        dist2 = xx * xx + yy * yy
        kernel = torch.exp(-dist2 / (2.0 * sigma2))
        kernel = kernel / (kernel.sum() + self.eps)
        return kernel  # (ks, ks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to have shape (B,C,H,W), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if h <= 1 or w <= 1:
            raise ValueError("H and W must be > 1 for blur to make sense")

        kernel2d = self._kernel_2d(device=x.device, dtype=x.dtype)  # (ks, ks)
        weight = kernel2d.view(1, 1, self.ks, self.ks).repeat(c, 1, 1, 1)  # (C,1,ks,ks)

        # Pad explicitly to keep output size same and to control boundary behavior.
        pad = (self.radius, self.radius, self.radius, self.radius)  # (left,right,top,bottom)
        if self.padding == "constant":
            x_pad = F.pad(x, pad, mode="constant", value=0.0)
        else:
            x_pad = F.pad(x, pad, mode=self.padding)

        y = F.conv2d(x_pad, weight=weight, bias=None, stride=1, padding=0, dilation=1, groups=c)
        return y

    @torch.no_grad()
    def set_sigma_(self, sigma_value: float) -> None:
        if sigma_value <= 0:
            raise ValueError("sigma_value must be > 0")
        self.log_sigma.copy_(torch.tensor(math.log(sigma_value), dtype=self.log_sigma.dtype, device=self.log_sigma.device))


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
