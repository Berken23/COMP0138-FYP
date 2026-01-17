# pnpflow/blind_degradations.py
"""
Learnable forward operators for blind inverse problems.
Each operator is a nn.Module with learnable parameters that can be optimized
during the blind reconstruction process.

Compatible with PnP-Flow framework using OT Flow Matching models.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_default_device():
    """Get default device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LearnableGaussianBlur(nn.Module):
    """
    Learnable Gaussian blur kernel.
    Parameterizes a Gaussian kernel with learnable sigma parameter.

    Sigma is parameterized in log space to ensure positivity.
    The blur kernel is applied via circular convolution to reduce boundary artifacts.

    Args:
        kernel_size (int): Size of the blur kernel (should be odd).
        num_channels (int): Number of image channels (3 for RGB).
        init_sigma (float): Initial sigma.
        device (str|torch.device|None): Optional device to move the module to.
    """

    def __init__(self, kernel_size=61, num_channels=3, init_sigma=1.0, device=None):
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = int(kernel_size)
        self.num_channels = int(num_channels)

        init_sigma_t = torch.tensor(float(init_sigma), dtype=torch.float32)
        self.log_sigma = nn.Parameter(torch.log(init_sigma_t))

        # Option B: accept device=, but DO NOT store a device string.
        # Buffers/params determine device; kernel tensors are created on sigma.device.
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)  # scalar tensor on module device
        kernel = self._create_gaussian_kernel(sigma)
        return self._apply_blur(x, kernel)

    def _create_gaussian_kernel(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            kernel: [C, 1, K, K] depthwise kernel on sigma.device
        """
        K = self.kernel_size
        device = sigma.device
        dtype = sigma.dtype

        ax = torch.arange(-(K // 2), K // 2 + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")

        kernel2d = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel2d = kernel2d / (kernel2d.sum() + 1e-12)

        kernel = kernel2d.view(1, 1, K, K).repeat(self.num_channels, 1, 1, 1)
        return kernel

    def _apply_blur(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
        return F.conv2d(x_padded, kernel, groups=self.num_channels)

    def get_sigma(self) -> float:
        return float(torch.exp(self.log_sigma).detach().cpu().item())

    def get_kernel(self) -> np.ndarray:
        with torch.no_grad():
            sigma = torch.exp(self.log_sigma)
            k = self._create_gaussian_kernel(sigma)
            return k[0, 0].detach().cpu().numpy()


class LearnableMotionBlur(nn.Module):
    """
    Learnable motion blur kernel.

    Parameterizes motion blur by length and angle.
    Uses a differentiable "soft line" kernel.

    Args:
        kernel_size (int): odd kernel size
        num_channels (int)
        init_length (float)
        init_angle (float) radians
        device (str|torch.device|None): Optional device to move the module to.
    """

    def __init__(
        self,
        kernel_size=61,
        num_channels=3,
        init_length=10.0,
        init_angle=0.0,
        device=None,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1

        self.kernel_size = int(kernel_size)
        self.num_channels = int(num_channels)

        self.log_length = nn.Parameter(torch.log(torch.tensor(float(init_length), dtype=torch.float32)))
        self.angle = nn.Parameter(torch.tensor(float(init_angle), dtype=torch.float32))

        # Create coordinate grid ON CPU, register as buffers so .to(device) moves them safely.
        ax = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("xx", xx, persistent=False)
        self.register_buffer("yy", yy, persistent=False)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = torch.exp(self.log_length)
        kernel = self._create_motion_kernel(length, self.angle)
        return self._apply_blur(x, kernel)

    def _create_motion_kernel(self, length: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        # Direction vector
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Buffers are already on the correct device.
        dist_to_line = torch.abs(self.xx * sin_a - self.yy * cos_a)
        dist_along = self.xx * cos_a + self.yy * sin_a

        line_width = 1.0
        perp_profile = torch.exp(-(dist_to_line**2) / (2.0 * line_width**2))

        half_len = length / 2.0
        steepness = 2.0
        along_profile = torch.sigmoid(steepness * (half_len - torch.abs(dist_along)))

        kernel2d = perp_profile * along_profile
        kernel2d = kernel2d / (kernel2d.sum() + 1e-8)

        kernel = kernel2d.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.num_channels, 1, 1, 1)
        return kernel

    def _apply_blur(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
        return F.conv2d(x_padded, kernel, groups=self.num_channels)

    def get_params(self) -> dict:
        return {
            "length": float(torch.exp(self.log_length).detach().cpu().item()),
            "angle": float(self.angle.detach().cpu().item()),
        }

    def get_kernel(self) -> np.ndarray:
        with torch.no_grad():
            length = torch.exp(self.log_length)
            k = self._create_motion_kernel(length, self.angle)
            return k[0, 0].detach().cpu().numpy()


class LearnableMask(nn.Module):
    """
    Learnable mask for inpainting.

    Args:
        image_shape (tuple): (C, H, W)
        init_ratio (float): initial masked ratio
        temperature (float): sigmoid temperature
        device (str|torch.device|None): Optional device to move the module to.
    """

    def __init__(self, image_shape, init_ratio=0.5, temperature=1.0, device=None):
        super().__init__()

        self.image_shape = tuple(image_shape)
        self.temperature = float(temperature)

        init_logits = torch.randn(*self.image_shape, dtype=torch.float32) * 0.1
        if init_ratio is not None:
            p = float(init_ratio)
            init_value = np.log(p / (1.0 - p + 1e-8))
            init_logits = init_logits + float(init_value)

        self.mask_logits = nn.Parameter(init_logits)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if hard:
            mask = (torch.sigmoid(self.mask_logits) > 0.5).float()
        else:
            mask = torch.sigmoid(self.mask_logits / self.temperature)

        mask = mask.unsqueeze(0)  # [1,C,H,W]
        return x * mask

    def get_mask(self, hard: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if hard:
                return (torch.sigmoid(self.mask_logits) > 0.5).float()
            return torch.sigmoid(self.mask_logits)

    def get_masked_ratio(self) -> float:
        with torch.no_grad():
            return float(self.get_mask(hard=True).mean().detach().cpu().item())


class LearnableDownsampling(nn.Module):
    """
    Learnable downsampling kernel.

    Args:
        scale_factor (int)
        num_channels (int)
        kernel_size (int|None)
        device (str|torch.device|None): Optional device to move the module to.
    """

    def __init__(self, scale_factor=2, num_channels=3, kernel_size=None, device=None):
        super().__init__()

        self.scale_factor = int(scale_factor)
        self.num_channels = int(num_channels)

        if kernel_size is None:
            kernel_size = 2 * self.scale_factor
        self.kernel_size = int(kernel_size)

        if self.kernel_size < self.scale_factor:
            raise ValueError("kernel_size must be >= scale_factor")

        init_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, dtype=torch.float32)
        init_kernel = init_kernel / (init_kernel.sum() + 1e-8)
        self.kernel = nn.Parameter(init_kernel)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel / (self.kernel.sum() + 1e-8)
        kernel = kernel.repeat(self.num_channels, 1, 1, 1)

        pad = (self.kernel_size - self.scale_factor) // 2
        if pad < 0:
            pad = 0

        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y = F.conv2d(x_padded, kernel, stride=self.scale_factor, groups=self.num_channels)

        target_h = x.shape[2] // self.scale_factor
        target_w = x.shape[3] // self.scale_factor
        if y.shape[2] != target_h or y.shape[3] != target_w:
            y = y[:, :, :target_h, :target_w]
        return y

    def get_kernel(self) -> np.ndarray:
        with torch.no_grad():
            k = self.kernel / (self.kernel.sum() + 1e-8)
            return k[0, 0].detach().cpu().numpy()


class CompositeOperator(nn.Module):
    """
    Composite of multiple operators applied sequentially.

    Args:
        operators (list[nn.Module])
        noise_level (float|None): additive Gaussian noise std (learnable in log-space if provided)
        device (str|torch.device|None): Optional device to move the module to.
    """

    def __init__(self, operators, noise_level=None, device=None):
        super().__init__()

        self.operators = nn.ModuleList(list(operators))

        if noise_level is not None:
            self.log_noise_std = nn.Parameter(torch.log(torch.tensor(float(noise_level), dtype=torch.float32)))
        else:
            self.log_noise_std = None

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        y = x
        for op in self.operators:
            y = op(y)

        if add_noise and self.log_noise_std is not None:
            noise_std = torch.exp(self.log_noise_std)
            y = y + noise_std * torch.randn_like(y)

        return y

    def get_noise_level(self) -> float:
        if self.log_noise_std is None:
            return 0.0
        return float(torch.exp(self.log_noise_std).detach().cpu().item())


# ============================================================================
# WRAPPER FOR COMPATIBILITY WITH PNP_FLOW DEGRADATION INTERFACE
# ============================================================================

class LearnableDegradation:
    """
    Wrapper class to expose H and H_adj for compatibility with existing code.
    """

    def __init__(self, operator: nn.Module, H_adj=None):
        self.operator = operator
        self._H_adj = H_adj

    def H(self, x):
        return self.operator(x)

    def H_adj(self, y):
        if self._H_adj is not None:
            return self._H_adj(y)
        return y

    def get_operator(self):
        return self.operator


# ============================================================================
# FACTORY / UTILS
# ============================================================================

def create_blind_operator(operator_type, device=None, **kwargs) -> nn.Module:
    if operator_type == "gaussian_blur":
        op = LearnableGaussianBlur(**kwargs)
    elif operator_type == "motion_blur":
        op = LearnableMotionBlur(**kwargs)
    elif operator_type == "mask":
        op = LearnableMask(**kwargs)
    elif operator_type == "downsample":
        op = LearnableDownsampling(**kwargs)
    elif operator_type == "composite":
        operators = kwargs.pop("operators", [])
        op = CompositeOperator(operators, **kwargs)
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")

    if device is not None:
        op = op.to(device)
    return op


def get_operator_parameters(operator: nn.Module) -> dict:
    params = {}

    if hasattr(operator, "get_sigma"):
        params["sigma"] = operator.get_sigma()

    if hasattr(operator, "get_params"):
        params.update(operator.get_params())

    if hasattr(operator, "get_masked_ratio"):
        params["mask_ratio"] = operator.get_masked_ratio()

    if hasattr(operator, "get_noise_level"):
        params["noise_level"] = operator.get_noise_level()

    return params
