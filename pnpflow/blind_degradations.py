"""
Learnable forward operators for blind inverse problems.

Fixes vs your original:
- No "self.device" strings inside modules; tensors are created on the correct runtime device.
- CompositeOperator does NOT add noise by default (noise belongs in the likelihood, not H).
- LearnableMask semantics are consistent: mask=1 means "observed/kept", and masked_ratio = 1 - keep_ratio.
- Safer numerics (eps in normalizations, sigma lower bound).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)


def _device_of(*tensors_or_params) -> torch.device:
    for obj in tensors_or_params:
        if isinstance(obj, torch.Tensor):
            return obj.device
        if isinstance(obj, nn.Parameter):
            return obj.device
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Learnable Gaussian Blur
# -----------------------------------------------------------------------------

class LearnableGaussianBlur(nn.Module):
    """
    Learnable Gaussian blur via depthwise convolution with circular padding.
    Sigma is parameterized in log space for positivity.
    """

    def __init__(self, kernel_size: int = 61, num_channels: int = 3, init_sigma: float = 1.0):
        super().__init__()
        self.kernel_size = _ensure_odd(int(kernel_size))
        self.num_channels = int(num_channels)

        init_sigma = float(init_sigma)
        if init_sigma <= 0:
            raise ValueError("init_sigma must be > 0")

        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma, dtype=torch.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma).clamp(min=1e-4)
        kernel = self._create_gaussian_kernel(sigma, device=x.device, dtype=x.dtype)
        return self._apply_blur(x, kernel)

    def _create_gaussian_kernel(self, sigma: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = self.kernel_size
        # Coordinate grid centered at 0 on the correct runtime device
        ax = torch.arange(-(k // 2), (k // 2) + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma.to(dtype=dtype) ** 2))
        kernel = kernel / (kernel.sum() + 1e-12)
        kernel = kernel.view(1, 1, k, k).repeat(self.num_channels, 1, 1, 1)  # [C,1,K,K]
        return kernel

    def _apply_blur(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
        return F.conv2d(x_padded, kernel, groups=self.num_channels)

    def get_sigma(self) -> float:
        return float(torch.exp(self.log_sigma).item())

    def get_kernel(self) -> np.ndarray:
        with torch.no_grad():
            # Create kernel on same device as parameter, then move to CPU numpy
            sigma = torch.exp(self.log_sigma).clamp(min=1e-4)
            device = sigma.device
            kernel = self._create_gaussian_kernel(sigma, device=device, dtype=torch.float32)
            return kernel[0, 0].detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Learnable Motion Blur
# -----------------------------------------------------------------------------

class LearnableMotionBlur(nn.Module):
    """
    Differentiable motion blur kernel parameterized by length and angle.
    Implemented as a soft line with smooth edges.
    """

    def __init__(
        self,
        kernel_size: int = 61,
        num_channels: int = 3,
        init_length: float = 10.0,
        init_angle: float = 0.0,
        line_width: float = 1.0,
        steepness: float = 2.0,
    ):
        super().__init__()
        self.kernel_size = _ensure_odd(int(kernel_size))
        self.num_channels = int(num_channels)

        init_length = float(init_length)
        if init_length <= 0:
            raise ValueError("init_length must be > 0")

        self.log_length = nn.Parameter(torch.log(torch.tensor(init_length, dtype=torch.float32)))
        self.angle = nn.Parameter(torch.tensor(float(init_angle), dtype=torch.float32))

        self.line_width = float(line_width)
        self.steepness = float(steepness)

        # Register coordinate grid as buffers so they move with .to(device)
        k = self.kernel_size
        ax = torch.arange(-(k // 2), (k // 2) + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        self.register_buffer("xx", xx)  # [K,K]
        self.register_buffer("yy", yy)  # [K,K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = torch.exp(self.log_length).clamp(min=1e-4)
        kernel = self._create_motion_kernel(length, self.angle, dtype=x.dtype, device=x.device)
        return self._apply_blur(x, kernel)

    def _create_motion_kernel(
        self,
        length: torch.Tensor,
        angle: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # Ensure buffers are on the same device as x (they usually are, but be safe)
        xx = self.xx.to(device=device, dtype=dtype)
        yy = self.yy.to(device=device, dtype=dtype)

        cos_a = torch.cos(angle.to(device=device, dtype=dtype))
        sin_a = torch.sin(angle.to(device=device, dtype=dtype))

        # Distance to line through origin with direction (cos_a, sin_a)
        dist_to_line = torch.abs(xx * sin_a - yy * cos_a)
        dist_along = xx * cos_a + yy * sin_a

        # Soft perpendicular profile
        lw = torch.tensor(self.line_width, device=device, dtype=dtype)
        perp = torch.exp(-(dist_to_line**2) / (2.0 * lw**2 + 1e-12))

        # Soft segment along the motion direction
        half_len = (length.to(device=device, dtype=dtype) / 2.0)
        s = torch.tensor(self.steepness, device=device, dtype=dtype)
        along = torch.sigmoid(s * (half_len - torch.abs(dist_along)))

        kernel = perp * along
        kernel = kernel / (kernel.sum() + 1e-12)

        k = self.kernel_size
        kernel = kernel.view(1, 1, k, k).repeat(self.num_channels, 1, 1, 1)
        return kernel

    def _apply_blur(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="circular")
        return F.conv2d(x_padded, kernel, groups=self.num_channels)

    def get_params(self) -> dict:
        return {"length": float(torch.exp(self.log_length).item()), "angle": float(self.angle.item())}

    def get_kernel(self) -> np.ndarray:
        with torch.no_grad():
            length = torch.exp(self.log_length).clamp(min=1e-4)
            kernel = self._create_motion_kernel(length, self.angle, dtype=torch.float32, device=length.device)
            return kernel[0, 0].detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Learnable Mask (inpainting)
# -----------------------------------------------------------------------------

class LearnableMask(nn.Module):
    """
    Learnable mask for inpainting.

    Convention (IMPORTANT):
    - mask = 1 means pixel is OBSERVED/KEPT
    - mask = 0 means pixel is MISSING
    init_keep_ratio controls the initial fraction of kept pixels.
    """

    def __init__(self, image_shape, init_keep_ratio: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.image_shape = tuple(image_shape)  # (C,H,W)
        self.temperature = float(temperature)

        if not (0.0 < init_keep_ratio < 1.0):
            raise ValueError("init_keep_ratio must be in (0,1)")

        init_logits = torch.randn(*self.image_shape, dtype=torch.float32) * 0.1
        init_value = np.log(init_keep_ratio / (1.0 - init_keep_ratio + 1e-8))
        init_logits = init_logits + float(init_value)

        self.mask_logits = nn.Parameter(init_logits)

    def forward(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if hard:
            mask = (torch.sigmoid(self.mask_logits) > 0.5).float()
        else:
            mask = torch.sigmoid(self.mask_logits / max(self.temperature, 1e-8))

        mask = mask.to(device=x.device, dtype=x.dtype).unsqueeze(0)  # [1,C,H,W]
        return x * mask

    def get_mask(self, hard: bool = True) -> torch.Tensor:
        with torch.no_grad():
            if hard:
                return (torch.sigmoid(self.mask_logits) > 0.5).float()
            return torch.sigmoid(self.mask_logits)

    def get_keep_ratio(self) -> float:
        with torch.no_grad():
            mask = (torch.sigmoid(self.mask_logits) > 0.5).float()
            return float(mask.mean().item())

    def get_masked_ratio(self) -> float:
        # masked ratio = fraction of missing pixels = 1 - keep ratio
        return float(1.0 - self.get_keep_ratio())


# -----------------------------------------------------------------------------
# Learnable Downsampling (super-resolution)
# -----------------------------------------------------------------------------

class LearnableDownsampling(nn.Module):
    """
    Learnable downsampling via depthwise conv with stride.
    Learns a kernel; normalization ensures kernel sum=1.
    """

    def __init__(self, scale_factor: int = 2, num_channels: int = 3, kernel_size: int | None = None):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.num_channels = int(num_channels)

        if kernel_size is None:
            kernel_size = 2 * self.scale_factor
        self.kernel_size = int(kernel_size)

        init_kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, dtype=torch.float32)
        init_kernel = init_kernel / init_kernel.sum()
        self.kernel = nn.Parameter(init_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel / (self.kernel.sum() + 1e-12)  # [1,1,K,K]
        k = k.to(device=x.device, dtype=x.dtype).repeat(self.num_channels, 1, 1, 1)

        pad = (self.kernel_size - self.scale_factor) // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")

        y = F.conv2d(x_padded, k, stride=self.scale_factor, groups=self.num_channels)

        target_h = x.shape[2] // self.scale_factor
        target_w = x.shape[3] // self.scale_factor
        if y.shape[2] != target_h or y.shape[3] != target_w:
            y = y[:, :, :target_h, :target_w]
        return y

    def get_kernel(self) -> np.ndarray:
        with torch.no_grad():
            k = self.kernel / (self.kernel.sum() + 1e-12)
            return k[0, 0].detach().cpu().numpy()


# -----------------------------------------------------------------------------
# Composite operator (no noise by default)
# -----------------------------------------------------------------------------

class CompositeOperator(nn.Module):
    """
    Sequential composition of operators: y = op_n(...op_2(op_1(x))...)
    NOTE: Noise is NOT part of H by default. Keep noise in the likelihood.
    """

    def __init__(self, operators, noise_level: float | None = None):
        super().__init__()
        self.operators = nn.ModuleList(list(operators))

        if noise_level is not None:
            noise_level = float(noise_level)
            if noise_level <= 0:
                raise ValueError("noise_level must be > 0")
            self.log_noise_std = nn.Parameter(torch.log(torch.tensor(noise_level, dtype=torch.float32)))
        else:
            self.log_noise_std = None

    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        y = x
        for op in self.operators:
            y = op(y)

        if add_noise and self.log_noise_std is not None:
            noise_std = torch.exp(self.log_noise_std).to(device=y.device, dtype=y.dtype)
            y = y + noise_std * torch.randn_like(y)

        return y

    def get_noise_level(self) -> float:
        if self.log_noise_std is not None:
            return float(torch.exp(self.log_noise_std).item())
        return 0.0


# -----------------------------------------------------------------------------
# Compatibility wrapper
# -----------------------------------------------------------------------------

class LearnableDegradation:
    """
    Wrapper to mimic the existing degradation interface (expects H and H_adj callables).
    """

    def __init__(self, operator: nn.Module, H_adj=None):
        self.operator = operator
        self._H_adj = H_adj

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return self.operator(x)

    def H_adj(self, y: torch.Tensor) -> torch.Tensor:
        return self._H_adj(y) if self._H_adj is not None else y

    def get_operator(self) -> nn.Module:
        return self.operator


# -----------------------------------------------------------------------------
# Factory + parameter extraction
# -----------------------------------------------------------------------------

def create_blind_operator(operator_type: str, **kwargs) -> nn.Module:
    """
    operator_type: 'gaussian_blur', 'motion_blur', 'mask', 'downsample', 'composite'
    """
    if operator_type == "gaussian_blur":
        return LearnableGaussianBlur(**kwargs)
    if operator_type == "motion_blur":
        return LearnableMotionBlur(**kwargs)
    if operator_type == "mask":
        # Backwards compatible: allow init_ratio as alias
        if "init_ratio" in kwargs and "init_keep_ratio" not in kwargs:
            kwargs["init_keep_ratio"] = kwargs.pop("init_ratio")
        return LearnableMask(**kwargs)
    if operator_type == "downsample":
        return LearnableDownsampling(**kwargs)
    if operator_type == "composite":
        operators = kwargs.pop("operators", [])
        return CompositeOperator(operators, **kwargs)
    raise ValueError(f"Unknown operator type: {operator_type}")


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
    if hasattr(operator, "get_keep_ratio"):
        params["keep_ratio"] = operator.get_keep_ratio()
    return params
