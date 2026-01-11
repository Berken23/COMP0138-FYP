"""
Learnable forward operators for blind inverse problems.
Each operator is a nn.Module with learnable parameters that can be optimized
during the blind reconstruction process.

Compatible with PnP-Flow framework using OT Flow Matching models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_default_device():
    """Get default device (cuda if available, else cpu)"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'

class LearnableGaussianBlur(nn.Module):
    """
    Learnable Gaussian blur kernel.
    Parameterizes a Gaussian kernel with learnable sigma parameter.
    
    The blur kernel is applied via circular convolution to avoid boundary artifacts.
    Sigma is parameterized in log space to ensure positivity.
    
    Args:
        kernel_size (int): Size of the blur kernel (should be odd)
        num_channels (int): Number of image channels (3 for RGB)
        init_sigma (float): Initial value of sigma
        device (str): Device to place parameters on ('cuda' or 'cpu')
    """
    
    def __init__(self, kernel_size=61, num_channels=3, init_sigma=1.0, device=None):
        super().__init__()
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.device = device if device is not None else get_default_device()
        
        # Use log parameterization to ensure sigma > 0
        # During optimization, sigma = exp(log_sigma) is always positive
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma, dtype=torch.float32)))
        
    def forward(self, x):
        """
        Apply learned Gaussian blur to input image.
        
        Args:
            x: Input image [B, C, H, W]
        
        Returns:
            Blurred image [B, C, H, W]
        """
        sigma = torch.exp(self.log_sigma)
        kernel = self._create_gaussian_kernel(sigma)
        return self._apply_blur(x, kernel)
    
    def _create_gaussian_kernel(self, sigma):
        """
        Create 2D Gaussian kernel from sigma parameter.
        
        Args:
            sigma: Standard deviation of Gaussian
            
        Returns:
            kernel: [num_channels, 1, kernel_size, kernel_size]
        """
        kernel_size = self.kernel_size
        
        # Create coordinate grid centered at 0
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., 
                         device=self.device, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # Gaussian formula: exp(-(x^2 + y^2) / (2 * sigma^2))
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Normalize so sum = 1
        kernel = kernel / kernel.sum()
        
        # Reshape for depthwise convolution: [out_channels, in_channels, H, W]
        # For depthwise conv, out_channels = num_channels, in_channels = 1
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(self.num_channels, 1, 1, 1)
        
        return kernel
    
    def _apply_blur(self, x, kernel):
        """
        Apply blur kernel via convolution with circular padding.
        
        Args:
            x: Input image [B, C, H, W]
            kernel: Blur kernel [C, 1, K, K]
            
        Returns:
            Blurred image [B, C, H, W]
        """
        pad = self.kernel_size // 2
        
        # Use circular padding to avoid boundary artifacts
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='circular')
        
        # Apply depthwise convolution (same kernel for each channel)
        blurred = F.conv2d(x_padded, kernel, groups=self.num_channels)
        
        return blurred
    
    def get_sigma(self):
        """Return current sigma value as Python float"""
        return torch.exp(self.log_sigma).item()
    
    def get_kernel(self):
        """Return current kernel as numpy array for visualization"""
        with torch.no_grad():
            sigma = torch.exp(self.log_sigma)
            kernel = self._create_gaussian_kernel(sigma)
            return kernel[0, 0].cpu().numpy()

class LearnableMotionBlur(nn.Module):
    """
    Learnable motion blur kernel.
    Parameterizes motion blur by length and angle of motion.
    
    Uses a differentiable approximation of motion blur by creating
    a soft line with Gaussian profile.
    
    Args:
        kernel_size (int): Size of the blur kernel (should be odd)
        num_channels (int): Number of image channels
        init_length (float): Initial motion length in pixels
        init_angle (float): Initial motion angle in radians
        device (str): Device to place parameters on
    """
    
    def __init__(self, kernel_size=61, num_channels=3, 
                 init_length=10.0, init_angle=0.0, device=None):
        super().__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.device = device if device is not None else get_default_device()
        
        # Learnable parameters
        # Length in log space for positivity
        self.log_length = nn.Parameter(torch.log(torch.tensor(init_length, dtype=torch.float32)))
        # Angle can be any value (periodic)
        self.angle = nn.Parameter(torch.tensor(init_angle, dtype=torch.float32))
        
        # Pre-compute coordinate grid (constant, doesn't need gradients)
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., 
                         device=self.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ax, ax, indexing='ij')
        # Register as buffer (moved with model but not trained)
        self.register_buffer('xx', xx)
        self.register_buffer('yy', yy)
        
    def forward(self, x):
        """Apply learned motion blur"""
        length = torch.exp(self.log_length)
        kernel = self._create_motion_kernel(length, self.angle)
        return self._apply_blur(x, kernel)
    
    def _create_motion_kernel(self, length, angle):
        """
        Create motion blur kernel from length and angle.
        Uses fully differentiable operations to preserve gradients.
        
        Args:
            length: Length of motion in pixels (differentiable)
            angle: Direction of motion in radians (differentiable)
            
        Returns:
            kernel: [num_channels, 1, kernel_size, kernel_size]
        """
        # Compute direction vector (differentiable)
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Distance from each point to the motion line
        # Line goes through origin in direction (cos_angle, sin_angle)
        # Distance to line: |x*sin - y*cos|
        dist_to_line = torch.abs(self.xx * sin_angle - self.yy * cos_angle)
        
        # Distance along the line (projection onto motion direction)
        dist_along_line = self.xx * cos_angle + self.yy * sin_angle
        
        # Create soft line perpendicular to motion
        line_width = 1.0  # Width of the motion line
        perp_profile = torch.exp(-dist_to_line**2 / (2 * line_width**2))
        
        # Create soft rectangular profile along motion direction
        # Use sigmoid to create smooth edges
        half_length = length / 2.0
        # Smooth step function using sigmoid
        steepness = 2.0  # Controls sharpness of edges
        along_profile = torch.sigmoid(steepness * (half_length - torch.abs(dist_along_line)))
        
        # Combine profiles
        kernel = perp_profile * along_profile
        
        # Normalize to sum to 1
        kernel = kernel / (kernel.sum() + 1e-8)
        
        # Reshape for convolution [num_channels, 1, H, W]
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.num_channels, 1, 1, 1)
        
        return kernel
    
    def _apply_blur(self, x, kernel):
        """Apply blur via convolution"""
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='circular')
        blurred = F.conv2d(x_padded, kernel, groups=self.num_channels)
        return blurred
    
    def get_params(self):
        """Return current parameters as dictionary"""
        return {
            'length': torch.exp(self.log_length).item(),
            'angle': self.angle.item()
        }
    
    def get_kernel(self):
        """Return current kernel as numpy array"""
        with torch.no_grad():
            length = torch.exp(self.log_length)
            kernel = self._create_motion_kernel(length, self.angle)
            return kernel[0, 0].cpu().numpy()

class LearnableMask(nn.Module):
    """
    Learnable binary mask for inpainting.
    Uses sigmoid + soft thresholding for differentiable binary mask.
    
    Args:
        image_shape (tuple): Shape of image (C, H, W)
        init_ratio (float): Initial ratio of masked pixels (0 to 1)
        temperature (float): Temperature for sigmoid (lower = more binary)
        device (str): Device to place parameters on
    """
    
    def __init__(self, image_shape, init_ratio=0.5, temperature=1.0, device=None):
        super().__init__()
        
        self.image_shape = image_shape  # (C, H, W)
        self.temperature = temperature
        self.device = device if device is not None else get_default_device()
        
        # Initialize mask logits
        # Logits are converted to probabilities via sigmoid
        init_logits = torch.randn(*image_shape, dtype=torch.float32) * 0.1
        
        if init_ratio is not None:
            # Initialize to approximately init_ratio of pixels masked
            # logit = log(p / (1-p)) where p = init_ratio
            init_value = np.log(init_ratio / (1 - init_ratio + 1e-8))
            init_logits = init_logits + init_value
        
        self.mask_logits = nn.Parameter(init_logits)
        
    def forward(self, x, hard=False):
        """
        Apply mask to input image.
        
        Args:
            x: Input image [B, C, H, W]
            hard: If True, use hard binary mask (non-differentiable)
                  If False, use soft mask (differentiable)
        
        Returns:
            Masked image [B, C, H, W]
        """
        if hard:
            # Hard binary mask (for inference)
            mask = (torch.sigmoid(self.mask_logits) > 0.5).float()
        else:
            # Soft mask (for training)
            mask = torch.sigmoid(self.mask_logits / self.temperature)
        
        # Broadcast mask to batch dimension
        mask = mask.unsqueeze(0)  # [1, C, H, W]
        
        return x * mask
    
    def get_mask(self, hard=True):
        """Get current mask as tensor"""
        with torch.no_grad():
            if hard:
                return (torch.sigmoid(self.mask_logits) > 0.5).float()
            else:
                return torch.sigmoid(self.mask_logits)
    
    def get_masked_ratio(self):
        """Return fraction of masked pixels"""
        with torch.no_grad():
            mask = self.get_mask(hard=True)
            return mask.mean().item()


class LearnableDownsampling(nn.Module):
    """
    Learnable downsampling operator for super-resolution.
    Learns a downsampling kernel instead of using fixed bicubic/bilinear.
    
    FIXED: Proper padding to ensure output size = input size // scale_factor
    
    Args:
        scale_factor (int): Downsampling scale (2 for 2x downsampling)
        num_channels (int): Number of image channels
        kernel_size (int): Size of downsampling kernel
        device (str): Device to place parameters on
    """
    
    def __init__(self, scale_factor=2, num_channels=3, kernel_size=None, device=None):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.num_channels = num_channels
        self.device = device if device is not None else get_default_device()
        
        if kernel_size is None:
            kernel_size = 2 * scale_factor
        self.kernel_size = kernel_size
        
        # Initialize with smooth kernel
        init_kernel = self._create_init_kernel()
        self.kernel = nn.Parameter(init_kernel)
        
    def forward(self, x):
        """
        Apply learned downsampling.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Downsampled image [B, C, H//scale, W//scale]
        """
        # Normalize kernel to sum to 1 (ensures no brightness change)
        kernel = self.kernel / (self.kernel.sum() + 1e-8)
        kernel = kernel.repeat(self.num_channels, 1, 1, 1)
        
        # FIXED: Calculate proper padding for exact downsampling
        # For stride=s and kernel=k, to get output size = input//s, we need:
        # output_size = (input_size + 2*pad - kernel_size) // stride + 1
        # We want: input_size // stride = (input_size + 2*pad - kernel_size) // stride + 1
        # This gives: pad = (kernel_size - stride) // 2
        
        pad = (self.kernel_size - self.scale_factor) // 2
        
        # Use reflect padding to avoid boundary artifacts
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        downsampled = F.conv2d(
            x_padded, 
            kernel, 
            stride=self.scale_factor,
            groups=self.num_channels
        )
        
        # Ensure exact size (crop if needed due to rounding)
        target_h = x.shape[2] // self.scale_factor
        target_w = x.shape[3] // self.scale_factor
        
        if downsampled.shape[2] != target_h or downsampled.shape[3] != target_w:
            downsampled = downsampled[:, :, :target_h, :target_w]
        
        return downsampled
    
    def _create_init_kernel(self):
        """Initialize with a smooth averaging kernel"""
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, 
                          dtype=torch.float32, device=self.device)
        kernel = kernel / kernel.sum()
        return kernel
    
    def get_kernel(self):
        """Get current kernel as numpy array"""
        with torch.no_grad():
            kernel = self.kernel / (self.kernel.sum() + 1e-8)
            return kernel[0, 0].cpu().numpy()


class CompositeOperator(nn.Module):
    """
    Composite of multiple operators applied sequentially.
    Useful for realistic degradation models (e.g., blur + downsample + noise).
    
    Args:
        operators (list): List of nn.Module operators to apply in sequence
        noise_level (float): Standard deviation of additive Gaussian noise
    """
    
    def __init__(self, operators, noise_level=None):
        super().__init__()
        
        self.operators = nn.ModuleList(operators)
        
        if noise_level is not None:
            self.log_noise_std = nn.Parameter(
                torch.log(torch.tensor(noise_level, dtype=torch.float32))
            )
        else:
            self.log_noise_std = None
    
    def forward(self, x, add_noise=True):
        """
        Apply operators sequentially.
        
        Args:
            x: Input image [B, C, H, W]
            add_noise: Whether to add noise at the end
            
        Returns:
            Degraded image
        """
        y = x
        
        # Apply each operator in sequence
        for op in self.operators:
            y = op(y)
        
        # Add noise if enabled
        if add_noise and self.log_noise_std is not None:
            noise_std = torch.exp(self.log_noise_std)
            y = y + noise_std * torch.randn_like(y)
        
        return y
    
    def get_noise_level(self):
        """Get current noise level"""
        if self.log_noise_std is not None:
            return torch.exp(self.log_noise_std).item()
        return 0.0


# ============================================================================
# WRAPPER CLASS FOR COMPATIBILITY WITH PNP_FLOW DEGRADATION INTERFACE
# ============================================================================

class LearnableDegradation:
    """
    Wrapper class to make learnable operators compatible with the 
    PNP_FLOW degradation interface (which expects H and H_adj functions).
    
    This allows blind operators to be used with the existing solve_ip method
    with minimal modifications.
    
    Args:
        operator: A learnable operator (nn.Module)
        H_adj: Adjoint operator (for initialization and gradient computation)
               Can be None if not needed
    """
    
    def __init__(self, operator, H_adj=None):
        self.operator = operator
        self._H_adj = H_adj
        
    def H(self, x):
        """Forward operator"""
        return self.operator(x)
    
    def H_adj(self, y):
        """Adjoint operator (transpose)"""
        if self._H_adj is not None:
            return self._H_adj(y)
        else:
            # Default: return input unchanged
            return y
    
    def get_operator(self):
        """Get the underlying learnable operator"""
        return self.operator


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_blind_operator(operator_type, device=None, **kwargs):
    """
    Factory function to create learnable operators.
    
    Args:
        operator_type (str): Type of operator ('gaussian_blur', 'motion_blur', 
                            'mask', 'downsample', 'composite')
        device (str): Device to place operator on
        **kwargs: Additional parameters for the operator
        
    Returns:
        Learnable operator (nn.Module)
        
    Example:
        >>> blur = create_blind_operator('gaussian_blur', init_sigma=2.0)
        >>> mask = create_blind_operator('mask', image_shape=(3, 128, 128))
    """
    if device is None:
        device = get_default_device()
    
    if operator_type == 'gaussian_blur':
        return LearnableGaussianBlur(device=device, **kwargs)
    
    elif operator_type == 'motion_blur':
        return LearnableMotionBlur(device=device, **kwargs)
    
    elif operator_type == 'mask':
        return LearnableMask(device=device, **kwargs)
    
    elif operator_type == 'downsample':
        return LearnableDownsampling(device=device, **kwargs)
    
    elif operator_type == 'composite':
        # For composite, operators should be passed in kwargs
        operators = kwargs.pop('operators', [])
        return CompositeOperator(operators, **kwargs)
    
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def get_operator_parameters(operator):
    """
    Extract current parameter values from a learnable operator.
    
    Args:
        operator: Learnable operator
        
    Returns:
        Dictionary of parameter names and values
    """
    params = {}
    
    if hasattr(operator, 'get_sigma'):
        params['sigma'] = operator.get_sigma()
    
    if hasattr(operator, 'get_params'):
        params.update(operator.get_params())
    
    if hasattr(operator, 'get_masked_ratio'):
        params['mask_ratio'] = operator.get_masked_ratio()
    
    if hasattr(operator, 'get_noise_level'):
        params['noise_level'] = operator.get_noise_level()
    
    return params