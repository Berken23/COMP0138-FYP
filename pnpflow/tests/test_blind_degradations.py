"""
Unit tests for learnable degradation operators.
Tests each operator individually and verifies:
1. Forward pass produces correct shapes
2. Gradients flow to learnable parameters
3. Parameters stay in valid ranges
4. Works on available devices (CPU/CUDA)
"""

import torch
import pytest
import sys
from pathlib import Path

# Add parent directory to path (robust method)
current_file = Path(__file__).resolve()
parent_dir = current_file.parent.parent
sys.path.insert(0, str(parent_dir))

from blind_degradations import (
    LearnableGaussianBlur,
    LearnableMotionBlur,
    LearnableMask,
    LearnableDownsampling,
    CompositeOperator,
    create_blind_operator,
    get_operator_parameters,
    LearnableDegradation,
    get_default_device
)

# Use CPU for testing if CUDA not available
TEST_DEVICE = get_default_device()
print(f"\n{'='*60}")
print(f"Using device: {TEST_DEVICE}")
print(f"{'='*60}")


def test_gaussian_blur():
    """Test learnable Gaussian blur operator"""
    print("\n--- Testing Gaussian Blur ---")
    
    # Create operator with specific initialization
    init_sigma = 2.0
    blur = LearnableGaussianBlur(
        kernel_size=61, 
        num_channels=3, 
        init_sigma=init_sigma,
        device=TEST_DEVICE
    )
    
    # Verify initialization
    sigma = blur.get_sigma()
    assert abs(sigma - init_sigma) < 0.01, f"Init sigma mismatch: {sigma} vs {init_sigma}"
    print(f"✓ Initialization correct: sigma = {sigma:.4f}")
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128, device=TEST_DEVICE)
    y = blur(x)
    
    # Check output shape
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"✓ Forward pass: input {x.shape} → output {y.shape}")
    
    # Check that blur actually changes the image
    assert not torch.allclose(x, y), "Blur should change the image"
    print(f"✓ Blur modifies image (MSE: {((x-y)**2).mean():.6f})")
    
    # Test gradient flow
    assert y.requires_grad, "Output should require gradients"
    loss = y.sum()
    loss.backward()
    
    assert blur.log_sigma.grad is not None, "log_sigma should have gradient"
    assert blur.log_sigma.grad.abs() > 1e-6, "Gradient should be non-zero"
    print(f"✓ Gradients flow correctly (grad norm: {blur.log_sigma.grad.abs():.6f})")
    
    # Test sigma remains positive after update
    with torch.no_grad():
        blur.log_sigma -= 0.1 * blur.log_sigma.grad
    new_sigma = blur.get_sigma()
    assert new_sigma > 0, "Sigma should remain positive after update"
    print(f"✓ Parameter update: {sigma:.4f} → {new_sigma:.4f}")
    
    # Test kernel extraction
    kernel = blur.get_kernel()
    assert kernel.shape == (61, 61), f"Kernel shape incorrect: {kernel.shape}"
    assert abs(kernel.sum() - 1.0) < 0.01, f"Kernel should sum to 1, got {kernel.sum()}"
    print(f"✓ Kernel extraction: shape {kernel.shape}, sum = {kernel.sum():.6f}")
    
    print("✓ Gaussian blur test PASSED\n")


def test_motion_blur():
    """Test learnable motion blur operator"""
    print("\n--- Testing Motion Blur ---")
    
    init_length = 10.0
    init_angle = 0.0
    blur = LearnableMotionBlur(
        kernel_size=61,
        num_channels=3,
        init_length=init_length,
        init_angle=init_angle,
        device=TEST_DEVICE
    )
    
    # Verify initialization
    params = blur.get_params()
    assert 'length' in params and 'angle' in params, "Missing parameters"
    assert abs(params['length'] - init_length) < 0.01, "Length init incorrect"
    assert abs(params['angle'] - init_angle) < 0.01, "Angle init incorrect"
    print(f"✓ Initialization: length={params['length']:.2f}, angle={params['angle']:.4f}")
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128, device=TEST_DEVICE)
    y = blur(x)
    
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    assert y.requires_grad, "Output should require gradients"
    print(f"✓ Forward pass successful")
    
    # Test gradient flow
    loss = y.sum()
    loss.backward()
    
    assert blur.log_length.grad is not None, "log_length should have gradient"
    assert blur.angle.grad is not None, "angle should have gradient"
    print(f"✓ Gradients: length_grad={blur.log_length.grad.abs():.6f}, angle_grad={blur.angle.grad.abs():.6f}")
    
    # Test parameter update
    with torch.no_grad():
        blur.log_length -= 0.1 * blur.log_length.grad
        blur.angle -= 0.1 * blur.angle.grad
    
    new_params = blur.get_params()
    assert new_params['length'] > 0, "Length should remain positive"
    print(f"✓ Parameters updated: length={new_params['length']:.2f}, angle={new_params['angle']:.4f}")
    
    # Test kernel extraction
    kernel = blur.get_kernel()
    assert kernel.shape == (61, 61), f"Kernel shape incorrect"
    assert abs(kernel.sum() - 1.0) < 0.01, f"Kernel should sum to 1"
    print(f"✓ Kernel valid: sum = {kernel.sum():.6f}")
    
    print("✓ Motion blur test PASSED\n")


def test_learnable_mask():
    """Test learnable mask operator"""
    print("\n--- Testing Learnable Mask ---")
    
    init_ratio = 0.5
    mask_op = LearnableMask(
        image_shape=(3, 128, 128),
        init_ratio=init_ratio,
        temperature=1.0,
        device=TEST_DEVICE
    )
    
    # Test initialization
    ratio = mask_op.get_masked_ratio()
    print(f"✓ Initialized with mask ratio: {ratio:.3f} (target: {init_ratio})")
    
    # Test soft mask (differentiable)
    x = torch.randn(2, 3, 128, 128, device=TEST_DEVICE)
    y_soft = mask_op(x, hard=False)
    
    assert y_soft.shape == x.shape, f"Shape mismatch"
    assert y_soft.requires_grad, "Soft mask should be differentiable"
    print(f"✓ Soft mask: differentiable, shape {y_soft.shape}")
    
    # Test hard mask (binary)
    y_hard = mask_op(x, hard=True)
    
    # Check that values are either 0 or original
    mask_binary = (y_hard != 0).float()
    reconstruction_error = torch.abs(y_hard - x * mask_binary).max()
    assert reconstruction_error < 1e-5, "Hard mask should be binary"
    print(f"✓ Hard mask: binary, max error = {reconstruction_error:.8f}")
    
    # Test gradient flow
    loss = y_soft.sum()
    loss.backward()
    
    assert mask_op.mask_logits.grad is not None, "mask_logits should have gradient"
    print(f"✓ Gradients flow (grad norm: {mask_op.mask_logits.grad.norm():.6f})")
    
    # Test mask extraction
    mask = mask_op.get_mask(hard=True)
    assert mask.shape == (3, 128, 128), f"Mask shape incorrect"
    assert torch.all((mask == 0) | (mask == 1)), "Mask should be binary"
    
    final_ratio = mask_op.get_masked_ratio()
    assert 0 <= final_ratio <= 1, f"Ratio should be in [0,1], got {final_ratio}"
    print(f"✓ Mask extraction: ratio = {final_ratio:.3f}")
    
    print("✓ Learnable mask test PASSED\n")


def test_downsampling():
    """Test learnable downsampling operator"""
    print("\n--- Testing Learnable Downsampling ---")
    
    scale = 2
    downsample = LearnableDownsampling(
        scale_factor=scale,
        num_channels=3,
        kernel_size=4,
        device=TEST_DEVICE
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128, device=TEST_DEVICE)
    y = downsample(x)
    
    expected_shape = (2, 3, 64, 64)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    print(f"✓ Downsampling: {x.shape} → {y.shape}")
    
    # Test gradients
    assert y.requires_grad, "Output should require gradients"
    loss = y.sum()
    loss.backward()
    
    assert downsample.kernel.grad is not None, "Kernel should have gradient"
    print(f"✓ Gradients flow (kernel grad norm: {downsample.kernel.grad.norm():.6f})")
    
    # Test kernel extraction
    kernel = downsample.get_kernel()
    assert kernel.shape == (4, 4), f"Kernel shape incorrect"
    print(f"✓ Kernel shape: {kernel.shape}, sum = {kernel.sum():.6f}")
    
    # Test different scales
    for test_scale in [2, 4]:
        ds = LearnableDownsampling(scale_factor=test_scale, device=TEST_DEVICE)
        y_test = ds(x)
        expected_h = 128 // test_scale
        expected_w = 128 // test_scale
        assert y_test.shape[2] == expected_h, f"Height mismatch for scale {test_scale}"
        assert y_test.shape[3] == expected_w, f"Width mismatch for scale {test_scale}"
        print(f"✓ Scale {test_scale}x: {x.shape} → {y_test.shape}")
    
    print("✓ Downsampling test PASSED\n")


def test_composite_operator():
    """Test composite of multiple operators"""
    print("\n--- Testing Composite Operator ---")
    
    # Create composite: blur + downsample
    blur = LearnableGaussianBlur(kernel_size=31, init_sigma=1.0, device=TEST_DEVICE)
    downsample = LearnableDownsampling(scale_factor=2, device=TEST_DEVICE)
    
    composite = CompositeOperator([blur, downsample], noise_level=0.01)
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128, device=TEST_DEVICE)
    y = composite(x, add_noise=True)
    
    expected_shape = (2, 3, 64, 64)
    assert y.shape == expected_shape, f"Expected {expected_shape}, got {y.shape}"
    print(f"✓ Composite forward: {x.shape} → {y.shape}")
    
    # Test gradients flow through all operators
    assert y.requires_grad, "Output should require gradients"
    loss = y.sum()
    loss.backward()
    
    assert blur.log_sigma.grad is not None, "Blur sigma should have gradient"
    assert downsample.kernel.grad is not None, "Downsample kernel should have gradient"
    print(f"✓ Gradients flow through all operators")
    
    # Test noise level
    noise_level = composite.get_noise_level()
    assert noise_level > 0, "Noise level should be positive"
    print(f"✓ Noise level: {noise_level:.4f}")
    
    # Test without noise
    y_no_noise = composite(x, add_noise=False)
    assert y_no_noise.shape == expected_shape, "Shape should be same without noise"
    print(f"✓ Composite without noise works")
    
    print("✓ Composite operator test PASSED\n")


def test_device_compatibility():
    """Test that operators work on different devices"""
    print("\n--- Testing Device Compatibility ---")
    
    # Always test on CPU
    blur_cpu = LearnableGaussianBlur(kernel_size=31, init_sigma=1.0, device='cpu')
    x_cpu = torch.randn(1, 3, 64, 64, device='cpu')
    y_cpu = blur_cpu(x_cpu)
    
    assert y_cpu.device.type == 'cpu', "Output should be on CPU"
    assert y_cpu.shape == x_cpu.shape, "Shape should be preserved"
    print("✓ CPU test passed")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        blur_cuda = LearnableGaussianBlur(kernel_size=31, init_sigma=1.0, device='cuda')
        x_cuda = torch.randn(1, 3, 64, 64, device='cuda')
        y_cuda = blur_cuda(x_cuda)
        
        assert y_cuda.device.type == 'cuda', "Output should be on CUDA"
        assert y_cuda.shape == x_cuda.shape, "Shape should be preserved"
        print("✓ CUDA test passed")
    else:
        print("↓ CUDA not available, skipping CUDA test")
    
    print("✓ Device compatibility test PASSED\n")


def test_parameter_gradients():
    """Test that all learnable parameters receive non-zero gradients"""
    print("\n--- Testing Parameter Gradients ---")
    
    # Test Gaussian blur
    blur = LearnableGaussianBlur(kernel_size=31, init_sigma=2.0, device=TEST_DEVICE)
    x = torch.randn(2, 3, 64, 64, device=TEST_DEVICE)
    y = blur(x)
    loss = y.sum()
    loss.backward()
    
    assert blur.log_sigma.grad is not None, "log_sigma should have gradient"
    assert blur.log_sigma.grad.abs() > 1e-6, "Gradient should be non-zero"
    print(f"✓ Gaussian blur: grad = {blur.log_sigma.grad.item():.6f}")
    
    # Test mask
    mask = LearnableMask(image_shape=(3, 64, 64), device=TEST_DEVICE)
    x_mask = torch.randn(2, 3, 64, 64, device=TEST_DEVICE)
    y_mask = mask(x_mask, hard=False)
    loss_mask = y_mask.sum()
    loss_mask.backward()
    
    assert mask.mask_logits.grad is not None, "mask_logits should have gradient"
    grad_norm = mask.mask_logits.grad.norm()
    assert grad_norm > 1e-6, "Gradient norm should be non-zero"
    print(f"✓ Mask: grad norm = {grad_norm:.6f}")
    
    # Test motion blur
    motion = LearnableMotionBlur(kernel_size=31, init_length=5.0, device=TEST_DEVICE)
    y_motion = motion(x)
    loss_motion = y_motion.sum()
    loss_motion.backward()
    
    assert motion.log_length.grad is not None, "log_length should have gradient"
    assert motion.angle.grad is not None, "angle should have gradient"
    print(f"✓ Motion blur: length_grad = {motion.log_length.grad.abs():.6f}, angle_grad = {motion.angle.grad.abs():.6f}")
    
    print("✓ Parameter gradients test PASSED\n")


def test_factory_function():
    """Test the create_blind_operator factory function"""
    print("\n--- Testing Factory Function ---")
    
    # Test creating each operator type
    operators = {
        'gaussian_blur': {'kernel_size': 31, 'init_sigma': 2.0},
        'motion_blur': {'kernel_size': 31, 'init_length': 10.0},
        'mask': {'image_shape': (3, 64, 64)},
        'downsample': {'scale_factor': 2}
    }
    
    for op_type, kwargs in operators.items():
        op = create_blind_operator(op_type, device=TEST_DEVICE, **kwargs)
        assert op is not None, f"Failed to create {op_type}"
        
        # Test forward pass
        x = torch.randn(1, 3, 64, 64, device=TEST_DEVICE)
        
        if op_type == 'mask':
            y = op(x, hard=False)
        else:
            y = op(x)
        
        assert y is not None, f"Forward pass failed for {op_type}"
        print(f"✓ Created and tested: {op_type}")
    
    # Test composite
    blur = create_blind_operator('gaussian_blur', init_sigma=1.0, device=TEST_DEVICE)
    ds = create_blind_operator('downsample', scale_factor=2, device=TEST_DEVICE)
    composite = create_blind_operator('composite', operators=[blur, ds], noise_level=0.01)
    
    x = torch.randn(1, 3, 128, 128, device=TEST_DEVICE)
    y = composite(x)
    assert y.shape == (1, 3, 64, 64), "Composite shape incorrect"
    print(f"✓ Created composite operator")
    
    print("✓ Factory function test PASSED\n")


def test_get_operator_parameters():
    """Test parameter extraction utility"""
    print("\n--- Testing Parameter Extraction ---")
    
    # Test with Gaussian blur
    blur = LearnableGaussianBlur(init_sigma=2.5, device=TEST_DEVICE)
    params = get_operator_parameters(blur)
    
    assert 'sigma' in params, "Sigma should be in parameters"
    assert abs(params['sigma'] - 2.5) < 0.01, "Sigma value incorrect"
    print(f"✓ Gaussian blur parameters: {params}")
    
    # Test with motion blur
    motion = LearnableMotionBlur(init_length=10.0, init_angle=1.0, device=TEST_DEVICE)
    params = get_operator_parameters(motion)
    
    assert 'length' in params and 'angle' in params, "Motion parameters missing"
    print(f"✓ Motion blur parameters: {params}")
    
    # Test with mask
    mask = LearnableMask(image_shape=(3, 64, 64), init_ratio=0.3, device=TEST_DEVICE)
    params = get_operator_parameters(mask)
    
    assert 'mask_ratio' in params, "Mask ratio should be in parameters"
    print(f"✓ Mask parameters: {params}")
    
    print("✓ Parameter extraction test PASSED\n")


def test_degradation_wrapper():
    """Test the LearnableDegradation wrapper for PNP_FLOW compatibility"""
    print("\n--- Testing Degradation Wrapper ---")
    
    # Create a learnable operator
    blur = LearnableGaussianBlur(init_sigma=2.0, device=TEST_DEVICE)
    
    # Create adjoint (identity for simplicity)
    def H_adj(y):
        return y
    
    # Wrap it
    degradation = LearnableDegradation(blur, H_adj=H_adj)
    
    # Test H (forward)
    x = torch.randn(1, 3, 64, 64, device=TEST_DEVICE)
    y = degradation.H(x)
    
    assert y.shape == x.shape, "H output shape incorrect"
    print(f"✓ H (forward) works: {x.shape} → {y.shape}")
    
    # Test H_adj (adjoint)
    y_adj = degradation.H_adj(y)
    assert y_adj.shape == y.shape, "H_adj output shape incorrect"
    print(f"✓ H_adj (adjoint) works: {y.shape} → {y_adj.shape}")
    
    # Test getting underlying operator
    op = degradation.get_operator()
    assert op is blur, "Should return underlying operator"
    print(f"✓ Can retrieve underlying operator")
    
    print("✓ Degradation wrapper test PASSED\n")


def test_parameter_constraints():
    """Test that parameters stay in valid ranges"""
    print("\n--- Testing Parameter Constraints ---")
    
    # Test sigma positivity
    blur = LearnableGaussianBlur(init_sigma=1.0, device=TEST_DEVICE)
    
    # Try to make sigma negative (shouldn't be possible due to log parameterization)
    with torch.no_grad():
        blur.log_sigma.data = torch.tensor(-10.0)  # Very negative log
    
    sigma = blur.get_sigma()
    assert sigma > 0, f"Sigma should always be positive, got {sigma}"
    print(f"✓ Sigma remains positive: {sigma:.8f}")
    
    # Test mask ratio stays in [0, 1]
    mask = LearnableMask(image_shape=(3, 64, 64), device=TEST_DEVICE)
    ratio = mask.get_masked_ratio()
    assert 0 <= ratio <= 1, f"Mask ratio should be in [0,1], got {ratio}"
    print(f"✓ Mask ratio in valid range: {ratio:.3f}")
    
    # Test length positivity
    motion = LearnableMotionBlur(init_length=5.0, device=TEST_DEVICE)
    with torch.no_grad():
        motion.log_length.data = torch.tensor(-5.0)
    
    length = motion.get_params()['length']
    assert length > 0, f"Length should be positive, got {length}"
    print(f"✓ Length remains positive: {length:.6f}")
    
    print("✓ Parameter constraints test PASSED\n")


def test_batch_processing():
    """Test that operators handle different batch sizes correctly"""
    print("\n--- Testing Batch Processing ---")
    
    blur = LearnableGaussianBlur(init_sigma=2.0, device=TEST_DEVICE)
    
    batch_sizes = [1, 2, 4, 8]
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 64, 64, device=TEST_DEVICE)
        y = blur(x)
        
        assert y.shape[0] == bs, f"Batch size mismatch: expected {bs}, got {y.shape[0]}"
        assert y.shape[1:] == (3, 64, 64), f"Non-batch dims incorrect"
        print(f"✓ Batch size {bs}: {x.shape} → {y.shape}")
    
    print("✓ Batch processing test PASSED\n")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("RUNNING BLIND DEGRADATION TESTS")
    print("="*60)
    
    tests = [
        ("Gaussian Blur", test_gaussian_blur),
        ("Motion Blur", test_motion_blur),
        ("Learnable Mask", test_learnable_mask),
        ("Downsampling", test_downsampling),
        ("Composite Operator", test_composite_operator),
        ("Device Compatibility", test_device_compatibility),
        ("Parameter Gradients", test_parameter_gradients),
        ("Factory Function", test_factory_function),
        ("Parameter Extraction", test_get_operator_parameters),
        ("Degradation Wrapper", test_degradation_wrapper),
        ("Parameter Constraints", test_parameter_constraints),
        ("Batch Processing", test_batch_processing),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            error_msg = f"FAILED: {test_name}\n  Error: {str(e)}"
            errors.append(error_msg)
            print(f"\n↓ {error_msg}\n")
        except Exception as e:
            failed += 1
            error_msg = f"ERROR: {test_name}\n  Exception: {str(e)}"
            errors.append(error_msg)
            print(f"\n↓ {error_msg}\n")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print("="*60)
    
    if failed > 0:
        print("\nFailed tests:")
        for error in errors:
            print(f"  • {error}")
        print("\nSOME TESTS FAILED")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Review the test outputs above")
        print("  2. Proceed to implement blind_pnp_flow.py")
        print("  3. Test on synthetic data")
        sys.exit(0)