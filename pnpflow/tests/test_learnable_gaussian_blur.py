import math
import torch
from pnpflow.blind_degradations import LearnableGaussianBlur

def test_shape_and_determinism():
    torch.manual_seed(0)
    blur = LearnableGaussianBlur(init_sigma=1.0, sigma_max=5.0, padding="reflect")
    x = torch.randn(2, 3, 64, 64)

    y1 = blur(x)
    y2 = blur(x)

    assert y1.shape == x.shape
    # Deterministic (no randomness inside forward)
    assert torch.allclose(y1, y2, atol=0.0, rtol=0.0)


def test_gradients_exist_for_log_sigma():
    torch.manual_seed(0)
    blur = LearnableGaussianBlur(init_sigma=1.2, sigma_max=5.0, padding="reflect")
    x = torch.randn(1, 3, 32, 32, requires_grad=True)

    y = blur(x)
    loss = y.mean()
    loss.backward()

    assert blur.log_sigma.grad is not None
    assert torch.isfinite(blur.log_sigma.grad).all()
    # Not guaranteed huge, but should not be exactly zero for generic x
    assert blur.log_sigma.grad.abs().item() > 0.0


def test_sigma_updates_in_right_direction_on_simple_target():
    """
    Build a target blurred with sigma_true and verify optimisation pushes sigma toward sigma_true.
    This catches: broken normalisation, broken kernel construction, padding mismatch, disconnected grads.
    """
    torch.manual_seed(0)
    sigma_true = 2.0
    sigma_init = 0.6

    # Use the SAME implementation for truth and learnable to avoid family mismatch
    true_op = LearnableGaussianBlur(init_sigma=sigma_true, sigma_max=5.0, padding="reflect")
    true_op.requires_grad_(False)

    learnable = LearnableGaussianBlur(init_sigma=sigma_init, sigma_max=5.0, padding="reflect")

    x = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y = true_op(x)

    opt = torch.optim.SGD(learnable.parameters(), lr=0.5)

    sigmas = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        pred = learnable(x)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        opt.step()
        sigmas.append(float(learnable.sigma().item()))

    # We expect sigma to increase from 0.6 towards 2.0
    assert sigmas[-1] > sigmas[0]
    # Don't require perfect convergence; require meaningful movement
    assert abs(sigmas[-1] - sigma_true) < abs(sigmas[0] - sigma_true)


def test_identity_limit_small_sigma_is_close_to_input():
    torch.manual_seed(0)
    blur = LearnableGaussianBlur(init_sigma=0.2, sigma_max=5.0, padding="reflect")
    x = torch.randn(1, 3, 64, 64)

    y = blur(x)
    # Small sigma should be close-ish to identity. Not exact due to discretisation.
    err = torch.mean((y - x) ** 2).item()
    assert err < 1e-2


def test_set_sigma_works():
    blur = LearnableGaussianBlur(init_sigma=1.0, sigma_max=5.0, padding="reflect")
    blur.set_sigma_(3.0)
    assert math.isclose(float(blur.sigma().item()), 3.0, rel_tol=1e-5, abs_tol=1e-5)
