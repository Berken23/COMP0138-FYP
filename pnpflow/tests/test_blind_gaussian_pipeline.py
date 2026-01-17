import math
import torch

from pnpflow.blind_data import make_blind_gaussian_blur_problem, operator_mse
from pnpflow.blind_degradations import LearnableGaussianBlur


def test_measurement_generation_matches_noise_level_when_sigma_is_correct():
    """
    If learnable op uses the true sigma, ||H(x_gt) - y||^2 should be ~ E[eps^2] = sigma_noise^2
    (up to sampling error), because y = H(x_gt) + eps.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 3, 64, 64, device=device)
    sigma_true = 2.0
    sigma_noise = 0.05
    sigma_max = 5.0

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=sigma_noise,
        sigma_max=sigma_max,
        padding="reflect",
        seed=123,
    )

    op = LearnableGaussianBlur(
        init_sigma=sigma_true,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)
    op.requires_grad_(False)

    loss = operator_mse(op, prob.x_gt, prob.y).item()
    expected = sigma_noise ** 2

    assert math.isfinite(loss)
    # Loose tolerance: finite sample + spatial correlation from blur
    assert abs(loss - expected) / expected < 0.5


def test_sigma_recovery_when_x_is_fixed_and_noise_zero():
    """
    Core Step 2 checkpoint:
    With noise=0 and x fixed to x_gt, optimising sigma alone should recover sigma_true.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 1, 64, 64, device=device)
    sigma_true = 2.5
    sigma_noise = 0.0
    sigma_max = 6.0

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=sigma_noise,
        sigma_max=sigma_max,
        padding="reflect",
        seed=999,
    )

    sigma_init = 0.8
    learnable = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(learnable.parameters(), lr=5e-2)

    x_fixed = prob.x_gt.detach()
    y = prob.y.detach()

    start_err = abs(float(learnable.sigma().item()) - sigma_true)

    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = operator_mse(learnable, x_fixed, y)
        loss.backward()
        opt.step()

    final_sigma = float(learnable.sigma().item())
    final_err = abs(final_sigma - sigma_true)

    assert final_err < start_err * 0.3
    assert final_err < 0.2


def test_true_op_is_frozen():
    torch.manual_seed(0)
    x_gt = torch.randn(1, 1, 16, 16)

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=1.5,
        sigma_noise=0.1,
        sigma_max=5.0,
        padding="reflect",
        seed=1,
    )

    for p in prob.true_op.parameters():
        assert not p.requires_grad

def test_sigma_recovery_from_far_initialisation():
    torch.manual_seed(0)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 1, 64, 64, device=device)
    sigma_true = 3.5
    sigma_init = 0.3   # very far
    sigma_max = 6.0

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=0.0,
        sigma_max=sigma_max,
        padding="reflect",
        seed=0,
    )

    learnable = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(learnable.parameters(), lr=5e-2)

    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = operator_mse(learnable, prob.x_gt, prob.y)
        loss.backward()
        opt.step()

    sigma_hat = learnable.sigma().item()

    # Must move strongly toward the correct scale
    assert sigma_hat > 2.5

    # Allow bias due to truncated kernel support
    assert abs(sigma_hat - sigma_true) < 0.6

def test_sigma_recovery_with_moderate_noise():
    torch.manual_seed(0)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 1, 64, 64, device=device)
    sigma_true = 2.0
    sigma_noise = 0.1
    sigma_max = 5.0

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=sigma_noise,
        sigma_max=sigma_max,
        padding="reflect",
        seed=42,
    )

    learnable = LearnableGaussianBlur(
        init_sigma=1.0,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(learnable.parameters(), lr=3e-2)

    for _ in range(400):
        opt.zero_grad(set_to_none=True)
        loss = operator_mse(learnable, prob.x_gt, prob.y)
        loss.backward()
        opt.step()

    # Expect bias but still reasonable
    assert abs(learnable.sigma().item() - sigma_true) < 0.5

def test_sigma_recovery_rgb():
    torch.manual_seed(0)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 3, 64, 64, device=device)
    sigma_true = 2.2
    sigma_max = 5.0

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=0.0,
        sigma_max=sigma_max,
        padding="reflect",
        seed=1,
    )

    learnable = LearnableGaussianBlur(
        init_sigma=0.7,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(learnable.parameters(), lr=5e-2)

    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = operator_mse(learnable, prob.x_gt, prob.y)
        loss.backward()
        opt.step()

    assert abs(learnable.sigma().item() - sigma_true) < 0.3
