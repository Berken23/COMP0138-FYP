# pnpflow/tests/test_blind_alternating_loop.py
#
# Step 3 stress tests (NO PnP / NO Flow yet).
# Contract at Step 3:
# - Alternating loop is stable and reproducible
# - sigma moves away from init in the correct direction
# - sigma does NOT explode/collapse
# - data-consistency loss decreases
# - schedule changes don't break learning
#
# NOTE: Without a prior, sigma is not expected to exactly recover sigma_true when x is free.
# These tests are intentionally framed around *stability and directional correctness*.

import math
import torch

from pnpflow.blind_data import make_blind_gaussian_blur_problem
from pnpflow.methods.blind_pnp_flow import blind_alternating_descent


def _avg(xs):
    return sum(xs) / max(1, len(xs))


def _run(
    *,
    seed: int,
    sigma_true: float,
    sigma_init: float,
    sigma_max: float,
    num_iters: int,
    operator_lr: float,
    image_lr: float,
    operator_update_freq: int,
    noise: float = 0.0,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 1, 64, 64, device=device)

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=noise,
        sigma_max=sigma_max,
        padding="reflect",
        seed=seed,
    )

    out = blind_alternating_descent(
        prob,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=num_iters,
        operator_lr=operator_lr,
        image_lr=image_lr,
        operator_update_freq=operator_update_freq,
    )
    return out


def test_blind_loop_sigma_moves_from_bad_init_and_is_stable():
    """
    Primary Step 3 test:
    - sigma must move substantially away from a bad init (directionally correct)
    - sigma must remain finite and in a sane range
    - do NOT require exact recovery (no prior yet)
    """
    sigma_true = 2.5
    sigma_init = 0.4
    sigma_max = 6.0

    out = _run(
        seed=0,
        sigma_true=sigma_true,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=300,
        operator_lr=5e-2,
        image_lr=1e-1,
        operator_update_freq=5,
        noise=0.0,
    )

    sigmas = out["sigma_history"]
    sigma_final = out["sigma_final"]

    assert all(math.isfinite(s) for s in sigmas)

    # Must move away from init (core directional correctness)
    assert sigma_final > sigma_init + 0.3

    # Must reach a meaningful scale (avoid trivial stall near init)
    assert sigma_final > 1.2

    # Must not collapse/explode
    assert min(sigmas) > 0.1
    assert max(sigmas) < sigma_max * 1.2


def test_sigma_moves_up_early_for_blur_case():
    """
    Early directional test:
    with blur present and sigma_init < sigma_true,
    sigma should increase in early iterations (not necessarily monotonically forever).
    """
    out = _run(
        seed=1,
        sigma_true=2.0,
        sigma_init=0.6,
        sigma_max=5.0,
        num_iters=120,
        operator_lr=5e-2,
        image_lr=1e-1,
        operator_update_freq=5,
        noise=0.0,
    )

    sigmas = out["sigma_history"]
    assert sigmas[10] > sigmas[0]
    assert sigmas[40] > sigmas[10]


def test_data_consistency_loss_decreases_on_average():
    """
    Data-consistency loss should decrease on average.
    We use an average-of-windows check to reduce sensitivity to small oscillations.
    """
    out = _run(
        seed=2,
        sigma_true=2.2,
        sigma_init=0.8,
        sigma_max=5.0,
        num_iters=220,
        operator_lr=5e-2,
        image_lr=1e-1,
        operator_update_freq=5,
        noise=0.0,
    )

    losses = out["loss_history"]
    early = _avg(losses[:30])
    mid = _avg(losses[80:110])
    late = _avg(losses[-30:])

    assert math.isfinite(early) and math.isfinite(late)

    # Should generally improve over time
    assert mid < early
    assert late < early


def test_sigma_does_not_go_numerically_bad_under_longer_run():
    """
    Longer run stress: sigma should remain finite and bounded.
    This catches graph leakage, bad detach logic, and optimizer instability.
    """
    sigma_true = 3.0
    sigma_init = 0.5
    sigma_max = 6.0

    out = _run(
        seed=3,
        sigma_true=sigma_true,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=450,
        operator_lr=3e-2,   # slightly safer for long run
        image_lr=1e-1,
        operator_update_freq=10,  # slower operator updates for stability
        noise=0.0,
    )

    sigmas = out["sigma_history"]
    assert all(math.isfinite(s) for s in sigmas)
    assert min(sigmas) > 0.1
    assert max(sigmas) < sigma_max * 1.2

    # Still should move away from init meaningfully
    assert out["sigma_final"] > sigma_init + 0.3
    assert out["sigma_final"] > 1.2


def test_schedule_sensitivity_operator_update_freq_does_not_break_learning():
    """
    Changing operator update frequency should not destroy learning.
    We only require directional movement and stability.
    """
    sigma_true = 2.5
    sigma_init = 0.7
    sigma_max = 6.0

    out_fast = _run(
        seed=4,
        sigma_true=sigma_true,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=300,
        operator_lr=3e-2,
        image_lr=1e-1,
        operator_update_freq=5,
        noise=0.0,
    )
    out_slow = _run(
        seed=4,
        sigma_true=sigma_true,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=300,
        operator_lr=3e-2,
        image_lr=1e-1,
        operator_update_freq=10,
        noise=0.0,
    )

    # Both should be stable and move away from init
    for out in (out_fast, out_slow):
        sigmas = out["sigma_history"]
        assert all(math.isfinite(s) for s in sigmas)
        assert out["sigma_final"] > sigma_init + 0.2
        assert out["sigma_final"] > 1.2
        assert min(sigmas) > 0.1
        assert max(sigmas) < sigma_max * 1.2

    # They can differ, but should be broadly similar scale
    assert abs(out_fast["sigma_final"] - out_slow["sigma_final"]) < 1.0


def test_moderate_noise_does_not_cause_sigma_collapse():
    """
    With moderate noise, sigma may be biased, but should not collapse/explode.
    This is a robustness test, not an accuracy test.
    """
    sigma_true = 2.0
    sigma_init = 0.8
    sigma_max = 5.0

    out = _run(
        seed=5,
        sigma_true=sigma_true,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=320,
        operator_lr=3e-2,
        image_lr=1e-1,
        operator_update_freq=10,
        noise=0.1,
    )

    sigmas = out["sigma_history"]
    assert all(math.isfinite(s) for s in sigmas)
    assert min(sigmas) > 0.1
    assert max(sigmas) < sigma_max * 1.2

    # Still should move away from init (directional learning)
    assert out["sigma_final"] > sigma_init + 0.2
