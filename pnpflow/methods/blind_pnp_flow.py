"""
Blind PnP-Flow — clean implementation.

Build layers:
  Layer 1: blind_alternating_descent     — data-fit only, no generative prior
  Layer 2: blind_pnp_flow_single         — single image, PnP-Flow prior (Aim 4, 5)
  Layer 3: blind_pnp_flow_multi          — shared sigma, B images (Aim 5, 6)

Mathematical setup
------------------
Forward model:  y_i = H_{theta}(x_i) + eps_i,   eps_i ~ N(0, sigma_noise^2 I)
H_{theta} = isotropic Gaussian blur with learnable sigma (theta = log_sigma).
H is self-adjoint for symmetric Gaussian kernels (H^T = H).

x update  (data gradient):  x <- x - lr * H(H(x) - y)
sigma update (operator):    Adam on  (1/B) sum_i || H_{sigma}(x_i) - y_i ||^2

Initialisation insight
----------------------
x_init = y  is intentional: with sigma_init << sigma_true,  H(y) is *more* blurry
than y, so H(y) - y < 0, and the gradient step sharpens x.  A sharper x than y
makes the sigma gradient point toward larger sigma, avoiding the identity trap.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from pnpflow.blind_degradations import LearnableGaussianBlur
from pnpflow.blind_data import BlindGaussianBlurProblem


# ---------------------------------------------------------------------------
# Internal: model forward pass  (supports "ot" U-Net and "rectified" NCSNpp)
# ---------------------------------------------------------------------------

def _model_velocity(model, x: torch.Tensor, t: torch.Tensor, model_type: str) -> torch.Tensor:
    """Return the flow velocity v_theta(x, t)."""
    if model_type == "ot":
        return model(x, t)
    elif model_type == "rectified":
        import pnpflow.image_generation.models.utils as mutils
        model_fn = mutils.get_model_fn(model, train=False)
        return model_fn(x.float(), t * 999)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'ot' or 'rectified'.")


# ---------------------------------------------------------------------------
# Internal: PnP-Flow building blocks
# ---------------------------------------------------------------------------

def _lr_schedule(lr: float, t: float, style: str, alpha: float = 1.0) -> float:
    """Step-size schedule gamma(t)."""
    if style == "1_minus_t":
        return lr * (1.0 - t)
    elif style == "sqrt_1_minus_t":
        return lr * (1.0 - t) ** 0.5
    elif style == "alpha_1_minus_t":
        return lr * (1.0 - t) ** alpha
    elif style == "constant":
        return lr
    else:
        raise ValueError(f"Unknown gamma_style '{style}'.")


def _interpolate(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Place x on the flow trajectory at time t:
        x_tilde = t * x + (1 - t) * eps,   eps ~ N(0, I)
    t shape: (B,) or (B,1,1,1)
    """
    t_vec = t.view(-1, 1, 1, 1)
    return t_vec * x + (1.0 - t_vec) * torch.randn_like(x)


def _denoiser(model, x: torch.Tensor, t: torch.Tensor, model_type: str) -> torch.Tensor:
    """
    Flow-matching denoiser at time t:
        D(x, t) = x + (1 - t) * v_theta(x, t)
    Returns a clean-image estimate.
    """
    t_vec = t.view(-1, 1, 1, 1)
    v = _model_velocity(model, x, t, model_type)
    return x + (1.0 - t_vec) * v


# ---------------------------------------------------------------------------
# Layer 1: blind_alternating_descent  (data-fit only, no prior)
#
# Purpose: establish the blind alternating loop, study update orderings and
# convergence without any generative prior.  Corresponds to Aim 4 (baseline).
#
# Contract guaranteed by tests (test_blind_alternating_loop.py):
#   - sigma moves in the correct direction (toward sigma_true)
#   - sigma stays finite and bounded
#   - data-consistency loss decreases on average
# ---------------------------------------------------------------------------

def blind_alternating_descent(
    prob: BlindGaussianBlurProblem,
    *,
    sigma_init: float,
    sigma_max: float,
    num_iters: int = 300,
    operator_lr: float = 5e-2,
    image_lr: float = 1e-1,
    operator_update_freq: int = 5,
    print_every: int = 0,
) -> Dict:
    """
    Alternating gradient descent between x and sigma — no generative prior.

    x  update (every iter):       x  <- x - image_lr * H(H(x) - y)
    sigma update (every freq):    Adam on  || H_sigma(x.detach()) - y ||^2

    Initialisation: x = y  (blurry observation).
    With sigma_init < sigma_true, H(y) is MORE blurry than y, so H(y)-y < 0,
    and the x step sharpens x slightly.  This makes the sigma gradient point
    toward larger sigma (away from the identity trap).

    Returns
    -------
    dict with keys: x, sigma_final, sigma_history, loss_history
    """
    device = prob.y.device
    y = prob.y.to(device)

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device=device, dtype=y.dtype)

    opt_sigma = torch.optim.Adam(op.parameters(), lr=operator_lr)

    # Initialise x at the blurry observation (see docstring for why this works)
    x = y.clone().detach()

    sigma_hist: List[float] = []
    loss_hist: List[float] = []

    for k in range(num_iters):

        # --- x update: gradient descent on (1/2)||H(x) - y||^2  ---
        # gradient = H^T(H(x) - y) = H(H(x) - y)  [H symmetric]
        with torch.no_grad():
            residual = op(x) - y          # H(x) - y
            grad_x = op(residual)         # H( H(x) - y )
            x = x - image_lr * grad_x

        # --- sigma update: Adam on ||H_sigma(x.detach()) - y||^2 ---
        if (k + 1) % operator_update_freq == 0:
            opt_sigma.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - y) ** 2)
            loss_op.backward()
            opt_sigma.step()
            op.clamp_params_()

        # logging
        with torch.no_grad():
            loss_val = torch.mean((op(x) - y) ** 2).item()
        sigma_hist.append(op.sigma().item())
        loss_hist.append(loss_val)

        if print_every > 0 and (k % print_every == 0 or k == num_iters - 1):
            print(
                f"[alt_descent {k:04d}/{num_iters}]  "
                f"sigma={op.sigma().item():.4f}  loss={loss_val:.4e}"
            )

    return {
        "x": x,
        "sigma_final": op.sigma().item(),
        "sigma_history": sigma_hist,
        "loss_history": loss_hist,
    }


# ---------------------------------------------------------------------------
# Layer 2: blind_pnp_flow_single  (single image, PnP-Flow prior)
#
# Alternating scheme:
#   x step:     run a full PnP-Flow trajectory (t: 0 -> 1) with sigma frozen
#   sigma step: Adam on || H_sigma(x.detach()) - y ||^2
#
# Corresponds to Aims 4 and 5.
# ---------------------------------------------------------------------------

def _pnp_flow_trajectory(
    model,
    *,
    x_init: torch.Tensor,
    y: torch.Tensor,
    op: LearnableGaussianBlur,
    sigma_noise: float,
    num_steps: int,
    lr: float,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
) -> torch.Tensor:
    """
    Run one full PnP-Flow trajectory for x with sigma frozen.

    For k = 0, ..., num_steps-1:
        t_k  = k / num_steps
        z    = x - lr(t_k) * H(H(x) - y) / sigma_noise^2
        x    = mean over S samples of  D( interp(z, t_k), t_k )
    """
    device = x_init.device
    x = x_init.clone()
    delta = 1.0 / num_steps

    with torch.no_grad():
        for k in range(num_steps):
            t_scalar = delta * k
            t = torch.full((len(x),), t_scalar, device=device)

            # data-fit gradient step: H^T(H(x) - y) = H(H(x) - y)
            residual = op(x) - y
            grad = op(residual)
            lr_k = _lr_schedule(lr, t_scalar, gamma_style, alpha)
            z = x - lr_k * grad

            # denoiser step (average S noise samples)
            x_new = torch.zeros_like(x)
            for _ in range(num_samples):
                z_tilde = _interpolate(z, t)
                x_new = x_new + _denoiser(model, z_tilde, t, model_type)
            x = x_new / num_samples

    return x


def blind_pnp_flow_single(
    prob: BlindGaussianBlurProblem,
    *,
    model,
    device,
    sigma_init: float,
    sigma_max: float,
    # outer alternating loop
    outer_iters: int = 60,
    # x update (PnP-Flow trajectory)
    pnp_steps: int = 50,
    pnp_lr: float = 0.05,
    sigma_noise: float = 0.05,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
    # sigma update
    operator_lr: float = 1e-2,
    sigma_update_every: int = 1,
    sigma_freeze_iters: int = 5,
    clip_grad_norm: float = 1.0,
    # debug
    print_every: int = 10,
) -> Dict:
    """
    Single-image blind PnP-Flow (Aims 4, 5).

    Outer loop:
      1. x  <- PnP-Flow trajectory  (pnp_steps steps, sigma frozen)
      2. sigma <- Adam step on  || H_sigma(x.detach()) - y ||^2

    Parameters
    ----------
    sigma_freeze_iters : int
        Number of outer iterations before sigma is allowed to update.
        This gives x time to stabilise before operator estimation begins.
    """
    y = prob.y.to(device)

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device=device, dtype=y.dtype)

    opt_sigma = torch.optim.Adam(op.parameters(), lr=operator_lr)

    x = y.clone().detach()

    sigma_hist: List[float] = []
    loss_hist: List[float] = []
    grad_sigma_hist: List[float] = []

    for outer in range(outer_iters):

        # --- x update ---
        x = _pnp_flow_trajectory(
            model,
            x_init=x,
            y=y,
            op=op,
            sigma_noise=sigma_noise,
            num_steps=pnp_steps,
            lr=pnp_lr,
            gamma_style=gamma_style,
            alpha=alpha,
            num_samples=num_samples,
            model_type=model_type,
        ).detach()

        # --- sigma update ---
        do_sigma = (
            outer >= sigma_freeze_iters
            and outer % max(1, sigma_update_every) == 0
        )

        if do_sigma:
            opt_sigma.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - y) ** 2)
            loss_op.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(op.parameters(), clip_grad_norm)
            opt_sigma.step()
            op.clamp_params_()
            grad_mag = (
                op.log_sigma.grad.abs().item()
                if op.log_sigma.grad is not None else float("nan")
            )
        else:
            grad_mag = float("nan")

        with torch.no_grad():
            loss_val = torch.mean((op(x) - y) ** 2).item()

        sigma_hist.append(op.sigma().item())
        loss_hist.append(loss_val)
        grad_sigma_hist.append(grad_mag)

        if print_every > 0 and (outer % print_every == 0 or outer == outer_iters - 1):
            status = f"|grad_sigma|={grad_mag:.3e}" if do_sigma else "sigma frozen"
            print(
                f"[blind_single {outer:03d}/{outer_iters}]  "
                f"sigma={op.sigma().item():.4f}  loss={loss_val:.4e}  {status}"
            )

    return {
        "x": x,
        "sigma_final": op.sigma().item(),
        "sigma_history": sigma_hist,
        "loss_history": loss_hist,
        "grad_sigma_history": grad_sigma_hist,
    }


# ---------------------------------------------------------------------------
# Layer 3: blind_pnp_flow_multi  (shared sigma, B images)
#
# Independent x updates per image (parallel), sigma updated via averaged loss.
# As B increases, the sigma gradient becomes more informative (Aim 5, 6).
# ---------------------------------------------------------------------------

def blind_pnp_flow_multi(
    probs: List[BlindGaussianBlurProblem],
    *,
    model,
    device,
    sigma_init: float,
    sigma_max: float,
    # outer loop
    outer_iters: int = 60,
    # x update
    pnp_steps: int = 50,
    pnp_lr: float = 0.05,
    sigma_noise: float = 0.05,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
    # sigma update
    operator_lr: float = 1e-2,
    sigma_update_every: int = 1,
    sigma_freeze_iters: int = 5,
    clip_grad_norm: float = 1.0,
    # debug
    print_every: int = 10,
) -> Dict:
    """
    Multi-image blind PnP-Flow with shared sigma (Aims 5, 6).

    Each image has its own latent x_i; sigma is shared and updated using
    the *averaged* data-term loss  (1/B) sum_i || H_sigma(x_i) - y_i ||^2.

    Increased B reduces ambiguity in sigma estimation: different x_i draw
    different evidence about the shared operator, making the average gradient
    more informative than any single-image gradient.
    """
    if len(probs) == 0:
        raise ValueError("probs must be non-empty")

    B = len(probs)
    ys = [p.y.to(device) for p in probs]

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device=device, dtype=ys[0].dtype)

    opt_sigma = torch.optim.Adam(op.parameters(), lr=operator_lr)

    xs = [y.clone().detach() for y in ys]

    sigma_hist: List[float] = []
    loss_hist: List[float] = []
    grad_sigma_hist: List[float] = []

    for outer in range(outer_iters):

        # --- x updates: independent PnP-Flow trajectories, shared sigma ---
        for i in range(B):
            xs[i] = _pnp_flow_trajectory(
                model,
                x_init=xs[i],
                y=ys[i],
                op=op,
                sigma_noise=sigma_noise,
                num_steps=pnp_steps,
                lr=pnp_lr,
                gamma_style=gamma_style,
                alpha=alpha,
                num_samples=num_samples,
                model_type=model_type,
            ).detach()

        # --- sigma update: averaged across all images ---
        do_sigma = (
            outer >= sigma_freeze_iters
            and outer % max(1, sigma_update_every) == 0
        )

        if do_sigma:
            opt_sigma.zero_grad(set_to_none=True)
            loss_op = sum(
                torch.mean((op(xs[i].detach()) - ys[i]) ** 2)
                for i in range(B)
            ) / float(B)
            loss_op.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(op.parameters(), clip_grad_norm)
            opt_sigma.step()
            op.clamp_params_()
            grad_mag = (
                op.log_sigma.grad.abs().item()
                if op.log_sigma.grad is not None else float("nan")
            )
        else:
            grad_mag = float("nan")

        with torch.no_grad():
            loss_val = sum(
                torch.mean((op(xs[i]) - ys[i]) ** 2).item()
                for i in range(B)
            ) / float(B)

        sigma_hist.append(op.sigma().item())
        loss_hist.append(loss_val)
        grad_sigma_hist.append(grad_mag)

        if print_every > 0 and (outer % print_every == 0 or outer == outer_iters - 1):
            status = f"|grad_sigma|={grad_mag:.3e}" if do_sigma else "sigma frozen"
            print(
                f"[blind_multi B={B} {outer:03d}/{outer_iters}]  "
                f"sigma={op.sigma().item():.4f}  loss={loss_val:.4e}  {status}"
            )

    return {
        "xs": xs,
        "sigma_final": op.sigma().item(),
        "sigma_history": sigma_hist,
        "loss_history": loss_hist,
        "grad_sigma_history": grad_sigma_hist,
        "batch_size": B,
    }
