"""Helpers shared across the experiment scripts.

Holds the trajectory variants needed by the diagnostic experiments, namely
the per step blind trajectory used by the A.14 baseline and the tracked
trajectory used by the straightness analysis. Also provides a lightweight
timing tracker.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Dict, List

import torch

from .forward import gaussian_blur_fft
from .reconstruction import _denoiser, _interpolate, _lr_schedule


class TimingTracker:
    """Accumulate timing measurements across multiple labelled blocks."""

    def __init__(self):
        self.records: Dict[str, List[float]] = {}

    @contextmanager
    def track(self, label: str):
        t0 = time.time()
        yield
        self.records.setdefault(label, []).append(time.time() - t0)

    def summary(self) -> str:
        lines = [
            f"{'Label':>30} | {'Count':>6} | {'Total (s)':>10} | {'Mean (s)':>10}",
            "-" * 65,
        ]
        for label, times in sorted(self.records.items()):
            n = len(times)
            total = sum(times)
            mean = total / n if n else 0.0
            lines.append(f"{label:>30} | {n:>6} | {total:>10.2f} | {mean:>10.2f}")
        return "\n".join(lines)


def pnp_flow_trajectory_tracked(
    model,
    *,
    x_init: torch.Tensor,
    y: torch.Tensor,
    sigma_blur: float,
    num_steps: int = 100,
    lr: float = 1.0,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
):
    """Run a PnP-Flow trajectory and return (final_x, snapshots, t_values).
    Used for the trajectory straightness analysis.
    """
    device = x_init.device
    x = x_init.clone()
    delta = 1.0 / num_steps
    trajectory = [x.detach().cpu().clone()]
    t_values = [0.0]

    with torch.no_grad():
        for k in range(num_steps):
            t_scalar = delta * k
            t = torch.full((len(x),), t_scalar, device=device)

            residual = gaussian_blur_fft(x, sigma_blur) - y
            grad = gaussian_blur_fft(residual, sigma_blur)
            lr_k = _lr_schedule(lr, t_scalar, gamma_style, alpha)
            z = x - lr_k * grad

            x_new = torch.zeros_like(x)
            for _ in range(num_samples):
                z_tilde = _interpolate(z, t)
                x_new = x_new + _denoiser(model, z_tilde, t, model_type)
            x = x_new / num_samples

            trajectory.append(x.detach().cpu().clone())
            t_values.append(t_scalar + delta)

    return x.detach(), trajectory, t_values


def pnp_flow_trajectory_blind(
    model,
    *,
    x_init: torch.Tensor,
    y: torch.Tensor,
    sigma_init: float,
    sigma_max: float,
    num_steps: int = 100,
    lr: float = 1.0,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    model_type: str = "ot",
    operator_lr: float = 1e-3,
    clip_grad_norm: float | None = 1.0,
):
    """A.14 per step Adam blind PnP-Flow used by the baseline experiment.
    Updates sigma with Adam at every iteration of the trajectory.

    Returns (x_final, sigma_final, sigma_history).
    """
    from pnpflow.blind_degradations import LearnableGaussianBlur

    device = x_init.device
    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device=device, dtype=x_init.dtype)
    opt_sigma = torch.optim.Adam(op.parameters(), lr=operator_lr)

    x = x_init.clone()
    delta = 1.0 / num_steps
    sigma_history: List[float] = []

    for k in range(num_steps):
        t_scalar = delta * k
        t = torch.full((len(x),), t_scalar, device=device)

        with torch.no_grad():
            residual = op(x) - y
            grad = op(residual)
            lr_k = _lr_schedule(lr, t_scalar, gamma_style, alpha)
            z = x - lr_k * grad

        with torch.no_grad():
            x_new = torch.zeros_like(x)
            for _ in range(num_samples):
                z_tilde = _interpolate(z, t)
                x_new = x_new + _denoiser(model, z_tilde, t, model_type)
            x = x_new / num_samples

        opt_sigma.zero_grad(set_to_none=True)
        loss_kernel = torch.mean((y - op(x.detach())) ** 2)
        loss_kernel.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(op.parameters(), clip_grad_norm)
        opt_sigma.step()
        op.clamp_params_()
        sigma_history.append(op.sigma().item())

    return x.detach(), op.sigma().item(), sigma_history


def find_repo_root() -> str:
    """Walk up from this file until we find the repository root."""
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(here, "pnpflow")) and os.path.isdir(
            os.path.join(here, "model")
        ):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate repository root containing pnpflow/ and model/.")


def load_model(repo_root: str, device: str, img_size: int = 128, num_channels: int = 3):
    """Load the pretrained CelebA flow matching model used throughout the
    blind deconvolution experiments.
    """
    import sys
    import types

    pnpflow_dir = os.path.join(repo_root, "pnpflow")
    if pnpflow_dir not in sys.path:
        sys.path.insert(0, pnpflow_dir)

    from pnpflow.utils import define_model, load_model as _load_model

    model_args = types.SimpleNamespace(
        model="ot",
        num_channels=num_channels,
        dim_image=img_size,
    )
    model, _ = define_model(model_args)
    _load_model(
        "ot",
        model,
        None,
        download=False,
        checkpoint_path=os.path.join(repo_root, "model", "celeba", "ot", "model_final.pt"),
        dataset=None,
        device=device,
    )
    return model.to(device).eval()


def save_json(path: str, obj) -> None:
    """JSON serialise with float coercion for numpy and torch scalars."""
    def _coerce(x):
        if hasattr(x, "item"):
            try:
                return x.item()
            except Exception:
                pass
        if isinstance(x, dict):
            return {k: _coerce(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_coerce(v) for v in x]
        return x

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_coerce(obj), f, indent=2)


SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
NOISE_LEVELS = [0.01, 0.05, 0.1]
SEED = 42
