"""
Blind PnP-Flow Matching

Steps:
0 - Minimal blind problem
1 - Learnable forward operator
2 - True operator
3 - Blind alternating descent (no prior)
4 - Blind PnP-Flow alternating descent (with flow prior)
"""

import torch
from typing import Dict
from types import SimpleNamespace

from pnpflow.blind_degradations import LearnableGaussianBlur
from pnpflow.blind_data import (
    BlindGaussianBlurProblem,
    make_blind_gaussian_blur_problem,
)
from pnpflow.methods.pnp_flow import PNP_FLOW
from pnpflow.utils import (
    load_cfg_from_cfg_file,
    define_model,
    load_model,
)


# ============================================================
# Step 3 — Baseline blind alternating descent (no prior)
# ============================================================
def blind_alternating_descent(
    prob: BlindGaussianBlurProblem,
    *,
    sigma_init: float,
    sigma_max: float,
    num_iters: int = 200,
    operator_lr: float = 5e-2,
    image_lr: float = 1e-1,
    operator_update_freq: int = 5,
) -> Dict:
    device = prob.x_gt.device
    x = torch.randn_like(prob.x_gt, requires_grad=True)

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(op.parameters(), lr=operator_lr)

    sigma_hist, loss_hist = [], []

    for it in range(num_iters):
        x.requires_grad_(True)
        loss_x = torch.mean((op(x) - prob.y) ** 2)
        grad_x = torch.autograd.grad(loss_x, x)[0]

        with torch.no_grad():
            x -= image_lr * grad_x
        x = x.detach()

        if it % operator_update_freq == 0:
            opt.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - prob.y) ** 2)
            loss_op.backward()
            opt.step()
            op.clamp_params_()

        sigma_hist.append(op.sigma().item())
        loss_hist.append(loss_x.item())

    return dict(
        x=x,
        sigma_history=sigma_hist,
        loss_history=loss_hist,
        sigma_final=op.sigma().item(),
    )


# ============================================================
# Step 4 — Blind PnP-Flow Matching
# ============================================================
class _BlindDegradation:
    """Adapter exposing H and H_adj for PNP_FLOW."""
    def __init__(self, op: LearnableGaussianBlur):
        self.op = op

    def H(self, x):
        return self.op(x)

    def H_adj(self, r):
        # Approximate adjoint (OK for debugging; reflect padding is not exactly self-adjoint).
        return self.op(r)


def _single_image_pnp_flow_update(
    pnp: PNP_FLOW,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    degradation: _BlindDegradation,
    steps: int,
) -> torch.Tensor:
    """
    Single-image PnP-Flow inner loop (blind-safe).

    - Neutralises internal /sigma_noise^2 scaling by setting sigma_noise = 1
    - No gradient tracking
    - Guards against NaNs/Infs
    """
    H, H_adj = degradation.H, degradation.H_adj
    pnp.args.sigma_noise = 1.0  # neutralise internal scaling

    delta = 1.0 / steps
    lr = float(pnp.args.lr_pnp)

    with torch.no_grad():
        for i in range(steps):
            t = torch.ones(len(x), device=x.device) * delta * i
            lr_t = pnp.learning_rate_strat(lr, t)

            grad = pnp.grad_datafit(x, y, H, H_adj)
            z = x - lr_t * grad

            if not torch.isfinite(z).all():
                raise FloatingPointError(
                    "Non-finite values detected in PnP inner loop. "
                    "Reduce pnp_lr_pnp."
                )

            x_new = torch.zeros_like(x)
            for _ in range(pnp.args.num_samples):
                z_tilde = pnp.interpolation_step(z, t.view(-1, 1, 1, 1))
                x_new += pnp.denoiser(z_tilde, t)

            x = x_new / pnp.args.num_samples

    return x


def blind_pnp_flow_alternating(
    prob: BlindGaussianBlurProblem,
    *,
    model,
    device,
    sigma_init: float,
    sigma_max: float,
    operator_lr: float = 3e-2,
    operator_update_freq: int = 10,    # ✅ IMPORTANT: stable schedule
    outer_iters: int = 80,
    pnp_lr_pnp: float = 0.05,
    pnp_steps_inner: int = 25,
    warmup_outer_iters: int = 1,
    warmup_pnp_steps: int = 200,
) -> Dict:
    args = SimpleNamespace(
        method="pnp_flow",
        model="ot",
        noise_type="gaussian",
        lr_pnp=pnp_lr_pnp,
        steps_pnp=pnp_steps_inner,
        num_samples=1,
        gamma_style="alpha_1_minus_t",
        alpha=1.0,
        sigma_noise=1.0,
    )

    pnp = PNP_FLOW(model=model, device=device, args=args)

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(op.parameters(), lr=operator_lr)

    y = prob.y.to(device)
    x = op(y).detach()  # H_adj(y) init

    sigma_hist, loss_hist = [], []

    for outer in range(outer_iters):
        degradation = _BlindDegradation(op)

        # Warmup: PnP only, fixed sigma (pure warmup)
        if outer < warmup_outer_iters:
            x = _single_image_pnp_flow_update(
                pnp,
                x=x,
                y=y,
                degradation=degradation,
                steps=warmup_pnp_steps,
            ).detach()
            continue

        # PnP update
        x = _single_image_pnp_flow_update(
            pnp,
            x=x,
            y=y,
            degradation=degradation,
            steps=pnp_steps_inner,
        ).detach()

        # Operator update (less frequent to prevent drift)
        if outer % operator_update_freq == 0:
            opt.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - y) ** 2)
            loss_op.backward()

            g = op.log_sigma.grad
            if g is None or not torch.isfinite(g).all():
                raise RuntimeError("log_sigma grad is None or non-finite")

            opt.step()
            op.clamp_params_()

            grad_mag = float(g.detach().abs().mean())
        else:
            grad_mag = float("nan")

        with torch.no_grad():
            loss_x = torch.mean((op(x) - y) ** 2).item()

        sigma_hist.append(op.sigma().item())
        loss_hist.append(loss_x)

        if outer % 5 == 0 or outer == outer_iters - 1:
            msg = (
                f"[Outer {outer:03d}/{outer_iters}] "
                f"sigma={op.sigma().item():.4f} "
                f"loss={loss_x:.4e}"
            )
            if outer % operator_update_freq == 0:
                msg += f" |gradσ|={grad_mag:.3e}"
            print(msg)

    return dict(
        x=x,
        sigma_final=op.sigma().item(),
        sigma_history=sigma_hist,
        loss_history=loss_hist,
    )


# ============================================================
# Diagnostics
# ============================================================
def _check_history_sanity(sigma_history, loss_history, name="Blind PnP-Flow"):
    print(f"\n=== {name} Summary ===")
    print("Sigma (first 10):", [round(s, 4) for s in sigma_history[:10]])
    print("Sigma (last 10): ", [round(s, 4) for s in sigma_history[-10:]])
    print("Loss  (first 10):", [round(l, 4) for l in loss_history[:10]])
    print("Loss  (last 10): ", [round(l, 4) for l in loss_history[-10:]])
    print(f"Δ sigma: {sigma_history[-1] - sigma_history[0]:.4f}")
    print(f"Δ loss:  {loss_history[-1] - loss_history[0]:.4e}")
    print("=============================\n")


# ============================================================
# __main__ — Step 4 sanity run
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_cfg_from_cfg_file("./config/main_config.yaml")
    cfg.update(load_cfg_from_cfg_file(cfg.root + f"config/dataset_config/{cfg.dataset}.yaml"))
    cfg.update(load_cfg_from_cfg_file(cfg.root + f"config/method_config/{cfg.method}.yaml"))

    cfg.model = "ot"
    cfg.method = "pnp_flow"
    cfg.num_channels = 3
    cfg.dim_image = 128

    model, state = define_model(cfg)
    load_model(
        "ot",
        model,
        state,
        download=False,
        checkpoint_path="./model/celeba/ot/model_final.pt",
        dataset=None,
        device=device,
    )
    model.eval()

    x_gt = torch.randn(1, 3, cfg.dim_image, cfg.dim_image, device=device)

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=1.2,
        sigma_noise=0.05,
        sigma_max=6.0,
        padding="reflect",
        seed=0,
    )

    out = blind_pnp_flow_alternating(
        prob,
        model=model,
        device=device,
        sigma_init=0.4,
        sigma_max=6.0,
    )

    _check_history_sanity(
        out["sigma_history"],
        out["loss_history"],
        name="Blind PnP-Flow (Gaussian Deblur)",
    )

    print(f"Final sigma: {out['sigma_final']:.4f} (true: 1.2)")
