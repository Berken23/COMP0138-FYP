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
# Step 3 — Baseline blind alternating descent (keep as-is)
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

    op_opt = torch.optim.Adam(op.parameters(), lr=operator_lr)

    sigma_history, loss_history = [], []

    for it in range(num_iters):
        # --- x-step (pure data consistency)
        x.requires_grad_(True)
        loss_x = torch.mean((op(x) - prob.y) ** 2)
        grad_x = torch.autograd.grad(loss_x, x)[0]

        with torch.no_grad():
            x -= image_lr * grad_x
        x = x.detach()

        # --- sigma-step
        if it % operator_update_freq == 0:
            op_opt.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - prob.y) ** 2)
            loss_op.backward()
            op_opt.step()

        sigma_history.append(op.sigma().item())
        loss_history.append(loss_x.item())

    return {
        "x": x,
        "sigma_history": sigma_history,
        "loss_history": loss_history,
        "sigma_final": op.sigma().item(),
    }


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
        # Gaussian blur ≈ self-adjoint
        return self.op(r)


def _single_image_pnp_flow_update(
    pnp: PNP_FLOW,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    degradation: _BlindDegradation,
    sigma_noise: float,
    steps: int,
) -> torch.Tensor:
    H, H_adj = degradation.H, degradation.H_adj
    pnp.args.sigma_noise = float(sigma_noise)

    delta = 1.0 / steps
    base_lr = float(pnp.args.lr_pnp)

    if pnp.args.noise_type == "gaussian":
        lr = (sigma_noise ** 2) * base_lr
    else:
        raise ValueError("Only gaussian noise supported here")

    with torch.no_grad():
        for i in range(steps):
            t = torch.ones(len(x), device=x.device) * delta * i
            lr_t = pnp.learning_rate_strat(lr, t)
            z = x - lr_t * pnp.grad_datafit(x, y, H, H_adj)

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
    operator_update_freq: int = 10,
    outer_iters: int = 30,
    pnp_lr_pnp: float = 1.0,
    pnp_steps_inner: int = 10,
    warmup_outer_iters: int = 1,
    warmup_pnp_steps: int = 100,
) -> Dict:
    from types import SimpleNamespace

    args = SimpleNamespace(
        method="pnp_flow",
        model="ot",
        noise_type="gaussian",
        lr_pnp=pnp_lr_pnp,
        steps_pnp=pnp_steps_inner,
        num_samples=5,
        gamma_style="alpha_1_minus_t",
        alpha=1.0,
        sigma_noise=prob.sigma_noise,
    )

    pnp = PNP_FLOW(model=model, device=device, args=args)

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    op_opt = torch.optim.Adam(op.parameters(), lr=operator_lr)

    y = prob.y.to(device)
    degradation = _BlindDegradation(op)
    x = degradation.H_adj(y).detach()

    sigma_history, loss_history = [], []

    for outer in range(outer_iters):
        degradation = _BlindDegradation(op)

        if outer < warmup_outer_iters:
            x = _single_image_pnp_flow_update(
                pnp, x=x, y=y, degradation=degradation,
                sigma_noise=prob.sigma_noise,
                steps=warmup_pnp_steps,
            ).detach()

        x = _single_image_pnp_flow_update(
            pnp, x=x, y=y, degradation=degradation,
            sigma_noise=prob.sigma_noise,
            steps=pnp_steps_inner,
        ).detach()

        if outer % operator_update_freq == 0:
            op_opt.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - y) ** 2)
            loss_op.backward()
            op_opt.step()

        with torch.no_grad():
            loss_x = torch.mean((op(x) - y) ** 2).item()

        sigma_history.append(op.sigma().item())
        loss_history.append(loss_x)
        
        if outer % 5 == 0 or outer == outer_iters - 1:
            print(
                f"[Outer {outer:03d}/{outer_iters}] "
                f"sigma={op.sigma().item():.4f} "
                f"loss={loss_x:.4e}"
            )


    return {
        "x": x,
        "sigma_final": op.sigma().item(),
        "sigma_history": sigma_history,
        "loss_history": loss_history,
    }


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
# __main__ — Step 4 sanity run (Option A)
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load EXACT training config ----
    cfg = load_cfg_from_cfg_file("./config/main_config.yaml")
    dataset_cfg = load_cfg_from_cfg_file(
        cfg.root + f"config/dataset_config/{cfg.dataset}.yaml"
    )
    cfg.update(dataset_cfg)
    method_cfg = load_cfg_from_cfg_file(
        cfg.root + f"config/method_config/{cfg.method}.yaml"
    )
    cfg.update(method_cfg)

    cfg.model = "ot"
    cfg.method = "pnp_flow"
    cfg.problem = "gaussian_deblurring_FFT"
    cfg.noise_type = "gaussian"
    cfg.num_channels = 3
    cfg.dim_image = 128  # change to 64 if your checkpoint is 64×64

    model, state = define_model(cfg)
    checkpoint_path = "./model/celeba/ot/model_final.pt"

    load_model(
        "ot",
        model,
        state,
        download=False,
        checkpoint_path=checkpoint_path,
        dataset=None,
        device=device,
    )
    model.eval()

    # ---- Synthetic blind problem (must match channels & size) ----
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
