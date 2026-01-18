"""
Blind PnP-Flow Matching — Step 5 Only (Fixed)

Fixes:
- Avoid identity trap (x=y + data-only warmup => sigma->0)
- Enforce sigma freeze while x moves under prior gating
- Optional real-image x_gt loading for in-distribution prior behaviour

Step 5 goals:
- Data-term-only warmup (no prior)
- Time-scale separation between image and operator updates
- Explicit prior gating (lambda ramp)
- Sigma updated ONLY from data term
- Sigma-neutral image initialisation
"""

import os
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
# Degradation adapter
# ============================================================
class _BlindDegradation:
    """Expose H and H_adj for PNP_FLOW."""
    def __init__(self, op: LearnableGaussianBlur):
        self.op = op

    def H(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

    """def H_adj(self, r: torch.Tensor) -> torch.Tensor:
        # Approximate adjoint (acceptable for Gaussian + reflect padding)
        return self.op(r)"""
        
    def H_adj(self, r: torch.Tensor) -> torch.Tensor:
        """
        Approximate true adjoint of Gaussian blur under reflect padding.
        Critical for unbiased sigma gradients.
        """
        sigma = self.op.sigma()
        k = self.op._make_kernel_2d(
            sigma,
            device=r.device,
            dtype=r.dtype
        )
        k = torch.flip(k, dims=[0, 1])  # transpose kernel

        c = r.shape[1]
        weight = k.view(1, 1, *k.shape).repeat(c, 1, 1, 1)
        pad = k.shape[-1] // 2

        r_pad = torch.nn.functional.pad(r, (pad, pad, pad, pad), mode="reflect")
        return torch.nn.functional.conv2d(r_pad, weight, groups=c)


# ============================================================
# Data-term-only image update (warmup + anchoring)
# ============================================================
def _data_only_update(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    degradation: _BlindDegradation,
    steps: int,
    image_lr: float,
) -> torch.Tensor:
    """
    Minimise ||H(x) - y||^2 using gradient descent.
    No prior, no denoiser.
    """
    H, H_adj = degradation.H, degradation.H_adj

    with torch.no_grad():
        for _ in range(int(steps)):
            r = H(x) - y
            grad = 2.0 * H_adj(r)
            x = x - float(image_lr) * grad

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    "Non-finite x in data-only update. Reduce image_lr."
                )
    return x


# ============================================================
# Gated PnP-Flow image update
# ============================================================
def _pnp_flow_update_gated(
    pnp: PNP_FLOW,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    degradation: _BlindDegradation,
    steps: int,
    lam: float,
) -> torch.Tensor:
    """
    Gated PnP update.

    lam = 0 → pure data step
    lam = 1 → pure PnP step
    """
    H, H_adj = degradation.H, degradation.H_adj
    pnp.args.sigma_noise = 1.0  # neutralise internal scaling

    delta = 1.0 / steps
    lr = float(pnp.args.lr_pnp)
    lam = float(max(0.0, min(1.0, lam)))

    with torch.no_grad():
        for i in range(int(steps)):
            t = torch.ones(len(x), device=x.device) * delta * i
            lr_t = pnp.learning_rate_strat(lr, t)

            grad = pnp.grad_datafit(x, y, H, H_adj)
            z = x - lr_t * grad

            if not torch.isfinite(z).all():
                raise FloatingPointError(
                    "Non-finite z in PnP update. Reduce pnp_lr_pnp."
                )

            x_pnp = torch.zeros_like(x)
            for _ in range(pnp.args.num_samples):
                z_tilde = pnp.interpolation_step(z, t.view(-1, 1, 1, 1))
                x_pnp += pnp.denoiser(z_tilde, t)
            x_pnp /= pnp.args.num_samples

            x = (1.0 - lam) * z + lam * x_pnp

            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    "Non-finite x after PnP gating. Reduce lam or lr."
                )

    return x


# ============================================================
# Step 5 — Identifiability-controlled blind PnP-Flow (fixed)
# ============================================================
def blind_pnp_flow_step5(
    prob: BlindGaussianBlurProblem,
    *,
    model,
    device,
    sigma_init: float,
    sigma_max: float,
    # operator updates
    operator_lr: float = 1e-2,
    sigma_update_every: int = 1,
    sigma_freeze_outer_iters: int = 10,    # ✅ KEY FIX: freeze sigma early
    # image updates
    outer_iters: int = 60,
    image_updates_per_outer: int = 10,     # ✅ many x updates per outer
    # data-only warmup / anchoring
    warmup_data_steps: int = 20,           # ✅ KEY FIX: do NOT overfit warmup
    data_only_between: int = 0,
    image_lr_data_only: float = 1e-1,
    # PnP
    pnp_lr_pnp: float = 0.05,
    pnp_steps_inner: int = 25,
    # prior gating
    lam_max: float = 0.8,
    lam_ramp_iters: int = 20,              # ✅ ramp faster so x leaves y
    # safety
    clip_sigma_grad_norm: float = 1.0,
    # debugging
    print_every: int = 5,
) -> Dict:
    """
    Fixed Step 5:
    - Initialise x away from y to avoid identity basin.
    - Do a short data-only warmup (not enough to make loss ~ 0).
    - Ramp prior (lam) while sigma is frozen, to move x into a prior-consistent region.
    - Only then allow sigma updates from data term.
    """

    # --- PnP setup ---
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

    # --- Learnable operator ---
    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    opt = torch.optim.Adam(op.parameters(), lr=operator_lr)

    # --- Data ---
    y = prob.y.to(device)

    # ✅ KEY FIX: sigma-neutral init (prevents "make H identity" trap)
    x = torch.randn_like(y)

    sigma_hist, loss_hist, grad_hist, lam_hist, x_y_hist = [], [], [], [], []

    degradation = _BlindDegradation(op)

    # --- Short data-only warmup (do NOT drive loss to ~0) ---
    if warmup_data_steps > 0:
        x = _data_only_update(
            x=x,
            y=y,
            degradation=degradation,
            steps=warmup_data_steps,
            image_lr=image_lr_data_only,
        ).detach()

    for outer in range(int(outer_iters)):
        degradation = _BlindDegradation(op)

        lam = float(lam_max) * float(min(1.0, outer / max(1, int(lam_ramp_iters))))
        lam_hist.append(lam)

        # multiple image updates per outer
        for _ in range(int(image_updates_per_outer)):
            if data_only_between > 0:
                x = _data_only_update(
                    x=x,
                    y=y,
                    degradation=degradation,
                    steps=data_only_between,
                    image_lr=image_lr_data_only,
                ).detach()

            x = _pnp_flow_update_gated(
                pnp,
                x=x,
                y=y,
                degradation=degradation,
                steps=pnp_steps_inner,
                lam=lam,
            ).detach()

        # sigma update: only after freeze period
        """do_sigma = (
            outer >= int(sigma_freeze_outer_iters)
            and (outer % max(1, int(sigma_update_every)) == 0)
        )"""
        
        # Identity-basin guard: do NOT update sigma if x ≈ y
        with torch.no_grad():
            x_y_dist = torch.mean(torch.abs(x - y)).item()

        do_sigma = (
            outer >= int(sigma_freeze_outer_iters)
            and (outer % max(1, int(sigma_update_every)) == 0)
            and (x_y_dist > 1e-3)   # <<< CRITICAL
        )


        if do_sigma:
            opt.zero_grad(set_to_none=True)
            loss_op = torch.mean((op(x.detach()) - y) ** 2)
            # loss_op.backward()
            sigma_grad_scale = 0.1   # conservative; try 0.01 if needed
            (loss_op * sigma_grad_scale).backward()

 
            g = op.log_sigma.grad
            if g is None or not torch.isfinite(g).all():
                raise RuntimeError("Invalid log_sigma gradient")

            if clip_sigma_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    op.parameters(), max_norm=float(clip_sigma_grad_norm)
                )

            opt.step()
            op.clamp_params_()
            grad_mag = float(g.detach().abs().mean())
        else:
            grad_mag = float("nan")

        with torch.no_grad():
            loss_x = torch.mean((op(x) - y) ** 2).item()
            x_y = torch.mean(torch.abs(x - y)).item()

        sigma_hist.append(op.sigma().item())
        loss_hist.append(loss_x)
        grad_hist.append(grad_mag)
        x_y_hist.append(x_y)

        if (outer % int(print_every) == 0) or (outer == outer_iters - 1):
            msg = (
                f"[Step5 {outer:03d}/{outer_iters}] "
                f"lam={lam:.3f} "
                f"sigma={op.sigma().item():.4f} "
                f"loss={loss_x:.4e} "
                f"|x-y|={x_y:.3e}"
            )
            if do_sigma:
                msg += f" |gradσ|={grad_mag:.3e}"
            else:
                msg += " |gradσ|=FROZEN"
            print(msg)

    return dict(
        x=x,
        sigma_final=op.sigma().item(),
        sigma_history=sigma_hist,
        loss_history=loss_hist,
        grad_sigma_history=grad_hist,
        lam_history=lam_hist,
        x_minus_y_history=x_y_hist,
    )


# ============================================================
# Optional: load a real image for in-distribution prior behaviour
# ============================================================
def _load_image_as_tensor(path: str, *, size: int, device) -> torch.Tensor:
    """
    Loads an RGB image from disk and returns (1,3,H,W) float tensor in [0,1].
    Uses torchvision/PIL if available.
    """
    try:
        from PIL import Image
        from torchvision import transforms
    except Exception as e:
        raise RuntimeError(
            "To load real images, install pillow + torchvision, "
            "or remove IMAGE_PATH to use synthetic random x_gt."
        ) from e

    img = Image.open(path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),  # [0,1], (C,H,W)
    ])
    x = tfm(img).unsqueeze(0).to(device=device)
    return x


# ============================================================
# __main__ — Step 5 run
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

    # Optional: set an environment variable IMAGE_PATH=/path/to/face.jpg
    image_path = os.environ.get("IMAGE_PATH", "").strip()

    if image_path:
        x_gt = _load_image_as_tensor(image_path, size=cfg.dim_image, device=device)
        print(f"Loaded x_gt from: {image_path}")
    else:
        # Still runs, but the flow prior is not a "prior" for Gaussian noise.
        x_gt = torch.randn(1, 3, cfg.dim_image, cfg.dim_image, device=device)
        print("WARNING: Using random x_gt. Flow prior is out-of-distribution; sigma behaviour may be misleading.")
        print("Set IMAGE_PATH=/path/to/image.jpg for an in-distribution test.")

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=1.2,
        sigma_noise=0.05,
        sigma_max=6.0,
        padding="reflect",
        seed=0,
    )

    out = blind_pnp_flow_step5(
        prob,
        model=model,
        device=device,
        sigma_init=0.4,
        sigma_max=6.0,
        operator_lr=1e-2,
        sigma_update_every=1,
        sigma_freeze_outer_iters=10,  
        outer_iters=40,
        image_updates_per_outer=10,    
        warmup_data_steps=20,          
        lam_max=0.8,
        lam_ramp_iters=20,
        print_every=5,
    )

    print(f"\nFinal sigma: {out['sigma_final']:.4f} (true: 1.2)")
