"""Classical iterative blind deconvolution baseline based on Krishnan et al.

Adapts Krishnan, Tay, and Fergus (CVPR 2011) to the parametric Gaussian
blur setting tested in this thesis. The original method jointly estimates
a free 2D kernel and an image under a normalised L1 over L2 sparsity prior
on image gradients. This implementation makes two simplifications, namely
(1) the kernel is parameterised as an isotropic Gaussian with unknown
sigma rather than a free 2D kernel, and (2) the prior on image gradients
is plain L1 rather than the non convex normalised L1 over L2.

The optimisation alternates between the following two steps until
convergence.

The outer loop is alternating minimisation between two convex sub
problems. The inner image update is solved by ADMM. The outer loop
itself is not ADMM, since ADMM is a convex optimisation solver and the
joint objective in (x, sigma) is non convex.

    1. x update (image, with sigma fixed).
            argmin_x  (1/2) || H_sigma x - y ||^2  +  lambda ( ||D_h x||_1 + ||D_v x||_1 )
       Solved by ADMM in the FFT domain, with FFT diagonal solves for the
       linear system and elementwise soft thresholding for the gradient
       auxiliaries.

    2. sigma update (parametric kernel, with x fixed).
            argmin_sigma  || H_sigma x - y ||^2
       Solved in two stages, namely a coarse 1D grid search to identify
       the basin of attraction (the objective is non convex in sigma when
       x is held fixed at an arbitrary point), followed by gradient
       descent on log sigma for refinement. The FFT Gaussian operator is
       differentiable in sigma through PyTorch autograd, so refinement
       requires no manual gradient derivation.

A continuation scheme on lambda starts at lambda_init and decreases
geometrically to lambda_final across outer iterations.

For each image the script reports two reconstructions, namely the
classical reconstruction produced by Krishnan style optimisation, and a
PnP-Flow reconstruction performed using Krishnan's estimated sigma. The
latter isolates whether the failure mode is in sigma estimation or in the
classical prior.

Outputs are written to results/blind/krishnan_baseline/<dataset>/sigma_<sigma>/.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.utils import save_image

from .._helpers import (
    SEED,
    TimingTracker,
    find_repo_root,
    load_model,
    save_json,
)
from ..data import load_dataset
from ..evaluation import evaluate, postprocess
from ..forward import gaussian_blur_fft
from ..reconstruction import pnp_flow_reconstruct


# ---------------------------------------------------------------------------
# Krishnan style optimisation primitives
# ---------------------------------------------------------------------------

def _soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0.0)


def _admm_l1_gradient(
    y: torch.Tensor,
    sigma: float,
    lam: float,
    rho: float = 1.0,
    n_iters: int = 30,
) -> torch.Tensor:
    """ADMM solver for the L1 image gradient prior with isotropic Gaussian
    blur. All operators are diagonal in the Fourier basis.

    minimise (1/2) || H_sigma x - y ||^2 + lambda ( ||D_h x||_1 + ||D_v x||_1 )
    """
    device = y.device
    squeeze = (y.ndim == 3)
    if squeeze:
        y = y.unsqueeze(0)
    _, _, H, W = y.shape

    fy = torch.fft.fftfreq(H, device=device, dtype=torch.float32)
    fx = torch.fft.fftfreq(W, device=device, dtype=torch.float32)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    freq_sq = (FY ** 2 + FX ** 2) * (2.0 * torch.pi) ** 2

    H_freq = torch.exp(-0.5 * sigma ** 2 * freq_sq).to(torch.complex64)
    H_sq_real = (H_freq.abs() ** 2).real

    Dh = (1.0 - torch.exp(-1j * 2.0 * torch.pi * FX)).to(torch.complex64)
    Dv = (1.0 - torch.exp(-1j * 2.0 * torch.pi * FY)).to(torch.complex64)
    Dh_sq_real = (Dh.abs() ** 2).real
    Dv_sq_real = (Dv.abs() ** 2).real

    Y = torch.fft.fft2(y.float())
    Hy_freq = H_freq * Y

    x = y.clone().float()
    z_h = torch.zeros_like(x)
    z_v = torch.zeros_like(x)
    u_h = torch.zeros_like(x)
    u_v = torch.zeros_like(x)

    denom = (H_sq_real + rho * (Dh_sq_real + Dv_sq_real)).unsqueeze(0).unsqueeze(0) + 1e-12
    Dh_b = Dh.unsqueeze(0).unsqueeze(0)
    Dv_b = Dv.unsqueeze(0).unsqueeze(0)

    for _ in range(n_iters):
        Zh_freq = torch.fft.fft2(z_h - u_h)
        Zv_freq = torch.fft.fft2(z_v - u_v)
        rhs = Hy_freq + rho * (Dh_b.conj() * Zh_freq + Dv_b.conj() * Zv_freq)
        X_freq = rhs / denom
        x = torch.fft.ifft2(X_freq).real

        Dh_x = torch.fft.ifft2(Dh_b * X_freq).real
        Dv_x = torch.fft.ifft2(Dv_b * X_freq).real

        z_h = _soft_threshold(Dh_x + u_h, lam / rho)
        z_v = _soft_threshold(Dv_x + u_v, lam / rho)

        u_h = u_h + Dh_x - z_h
        u_v = u_v + Dv_x - z_v

    return x.squeeze(0) if squeeze else x


def _grid_init_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_min: float,
    sigma_max: float,
    n_grid: int = 30,
) -> float:
    """Coarse grid evaluation of the data fit residual, used to bracket
    the gradient refinement. Selects the grid point with smallest
    residual, no parabolic fit (the gradient step does the refinement).
    """
    grid = torch.linspace(sigma_min, sigma_max, n_grid)
    losses: List[float] = []
    for s in grid:
        with torch.no_grad():
            losses.append(float(torch.mean((gaussian_blur_fft(x, float(s)) - y) ** 2).item()))
    j = int(np.argmin(losses))
    return float(grid[j])


def _grad_step_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_init: float,
    sigma_min: float = 0.1,
    sigma_max: float = 8.0,
    n_steps: int = 50,
    lr: float = 0.01,
) -> Tuple[float, float]:
    """Gradient descent on log sigma to refine the sigma estimate.

    Refinement only, expects the caller to have already bracketed the
    correct basin via grid search. The FFT Gaussian operator is
    differentiable in sigma through PyTorch autograd.

    Returns the refined sigma and the magnitude of the final gradient,
    which serves as a convergence diagnostic.
    """
    x = x.detach()
    y = y.detach()
    log_sigma = torch.tensor(
        float(np.log(max(sigma_init, sigma_min))),
        device=x.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    log_sigma_min = float(np.log(sigma_min))
    log_sigma_max = float(np.log(sigma_max))
    opt = torch.optim.Adam([log_sigma], lr=lr)
    final_grad_mag = 0.0
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        sigma_t = torch.exp(log_sigma)
        loss = torch.mean((gaussian_blur_fft(x, sigma_t) - y) ** 2)
        loss.backward()
        if log_sigma.grad is not None:
            final_grad_mag = float(log_sigma.grad.abs().item())
        opt.step()
        with torch.no_grad():
            log_sigma.clamp_(log_sigma_min, log_sigma_max)
    return float(torch.exp(log_sigma.detach()).item()), final_grad_mag


def _update_sigma(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma_min: float = 0.1,
    sigma_max: float = 8.0,
    n_grid: int = 30,
    n_grad_steps: int = 50,
    grad_lr: float = 0.01,
) -> Tuple[float, float]:
    """Combined sigma update: coarse grid init then gradient refinement."""
    sigma_init = _grid_init_sigma(x, y, sigma_min, sigma_max, n_grid)
    return _grad_step_sigma(
        x, y, sigma_init,
        sigma_min=sigma_min, sigma_max=sigma_max,
        n_steps=n_grad_steps, lr=grad_lr,
    )


def krishnan_blind_deconv(
    y: torch.Tensor,
    sigma_init: float = 3.0,
    sigma_min: float = 0.1,
    sigma_max: float = 8.0,
    outer_iters: int = 5,
    inner_iters: int = 30,
    lambda_init: float = 0.05,
    lambda_final: float = 0.001,
    rho: float = 1.0,
    sigma_grad_steps: int = 50,
    sigma_grad_lr: float = 0.01,
    x_clean: torch.Tensor | None = None,
    sigma_true: float | None = None,
) -> Tuple[torch.Tensor, float, Dict[str, List[float]]]:
    """Run the alternating Krishnan style blind deconvolution.

    Returns the classical reconstruction, the estimated sigma, and a
    dictionary of per outer iteration convergence diagnostics:

        sigma_history          : sigma value after each outer iteration
        objective_history      : || H_sigma(x) - y ||^2 after each outer iteration
        sigma_grad_norm_history: |grad sigma| at the end of the sigma update
        sigma_error_history    : |sigma_est - sigma_true| (if sigma_true given)
        x_l2_error_history     : ||x - x_clean||_2 (if x_clean given)
        psnr_history           : PSNR(x, x_clean) (if x_clean given)
    """
    sigma_current = sigma_init
    x = y.clone().float()
    history: Dict[str, List[float]] = {
        "sigma_history": [sigma_current],
        "objective_history": [],
        "sigma_grad_norm_history": [],
        "sigma_error_history": [],
        "x_l2_error_history": [],
        "psnr_history": [],
    }

    for outer in range(outer_iters):
        if outer_iters > 1:
            lam = lambda_init * (lambda_final / lambda_init) ** (outer / (outer_iters - 1))
        else:
            lam = lambda_final

        x = _admm_l1_gradient(y, sigma_current, lam, rho, n_iters=inner_iters)
        sigma_current, grad_mag = _update_sigma(
            x, y,
            sigma_min=sigma_min, sigma_max=sigma_max,
            n_grad_steps=sigma_grad_steps, grad_lr=sigma_grad_lr,
        )

        # Convergence metrics for this outer iteration.
        with torch.no_grad():
            obj = float(torch.mean((gaussian_blur_fft(x, sigma_current) - y) ** 2).item())
        history["sigma_history"].append(sigma_current)
        history["objective_history"].append(obj)
        history["sigma_grad_norm_history"].append(grad_mag)
        if sigma_true is not None:
            history["sigma_error_history"].append(abs(sigma_current - sigma_true))
        if x_clean is not None:
            with torch.no_grad():
                x_clamped = x.clamp(-1, 1)
                l2 = float(torch.linalg.vector_norm(x_clamped - x_clean).item())
                psnr_postproc = ((x_clamped + 1) / 2 - (x_clean + 1) / 2).clamp(-1, 1)
                mse = float(torch.mean(psnr_postproc ** 2).item())
                psnr = float("inf") if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)
            history["x_l2_error_history"].append(l2)
            history["psnr_history"].append(psnr)

    return x, sigma_current, history


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run(
    dataset: str,
    sigma_true: float,
    noise_std: float,
    num_images: int,
    output_dir: str,
    device: str,
    use_pnpflow_for_recon: bool,
    pnp_steps: int,
    outer_iters: int,
    inner_iters: int,
):
    repo_root = find_repo_root()
    print(f"Loading {dataset}")
    images = load_dataset(dataset, repo_root, device, num_celeba=num_images)
    images = images[:num_images]
    print(f"Using {len(images)} images")

    model = None
    if use_pnpflow_for_recon:
        print("Loading PnP-Flow model for cross check")
        model = load_model(repo_root, device)

    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)

    tt = TimingTracker()
    records: List[Dict] = []

    for idx, x_gt in enumerate(images):
        torch.manual_seed(SEED + idx)
        y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std

        with tt.track("krishnan_full"):
            x_classical, sigma_est, history = krishnan_blind_deconv(
                y,
                outer_iters=outer_iters,
                inner_iters=inner_iters,
                x_clean=x_gt,
                sigma_true=sigma_true,
            )
        x_classical = x_classical.clamp(-1, 1)
        m_classical = evaluate(x_classical, x_gt)

        m_pnp = None
        x_pnp = None
        if use_pnpflow_for_recon:
            torch.manual_seed(SEED + idx * 100)
            with tt.track("pnp_with_krishnan_sigma"):
                x_pnp = pnp_flow_reconstruct(
                    model, y=y, sigma_blur=sigma_est, num_steps=pnp_steps,
                )
            m_pnp = evaluate(x_pnp, x_gt)

        # Oracle for headline gap reporting
        if model is not None:
            torch.manual_seed(SEED + idx * 100)
            with tt.track("oracle"):
                x_oracle = pnp_flow_reconstruct(
                    model, y=y, sigma_blur=sigma_true, num_steps=pnp_steps,
                )
            m_oracle = evaluate(x_oracle, x_gt)
        else:
            m_oracle = None

        # Per dataset cap on qualitative count: 30 for CelebA, 18 for BSD68,
        # 12 for Set12 (full set since it only has 12 images).
        if dataset == "CelebA":
            qual_limit = 30
        elif dataset == "BSD68":
            qual_limit = 18
        else:
            qual_limit = 12
        if idx < qual_limit:
            save_image(postprocess(x_gt), os.path.join(qual_dir, f"{idx:04d}_clean.png"))
            save_image(postprocess(y), os.path.join(qual_dir, f"{idx:04d}_observed.png"))
            save_image(postprocess(x_classical), os.path.join(qual_dir, f"{idx:04d}_krishnan.png"))
            if x_pnp is not None:
                save_image(postprocess(x_pnp), os.path.join(qual_dir, f"{idx:04d}_pnp_with_krishnan_sigma.png"))

        record = {
            "index": idx,
            "sigma_est": sigma_est,
            "sigma_error": abs(sigma_est - sigma_true),
            "convergence": history,
            "psnr_classical": m_classical["psnr"],
            "ssim_classical": m_classical["ssim"],
            "lpips_classical": m_classical["lpips"],
        }
        if m_pnp is not None:
            record.update({
                "psnr_pnp_with_krishnan_sigma": m_pnp["psnr"],
                "ssim_pnp_with_krishnan_sigma": m_pnp["ssim"],
                "lpips_pnp_with_krishnan_sigma": m_pnp["lpips"],
            })
        if m_oracle is not None:
            record.update({
                "psnr_oracle": m_oracle["psnr"],
                "ssim_oracle": m_oracle["ssim"],
                "lpips_oracle": m_oracle["lpips"],
                "psnr_gap_classical": m_oracle["psnr"] - m_classical["psnr"],
            })
            if m_pnp is not None:
                record["psnr_gap_pnp_with_krishnan_sigma"] = m_oracle["psnr"] - m_pnp["psnr"]
        records.append(record)

        msg = (
            f"[{idx+1:>3}/{len(images)}] sigma_est={sigma_est:.3f} "
            f"err={record['sigma_error']:.3f} "
            f"PSNR_classical={m_classical['psnr']:.2f}"
        )
        if m_pnp is not None:
            msg += f" PSNR_pnp(krishnan_sigma)={m_pnp['psnr']:.2f}"
        if m_oracle is not None:
            msg += f" PSNR_oracle={m_oracle['psnr']:.2f}"
        print(msg)

    summary = {
        "approach": "krishnan_baseline",
        "dataset": dataset,
        "sigma_true": sigma_true,
        "noise_std": noise_std,
        "num_images": len(records),
        "outer_iters": outer_iters,
        "inner_iters": inner_iters,
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "sigma_error_std": float(np.std([r["sigma_error"] for r in records])),
        "psnr_classical_mean": float(np.mean([r["psnr_classical"] for r in records])),
        "ssim_classical_mean": float(np.mean([r["ssim_classical"] for r in records])),
        "lpips_classical_mean": float(np.mean([r["lpips_classical"] for r in records])),
    }
    if records and "psnr_pnp_with_krishnan_sigma" in records[0]:
        summary.update({
            "psnr_pnp_mean": float(np.mean([r["psnr_pnp_with_krishnan_sigma"] for r in records])),
            "ssim_pnp_mean": float(np.mean([r["ssim_pnp_with_krishnan_sigma"] for r in records])),
            "lpips_pnp_mean": float(np.mean([r["lpips_pnp_with_krishnan_sigma"] for r in records])),
        })
    if records and "psnr_oracle" in records[0]:
        summary.update({
            "psnr_oracle_mean": float(np.mean([r["psnr_oracle"] for r in records])),
            "psnr_gap_classical_mean": float(np.mean([r["psnr_gap_classical"] for r in records])),
        })
        if "psnr_gap_pnp_with_krishnan_sigma" in records[0]:
            summary["psnr_gap_pnp_mean"] = float(
                np.mean([r["psnr_gap_pnp_with_krishnan_sigma"] for r in records])
            )

    save_json(os.path.join(output_dir, "per_image.json"), records)
    save_json(os.path.join(output_dir, "summary.json"), summary)

    print()
    print(f"Wrote results to {output_dir}")
    print(f"sigma error mean: {summary['sigma_error_mean']:.4f}")
    print(f"PSNR classical:   {summary['psnr_classical_mean']:.2f}")
    if "psnr_pnp_mean" in summary:
        print(f"PSNR PnP (krishnan_sigma): {summary['psnr_pnp_mean']:.2f}")
    if "psnr_oracle_mean" in summary:
        print(f"PSNR oracle:      {summary['psnr_oracle_mean']:.2f}")
        print(f"PSNR gap classical:        {summary['psnr_gap_classical_mean']:+.2f}")
        if "psnr_gap_pnp_mean" in summary:
            print(f"PSNR gap PnP(krishnan):    {summary['psnr_gap_pnp_mean']:+.2f}")
    print(tt.summary())
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Krishnan inspired classical blind deconvolution baseline"
    )
    p.add_argument("--dataset", choices=["CelebA", "BSD68", "Set12"], default="CelebA")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--num-images", type=int, default=50)
    p.add_argument("--outer-iters", type=int, default=5)
    p.add_argument("--inner-iters", type=int, default=30)
    p.add_argument("--pnp-steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-pnpflow", action="store_true",
                   help="Skip PnP-Flow reconstruction with Krishnan's sigma estimate")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        find_repo_root(), "results", "blind", "krishnan_baseline",
        args.dataset, f"sigma_{args.sigma:.2f}",
    )
    os.makedirs(output_dir, exist_ok=True)
    run(
        dataset=args.dataset,
        sigma_true=args.sigma,
        noise_std=args.noise_std,
        num_images=args.num_images,
        output_dir=output_dir,
        device=args.device,
        use_pnpflow_for_recon=not args.no_pnpflow,
        pnp_steps=args.pnp_steps,
        outer_iters=args.outer_iters,
        inner_iters=args.inner_iters,
    )


if __name__ == "__main__":
    main()
