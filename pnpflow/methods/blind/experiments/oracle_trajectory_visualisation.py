"""Composite oracle trajectory visualisation for the non-blind oracle section.

For each of CelebA, BSD68, and Set12, runs a single PnP-Flow oracle
reconstruction trajectory and saves the clean image, the degraded
observation, and the iterate at six points along the trajectory
(t = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0) as individual PNG files. Per-cell
PSNR values are written to a separate `psnrs.txt` for inclusion in the
figure caption.

All six trajectory snapshots in a dataset come from a single PnP-Flow
run, not six independent runs. Reconstructions use the oracle blur
parameter sigma = 1.5 and supply the true sigma directly to the
data-fidelity step (no Blur-SURE).

Configuration: sigma = 1.5, nu = 0.05, num_steps = 100, lr = 1.0,
gamma = 1 - t, cold start from x_0 = y, num_samples = 1, outer = 1.

Deterministic seeds (matches the experimental setup):
- observation generation seed for image index i: SEED + i
- oracle reconstruction seed for image index i:  SEED + 100 * i

Usage
-----
    python -m pnpflow.methods.blind.experiments.oracle_trajectory_visualisation
    python -m pnpflow.methods.blind.experiments.oracle_trajectory_visualisation \
        --celeba-idx 4 --bsd68-idx 10 --set12-idx 2

Outputs
-------
results/blind/oracle_trajectory_visualisation/
    celeba_clean.png     celeba_degraded.png
    celeba_t_00.png      celeba_t_02.png      celeba_t_04.png
    celeba_t_06.png      celeba_t_08.png      celeba_t_10.png
    bsd68_clean.png      ...
    set12_clean.png      ...
    psnrs.txt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torchvision.utils import save_image

from .._helpers import (
    SEED,
    find_repo_root,
    load_model,
    pnp_flow_trajectory_tracked,
)
from ..data import load_bsd68, load_celeba, load_set12
from ..evaluation import postprocess
from ..forward import gaussian_blur_fft


SNAPSHOT_INDICES = [0, 20, 40, 60, 80, 100]
SNAPSHOT_T_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _psnr_db(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [-1, 1], post-processed to [0, 1]."""
    a = postprocess(x_hat.detach().cpu())
    b = postprocess(x_gt.detach().cpu())
    mse = torch.mean((a - b) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _load_image(dataset: str, repo_root: Path, device: str, idx: int) -> torch.Tensor:
    if dataset == "CelebA":
        imgs = load_celeba(str(repo_root), device, num_images=idx + 1, seed=SEED)
    elif dataset == "BSD68":
        imgs = load_bsd68(str(repo_root), device, img_size=128)
    elif dataset == "Set12":
        imgs = load_set12(str(repo_root), device, img_size=128)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")
    if idx >= len(imgs):
        raise IndexError(f"{dataset} index {idx} out of range (size {len(imgs)})")
    return imgs[idx]


def _compute_row(
    model,
    x_gt: torch.Tensor,
    image_idx: int,
    sigma: float,
    noise: float,
    num_steps: int,
):
    """Run one oracle trajectory and return (degraded, snapshots, psnrs)."""
    device = x_gt.device

    torch.manual_seed(SEED + image_idx)
    y = gaussian_blur_fft(x_gt, sigma) + torch.randn_like(x_gt) * noise

    torch.manual_seed(SEED + image_idx * 100)
    x_final, trajectory, t_values = pnp_flow_trajectory_tracked(
        model,
        x_init=y.clone(),
        y=y,
        sigma_blur=sigma,
        num_steps=num_steps,
    )

    snapshots = [trajectory[k].to(device) for k in SNAPSHOT_INDICES]
    psnr_degraded = _psnr_db(y, x_gt)
    psnr_snaps = [_psnr_db(s, x_gt) for s in snapshots]

    return y, snapshots, psnr_degraded, psnr_snaps


def run(
    celeba_idx: int = 2,
    bsd68_idx: int = 0,
    set12_idx: int = 2,
    sigma: float = 1.5,
    noise: float = 0.05,
    num_steps: int = 100,
    output_dir: Path | None = None,
    device: str | None = None,
):
    repo_root = Path(find_repo_root())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = (
            repo_root / "results" / "blind" / "oracle_trajectory_visualisation"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(
        f"Configuration:    sigma={sigma}, nu={noise}, num_steps={num_steps}, "
        f"lr=1.0, gamma=1-t, cold start"
    )
    print(f"Loading model on {device}")
    model = load_model(repo_root, device)

    dataset_specs = [
        ("CelebA", celeba_idx),
        ("BSD68", bsd68_idx),
        ("Set12", set12_idx),
    ]

    summary_lines = [
        "Non-blind oracle trajectory visualisation",
        f"Configuration: sigma = {sigma}, nu = {noise}, num_steps = {num_steps}, "
        f"lr = 1.0, gamma = 1 - t, cold start",
        "",
    ]

    for dataset, idx in dataset_specs:
        print(f"\nProcessing {dataset}[index = {idx}] ...")
        x_gt = _load_image(dataset, repo_root, device, idx)
        y, snapshots, psnr_deg, psnr_snaps = _compute_row(
            model,
            x_gt,
            image_idx=idx,
            sigma=sigma,
            noise=noise,
            num_steps=num_steps,
        )

        prefix = dataset.lower()
        save_image(postprocess(x_gt), output_dir / f"{prefix}_clean.png")
        save_image(postprocess(y), output_dir / f"{prefix}_degraded.png")
        for t_val, snap in zip(SNAPSHOT_T_VALUES, snapshots):
            fname = f"{prefix}_t_{int(round(t_val * 10)):02d}.png"
            save_image(postprocess(snap), output_dir / fname)

        obs_seed = SEED + idx
        recon_seed = SEED + idx * 100
        summary_lines.append(
            f"{dataset} (index {idx}, obs seed {obs_seed}, recon seed {recon_seed}):"
        )
        summary_lines.append(f"  degraded:      PSNR = {psnr_deg:.2f} dB")
        for t_val, psnr in zip(SNAPSHOT_T_VALUES, psnr_snaps):
            summary_lines.append(f"  t = {t_val:.1f}:       PSNR = {psnr:.2f} dB")
        summary_lines.append("")

        print(f"  observation seed   = {obs_seed}")
        print(f"  reconstruction seed = {recon_seed}")
        print(f"  PSNR degraded      = {psnr_deg:.2f} dB")
        for t_val, psnr in zip(SNAPSHOT_T_VALUES, psnr_snaps):
            print(f"  PSNR t = {t_val:.1f}      = {psnr:.2f} dB")

    psnrs_path = output_dir / "psnrs.txt"
    psnrs_path.write_text("\n".join(summary_lines))

    print()
    print(f"Wrote {3 * 8} PNGs and psnrs.txt to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate composite oracle trajectory visualisation "
        "(3 datasets x 8 panels per row, saved as individual PNGs)"
    )
    p.add_argument(
        "--celeba-idx",
        type=int,
        default=2,
        help="CelebA index within the deterministic-seed sample (default 2)",
    )
    p.add_argument(
        "--bsd68-idx",
        type=int,
        default=0,
        help="BSD68 image index in sorted-filename order (default 0)",
    )
    p.add_argument(
        "--set12-idx",
        type=int,
        default=2,
        help="Set12 image index in sorted-filename order (default 2, typically Cameraman)",
    )
    p.add_argument("--sigma", type=float, default=1.5, help="True blur sigma (oracle)")
    p.add_argument("--noise-std", type=float, default=0.05, help="Additive noise std")
    p.add_argument(
        "--num-steps", type=int, default=100, help="PnP-Flow trajectory steps"
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Override output directory"
    )
    p.add_argument("--device", default=None, help="Override device (cuda or cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        celeba_idx=args.celeba_idx,
        bsd68_idx=args.bsd68_idx,
        set12_idx=args.set12_idx,
        sigma=args.sigma,
        noise=args.noise_std,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
