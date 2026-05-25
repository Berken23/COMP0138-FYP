"""Generate a multi-image non-blind-oracle trajectory figure.

For each of N images from a chosen dataset (CelebA, BSD68, or Set12),
renders the PnP-Flow reconstruction trajectory at t in {0.0, 0.2, 0.4,
0.6, 0.8, 1.0} when supplied with the true blur parameter (i.e. the
non-blind oracle setting). Each row of the resulting figure consists of:
clean | degraded | t=0.0 | t=0.2 | ... | t=1.0, illustrating the
progression of deblurring under standard PnP-Flow at known sigma.

Usage
-----
    python -m pnpflow.methods.blind.experiments.oracle_trajectory_grid
    python -m pnpflow.methods.blind.experiments.oracle_trajectory_grid --dataset BSD68
    python -m pnpflow.methods.blind.experiments.oracle_trajectory_grid --dataset Set12 --num 5

Outputs
-------
results/blind/figures/oracle_trajectory_grid/<dataset>/
    img00_clean.png       ... ground-truth reference
    img00_degraded.png    ... blurred and noisy observation y
    img00_t_00.png        ... iterate at t = 0.0
    img00_t_02.png        ... iterate at t = 0.2
    ...
    img00_t_10.png        ... iterate at t = 1.0
    img01_*.png           ... same set for image 1
    ...
    psnrs.txt             ... PSNR for every saved panel
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


def _psnr_db(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [-1, 1]."""
    mse = torch.mean(
        (postprocess(x_hat) - postprocess(x_gt.to(x_hat.device))) ** 2
    ).item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _load_dataset_images(
    dataset: str, repo_root: Path, device: str, num_images: int, img_size: int = 128
):
    if dataset == "CelebA":
        return load_celeba(str(repo_root), device, num_images=num_images, seed=SEED)
    if dataset == "BSD68":
        return load_bsd68(str(repo_root), device, img_size=img_size)[:num_images]
    if dataset == "Set12":
        return load_set12(str(repo_root), device, img_size=img_size)[:num_images]
    raise ValueError(f"Unknown dataset '{dataset}'. Choose CelebA, BSD68, or Set12.")


def run(
    dataset: str = "CelebA",
    num_images: int = 5,
    sigma: float = 1.5,
    noise: float = 0.05,
    snapshot_t: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    num_steps: int = 100,
    output_dir: Path | None = None,
    device: str | None = None,
):
    repo_root = Path(find_repo_root())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = (
            repo_root
            / "results"
            / "blind"
            / "figures"
            / "oracle_trajectory_grid"
            / dataset.lower()
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(
        f"Configuration:    {dataset}, sigma={sigma}, nu={noise}, num_steps={num_steps}"
    )
    print(f"Loading model on {device}")
    model = load_model(repo_root, device)

    print(f"Loading {num_images} {dataset} images")
    images = _load_dataset_images(dataset, repo_root, device, num_images)
    if len(images) < num_images:
        print(
            f"  warning: requested {num_images} images but {dataset} only provided "
            f"{len(images)}"
        )

    summary_lines = [
        "Non-blind oracle trajectory grid",
        f"Configuration: {dataset}, sigma = {sigma}, nu = {noise}, num_steps = {num_steps}",
        f"Snapshots at t = {list(snapshot_t)}",
        "",
    ]

    for image_idx in range(len(images)):
        x_gt = images[image_idx].to(device)

        # Generate the degraded observation with the same seeding convention
        # used throughout the chapter.
        torch.manual_seed(SEED + image_idx)
        y = gaussian_blur_fft(x_gt, sigma) + torch.randn_like(x_gt) * noise

        # Run the trajectory-tracked reconstruction at the oracle sigma.
        torch.manual_seed(SEED + image_idx * 100)
        x_final, trajectory, t_values = pnp_flow_trajectory_tracked(
            model,
            x_init=y.clone(),
            y=y,
            sigma_blur=sigma,
            num_steps=num_steps,
        )

        prefix = f"img{image_idx:02d}"
        save_image(postprocess(x_gt), output_dir / f"{prefix}_clean.png")
        save_image(postprocess(y), output_dir / f"{prefix}_degraded.png")
        psnr_deg = _psnr_db(y, x_gt)

        summary_lines.append(f"Image {image_idx}:")
        summary_lines.append(f"  degraded:           PSNR = {psnr_deg:.2f} dB")

        for t_target in snapshot_t:
            idx = min(range(len(t_values)), key=lambda i: abs(t_values[i] - t_target))
            x_snap = trajectory[idx].to(device)
            fname = f"{prefix}_t_{int(round(t_target * 10)):02d}.png"
            save_image(postprocess(x_snap), output_dir / fname)
            psnr_snap = _psnr_db(x_snap, x_gt)
            summary_lines.append(
                f"  t = {t_target:.1f} ({fname}):  PSNR = {psnr_snap:.2f} dB"
            )

        psnr_final = _psnr_db(x_final, x_gt)
        summary_lines.append(f"  final (t=1.0):      PSNR = {psnr_final:.2f} dB")
        summary_lines.append("")

        print(
            f"  image {image_idx}: degraded={psnr_deg:.2f} dB, final={psnr_final:.2f} dB"
        )

    summary_path = output_dir / "psnrs.txt"
    summary_path.write_text("\n".join(summary_lines))

    print()
    print(
        f"Done. Wrote {len(images)} images x {len(snapshot_t) + 2} panels to {output_dir}"
    )
    print(f"PSNR summary in {summary_path}")
    print(
        "To use in LaTeX: copy this folder to <thesis_root>/figures/ and assemble a"
    )
    print(
        "tabular figure with rows = images and columns = (clean, degraded, t=0.0, ...)."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a multi-image non-blind-oracle trajectory figure"
    )
    p.add_argument(
        "--dataset",
        choices=["CelebA", "BSD68", "Set12"],
        default="CelebA",
        help="Dataset to draw images from (default CelebA)",
    )
    p.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of images to render (default 5). Capped by dataset size.",
    )
    p.add_argument("--sigma", type=float, default=1.5, help="True blur sigma")
    p.add_argument("--noise-std", type=float, default=0.05, help="Additive noise std")
    p.add_argument(
        "--num-steps", type=int, default=100, help="PnP-Flow trajectory steps"
    )
    p.add_argument(
        "--snapshots",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Trajectory time points at which to save snapshots",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    p.add_argument("--device", default=None, help="Override device (cuda or cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        dataset=args.dataset,
        num_images=args.num,
        sigma=args.sigma,
        noise=args.noise_std,
        snapshot_t=tuple(args.snapshots),
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
