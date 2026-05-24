"""Generate the PnP-Flow trajectory progression figure for Section 4.5.

Renders one representative blind reconstruction as a sequence of snapshots
at t in {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}, alongside the degraded observation
and the clean reference. Each snapshot is saved as an individual PNG file
in a folder ready to drop into a thesis figures directory.

Usage
-----
    python -m pnpflow.methods.blind.experiments.trajectory_figure

Outputs
-------
results/blind/figures/trajectory_progression/
    clean.png       ground-truth reference (for the caption)
    degraded.png    blurred and noisy observation y
    t_00.png        iterate at t = 0.0
    t_02.png        iterate at t = 0.2
    t_04.png        iterate at t = 0.4
    t_06.png        iterate at t = 0.6
    t_08.png        iterate at t = 0.8
    t_10.png        iterate at t = 1.0
    psnrs.txt       PSNR value for each panel (paste into LaTeX caption)
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
from ..data import load_celeba
from ..evaluation import postprocess
from ..forward import gaussian_blur_fft


def _psnr_db(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [-1, 1]."""
    mse = torch.mean((postprocess(x_hat) - postprocess(x_gt.to(x_hat.device))) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def run(
    image_idx: int = 2,
    sigma: float = 1.5,
    noise: float = 0.05,
    snapshot_t: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    output_dir: Path | None = None,
    device: str | None = None,
):
    repo_root = Path(find_repo_root())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = repo_root / "results" / "blind" / "figures" / "trajectory_progression"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load model and image.
    model = load_model(repo_root, device)
    images = load_celeba(repo_root, device, num_images=image_idx + 1, seed=SEED)
    x_gt = images[image_idx].to(device)

    # Generate the degraded observation with the same seeding convention used
    # throughout the chapter.
    torch.manual_seed(SEED + image_idx)
    y = gaussian_blur_fft(x_gt, sigma) + torch.randn_like(x_gt) * noise

    # Run the trajectory-tracked reconstruction.  Use the matched-RNG-state
    # convention so the trajectory is reproducible.
    torch.manual_seed(SEED + image_idx * 100)
    x_final, trajectory, t_values = pnp_flow_trajectory_tracked(
        model,
        x_init=y.clone(),
        y=y,
        sigma_blur=sigma,
        num_steps=100,
    )

    # Save clean and degraded references.
    save_image(postprocess(x_gt), output_dir / "clean.png")
    save_image(postprocess(y), output_dir / "degraded.png")
    psnr_degraded = _psnr_db(y, x_gt)

    # Save snapshots at each requested t and compute PSNRs.
    snapshot_records = []
    for t_target in snapshot_t:
        idx = min(range(len(t_values)), key=lambda i: abs(t_values[i] - t_target))
        x_snap = trajectory[idx].to(device)
        fname = f"t_{int(round(t_target * 10)):02d}.png"
        save_image(postprocess(x_snap), output_dir / fname)
        snapshot_records.append((t_target, fname, _psnr_db(x_snap, x_gt)))

    # Write the PSNR values to a text file for easy reference when finalising
    # the LaTeX caption.
    psnrs_path = output_dir / "psnrs.txt"
    with open(psnrs_path, "w") as f:
        f.write(f"Configuration: CelebA image {image_idx}, sigma = {sigma}, nu = {noise}\n")
        f.write(f"Degraded y:  PSNR = {psnr_degraded:.2f} dB\n")
        for t, fname, psnr in snapshot_records:
            f.write(f"t = {t:.1f} ({fname}):  PSNR = {psnr:.2f} dB\n")
        f.write(f"Final x (t=1.0):  PSNR = {_psnr_db(x_final, x_gt):.2f} dB\n")

    print()
    print(f"PSNR summary (also written to {psnrs_path}):")
    print(f"  Degraded y:    {psnr_degraded:.2f} dB")
    for t, fname, psnr in snapshot_records:
        print(f"  t = {t:.1f} ({fname}):  {psnr:.2f} dB")
    print()
    print("To use in LaTeX: copy this folder to <thesis_root>/figures/trajectory_progression/")
    print("and paste the figure environment from the conversation alongside it.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PnP-Flow trajectory progression figure")
    p.add_argument("--image-idx", type=int, default=2, help="CelebA test image index")
    p.add_argument("--sigma", type=float, default=1.5, help="True blur sigma")
    p.add_argument("--noise-std", type=float, default=0.05, help="Additive noise std")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    p.add_argument("--device", default=None, help="Override device (cuda or cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        image_idx=args.image_idx,
        sigma=args.sigma,
        noise=args.noise_std,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
