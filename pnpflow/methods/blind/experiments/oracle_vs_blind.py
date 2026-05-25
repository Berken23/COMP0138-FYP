"""Generate oracle-vs-blind reconstruction comparison images.

For each image, produces 4 panels — clean, degraded, oracle reconstruction
(PnP-Flow with true sigma), and blind reconstruction (PnP-Flow with the
Blur-SURE estimated sigma) — saved as individual PNG files for use in the
thesis figures.

Default: 5 images from CelebA, 3 from BSD68, 1 from Set12. All processed at
sigma = 1.5, nu = 0.05.

Configuration: sigma = 1.5, nu = 0.05, num_steps = 100, lr = 1.0,
gamma = 1 - t, cold start from x_0 = y. Both oracle and blind use the
matched-RNG-state convention (same reconstruction seed), so the only
difference between the two reconstructions is the value of sigma supplied
to the data-fidelity step.

Deterministic seeds:
- Observation generation for image i in a dataset: SEED + i
- Oracle and blind reconstruction for image i:     SEED + 100 * i

Usage
-----
    python -m pnpflow.methods.blind.experiments.oracle_vs_blind
    python -m pnpflow.methods.blind.experiments.oracle_vs_blind --celeba-num 5 --bsd68-num 3 --set12-num 1

Outputs
-------
results/blind/figures/oracle_vs_blind/
    celeba_00_clean.png   celeba_00_degraded.png
    celeba_00_oracle.png  celeba_00_blind.png
    celeba_01_*.png ... celeba_04_*.png
    bsd68_00_*.png  bsd68_01_*.png  bsd68_02_*.png
    set12_00_*.png
    psnrs.txt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image

from .._helpers import SEED, find_repo_root, load_model
from ..data import load_bsd68, load_celeba, load_set12
from ..evaluation import non_blind_oracle, postprocess
from ..forward import gaussian_blur_fft
from ..reconstruction import pnp_flow_reconstruct
from ..sigma_estimation import estimate_sigma_blur_sure


def _psnr_db(x_hat: torch.Tensor, x_gt: torch.Tensor) -> float:
    """PSNR in dB between two tensors in [-1, 1]."""
    a = postprocess(x_hat.detach().cpu())
    b = postprocess(x_gt.detach().cpu())
    mse = torch.mean((a - b) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def _load_images(
    dataset: str,
    repo_root: Path,
    device: str,
    num: int,
    offset: int = 0,
    celeba_seed: int = SEED,
) -> List[torch.Tensor]:
    if num <= 0:
        return []
    if dataset == "celeba":
        # CelebA's loader samples deterministically from the test pool using
        # the seed. Different seeds produce different samples; the offset
        # then takes a slice within the seeded sample.
        imgs = load_celeba(
            str(repo_root), device, num_images=offset + num, seed=celeba_seed
        )
        return imgs[offset : offset + num]
    if dataset == "bsd68":
        return load_bsd68(str(repo_root), device, img_size=128)[
            offset : offset + num
        ]
    if dataset == "set12":
        return load_set12(str(repo_root), device, img_size=128)[
            offset : offset + num
        ]
    raise ValueError(f"Unknown dataset '{dataset}'")


def _process_image(
    model,
    x_gt: torch.Tensor,
    image_idx: int,
    sigma_true: float,
    noise: float,
    num_steps: int,
):
    """Run Blur-SURE + blind + oracle. Returns (y, sigma_hat, x_blind, x_oracle)."""
    torch.manual_seed(SEED + image_idx)
    y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise

    sure_out = estimate_sigma_blur_sure(y, noise_var=noise ** 2)
    sigma_hat = sure_out["sigma_star"]

    torch.manual_seed(SEED + image_idx * 100)
    x_oracle = non_blind_oracle(
        model, y=y, sigma_true=sigma_true, num_steps=num_steps
    )

    torch.manual_seed(SEED + image_idx * 100)
    x_blind = pnp_flow_reconstruct(
        model, y=y, sigma_blur=sigma_hat, num_steps=num_steps
    )

    return y, sigma_hat, x_blind, x_oracle


def run(
    celeba_num: int = 5,
    bsd68_num: int = 3,
    set12_num: int = 1,
    celeba_offset: int = 0,
    bsd68_offset: int = 0,
    set12_offset: int = 0,
    celeba_seed: int = SEED,
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
            repo_root / "results" / "blind" / "figures" / "oracle_vs_blind"
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(
        f"Configuration:    sigma={sigma}, nu={noise}, num_steps={num_steps}"
    )
    print(f"Loading model on {device}")
    model = load_model(repo_root, device)

    dataset_specs = [
        ("celeba", celeba_num, celeba_offset),
        ("bsd68", bsd68_num, bsd68_offset),
        ("set12", set12_num, set12_offset),
    ]

    summary_lines = [
        "Oracle-vs-blind reconstruction comparison",
        f"Configuration: sigma = {sigma}, nu = {noise}, num_steps = {num_steps}",
        "",
    ]

    total = celeba_num + bsd68_num + set12_num
    counter = 0

    for dataset, num, offset in dataset_specs:
        if num <= 0:
            continue
        print(
            f"\nLoading {num} {dataset.upper()} images "
            f"(offset {offset}{', seed ' + str(celeba_seed) if dataset == 'celeba' else ''})"
        )
        images = _load_images(
            dataset, repo_root, device, num, offset=offset, celeba_seed=celeba_seed
        )
        for i, x_gt in enumerate(images):
            counter += 1
            print(f"  [{counter}/{total}] {dataset}_{i:02d} ...")
            x_gt = x_gt.to(device)

            y, sigma_hat, x_blind, x_oracle = _process_image(
                model,
                x_gt,
                image_idx=i,
                sigma_true=sigma,
                noise=noise,
                num_steps=num_steps,
            )

            prefix = f"{dataset}_{i:02d}"
            save_image(postprocess(x_gt), output_dir / f"{prefix}_clean.png")
            save_image(postprocess(y), output_dir / f"{prefix}_degraded.png")
            save_image(postprocess(x_oracle), output_dir / f"{prefix}_oracle.png")
            save_image(postprocess(x_blind), output_dir / f"{prefix}_blind.png")

            psnr_deg = _psnr_db(y, x_gt)
            psnr_orc = _psnr_db(x_oracle, x_gt)
            psnr_bli = _psnr_db(x_blind, x_gt)

            summary_lines.append(
                f"{prefix}:  sigma_hat = {sigma_hat:.3f}  "
                f"degraded = {psnr_deg:.2f} dB  "
                f"oracle = {psnr_orc:.2f} dB  "
                f"blind = {psnr_bli:.2f} dB"
            )
            print(
                f"    sigma_hat={sigma_hat:.3f}  "
                f"degraded={psnr_deg:.2f}  oracle={psnr_orc:.2f}  blind={psnr_bli:.2f}"
            )

    psnrs_path = output_dir / "psnrs.txt"
    psnrs_path.write_text("\n".join(summary_lines))

    print()
    print(f"Done. Wrote {total} x 4 = {total * 4} PNGs and psnrs.txt to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate oracle-vs-blind 4-panel reconstruction comparison images"
    )
    p.add_argument(
        "--celeba-num",
        type=int,
        default=5,
        help="Number of CelebA images to process (default 5)",
    )
    p.add_argument(
        "--bsd68-num",
        type=int,
        default=3,
        help="Number of BSD68 images to process (default 3)",
    )
    p.add_argument(
        "--set12-num",
        type=int,
        default=1,
        help="Number of Set12 images to process (default 1)",
    )
    p.add_argument(
        "--celeba-offset",
        type=int,
        default=0,
        help="Skip the first N CelebA images from the seeded sample (default 0)",
    )
    p.add_argument(
        "--bsd68-offset",
        type=int,
        default=0,
        help="Skip the first N BSD68 images in sorted-filename order (default 0)",
    )
    p.add_argument(
        "--set12-offset",
        type=int,
        default=0,
        help="Skip the first N Set12 images in sorted-filename order (default 0)",
    )
    p.add_argument(
        "--celeba-seed",
        type=int,
        default=SEED,
        help="Random seed for CelebA sampling (default matches the experimental setup)",
    )
    p.add_argument("--sigma", type=float, default=1.5, help="True blur sigma")
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
        celeba_num=args.celeba_num,
        bsd68_num=args.bsd68_num,
        set12_num=args.set12_num,
        celeba_offset=args.celeba_offset,
        bsd68_offset=args.bsd68_offset,
        set12_offset=args.set12_offset,
        celeba_seed=args.celeba_seed,
        sigma=args.sigma,
        noise=args.noise_std,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
