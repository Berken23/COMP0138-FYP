"""Save sample images from each dataset for inclusion in the report.

Loads N images from each of CelebA, BSD68, and Set12 at the resolution used
throughout the blind-deconvolution experiments (128x128), postprocesses them
to [0, 1] for display, and writes them to a single output folder ready to be
copied into the thesis figures directory.

Usage
-----
    python -m pnpflow.methods.blind.experiments.dataset_samples
    python -m pnpflow.methods.blind.experiments.dataset_samples --num 6

Outputs
-------
results/blind/figures/dataset_samples/
    celeba_00.png ... celeba_03.png
    bsd68_00.png  ... bsd68_03.png
    set12_00.png  ... set12_03.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from .._helpers import SEED, find_repo_root
from ..data import load_bsd68, load_celeba, load_set12
from ..evaluation import postprocess


def run(
    num_per_dataset: int = 4,
    img_size: int = 128,
    output_dir: Path | None = None,
    device: str | None = None,
):
    repo_root = Path(find_repo_root())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_dir is None:
        output_dir = repo_root / "results" / "blind" / "figures" / "dataset_samples"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Image size:       {img_size}x{img_size}")
    print(f"Images per set:   {num_per_dataset}")
    print()

    loaders = [
        (
            "celeba",
            lambda: load_celeba(
                str(repo_root), device, num_images=num_per_dataset, seed=SEED
            ),
        ),
        ("bsd68", lambda: load_bsd68(str(repo_root), device, img_size=img_size)),
        ("set12", lambda: load_set12(str(repo_root), device, img_size=img_size)),
    ]

    for name, fn in loaders:
        print(f"Loading {name} ...")
        images = fn()
        print(f"  found {len(images)} images")
        n = min(num_per_dataset, len(images))
        for i in range(n):
            fname = f"{name}_{i:02d}.png"
            save_image(postprocess(images[i]), output_dir / fname)
            print(f"  wrote {fname}")
        print()

    print(f"Done. {num_per_dataset} images per dataset written to {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Save sample images from each dataset for report figures"
    )
    p.add_argument(
        "--num", type=int, default=4, help="Number of images per dataset (default 4)"
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Resize / centre-crop size (default 128, matches the experiments)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Override output directory"
    )
    p.add_argument("--device", default=None, help="Override device (cuda or cpu)")
    return p.parse_args()


def main():
    args = parse_args()
    run(
        num_per_dataset=args.num,
        img_size=args.img_size,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
