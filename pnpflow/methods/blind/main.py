"""Entry point for the blind deconvolution evaluation.

Runs the decoupled two stage pipeline on a chosen dataset for a chosen
blur level. For each image the script estimates sigma with blur-SURE,
reconstructs with PnP-Flow under the estimated sigma, runs the non blind
oracle with the true sigma, and writes per image metrics and aggregate
summaries to results/blind/<dataset>/sigma_<sigma>/.

Usage
-----
python -m pnpflow.methods.blind.main --dataset CelebA --sigma 1.5 --num-images 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from typing import Dict, List

import torch
from torchvision.utils import save_image

from .data import load_dataset
from .evaluation import compute_lpips, compute_ssim, evaluate, non_blind_oracle, postprocess, psnr_db
from .forward import gaussian_blur_fft
from .reconstruction import pnp_flow_reconstruct
from .sigma_estimation import estimate_sigma_blur_sure


def _find_repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(here, "pnpflow")) and os.path.isdir(
            os.path.join(here, "model")
        ):
            return here
        here = os.path.dirname(here)
    raise RuntimeError("Could not locate repository root containing pnpflow/ and model/.")


def _load_model(repo_root: str, device: str, img_size: int = 128, num_channels: int = 3):
    pnpflow_dir = os.path.join(repo_root, "pnpflow")
    if pnpflow_dir not in sys.path:
        sys.path.insert(0, pnpflow_dir)

    from pnpflow.utils import define_model, load_model

    model_args = types.SimpleNamespace(
        model="ot",
        num_channels=num_channels,
        dim_image=img_size,
    )
    model, _ = define_model(model_args)
    load_model(
        "ot",
        model,
        None,
        download=False,
        checkpoint_path=os.path.join(repo_root, "model", "celeba", "ot", "model_final.pt"),
        dataset=None,
        device=device,
    )
    return model.to(device).eval()


def _aggregate(records: List[Dict]) -> Dict[str, float]:
    keys = ["sigma_estimated", "sigma_error", "psnr_blind", "ssim_blind", "lpips_blind",
            "psnr_oracle", "ssim_oracle", "lpips_oracle", "psnr_gap"]
    return {
        k: float(sum(r[k] for r in records) / len(records))
        for k in keys
        if records and k in records[0]
    }


def run(
    dataset: str,
    sigma_true: float,
    num_images: int,
    output_dir: str,
    device: str,
    pnp_steps: int,
    pnp_lr: float,
    noise_std: float,
    sigma_seed: int,
) -> Dict:
    repo_root = _find_repo_root()
    print(f"Repo root: {repo_root}")
    print(f"Loading model on {device}")
    model = _load_model(repo_root, device)

    print(f"Loading {dataset}")
    images = load_dataset(dataset, repo_root, device, num_celeba=num_images)
    if num_images is not None:
        images = images[:num_images]
    print(f"Using {len(images)} images")

    out_dir = os.path.join(output_dir, dataset, f"sigma_{sigma_true:.2f}")
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    records: List[Dict] = []
    t_start = time.time()

    for i, x_clean in enumerate(images):
        x_clean = x_clean.to(device)
        torch.manual_seed(sigma_seed + i)
        y = gaussian_blur_fft(x_clean, sigma_true) + torch.randn_like(x_clean) * noise_std

        sure_out = estimate_sigma_blur_sure(y, noise_var=noise_std ** 2)
        sigma_hat = sure_out["sigma_star"]

        x_blind = pnp_flow_reconstruct(
            model, y=y, sigma_blur=sigma_hat, num_steps=pnp_steps, lr=pnp_lr,
        )
        x_oracle = non_blind_oracle(
            model, y=y, sigma_true=sigma_true, num_steps=pnp_steps, lr=pnp_lr,
        )

        m_blind = evaluate(x_blind, x_clean)
        m_oracle = evaluate(x_oracle, x_clean)

        record = {
            "index": i,
            "sigma_true": sigma_true,
            "sigma_estimated": sigma_hat,
            "sigma_error": abs(sigma_hat - sigma_true),
            "psnr_blind": m_blind["psnr"],
            "ssim_blind": m_blind["ssim"],
            "lpips_blind": m_blind["lpips"],
            "psnr_oracle": m_oracle["psnr"],
            "ssim_oracle": m_oracle["ssim"],
            "lpips_oracle": m_oracle["lpips"],
            "psnr_gap": m_oracle["psnr"] - m_blind["psnr"],
        }
        records.append(record)

        save_image(postprocess(x_clean), os.path.join(images_dir, f"{i:04d}_clean.png"))
        save_image(postprocess(y), os.path.join(images_dir, f"{i:04d}_observed.png"))
        save_image(postprocess(x_blind), os.path.join(images_dir, f"{i:04d}_blind.png"))
        save_image(postprocess(x_oracle), os.path.join(images_dir, f"{i:04d}_oracle.png"))

        print(
            f"[{i+1:>3}/{len(images)}] sigma_hat={sigma_hat:.3f} "
            f"psnr_blind={m_blind['psnr']:.2f} psnr_oracle={m_oracle['psnr']:.2f} "
            f"gap={record['psnr_gap']:+.2f}"
        )

    summary = _aggregate(records)
    summary["dataset"] = dataset
    summary["sigma_true"] = sigma_true
    summary["num_images"] = len(records)
    summary["pnp_steps"] = pnp_steps
    summary["pnp_lr"] = pnp_lr
    summary["noise_std"] = noise_std
    summary["elapsed_seconds"] = time.time() - t_start

    with open(os.path.join(out_dir, "per_image.json"), "w") as f:
        json.dump(records, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Wrote {len(records)} per image records and summary to {out_dir}")
    print(
        f"Average  psnr_blind={summary['psnr_blind']:.2f}  "
        f"psnr_oracle={summary['psnr_oracle']:.2f}  "
        f"gap={summary['psnr_gap']:+.2f}  "
        f"sigma_err={summary['sigma_error']:.3f}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blind PnP-Flow deconvolution evaluation")
    p.add_argument("--dataset", choices=["CelebA", "BSD68", "Set12"], default="CelebA")
    p.add_argument("--sigma", type=float, default=1.5, help="True blur sigma")
    p.add_argument("--num-images", type=int, default=10, help="Number of images to evaluate")
    p.add_argument("--noise-std", type=float, default=0.05, help="Additive noise std")
    p.add_argument("--pnp-steps", type=int, default=100, help="PnP-Flow trajectory steps")
    p.add_argument("--pnp-lr", type=float, default=1.0, help="PnP-Flow data fit step size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <repo_root>/results/blind/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(_find_repo_root(), "results", "blind")

    run(
        dataset=args.dataset,
        sigma_true=args.sigma,
        num_images=args.num_images,
        output_dir=output_dir,
        device=args.device,
        pnp_steps=args.pnp_steps,
        pnp_lr=args.pnp_lr,
        noise_std=args.noise_std,
        sigma_seed=args.seed,
    )


if __name__ == "__main__":
    main()
