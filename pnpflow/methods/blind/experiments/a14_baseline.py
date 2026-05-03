"""A.14 baseline experiment.

Reproduces the failed joint estimation baseline from the thesis. At every
PnP-Flow step the operator parameter sigma is updated via Adam against
||y - H_sigma(x)||^2. This is the approach the thesis diagnoses as
structurally biased due to the over sharpening of the denoiser output.

Outputs are written to results/blind/a14_baseline/.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torchvision.utils import save_image

from .._helpers import (
    SEED,
    TimingTracker,
    find_repo_root,
    load_model,
    pnp_flow_trajectory_blind,
    save_json,
)
from ..data import load_celeba
from ..evaluation import evaluate, postprocess
from ..forward import gaussian_blur_fft


N_PROGRESSION_IMAGES = 4


def run(
    num_images: int,
    sigma_true: float,
    sigma_init: float,
    sigma_max: float,
    noise_std: float,
    outer_iters: int,
    pnp_steps: int,
    output_dir: str,
    device: str,
):
    repo_root = find_repo_root()
    model = load_model(repo_root, device)
    images = load_celeba(repo_root, device, num_images=num_images, seed=SEED)

    tt = TimingTracker()
    records = []

    for idx, x_gt in enumerate(images):
        torch.manual_seed(SEED + idx)
        y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std

        sigma_hist_outer = []
        x = y.clone()

        with tt.track("a14_full"):
            for _ in range(outer_iters):
                x, sigma_final, _ = pnp_flow_trajectory_blind(
                    model,
                    x_init=x,
                    y=y,
                    sigma_init=sigma_hist_outer[-1] if sigma_hist_outer else sigma_init,
                    sigma_max=sigma_max,
                    num_steps=pnp_steps,
                )
                sigma_hist_outer.append(sigma_final)

        m = evaluate(x, x_gt)
        records.append({
            "index": idx,
            "sigma_final": sigma_final,
            "sigma_error": abs(sigma_final - sigma_true),
            "sigma_history_outer": sigma_hist_outer,
            "psnr": m["psnr"],
            "ssim": m["ssim"],
            "lpips": m["lpips"],
        })
        print(
            f"[{idx+1:>3}/{num_images}] sigma={sigma_final:.4f} "
            f"err={records[-1]['sigma_error']:.4f} PSNR={m['psnr']:.2f}"
        )

    summary = {
        "experiment": "a14_baseline",
        "sigma_true": sigma_true,
        "sigma_init": sigma_init,
        "noise_std": noise_std,
        "outer_iters": outer_iters,
        "pnp_steps": pnp_steps,
        "num_images": len(records),
        "sigma_error_mean": float(np.mean([r["sigma_error"] for r in records])),
        "sigma_error_std": float(np.std([r["sigma_error"] for r in records])),
        "psnr_mean": float(np.mean([r["psnr"] for r in records])),
        "ssim_mean": float(np.mean([r["ssim"] for r in records])),
        "lpips_mean": float(np.mean([r["lpips"] for r in records])),
    }

    save_json(os.path.join(output_dir, "per_image.json"), records)
    save_json(os.path.join(output_dir, "summary.json"), summary)

    print()
    print(f"Wrote results to {output_dir}")
    print(f"sigma_error mean = {summary['sigma_error_mean']:.4f}")
    print(f"PSNR mean        = {summary['psnr_mean']:.2f} dB")

    print(f"\nIteration progression snapshots for first {min(N_PROGRESSION_IMAGES, num_images)} images")
    progression_outers = sorted(set([0, 5, 10, 15, 20, 25, outer_iters - 1]) - {None})
    progression_outers = [o for o in progression_outers if 0 <= o < outer_iters]
    progression_dir = os.path.join(output_dir, "iteration_progression")
    os.makedirs(progression_dir, exist_ok=True)
    progression_meta = []
    n_show = min(N_PROGRESSION_IMAGES, len(images))
    for idx in range(n_show):
        x_gt = images[idx]
        torch.manual_seed(SEED + idx)
        y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std
        x = y.clone()
        sigma_current = sigma_init
        save_image(postprocess(x_gt), os.path.join(progression_dir, f"img{idx}_clean.png"))
        save_image(postprocess(y), os.path.join(progression_dir, f"img{idx}_observed.png"))
        for outer in range(outer_iters):
            x, sigma_current, _ = pnp_flow_trajectory_blind(
                model,
                x_init=x,
                y=y,
                sigma_init=sigma_current,
                sigma_max=sigma_max,
                num_steps=pnp_steps,
            )
            if outer in progression_outers:
                save_image(
                    postprocess(x),
                    os.path.join(progression_dir, f"img{idx}_outer{outer:03d}.png"),
                )
                progression_meta.append({
                    "image": idx,
                    "outer": outer,
                    "sigma": float(sigma_current),
                })
    save_json(os.path.join(output_dir, "iteration_progression.json"), progression_meta)

    print(tt.summary())
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.14 baseline (failed joint estimation)")
    p.add_argument("--num-images", type=int, default=10)
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--sigma-init", type=float, default=3.0)
    p.add_argument("--sigma-max", type=float, default=5.0)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--outer-iters", type=int, default=30)
    p.add_argument("--pnp-steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(find_repo_root(), "results", "blind", "a14_baseline")
    run(
        num_images=args.num_images,
        sigma_true=args.sigma,
        sigma_init=args.sigma_init,
        sigma_max=args.sigma_max,
        noise_std=args.noise_std,
        outer_iters=args.outer_iters,
        pnp_steps=args.pnp_steps,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
