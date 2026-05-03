"""Full blur-SURE evaluation on CelebA.

Reproduces the main result of the thesis. Stage 1 calibrates the lambda
multiplier on synthetic data. Stage 2 estimates sigma with blur-SURE on
each CelebA image, reconstructs with PnP-Flow under the estimated sigma,
runs the non blind oracle, and reports the PSNR gap. Sweeps over the full
sigma and noise grid with three reconstruction seeds per image.

Outputs are written to results/blind/blur_sure_full/.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import torch
from torchvision.utils import save_image

from .._helpers import (
    NOISE_LEVELS,
    SEED,
    SIGMA_VALUES,
    TimingTracker,
    find_repo_root,
    load_model,
    save_json,
)
from ..data import load_celeba
from ..evaluation import evaluate, postprocess
from ..forward import gaussian_blur_fft
from ..reconstruction import pnp_flow_reconstruct
from ..sigma_estimation import estimate_sigma_blur_sure


QUAL_SIGMA = 1.5
QUAL_NOISE = 0.05
QUAL_NUM_IMAGES = 8
QUAL_NUM_SURE_CURVES = 3


def calibrate_lambda(noise_levels: List[float], img_size: int, device: str) -> Dict[float, float]:
    """Stage 1: calibrate the lambda multiplier on synthetic data."""
    LAM_MULTIPLIERS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    CALIB_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    N_CALIB = 10

    best: Dict[float, float] = {}
    for noise_std in noise_levels:
        noise_var = noise_std ** 2
        best_mult = 1.0
        best_err = float("inf")
        for mult in LAM_MULTIPLIERS:
            lam = noise_var * mult
            errors = []
            for st in CALIB_SIGMAS:
                for trial in range(N_CALIB):
                    torch.manual_seed(SEED + 500 + trial)
                    x_synth = torch.randn(3, img_size, img_size, device=device) * 0.3
                    x_synth = x_synth.clamp(-1, 1).unsqueeze(0)
                    y_synth = gaussian_blur_fft(x_synth, st) + torch.randn_like(x_synth) * noise_std
                    res = estimate_sigma_blur_sure(y_synth, noise_var, lam=lam)
                    errors.append(abs(res["sigma_star"] - st))
            mean_err = float(np.mean(errors))
            if mean_err < best_err:
                best_err = mean_err
                best_mult = mult
        best[noise_std] = best_mult
        print(f"  noise={noise_std}: best_mult={best_mult} err={best_err:.4f}")
    return best


def run(
    num_images: int,
    n_seeds: int,
    pnp_steps: int,
    output_dir: str,
    device: str,
    skip_calibration: bool,
):
    repo_root = find_repo_root()
    model = load_model(repo_root, device)
    images = load_celeba(repo_root, device, num_images=num_images, seed=SEED)
    print(f"Loaded {len(images)} CelebA images")

    if skip_calibration:
        print("Skipping calibration, using lambda multiplier 10.0")
        best_lam_mult = {ns: 10.0 for ns in NOISE_LEVELS}
    else:
        print("Stage 1: calibrating lambda multiplier on synthetic data")
        best_lam_mult = calibrate_lambda(NOISE_LEVELS, img_size=128, device=device)

    save_json(os.path.join(output_dir, "lambda_calibration.json"), best_lam_mult)

    print("\nStage 2: blur-SURE evaluation")
    tt = TimingTracker()
    all_results: Dict = {}
    qual_dir = os.path.join(output_dir, "qualitative")
    os.makedirs(qual_dir, exist_ok=True)
    sure_curves_dump: List[Dict] = []

    for sigma_true in SIGMA_VALUES:
        for noise_std in NOISE_LEVELS:
            noise_var = noise_std ** 2
            lam = noise_var * best_lam_mult[noise_std]
            cfg_key = f"sigma={sigma_true}/noise={noise_std}"
            print(f"\n--- {cfg_key} (lam={lam:.2e}) ---")

            is_qual_cfg = (sigma_true == QUAL_SIGMA and noise_std == QUAL_NOISE)
            per_image = []
            for idx, x_gt in enumerate(images):
                torch.manual_seed(SEED + idx)
                y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std

                with tt.track("sigma_estimation"):
                    sure_res = estimate_sigma_blur_sure(y, noise_var, lam=lam)
                sigma_est = sure_res["sigma_star"]

                if is_qual_cfg and idx < QUAL_NUM_SURE_CURVES:
                    sure_curves_dump.append({
                        "index": idx,
                        "sigma_grid": sure_res["sigma_grid"].tolist(),
                        "sure_curve": sure_res["sure_curve"].tolist(),
                        "sigma_star": sigma_est,
                        "sigma_true": sigma_true,
                    })

                seed_metrics = {"psnr": [], "ssim": [], "lpips": []}
                first_x_rec = None
                for so in range(n_seeds):
                    torch.manual_seed(SEED + idx * 100 + so)
                    with tt.track("reconstruction"):
                        x_rec = pnp_flow_reconstruct(
                            model, y=y, sigma_blur=sigma_est, num_steps=pnp_steps,
                        )
                    m = evaluate(x_rec, x_gt)
                    for k in seed_metrics:
                        seed_metrics[k].append(m[k])
                    if so == 0:
                        first_x_rec = x_rec

                torch.manual_seed(SEED + idx * 100)
                x_oracle = pnp_flow_reconstruct(
                    model, y=y, sigma_blur=sigma_true, num_steps=pnp_steps,
                )
                m_oracle = evaluate(x_oracle, x_gt)

                if is_qual_cfg and idx < QUAL_NUM_IMAGES:
                    save_image(postprocess(x_gt), os.path.join(qual_dir, f"{idx:04d}_clean.png"))
                    save_image(postprocess(y), os.path.join(qual_dir, f"{idx:04d}_observed.png"))
                    save_image(postprocess(first_x_rec), os.path.join(qual_dir, f"{idx:04d}_blind.png"))
                    save_image(postprocess(x_oracle), os.path.join(qual_dir, f"{idx:04d}_oracle.png"))

                per_image.append({
                    "index": idx,
                    "sigma_est": sigma_est,
                    "sigma_error": abs(sigma_est - sigma_true),
                    "psnr_mean": float(np.mean(seed_metrics["psnr"])),
                    "psnr_std": float(np.std(seed_metrics["psnr"])),
                    "ssim_mean": float(np.mean(seed_metrics["ssim"])),
                    "lpips_mean": float(np.mean(seed_metrics["lpips"])),
                    "oracle_psnr": m_oracle["psnr"],
                    "oracle_ssim": m_oracle["ssim"],
                    "oracle_lpips": m_oracle["lpips"],
                })

            psnr_mean = float(np.mean([p["psnr_mean"] for p in per_image]))
            oracle_psnr = float(np.mean([p["oracle_psnr"] for p in per_image]))
            all_results[cfg_key] = {
                "per_image": per_image,
                "sigma_error_mean": float(np.mean([p["sigma_error"] for p in per_image])),
                "sigma_error_std": float(np.std([p["sigma_error"] for p in per_image])),
                "psnr_mean": psnr_mean,
                "ssim_mean": float(np.mean([p["ssim_mean"] for p in per_image])),
                "lpips_mean": float(np.mean([p["lpips_mean"] for p in per_image])),
                "oracle_psnr_mean": oracle_psnr,
                "oracle_ssim_mean": float(np.mean([p["oracle_ssim"] for p in per_image])),
                "oracle_lpips_mean": float(np.mean([p["oracle_lpips"] for p in per_image])),
                "psnr_gap": oracle_psnr - psnr_mean,
            }
            r = all_results[cfg_key]
            print(
                f"  err={r['sigma_error_mean']:.4f} PSNR={r['psnr_mean']:.2f} "
                f"oracle={r['oracle_psnr_mean']:.2f} gap={r['psnr_gap']:+.2f}"
            )

    save_json(os.path.join(output_dir, "results.json"), all_results)
    save_json(os.path.join(output_dir, "sure_curves.json"), sure_curves_dump)
    print(f"\nWrote {output_dir}/results.json and sure_curves.json")
    print(f"Qualitative images at {qual_dir}")
    print(tt.summary())
    return all_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full blur-SURE evaluation on CelebA")
    p.add_argument("--num-images", type=int, default=100)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--pnp-steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--skip-calibration", action="store_true",
                   help="Skip stage 1 and use lambda=10*eta^2 directly")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(find_repo_root(), "results", "blind", "blur_sure_full")
    run(
        num_images=args.num_images,
        n_seeds=args.n_seeds,
        pnp_steps=args.pnp_steps,
        output_dir=output_dir,
        device=args.device,
        skip_calibration=args.skip_calibration,
    )


if __name__ == "__main__":
    main()
