"""Multi image blur-SURE evaluation.

Demonstrates the variance reduction of blur-SURE when curves are averaged
over B independent observations. Part 1 sweeps sigma estimation accuracy
over B for the full sigma and noise grid. Part 2 reconstructs eight
CelebA images at sigma=1.5, noise=0.05 for each B value to show how
reconstruction quality tracks estimation accuracy.

Outputs are written to results/blind/multi_image/.
"""
from __future__ import annotations

import argparse
import os
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
from ..sigma_estimation import estimate_sigma_multi_sure


def run(
    b_values: List[int],
    n_trials: int,
    n_recon_images: int,
    n_seeds: int,
    lam_multiplier: float,
    pnp_steps: int,
    output_dir: str,
    device: str,
):
    repo_root = find_repo_root()
    model = load_model(repo_root, device)
    images = load_celeba(repo_root, device, num_images=max(250, max(b_values)), seed=SEED)
    print(f"Loaded {len(images)} CelebA images")

    tt = TimingTracker()

    print("Part 1: sigma estimation vs B (across all sigma and noise)")
    est_results: Dict[str, Dict] = {}
    for sigma_true in SIGMA_VALUES:
        for noise_std in NOISE_LEVELS:
            noise_var = noise_std ** 2
            lam = noise_var * lam_multiplier
            for B in b_values:
                errors: List[float] = []
                ests: List[float] = []
                for trial in range(n_trials):
                    start_idx = (trial * B) % len(images)
                    sub = images[start_idx:start_idx + B]
                    if len(sub) < B:
                        sub = sub + images[: B - len(sub)]
                    ys = []
                    for i, x_gt in enumerate(sub):
                        torch.manual_seed(
                            SEED + 3000 + int(sigma_true * 1000) + int(noise_std * 100) + trial * B + i
                        )
                        y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std
                        ys.append(y)
                    res = estimate_sigma_multi_sure(ys, noise_var, lam=lam)
                    sigma_est = res["sigma_star"]
                    ests.append(sigma_est)
                    errors.append(abs(sigma_est - sigma_true))
                key = f"sigma={sigma_true}/noise={noise_std}/B={B}"
                est_results[key] = {
                    "sigma_estimates": ests,
                    "sigma_errors": errors,
                    "mean_error": float(np.mean(errors)),
                    "std_error": float(np.std(errors)),
                }
                print(f"  {key}: err={np.mean(errors):.4f}+/-{np.std(errors):.4f}")

    print("\nPart 2: reconstruction comparison at sigma=1.5, noise=0.05")
    RECON_SIGMA = 1.5
    RECON_NOISE = 0.05
    recon_results: Dict[int, Dict] = {}
    for B in b_values:
        noise_var = RECON_NOISE ** 2
        lam = noise_var * lam_multiplier

        torch.manual_seed(SEED + 5000)
        observations = []
        for i in range(max(B, n_recon_images)):
            x_gt = images[i % len(images)]
            y = gaussian_blur_fft(x_gt, RECON_SIGMA) + torch.randn_like(x_gt) * RECON_NOISE
            observations.append({"x_gt": x_gt, "y": y})

        ys_for_est = [obs["y"] for obs in observations[:B]]
        with tt.track(f"est_B{B}"):
            res = estimate_sigma_multi_sure(ys_for_est, noise_var, lam=lam)
        sigma_est = res["sigma_star"]
        err = abs(sigma_est - RECON_SIGMA)

        qual_save_for_B = B in {1, 4, 16, max(b_values)}
        qual_dir = os.path.join(output_dir, "qualitative", f"B={B}")
        if qual_save_for_B:
            os.makedirs(qual_dir, exist_ok=True)

        psnr_all, ssim_all, lpips_all = [], [], []
        orc_psnr_all, orc_ssim_all, orc_lpips_all = [], [], []
        for i in range(n_recon_images):
            x_gt = observations[i]["x_gt"]
            y = observations[i]["y"]
            seed_m = {"psnr": [], "ssim": [], "lpips": []}
            first_x_rec = None
            for so in range(n_seeds):
                torch.manual_seed(SEED + i * 100 + so)
                with tt.track(f"recon_B{B}"):
                    x_rec = pnp_flow_reconstruct(
                        model, y=y, sigma_blur=sigma_est, num_steps=pnp_steps,
                    )
                m = evaluate(x_rec, x_gt)
                for k in seed_m:
                    seed_m[k].append(m[k])
                if so == 0:
                    first_x_rec = x_rec
            psnr_all.append(float(np.mean(seed_m["psnr"])))
            ssim_all.append(float(np.mean(seed_m["ssim"])))
            lpips_all.append(float(np.mean(seed_m["lpips"])))

            torch.manual_seed(SEED + i * 100)
            x_orc = pnp_flow_reconstruct(
                model, y=y, sigma_blur=RECON_SIGMA, num_steps=pnp_steps,
            )
            m_orc = evaluate(x_orc, x_gt)
            orc_psnr_all.append(m_orc["psnr"])
            orc_ssim_all.append(m_orc["ssim"])
            orc_lpips_all.append(m_orc["lpips"])

            if qual_save_for_B:
                save_image(postprocess(x_gt), os.path.join(qual_dir, f"{i:04d}_clean.png"))
                save_image(postprocess(y), os.path.join(qual_dir, f"{i:04d}_observed.png"))
                save_image(postprocess(first_x_rec), os.path.join(qual_dir, f"{i:04d}_blind.png"))
                save_image(postprocess(x_orc), os.path.join(qual_dir, f"{i:04d}_oracle.png"))

        gap_all = [o - b for o, b in zip(orc_psnr_all, psnr_all)]
        recon_results[B] = {
            "sigma_est": sigma_est,
            "sigma_error": err,
            "psnr_mean": float(np.mean(psnr_all)),
            "psnr_std": float(np.std(psnr_all)),
            "ssim_mean": float(np.mean(ssim_all)),
            "ssim_std": float(np.std(ssim_all)),
            "lpips_mean": float(np.mean(lpips_all)),
            "lpips_std": float(np.std(lpips_all)),
            "orc_psnr_mean": float(np.mean(orc_psnr_all)),
            "orc_psnr_std": float(np.std(orc_psnr_all)),
            "orc_ssim_mean": float(np.mean(orc_ssim_all)),
            "orc_ssim_std": float(np.std(orc_ssim_all)),
            "orc_lpips_mean": float(np.mean(orc_lpips_all)),
            "orc_lpips_std": float(np.std(orc_lpips_all)),
            "psnr_gap": float(np.mean(orc_psnr_all) - np.mean(psnr_all)),
            "psnr_gap_std": float(np.std(gap_all)),
        }
        r = recon_results[B]
        print(
            f"  B={B:>2}: sigma_est={sigma_est:.4f} err={err:.4f} "
            f"PSNR={r['psnr_mean']:.2f} oracle={r['orc_psnr_mean']:.2f} gap={r['psnr_gap']:+.2f}"
        )

    save_json(os.path.join(output_dir, "estimation.json"), est_results)
    save_json(os.path.join(output_dir, "reconstruction.json"), recon_results)
    print(f"\nWrote {output_dir}/estimation.json and reconstruction.json")
    print(tt.summary())
    return {"estimation": est_results, "reconstruction": recon_results}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi image blur-SURE evaluation")
    p.add_argument("--b-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--n-trials", type=int, default=5)
    p.add_argument("--n-recon-images", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--lam-multiplier", type=float, default=10.0)
    p.add_argument("--pnp-steps", type=int, default=100)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(find_repo_root(), "results", "blind", "multi_image")
    run(
        b_values=args.b_values,
        n_trials=args.n_trials,
        n_recon_images=args.n_recon_images,
        n_seeds=args.n_seeds,
        lam_multiplier=args.lam_multiplier,
        pnp_steps=args.pnp_steps,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
