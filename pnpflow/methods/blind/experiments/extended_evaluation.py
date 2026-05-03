"""Extended blur-SURE evaluation on BSD68 and Set12.

Mirrors the protocol used for CelebA in blur_sure_full.py with the same
sigma and noise grid. Lambda is fixed at 10*eta^2 following the
calibration result. Three reconstruction seeds per image, all three
metrics reported, oracle gap computed.

Outputs are written to results/blind/extended_evaluation/.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict

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
from ..data import load_bsd68, load_set12
from ..evaluation import evaluate, postprocess
from ..forward import gaussian_blur_fft
from ..reconstruction import pnp_flow_reconstruct
from ..sigma_estimation import estimate_sigma_blur_sure


QUAL_SIGMA = 1.5
QUAL_NOISE = 0.05
QUAL_NUM_IMAGES = 8


def run(
    n_seeds: int,
    pnp_steps: int,
    lam_multiplier: float,
    output_dir: str,
    device: str,
):
    repo_root = find_repo_root()
    model = load_model(repo_root, device)

    datasets = {
        "BSD68": load_bsd68(repo_root, device),
        "Set12": load_set12(repo_root, device),
    }
    for name, imgs in datasets.items():
        print(f"{name}: {len(imgs)} images")

    tt = TimingTracker()
    all_results: Dict = {}

    for dname, imgs in datasets.items():
        print(f"\n{'='*60}")
        print(f"DATASET: {dname}")
        print(f"{'='*60}")

        for sigma_true in SIGMA_VALUES:
            for noise_std in NOISE_LEVELS:
                noise_var = noise_std ** 2
                lam = noise_var * lam_multiplier
                cfg_key = f"{dname}/sigma={sigma_true}/noise={noise_std}"
                print(f"\n--- {cfg_key} ---")

                is_qual_cfg = (sigma_true == QUAL_SIGMA and noise_std == QUAL_NOISE)
                qual_dir = os.path.join(output_dir, "qualitative", dname)
                if is_qual_cfg:
                    os.makedirs(qual_dir, exist_ok=True)

                per_image = []
                for idx, x_gt in enumerate(imgs):
                    torch.manual_seed(SEED + idx)
                    y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std

                    with tt.track(f"{dname}_est"):
                        sure_res = estimate_sigma_blur_sure(y, noise_var, lam=lam)
                    sigma_est = sure_res["sigma_star"]

                    seed_m = {"psnr": [], "ssim": [], "lpips": []}
                    first_x_rec = None
                    for so in range(n_seeds):
                        torch.manual_seed(SEED + idx * 100 + so)
                        with tt.track(f"{dname}_recon"):
                            x_rec = pnp_flow_reconstruct(
                                model, y=y, sigma_blur=sigma_est, num_steps=pnp_steps,
                            )
                        m = evaluate(x_rec, x_gt)
                        for k in seed_m:
                            seed_m[k].append(m[k])
                        if so == 0:
                            first_x_rec = x_rec

                    torch.manual_seed(SEED + idx * 100)
                    x_orc = pnp_flow_reconstruct(
                        model, y=y, sigma_blur=sigma_true, num_steps=pnp_steps,
                    )
                    m_orc = evaluate(x_orc, x_gt)

                    if is_qual_cfg and idx < QUAL_NUM_IMAGES:
                        save_image(postprocess(x_gt), os.path.join(qual_dir, f"{idx:04d}_clean.png"))
                        save_image(postprocess(y), os.path.join(qual_dir, f"{idx:04d}_observed.png"))
                        save_image(postprocess(first_x_rec), os.path.join(qual_dir, f"{idx:04d}_blind.png"))
                        save_image(postprocess(x_orc), os.path.join(qual_dir, f"{idx:04d}_oracle.png"))

                    per_image.append({
                        "index": idx,
                        "sigma_est": sigma_est,
                        "sigma_error": abs(sigma_est - sigma_true),
                        "psnr_mean": float(np.mean(seed_m["psnr"])),
                        "psnr_std": float(np.std(seed_m["psnr"])),
                        "ssim_mean": float(np.mean(seed_m["ssim"])),
                        "lpips_mean": float(np.mean(seed_m["lpips"])),
                        "oracle_psnr": m_orc["psnr"],
                        "oracle_ssim": m_orc["ssim"],
                        "oracle_lpips": m_orc["lpips"],
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
    print(f"\nWrote {output_dir}/results.json")
    print(tt.summary())
    return all_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extended blur-SURE evaluation on BSD68 and Set12")
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--pnp-steps", type=int, default=100)
    p.add_argument("--lam-multiplier", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(find_repo_root(), "results", "blind", "extended_evaluation")
    run(
        n_seeds=args.n_seeds,
        pnp_steps=args.pnp_steps,
        lam_multiplier=args.lam_multiplier,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
