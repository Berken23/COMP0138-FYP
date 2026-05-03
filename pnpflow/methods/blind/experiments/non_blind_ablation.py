"""Non blind hyperparameter ablation.

Sweeps gamma schedule, number of trajectory steps, learning rate, number of
denoiser samples, outer iterations, and warm start, evaluated with the true
sigma so reconstruction quality is isolated from sigma estimation. Establishes
the oracle ceiling for each configuration.

Outputs are written to results/blind/non_blind_ablation/.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch

from .._helpers import SEED, TimingTracker, find_repo_root, load_model, save_json
from ..data import load_dataset
from ..evaluation import evaluate
from ..forward import gaussian_blur_fft
from ..reconstruction import pnp_flow_reconstruct


def run_eval(
    model,
    imgs: List[torch.Tensor],
    sigma_true: float,
    noise_std: float,
    *,
    num_steps: int = 100,
    lr: float = 1.0,
    gamma_style: str = "1_minus_t",
    alpha: float = 1.0,
    num_samples: int = 1,
    outer_iters: int = 1,
    warm_start: bool = False,
    max_imgs: int = 10,
) -> Dict[str, float]:
    sub = imgs[:max_imgs]
    metrics = {"psnr": [], "ssim": [], "lpips": []}
    for idx, x_gt in enumerate(sub):
        torch.manual_seed(SEED + idx)
        y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std
        x = y.clone()
        for _ in range(outer_iters):
            x = pnp_flow_reconstruct(
                model,
                y=y,
                sigma_blur=sigma_true,
                num_steps=num_steps,
                lr=lr,
                gamma_style=gamma_style,
                alpha=alpha,
                num_samples=num_samples,
                x_init=x if warm_start else y.clone(),
            )
        m = evaluate(x, x_gt)
        for k in metrics:
            metrics[k].append(m[k])
    return {f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()} | {
        f"{k}_std": float(np.std(v)) for k, v in metrics.items()
    }


def run(
    sigma_true: float,
    noise_std: float,
    max_imgs: int,
    output_dir: str,
    device: str,
):
    repo_root = find_repo_root()
    model = load_model(repo_root, device)

    datasets = {name: load_dataset(name, repo_root, device, num_celeba=max_imgs)
                for name in ["CelebA", "BSD68", "Set12"]}

    tt = TimingTracker()
    out: Dict[str, Dict] = {}

    GAMMA_SCHEDULES = ["1_minus_t", "sqrt_1_minus_t", "constant"]
    ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]
    NUM_STEPS_LIST = [25, 50, 100, 200]
    LR_LIST = [0.5, 1.0, 2.0]
    NUM_SAMPLES_LIST = [1, 3, 5]
    OUTER_ITERS_LIST = [1, 5, 10, 20, 30]
    WARM_START_OPTIONS = [False, True]

    print("Ablation 1: gamma schedules")
    out["gamma"] = {}
    for gamma in GAMMA_SCHEDULES:
        alphas = ALPHA_VALUES if gamma == "alpha_1_minus_t" else [1.0]
        for a in alphas:
            label = gamma if gamma != "alpha_1_minus_t" else f"alpha_1_minus_t(a={a})"
            out["gamma"][label] = {}
            for dname, imgs in datasets.items():
                with tt.track(f"gamma_{label}_{dname}"):
                    out["gamma"][label][dname] = run_eval(
                        model, imgs, sigma_true, noise_std,
                        gamma_style=gamma, alpha=a, max_imgs=max_imgs,
                    )
            psnrs = " | ".join(f"{d}={out['gamma'][label][d]['psnr_mean']:.2f}" for d in datasets)
            print(f"  {label:<28}: {psnrs}")

    print("\nAblation 2: number of steps")
    out["num_steps"] = {}
    for ns in NUM_STEPS_LIST:
        out["num_steps"][ns] = {}
        for dname, imgs in datasets.items():
            with tt.track(f"steps_{ns}_{dname}"):
                out["num_steps"][ns][dname] = run_eval(
                    model, imgs, sigma_true, noise_std,
                    num_steps=ns, max_imgs=max_imgs,
                )
        psnrs = " | ".join(f"{d}={out['num_steps'][ns][d]['psnr_mean']:.2f}" for d in datasets)
        print(f"  steps={ns:<5}: {psnrs}")

    print("\nAblation 3: learning rate")
    out["lr"] = {}
    for lr in LR_LIST:
        out["lr"][lr] = {}
        for dname, imgs in datasets.items():
            with tt.track(f"lr_{lr}_{dname}"):
                out["lr"][lr][dname] = run_eval(
                    model, imgs, sigma_true, noise_std,
                    lr=lr, max_imgs=max_imgs,
                )
        psnrs = " | ".join(f"{d}={out['lr'][lr][d]['psnr_mean']:.2f}" for d in datasets)
        print(f"  lr={lr:<5}: {psnrs}")

    print("\nAblation 4: number of denoiser samples")
    out["num_samples"] = {}
    for ns in NUM_SAMPLES_LIST:
        out["num_samples"][ns] = {}
        for dname, imgs in datasets.items():
            with tt.track(f"nsamples_{ns}_{dname}"):
                out["num_samples"][ns][dname] = run_eval(
                    model, imgs, sigma_true, noise_std,
                    num_samples=ns, max_imgs=max_imgs,
                )
        psnrs = " | ".join(f"{d}={out['num_samples'][ns][d]['psnr_mean']:.2f}" for d in datasets)
        print(f"  S={ns:<3}: {psnrs}")

    print("\nAblation 5: outer iterations")
    out["outer_iters"] = {}
    for oi in OUTER_ITERS_LIST:
        out["outer_iters"][oi] = {}
        for dname, imgs in datasets.items():
            with tt.track(f"oi_{oi}_{dname}"):
                out["outer_iters"][oi][dname] = run_eval(
                    model, imgs, sigma_true, noise_std,
                    outer_iters=oi, max_imgs=max_imgs,
                )
        psnrs = " | ".join(f"{d}={out['outer_iters'][oi][d]['psnr_mean']:.2f}" for d in datasets)
        print(f"  outer_iters={oi:<3}: {psnrs}")

    print("\nAblation 6: warm start")
    out["warm_start"] = {}
    for ws in WARM_START_OPTIONS:
        label = str(ws)
        out["warm_start"][label] = {}
        for dname, imgs in datasets.items():
            with tt.track(f"ws_{label}_{dname}"):
                out["warm_start"][label][dname] = run_eval(
                    model, imgs, sigma_true, noise_std,
                    warm_start=ws, outer_iters=5, max_imgs=max_imgs,
                )
        psnrs = " | ".join(f"{d}={out['warm_start'][label][d]['psnr_mean']:.2f}" for d in datasets)
        print(f"  warm_start={label:<5}: {psnrs}")

    save_json(os.path.join(output_dir, "ablation_results.json"), out)
    print(f"\nWrote {output_dir}/ablation_results.json")
    print(tt.summary())
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non blind hyperparameter ablation")
    p.add_argument("--sigma", type=float, default=1.5)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--max-imgs", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(find_repo_root(), "results", "blind", "non_blind_ablation")
    run(
        sigma_true=args.sigma,
        noise_std=args.noise_std,
        max_imgs=args.max_imgs,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
