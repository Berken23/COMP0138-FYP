"""Trajectory straightness analysis.

For each combination of dataset and sigma, runs a tracked PnP-Flow
trajectory and reports the RMSE deviation from the straight line between
x_0 and x_T at every flow time. Demonstrates the rectified flow assumption
in the blind setting and is referenced in the methodology to justify the
linear interpolation approximation.

Outputs are written to results/blind/trajectory_straightness/.
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from torchvision.utils import save_image

from .._helpers import (
    NOISE_LEVELS,
    SEED,
    SIGMA_VALUES,
    find_repo_root,
    load_model,
    pnp_flow_trajectory_tracked,
    save_json,
)
from ..data import load_dataset
from ..evaluation import postprocess
from ..forward import gaussian_blur_fft


SNAPSHOT_T_INDICES = [0, 10, 25, 50, 75, 100]


def compute_deviation(trajectory: List[torch.Tensor], t_values: List[float]) -> List[float]:
    x_0 = trajectory[0]
    x_T = trajectory[-1]
    deviations = []
    for x_t, t in zip(trajectory, t_values):
        x_lin = (1.0 - t) * x_0 + t * x_T
        deviations.append(torch.sqrt(torch.mean((x_t - x_lin) ** 2)).item())
    return deviations


def run(
    n_imgs: int,
    num_steps: int,
    noise_std: float,
    output_dir: str,
    device: str,
):
    repo_root = find_repo_root()
    model = load_model(repo_root, device)
    datasets = {
        name: load_dataset(name, repo_root, device, num_celeba=n_imgs)
        for name in ["CelebA", "BSD68", "Set12"]
    }

    out = {"by_sigma": {}, "by_schedule": {}}
    schedules = ["1_minus_t", "sqrt_1_minus_t", "constant"]

    print("Experiment 1: deviation vs sigma (gamma=1_minus_t)")
    for dname, imgs in datasets.items():
        sub = imgs[:n_imgs]
        for sigma_true in SIGMA_VALUES:
            all_devs = []
            t_vals_ref = None
            for idx, x_gt in enumerate(sub):
                torch.manual_seed(SEED + idx)
                y = gaussian_blur_fft(x_gt, sigma_true) + torch.randn_like(x_gt) * noise_std
                _, traj, t_vals = pnp_flow_trajectory_tracked(
                    model, x_init=y.clone(), y=y, sigma_blur=sigma_true,
                    num_steps=num_steps, gamma_style="1_minus_t",
                )
                all_devs.append(compute_deviation(traj, t_vals))
                t_vals_ref = t_vals
            avg = np.mean(all_devs, axis=0)
            peak = int(np.argmax(avg))
            out["by_sigma"][f"{dname}/sigma={sigma_true}"] = {
                "avg_deviation": avg.tolist(),
                "t_values": t_vals_ref,
                "mean_deviation": float(np.mean(avg)),
                "peak_deviation": float(np.max(avg)),
                "peak_t": float(t_vals_ref[peak]),
            }
            print(f"  {dname:<8} sigma={sigma_true:>4.1f}: "
                  f"mean={np.mean(avg):.4f} peak={np.max(avg):.4f}@t={t_vals_ref[peak]:.2f}")

    print("\nExperiment 2: deviation vs schedule (sigma=1.5)")
    sigma_fixed = 1.5
    for dname, imgs in datasets.items():
        sub = imgs[:n_imgs]
        for gamma in schedules:
            all_devs = []
            t_vals_ref = None
            for idx, x_gt in enumerate(sub):
                torch.manual_seed(SEED + idx)
                y = gaussian_blur_fft(x_gt, sigma_fixed) + torch.randn_like(x_gt) * noise_std
                _, traj, t_vals = pnp_flow_trajectory_tracked(
                    model, x_init=y.clone(), y=y, sigma_blur=sigma_fixed,
                    num_steps=num_steps, gamma_style=gamma,
                )
                all_devs.append(compute_deviation(traj, t_vals))
                t_vals_ref = t_vals
            avg = np.mean(all_devs, axis=0)
            peak = int(np.argmax(avg))
            out["by_schedule"][f"{dname}/{gamma}"] = {
                "avg_deviation": avg.tolist(),
                "t_values": t_vals_ref,
                "mean_deviation": float(np.mean(avg)),
                "peak_deviation": float(np.max(avg)),
                "peak_t": float(t_vals_ref[peak]),
            }
            print(f"  {dname:<8} {gamma:<18}: "
                  f"mean={np.mean(avg):.4f} peak={np.max(avg):.4f}@t={t_vals_ref[peak]:.2f}")

    print("\nExperiment 3: visual snapshot grid (one image per dataset, sigma=1.5)")
    snap_dir = os.path.join(output_dir, "snapshots")
    snapshot_meta = []
    for dname, imgs in datasets.items():
        torch.manual_seed(SEED)
        x_gt = imgs[0]
        y = gaussian_blur_fft(x_gt, sigma_fixed) + torch.randn_like(x_gt) * noise_std
        _, traj, t_vals = pnp_flow_trajectory_tracked(
            model, x_init=y.clone(), y=y, sigma_blur=sigma_fixed,
            num_steps=num_steps, gamma_style="1_minus_t",
        )
        x_0 = traj[0]
        x_T = traj[-1]
        for k in SNAPSHOT_T_INDICES:
            if k >= len(traj):
                continue
            t = t_vals[k]
            x_t = traj[k]
            x_lin = (1.0 - t) * x_0 + t * x_T
            diff = (x_t - x_lin).abs()
            dev = float(torch.sqrt(torch.mean(diff ** 2)).item())

            os.makedirs(snap_dir, exist_ok=True)
            base = f"{dname}_k{k:03d}"
            save_image(postprocess(x_t), os.path.join(snap_dir, f"{base}_actual.png"))
            save_image(postprocess(x_lin), os.path.join(snap_dir, f"{base}_linear.png"))
            save_image(diff.mean(0, keepdim=True).clamp(0, 1), os.path.join(snap_dir, f"{base}_diff.png"))
            snapshot_meta.append({
                "dataset": dname,
                "k": k,
                "t": float(t),
                "deviation": dev,
            })
    save_json(os.path.join(output_dir, "snapshots.json"), snapshot_meta)
    print(f"Saved snapshots and metadata for {len(snapshot_meta)} (dataset, t) pairs")

    save_json(os.path.join(output_dir, "straightness.json"), out)
    print(f"\nWrote {output_dir}/straightness.json")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trajectory straightness analysis")
    p.add_argument("--n-imgs", type=int, default=5)
    p.add_argument("--num-steps", type=int, default=100)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(find_repo_root(), "results", "blind", "trajectory_straightness")
    run(
        n_imgs=args.n_imgs,
        num_steps=args.num_steps,
        noise_std=args.noise_std,
        output_dir=output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
