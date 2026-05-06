"""Reproduce every figure used in the thesis from the JSON outputs of the
experiment scripts.

Each plot family is implemented as a separate function so individual
figures can be regenerated without re running every experiment. The CLI
entry point detects which result files are present under
results/blind/ and renders only the corresponding figures.

Output figures land in results/blind/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .._helpers import NOISE_LEVELS, SIGMA_VALUES, find_repo_root


def _load(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_naive_joint_estimation(results_dir: str, fig_dir: str) -> None:
    per_image = _load(os.path.join(results_dir, "per_image.json"))
    summary = _load(os.path.join(results_dir, "summary.json"))
    sigma_true = summary["sigma_true"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for r in per_image:
        ax.plot(r["sigma_history_outer"], alpha=0.7, label=f"img {r['index']}")
    ax.axhline(sigma_true, color="red", linestyle="--", linewidth=2, label=f"sigma_true={sigma_true}")
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("sigma")
    ax.set_title("Sigma trajectory (naive joint estimation)")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

    ax = axes[1]
    errs = [r["sigma_error"] for r in per_image]
    ax.bar(range(len(errs)), errs, alpha=0.7)
    ax.axhline(np.mean(errs), color="red", linestyle="--", label=f"mean={np.mean(errs):.3f}")
    ax.set_xlabel("Image")
    ax.set_ylabel("|sigma error|")
    ax.set_title("Sigma error per image")
    ax.legend()

    ax = axes[2]
    psnrs = [r["psnr"] for r in per_image]
    ax.bar(range(len(psnrs)), psnrs, alpha=0.7)
    ax.axhline(np.mean(psnrs), color="red", linestyle="--", label=f"mean={np.mean(psnrs):.2f}")
    ax.set_xlabel("Image")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR per image")
    ax.legend()

    plt.suptitle("Naive joint estimation (A.14 per step Adam)", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "naive_joint_estimation.png"))

    # Iteration progression grid
    prog_meta_path = os.path.join(results_dir, "iteration_progression.json")
    prog_dir = os.path.join(results_dir, "iteration_progression")
    if os.path.exists(prog_meta_path) and os.path.isdir(prog_dir):
        meta = _load(prog_meta_path)
        by_image: Dict[int, List[Dict]] = {}
        for m in meta:
            by_image.setdefault(m["image"], []).append(m)
        for img_idx, entries in sorted(by_image.items()):
            entries = sorted(entries, key=lambda e: e["outer"])
            outers = [e["outer"] for e in entries]
            n_cols = 2 + len(outers)
            fig, axes = plt.subplots(1, n_cols, figsize=(2.6 * n_cols, 3))
            clean_path = os.path.join(prog_dir, f"img{img_idx}_clean.png")
            obs_path = os.path.join(prog_dir, f"img{img_idx}_observed.png")
            if os.path.exists(clean_path):
                axes[0].imshow(mpimg.imread(clean_path))
            axes[0].set_title("GT", fontsize=8)
            axes[0].axis("off")
            if os.path.exists(obs_path):
                axes[1].imshow(mpimg.imread(obs_path))
            axes[1].set_title("blurred", fontsize=8)
            axes[1].axis("off")
            for j, e in enumerate(entries):
                snap_path = os.path.join(prog_dir, f"img{img_idx}_outer{e['outer']:03d}.png")
                if os.path.exists(snap_path):
                    axes[2 + j].imshow(mpimg.imread(snap_path))
                axes[2 + j].set_title(f"iter {e['outer']}\ns={e['sigma']:.2f}", fontsize=7)
                axes[2 + j].axis("off")
            plt.suptitle(
                f"Image {img_idx}: naive joint estimation progression (sigma drifts {summary['sigma_init']}->{entries[-1]['sigma']:.2f})",
                fontsize=10,
            )
            plt.tight_layout()
            _save(fig, os.path.join(fig_dir, f"naive_joint_progression_img{img_idx}.png"))


def plot_trajectory_straightness(results_dir: str, fig_dir: str) -> None:
    data = _load(os.path.join(results_dir, "straightness.json"))
    by_sigma = data["by_sigma"]
    by_schedule = data["by_schedule"]

    datasets = sorted({k.split("/")[0] for k in by_sigma})

    # Deviation vs t per dataset (overlay sigma)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for di, dname in enumerate(datasets):
        ax = axes[0][di]
        for sigma in SIGMA_VALUES:
            key = f"{dname}/sigma={sigma}"
            if key not in by_sigma:
                continue
            r = by_sigma[key]
            ax.plot(r["t_values"], r["avg_deviation"], linewidth=1.5, label=f"sigma={sigma}")
        ax.set_xlabel("flow time t")
        ax.set_ylabel("RMSE deviation")
        ax.set_title(f"{dname}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Trajectory deviation by sigma", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "straightness_by_sigma.png"))

    # Deviation vs t per dataset (overlay schedule)
    schedules = sorted({k.split("/")[1] for k in by_schedule})
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)
    for di, dname in enumerate(datasets):
        ax = axes[0][di]
        for gamma in schedules:
            key = f"{dname}/{gamma}"
            if key not in by_schedule:
                continue
            r = by_schedule[key]
            ax.plot(r["t_values"], r["avg_deviation"], linewidth=1.5, label=gamma)
        ax.set_xlabel("flow time t")
        ax.set_ylabel("RMSE deviation")
        ax.set_title(f"{dname}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Trajectory deviation by schedule (sigma=1.5)", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "straightness_by_schedule.png"))

    # Mean deviation heatmap dataset x sigma
    matrix = np.zeros((len(datasets), len(SIGMA_VALUES)))
    for di, dname in enumerate(datasets):
        for si, sigma in enumerate(SIGMA_VALUES):
            key = f"{dname}/sigma={sigma}"
            if key in by_sigma:
                matrix[di, si] = by_sigma[key]["mean_deviation"]
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(SIGMA_VALUES)))
    ax.set_xticklabels([str(s) for s in SIGMA_VALUES])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel("sigma_true")
    ax.set_ylabel("dataset")
    ax.set_title("Mean trajectory deviation (gamma=1_minus_t)")
    for di in range(len(datasets)):
        for si in range(len(SIGMA_VALUES)):
            ax.text(si, di, f"{matrix[di, si]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "straightness_heatmap.png"))

    # Bar chart: mean deviation by schedule, grouped by dataset
    schedules_set = sorted({k.split("/")[1] for k in by_schedule})
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(schedules_set))
    width = 0.8 / max(1, len(datasets))
    for di, dname in enumerate(datasets):
        vals = [by_schedule.get(f"{dname}/{g}", {}).get("mean_deviation", 0.0) for g in schedules_set]
        ax.bar(x_pos + di * width, vals, width, label=dname, alpha=0.8)
    ax.set_xticks(x_pos + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(schedules_set, fontsize=9)
    ax.set_ylabel("mean deviation")
    ax.set_title("Mean trajectory deviation by schedule (sigma=1.5)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "straightness_schedule_bar.png"))

    # Visual snapshot grid (one row per dataset)
    snap_meta_path = os.path.join(results_dir, "snapshots.json")
    snap_dir = os.path.join(results_dir, "snapshots")
    if os.path.exists(snap_meta_path) and os.path.isdir(snap_dir):
        meta = _load(snap_meta_path)
        per_dataset: Dict[str, List[Dict]] = {}
        for m in meta:
            per_dataset.setdefault(m["dataset"], []).append(m)
        for dname, entries in per_dataset.items():
            entries = sorted(entries, key=lambda e: e["k"])
            n = len(entries)
            if n == 0:
                continue
            fig, axes = plt.subplots(3, n, figsize=(3.0 * n, 9))
            if n == 1:
                axes = axes.reshape(3, 1)
            for j, e in enumerate(entries):
                base = f"{dname}_k{e['k']:03d}"
                actual_path = os.path.join(snap_dir, f"{base}_actual.png")
                linear_path = os.path.join(snap_dir, f"{base}_linear.png")
                diff_path = os.path.join(snap_dir, f"{base}_diff.png")
                if os.path.exists(actual_path):
                    axes[0, j].imshow(mpimg.imread(actual_path))
                axes[0, j].set_title(f"t={e['t']:.2f}\nactual", fontsize=8)
                axes[0, j].axis("off")
                if os.path.exists(linear_path):
                    axes[1, j].imshow(mpimg.imread(linear_path))
                axes[1, j].set_title("straight line", fontsize=8)
                axes[1, j].axis("off")
                if os.path.exists(diff_path):
                    axes[2, j].imshow(mpimg.imread(diff_path), cmap="hot")
                axes[2, j].set_title(f"dev={e['deviation']:.4f}", fontsize=8)
                axes[2, j].axis("off")
            plt.suptitle(f"{dname}: trajectory vs straight line (sigma=1.5)", fontsize=12)
            plt.tight_layout()
            _save(fig, os.path.join(fig_dir, f"straightness_snapshots_{dname}.png"))


def plot_blur_sure_full(results_dir: str, fig_dir: str) -> None:
    results = _load(os.path.join(results_dir, "results.json"))

    # Build matrices indexed by sigma x noise
    err_matrix = np.zeros((len(SIGMA_VALUES), len(NOISE_LEVELS)))
    gap_matrix = np.zeros_like(err_matrix)
    for si, sigma in enumerate(SIGMA_VALUES):
        for ni, noise in enumerate(NOISE_LEVELS):
            key = f"sigma={sigma}/noise={noise}"
            if key not in results:
                continue
            err_matrix[si, ni] = results[key]["sigma_error_mean"]
            gap_matrix[si, ni] = results[key]["psnr_gap"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, matrix, title, fmt in [
        (axes[0], err_matrix, "Sigma estimation error", ".3f"),
        (axes[1], gap_matrix, "PSNR gap (oracle - blind)", ".2f"),
    ]:
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels([str(n) for n in NOISE_LEVELS])
        ax.set_yticks(range(len(SIGMA_VALUES)))
        ax.set_yticklabels([str(s) for s in SIGMA_VALUES])
        ax.set_xlabel("noise_std")
        ax.set_ylabel("sigma_true")
        ax.set_title(title)
        for si in range(len(SIGMA_VALUES)):
            for ni in range(len(NOISE_LEVELS)):
                ax.text(ni, si, format(matrix[si, ni], fmt), ha="center", va="center",
                        fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax)
    plt.suptitle("Blur-SURE on CelebA: estimation error and reconstruction gap", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "blur_sure_heatmaps.png"))

    # SURE curves
    sure_path = os.path.join(results_dir, "sure_curves.json")
    if os.path.exists(sure_path):
        curves = _load(sure_path)
        if curves:
            fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4), squeeze=False)
            for j, c in enumerate(curves):
                ax = axes[0][j]
                ax.plot(c["sigma_grid"], c["sure_curve"], "b-", linewidth=1.5)
                ax.axvline(c["sigma_star"], color="green", linestyle="--",
                           label=f"est={c['sigma_star']:.3f}")
                ax.axvline(c["sigma_true"], color="red", linestyle=":",
                           label=f"true={c['sigma_true']}")
                ax.set_xlabel("sigma")
                ax.set_ylabel("blur-SURE")
                ax.set_title(f"Image {c['index']}")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            plt.suptitle("SURE curves at sigma=1.5, noise=0.05", fontsize=12)
            plt.tight_layout()
            _save(fig, os.path.join(fig_dir, "sure_curves.png"))

    # Per image sigma estimate scatter at sigma=1.5, noise=0.05
    key = "sigma=1.5/noise=0.05"
    if key in results:
        per = results[key]["per_image"]
        ests = [p["sigma_est"] for p in per]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.scatter(range(len(ests)), ests, s=15, alpha=0.6)
        ax.axhline(1.5, color="red", linestyle="--", linewidth=2, label="sigma_true=1.5")
        ax.set_xlabel("image index")
        ax.set_ylabel("estimated sigma")
        ax.set_title("Per image sigma estimates (sigma=1.5, noise=0.05)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, os.path.join(fig_dir, "sigma_estimates_scatter.png"))

        # PSNR errorbar
        psnr_means = [p["psnr_mean"] for p in per]
        psnr_stds = [p["psnr_std"] for p in per]
        oracle_psnrs = [p["oracle_psnr"] for p in per]
        fig, ax = plt.subplots(figsize=(12, 4))
        x_idx = list(range(len(psnr_means)))
        ax.errorbar(x_idx, psnr_means, yerr=psnr_stds, fmt="o", markersize=3, capsize=2,
                    label="Blur-SURE", alpha=0.7)
        ax.scatter(x_idx, oracle_psnrs, s=10, marker="x", color="red", alpha=0.5, label="Oracle")
        ax.set_xlabel("image index")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title("Per image PSNR with reconstruction confidence (sigma=1.5, noise=0.05)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, os.path.join(fig_dir, "psnr_per_image.png"))

    # Visual comparison grid composed from saved quadruplets
    qual_dir = os.path.join(results_dir, "qualitative")
    if os.path.isdir(qual_dir):
        files = sorted(f for f in os.listdir(qual_dir) if f.endswith("_clean.png"))
        n_show = min(4, len(files))
        if n_show > 0:
            fig, axes = plt.subplots(n_show, 4, figsize=(14, 3.2 * n_show), squeeze=False)
            for row, clean_file in enumerate(files[:n_show]):
                base = clean_file.replace("_clean.png", "")
                paths = [
                    (os.path.join(qual_dir, f"{base}_clean.png"), "Ground truth"),
                    (os.path.join(qual_dir, f"{base}_observed.png"), "Blurred + noisy"),
                    (os.path.join(qual_dir, f"{base}_oracle.png"), "Oracle"),
                    (os.path.join(qual_dir, f"{base}_blind.png"), "Blur-SURE"),
                ]
                for col, (path, title) in enumerate(paths):
                    if os.path.exists(path):
                        axes[row, col].imshow(mpimg.imread(path))
                    axes[row, col].set_title(title, fontsize=9)
                    axes[row, col].axis("off")
            plt.suptitle("Blur-SURE vs Oracle (sigma=1.5, noise=0.05)", fontsize=12)
            plt.tight_layout()
            _save(fig, os.path.join(fig_dir, "qualitative_grid.png"))


def plot_multi_image(results_dir: str, fig_dir: str) -> None:
    est = _load(os.path.join(results_dir, "estimation.json"))
    recon = _load(os.path.join(results_dir, "reconstruction.json"))

    b_values = sorted({int(k.split("B=")[1]) for k in est})

    # Sigma error vs B per noise level (averaged over sigma)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    for noise in NOISE_LEVELS:
        means = []
        stds = []
        for B in b_values:
            errs: List[float] = []
            for sigma in SIGMA_VALUES:
                k = f"sigma={sigma}/noise={noise}/B={B}"
                if k in est:
                    errs.extend(est[k]["sigma_errors"])
            means.append(float(np.mean(errs)) if errs else 0.0)
            stds.append(float(np.std(errs)) if errs else 0.0)
        ax.errorbar(b_values, means, yerr=stds, marker="o", capsize=4, linewidth=2,
                    label=f"noise={noise}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b_values)
    ax.set_xticklabels([str(b) for b in b_values])
    ax.set_xlabel("B")
    ax.set_ylabel("sigma error")
    ax.set_title("Sigma error vs B (averaged over sigma)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Sigma error vs B per sigma at noise=0.05
    ax = axes[0, 1]
    for sigma in SIGMA_VALUES:
        means = []
        stds = []
        for B in b_values:
            k = f"sigma={sigma}/noise=0.05/B={B}"
            if k in est:
                means.append(est[k]["mean_error"])
                stds.append(est[k]["std_error"])
            else:
                means.append(0.0)
                stds.append(0.0)
        ax.errorbar(b_values, means, yerr=stds, marker="o", capsize=4, linewidth=2,
                    label=f"sigma={sigma}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b_values)
    ax.set_xticklabels([str(b) for b in b_values])
    ax.set_xlabel("B")
    ax.set_ylabel("sigma error")
    ax.set_title("Sigma error vs B (noise=0.05)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # PSNR vs B
    ax = axes[1, 0]
    psnrs = [recon[str(B)]["psnr_mean"] for B in b_values if str(B) in recon]
    oracle = [recon[str(B)]["orc_psnr_mean"] for B in b_values if str(B) in recon]
    ax.plot(b_values, psnrs, "bo-", linewidth=2, markersize=6, label="Blur-SURE")
    ax.plot(b_values, oracle, "r--", linewidth=2, label="Oracle")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b_values)
    ax.set_xticklabels([str(b) for b in b_values])
    ax.set_xlabel("B")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs B (sigma=1.5, noise=0.05)")
    ax.legend()
    ax.grid(alpha=0.3)

    # PSNR gap vs B
    ax = axes[1, 1]
    gaps = [recon[str(B)]["psnr_gap"] for B in b_values if str(B) in recon]
    ax.bar([str(B) for B in b_values], gaps, alpha=0.7)
    ax.set_xlabel("B")
    ax.set_ylabel("PSNR gap (dB)")
    ax.set_title("Oracle minus blind gap vs B")
    ax.grid(alpha=0.3, axis="y")
    for i, (b, g) in enumerate(zip(b_values, gaps)):
        ax.text(i, g + 0.005, f"{g:.3f}", ha="center", fontsize=8)

    plt.suptitle("Multi image blur-SURE", fontsize=14)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "multi_image.png"))

    # Std vs B per sigma (averaged over noise)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in SIGMA_VALUES:
        stds = []
        for B in b_values:
            errs: List[float] = []
            for noise in NOISE_LEVELS:
                k = f"sigma={sigma}/noise={noise}/B={B}"
                if k in est:
                    errs.extend(est[k]["sigma_errors"])
            stds.append(float(np.std(errs)) if errs else 0.0)
        ax.plot(b_values, stds, "o-", linewidth=2, markersize=5, label=f"sigma={sigma}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(b_values)
    ax.set_xticklabels([str(b) for b in b_values])
    ax.set_xlabel("B")
    ax.set_ylabel("std of sigma error")
    ax.set_title("Estimation consistency vs B")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "multi_image_std_vs_B.png"))

    # Sigma estimates scatter at sigma=1.5, noise=0.05 across B
    fig, ax = plt.subplots(figsize=(10, 5))
    rng = np.random.default_rng(0)
    for bi, B in enumerate(b_values):
        k = f"sigma=1.5/noise=0.05/B={B}"
        if k not in est:
            continue
        ests = est[k]["sigma_estimates"]
        jitter = rng.uniform(-0.15, 0.15, size=len(ests))
        ax.scatter([bi + j for j in jitter], ests, alpha=0.6, s=30)
        ax.plot(bi, float(np.mean(ests)), "kx", markersize=10, markeredgewidth=2)
    ax.axhline(1.5, color="red", linestyle="--", linewidth=2, label="sigma_true=1.5")
    ax.set_xticks(range(len(b_values)))
    ax.set_xticklabels([str(b) for b in b_values])
    ax.set_xlabel("B")
    ax.set_ylabel("sigma_est")
    ax.set_title("Sigma estimates at sigma=1.5, noise=0.05")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "multi_image_estimates_scatter.png"))

    # B=1 vs B=max heatmap
    bmax = max(b_values)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, B in zip(axes, [1, bmax]):
        matrix = np.zeros((len(SIGMA_VALUES), len(NOISE_LEVELS)))
        for si, sigma in enumerate(SIGMA_VALUES):
            for ni, noise in enumerate(NOISE_LEVELS):
                k = f"sigma={sigma}/noise={noise}/B={B}"
                if k in est:
                    matrix[si, ni] = est[k]["mean_error"]
        vmax = max(0.5, matrix.max())
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(NOISE_LEVELS)))
        ax.set_xticklabels([str(n) for n in NOISE_LEVELS])
        ax.set_yticks(range(len(SIGMA_VALUES)))
        ax.set_yticklabels([str(s) for s in SIGMA_VALUES])
        ax.set_xlabel("noise")
        ax.set_ylabel("sigma")
        ax.set_title(f"B={B}")
        for si in range(len(SIGMA_VALUES)):
            for ni in range(len(NOISE_LEVELS)):
                ax.text(ni, si, f"{matrix[si, ni]:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold")
        plt.colorbar(im, ax=ax)
    plt.suptitle(f"Sigma error: B=1 vs B={bmax}", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "multi_image_heatmap.png"))


def plot_extended_evaluation(results_dir: str, fig_dir: str) -> None:
    results = _load(os.path.join(results_dir, "results.json"))

    # Optionally include CelebA results from blur_sure_full for cross-dataset plots
    celeba_results = None
    celeba_path = os.path.join(os.path.dirname(results_dir), "blur_sure_full", "results.json")
    if os.path.exists(celeba_path):
        celeba_results = _load(celeba_path)

    datasets = sorted({k.split("/")[0] for k in results})
    fig, axes = plt.subplots(2, len(datasets), figsize=(7 * len(datasets), 9), squeeze=False)
    for di, dname in enumerate(datasets):
        err_matrix = np.zeros((len(SIGMA_VALUES), len(NOISE_LEVELS)))
        gap_matrix = np.zeros_like(err_matrix)
        for si, sigma in enumerate(SIGMA_VALUES):
            for ni, noise in enumerate(NOISE_LEVELS):
                key = f"{dname}/sigma={sigma}/noise={noise}"
                if key in results:
                    err_matrix[si, ni] = results[key]["sigma_error_mean"]
                    gap_matrix[si, ni] = results[key]["psnr_gap"]

        for ax, matrix, title, fmt in [
            (axes[0][di], err_matrix, f"{dname} sigma error", ".3f"),
            (axes[1][di], gap_matrix, f"{dname} PSNR gap", ".2f"),
        ]:
            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
            ax.set_xticks(range(len(NOISE_LEVELS)))
            ax.set_xticklabels([str(n) for n in NOISE_LEVELS])
            ax.set_yticks(range(len(SIGMA_VALUES)))
            ax.set_yticklabels([str(s) for s in SIGMA_VALUES])
            ax.set_xlabel("noise_std")
            ax.set_ylabel("sigma_true")
            ax.set_title(title)
            for si in range(len(SIGMA_VALUES)):
                for ni in range(len(NOISE_LEVELS)):
                    ax.text(ni, si, format(matrix[si, ni], fmt), ha="center", va="center",
                            fontsize=8, fontweight="bold")
            plt.colorbar(im, ax=ax)
    plt.suptitle("Extended evaluation: BSD68 and Set12", fontsize=14)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "extended_evaluation.png"))

    # Cross dataset comparison line plots (PSNR gap and sigma error per noise)
    cross_datasets = list(datasets)
    if celeba_results is not None:
        cross_datasets = ["CelebA"] + cross_datasets

    def _cross_value(dname: str, sigma: float, noise: float, field: str) -> float:
        if dname == "CelebA" and celeba_results is not None:
            r = celeba_results.get(f"sigma={sigma}/noise={noise}", {})
            return float(r.get(field, np.nan))
        r = results.get(f"{dname}/sigma={sigma}/noise={noise}", {})
        return float(r.get(field, np.nan))

    fig, axes = plt.subplots(1, len(NOISE_LEVELS), figsize=(6 * len(NOISE_LEVELS), 5), squeeze=False)
    for ni, noise in enumerate(NOISE_LEVELS):
        ax = axes[0][ni]
        for dname in cross_datasets:
            gaps = [_cross_value(dname, s, noise, "psnr_gap") for s in SIGMA_VALUES]
            ax.plot(SIGMA_VALUES, gaps, "o-", linewidth=2, markersize=6, label=dname)
        ax.axhline(0, color="black", linestyle="-", alpha=0.2)
        ax.set_xlabel("sigma_true")
        ax.set_ylabel("PSNR gap (dB)")
        ax.set_title(f"noise={noise}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("PSNR gap across datasets", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "cross_dataset_psnr_gap.png"))

    fig, axes = plt.subplots(1, len(NOISE_LEVELS), figsize=(6 * len(NOISE_LEVELS), 5), squeeze=False)
    for ni, noise in enumerate(NOISE_LEVELS):
        ax = axes[0][ni]
        for dname in cross_datasets:
            errs = [_cross_value(dname, s, noise, "sigma_error_mean") for s in SIGMA_VALUES]
            ax.plot(SIGMA_VALUES, errs, "o-", linewidth=2, markersize=6, label=dname)
        ax.set_xlabel("sigma_true")
        ax.set_ylabel("sigma error")
        ax.set_title(f"noise={noise}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle("Sigma error across datasets", fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "cross_dataset_sigma_error.png"))


def plot_convergence(results_root: str, fig_dir: str) -> None:
    """Render convergence diagnostics for the Krishnan baseline, naive
    joint estimation, and PnP-Flow reconstruction. Shows L2 error,
    objective, sigma error, and where applicable gradient norm versus
    iteration on semilog axes.
    """
    panels: List[Dict] = []

    naive_per_image = os.path.join(results_root, "naive_joint_estimation", "per_image.json")
    if os.path.exists(naive_per_image):
        records = _load(naive_per_image)
        if records and "objective_history" in records[0]:
            panels.append({
                "name": "Naive joint estimation",
                "records": records,
                "xlabel": "outer iteration",
            })

    krishnan_glob = os.path.join(results_root, "krishnan_baseline")
    if os.path.isdir(krishnan_glob):
        for dataset in os.listdir(krishnan_glob):
            ds_dir = os.path.join(krishnan_glob, dataset)
            if not os.path.isdir(ds_dir):
                continue
            for cfg in os.listdir(ds_dir):
                cfg_dir = os.path.join(ds_dir, cfg)
                per_image_path = os.path.join(cfg_dir, "per_image.json")
                if not os.path.exists(per_image_path):
                    continue
                records = _load(per_image_path)
                if not records or "convergence" not in records[0]:
                    continue
                # Flatten to the structure expected by the plotter below.
                flat = []
                for r in records:
                    h = r["convergence"]
                    flat.append({
                        "objective_history": h.get("objective_history", []),
                        "x_l2_error_history": h.get("x_l2_error_history", []),
                        "sigma_error_history": h.get("sigma_error_history", []),
                        "sigma_grad_norm_history": h.get("sigma_grad_norm_history", []),
                    })
                panels.append({
                    "name": f"Krishnan {dataset} {cfg}",
                    "records": flat,
                    "xlabel": "outer iteration",
                })

    if not panels:
        return

    for panel in panels:
        records = panel["records"]
        # Average across images.
        max_len = max(
            len(r.get("objective_history", [])) for r in records if r.get("objective_history")
        )
        if max_len == 0:
            continue

        def avg_curve(field: str):
            curves = [r.get(field, []) for r in records if r.get(field)]
            if not curves:
                return None
            curves = [c[:max_len] + [c[-1]] * (max_len - len(c)) for c in curves]
            return np.mean(np.array(curves), axis=0)

        obj = avg_curve("objective_history")
        l2 = avg_curve("x_l2_error_history")
        sig_err = avg_curve("sigma_error_history")
        grad_n = avg_curve("sigma_grad_norm_history")

        n_subpanels = sum(c is not None for c in [obj, l2, sig_err, grad_n])
        fig, axes = plt.subplots(1, n_subpanels, figsize=(5 * n_subpanels, 4), squeeze=False)
        col = 0
        if obj is not None:
            ax = axes[0][col]
            ax.semilogy(range(1, len(obj) + 1), obj, "b-o", markersize=4)
            ax.set_xlabel(panel["xlabel"])
            ax.set_ylabel("objective ||H_sigma(x) - y||^2 (log scale)")
            ax.set_title("Objective convergence")
            ax.grid(alpha=0.3, which="both")
            col += 1
        if l2 is not None:
            ax = axes[0][col]
            ax.semilogy(range(1, len(l2) + 1), l2, "g-s", markersize=4)
            ax.set_xlabel(panel["xlabel"])
            ax.set_ylabel("||x - x_clean||_2 (log scale)")
            ax.set_title("Image error vs ground truth")
            ax.grid(alpha=0.3, which="both")
            col += 1
        if sig_err is not None:
            ax = axes[0][col]
            sig_err_safe = np.maximum(sig_err, 1e-6)
            ax.semilogy(range(1, len(sig_err) + 1), sig_err_safe, "r-^", markersize=4)
            ax.set_xlabel(panel["xlabel"])
            ax.set_ylabel("|sigma_est - sigma_true| (log scale)")
            ax.set_title("Sigma error vs ground truth")
            ax.grid(alpha=0.3, which="both")
            col += 1
        if grad_n is not None:
            ax = axes[0][col]
            grad_safe = np.maximum(grad_n, 1e-12)
            ax.semilogy(range(1, len(grad_n) + 1), grad_safe, "m-d", markersize=4)
            ax.set_xlabel(panel["xlabel"])
            ax.set_ylabel("|grad sigma| (log scale)")
            ax.set_title("Sigma gradient norm")
            ax.grid(alpha=0.3, which="both")
            col += 1
        plt.suptitle(panel["name"], fontsize=13)
        plt.tight_layout()
        safe_name = panel["name"].replace(" ", "_").replace("/", "_")
        _save(fig, os.path.join(fig_dir, f"convergence_{safe_name}.png"))

    # Also render PnP-Flow reconstruction trajectory convergence if present.
    bsf_conv_path = os.path.join(results_root, "blur_sure_full", "convergence.json")
    if os.path.exists(bsf_conv_path):
        conv = _load(bsf_conv_path)
        if conv:
            blind = [c for c in conv if c.get("kind") == "blind"]
            oracle = [c for c in conv if c.get("kind") == "oracle"]

            def avg_history(items: List[Dict], field: str):
                if not items:
                    return None
                curves = [it["history"].get(field, []) for it in items if it.get("history")]
                if not curves:
                    return None
                m = max(len(c) for c in curves)
                curves = [c + [c[-1]] * (m - len(c)) for c in curves]
                return np.mean(np.array(curves), axis=0)

            fields = [
                ("psnr_history", "PSNR (dB)", "linear"),
                ("x_l2_error_history", "||x_t - x_clean||_2", "log"),
                ("objective_history", "||H_sigma(x_t) - y||^2", "log"),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(16, 4), squeeze=False)
            for ci, (field, ylabel, scale) in enumerate(fields):
                ax = axes[0][ci]
                blind_curve = avg_history(blind, field)
                oracle_curve = avg_history(oracle, field)
                if blind_curve is not None:
                    if scale == "log":
                        ax.semilogy(range(len(blind_curve)), np.maximum(blind_curve, 1e-12), "b-", label="blind (blur-SURE sigma)")
                    else:
                        ax.plot(range(len(blind_curve)), blind_curve, "b-", label="blind (blur-SURE sigma)")
                if oracle_curve is not None:
                    if scale == "log":
                        ax.semilogy(range(len(oracle_curve)), np.maximum(oracle_curve, 1e-12), "r--", label="oracle (true sigma)")
                    else:
                        ax.plot(range(len(oracle_curve)), oracle_curve, "r--", label="oracle (true sigma)")
                ax.set_xlabel("PnP-Flow trajectory step")
                ax.set_ylabel(ylabel)
                ax.set_title(field.replace("_history", ""))
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3, which="both")
            plt.suptitle("PnP-Flow reconstruction convergence (blur-SURE sigma vs oracle)", fontsize=13)
            plt.tight_layout()
            _save(fig, os.path.join(fig_dir, "convergence_pnp_flow.png"))


def plot_failed_approaches(results_dir: str, fig_dir: str) -> None:
    """Render a sigma trajectory grid plus a comparative bar chart of final
    sigma errors for every failed approach with available results."""
    files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f != "summary.json"]
    if not files:
        return

    runs: Dict[str, Dict] = {}
    for f in files:
        name = f.replace(".json", "")
        runs[name] = _load(os.path.join(results_dir, f))

    # Sigma trajectory grid: one panel per approach, average over images
    n = len(runs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4 * rows), squeeze=False)
    for ax_idx, (name, run) in enumerate(sorted(runs.items())):
        ax = axes[ax_idx // cols][ax_idx % cols]
        per_image = run.get("per_image", [])
        for r in per_image:
            history = r.get("sigma_history", [])
            if history:
                ax.plot(history, alpha=0.6)
        ax.axhline(1.5, color="red", linestyle="--", linewidth=1.5, label="sigma_true=1.5")
        ax.set_xlabel("iteration")
        ax.set_ylabel("sigma")
        ax.set_title(name)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for k in range(len(runs), rows * cols):
        axes[k // cols][k % cols].axis("off")
    plt.suptitle("Failed approaches: sigma trajectories", fontsize=14)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "failed_sigma_trajectories.png"))

    # Bar chart of final sigma errors
    names = sorted(runs.keys())
    errors = [runs[n].get("sigma_error_mean", float("nan")) for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, errors, color="steelblue", alpha=0.8)
    ax.set_ylabel("mean sigma error")
    ax.set_title("Final sigma error by failed approach")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(errors):
        if v == v:
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "failed_sigma_errors.png"))


def plot_non_blind_ablation(results_dir: str, fig_dir: str) -> None:
    data = _load(os.path.join(results_dir, "ablation_results.json"))
    datasets_present = set()
    for group in data.values():
        for cfg in group.values():
            datasets_present.update(cfg.keys())
    datasets = sorted(datasets_present)

    panels = [k for k in ["gamma", "num_steps", "lr", "num_samples", "outer_iters", "warm_start"]
              if k in data]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax_idx, panel in enumerate(panels):
        ax = axes[ax_idx]
        labels = list(data[panel].keys())
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(datasets))
        for di, dname in enumerate(datasets):
            psnrs = [data[panel][label].get(dname, {}).get("psnr_mean", 0.0) for label in labels]
            ax.bar(x + di * width, psnrs, width, label=dname, alpha=0.8)
        ax.set_xticks(x + width * (len(datasets) - 1) / 2)
        ax.set_xticklabels([str(label) for label in labels], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(panel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    for k in range(len(panels), len(axes)):
        axes[k].axis("off")
    plt.suptitle("Non blind ablation (true sigma)", fontsize=14)
    plt.tight_layout()
    _save(fig, os.path.join(fig_dir, "non_blind_ablation.png"))


def main() -> None:
    p = argparse.ArgumentParser(description="Render figures from experiment JSON outputs")
    p.add_argument("--results-root", default=None,
                   help="Defaults to <repo_root>/results/blind/")
    p.add_argument("--fig-dir", default=None,
                   help="Defaults to <results_root>/figures/")
    args = p.parse_args()

    repo_root = find_repo_root()
    results_root = args.results_root or os.path.join(repo_root, "results", "blind")
    fig_dir = args.fig_dir or os.path.join(results_root, "figures")
    _ensure_dir(fig_dir)

    # Convergence plots aggregate across multiple result directories.
    plot_convergence(results_root, fig_dir)

    plotters = [
        ("naive_joint_estimation", plot_naive_joint_estimation),
        ("non_blind_ablation", plot_non_blind_ablation),
        ("trajectory_straightness", plot_trajectory_straightness),
        ("failed_approaches", plot_failed_approaches),
        ("blur_sure_full", plot_blur_sure_full),
        ("multi_image", plot_multi_image),
        ("extended_evaluation", plot_extended_evaluation),
    ]
    for name, fn in plotters:
        path = os.path.join(results_root, name)
        if not os.path.isdir(path):
            print(f"Skipping {name}, no results directory at {path}")
            continue
        print(f"Rendering {name} from {path}")
        try:
            fn(path, fig_dir)
        except FileNotFoundError as e:
            print(f"  missing input file: {e}")
        except Exception as e:
            print(f"  failed: {e}")


if __name__ == "__main__":
    main()
