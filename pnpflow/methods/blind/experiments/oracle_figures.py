"""Generate Figures 1 and 2 for the Non-Blind Oracle Performance subsection.

Figure 1: qualitative oracle reconstructions on CelebA across sigma values.
    A grid of 3 example images (rows: example image) by 4 columns (ground
    truth, then observed and oracle for each sigma in {0.5, 2.0, 4.0}) at
    fixed eta = 0.05. Demonstrates the oracle's behaviour across the
    operating range visually.

Figure 2: oracle PSNR heatmap for the three datasets.
    A three panel heatmap where each panel shows the 6 by 3 oracle PSNR
    grid for one dataset. Reads from the existing JSON outputs of
    blur_sure_full and extended_evaluation.

Outputs land in results/blind/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image

from .._helpers import SEED, find_repo_root, load_model
from ..data import load_celeba
from ..evaluation import postprocess
from ..forward import gaussian_blur_fft
from ..reconstruction import pnp_flow_reconstruct


SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
NOISES = [0.01, 0.05, 0.1]


# ---------------------------------------------------------------------------
# Figure 2: heatmap from existing JSON results
# ---------------------------------------------------------------------------

def render_oracle_heatmap(results_root: str, fig_dir: str) -> None:
    bsf = json.load(open(os.path.join(results_root, "blur_sure_full", "results.json")))
    ext = json.load(open(os.path.join(results_root, "extended_evaluation", "results.json")))

    panels = [("CelebA", bsf, ""), ("BSD68", ext, "BSD68/"), ("Set12", ext, "Set12/")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    vmin, vmax = float("inf"), float("-inf")
    matrices = []
    for label, data, prefix in panels:
        m = np.zeros((len(SIGMAS), len(NOISES)))
        for si, s in enumerate(SIGMAS):
            for ni, n in enumerate(NOISES):
                cell = data.get(f"{prefix}sigma={s}/noise={n}", {})
                m[si, ni] = cell.get("oracle_psnr_mean", np.nan)
        matrices.append(m)
        vmin = min(vmin, np.nanmin(m))
        vmax = max(vmax, np.nanmax(m))

    for ax, (label, _, _), m in zip(axes, panels, matrices):
        im = ax.imshow(m, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(NOISES)))
        ax.set_xticklabels([str(n) for n in NOISES])
        ax.set_yticks(range(len(SIGMAS)))
        ax.set_yticklabels([str(s) for s in SIGMAS])
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\sigma$ (true)")
        ax.set_title(label)
        for si in range(len(SIGMAS)):
            for ni in range(len(NOISES)):
                ax.text(ni, si, f"{m[si, ni]:.1f}", ha="center", va="center",
                        fontsize=9, color="white" if m[si, ni] < (vmin + vmax) / 2 else "black")

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.025, pad=0.02,
                 label="Oracle PSNR (dB)")
    plt.suptitle("Non-Blind Oracle PSNR across Datasets", fontsize=13)
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "fig2_oracle_psnr_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 1: qualitative oracle reconstructions across sigma values
# ---------------------------------------------------------------------------

def render_qualitative_grid(
    results_root: str,
    fig_dir: str,
    device: str,
    n_examples: int = 3,
    sigmas: List[float] = (0.5, 2.0, 4.0),
    eta: float = 0.05,
    pnp_steps: int = 100,
) -> None:
    repo_root = find_repo_root()
    print(f"Loading model on {device}")
    model = load_model(repo_root, device)
    print(f"Loading {n_examples} CelebA images")
    images = load_celeba(repo_root, device, num_images=n_examples, seed=SEED)
    images = images[:n_examples]

    # Build the panel grid: rows = examples, cols = (clean) + (observed, oracle) per sigma
    n_cols = 1 + 2 * len(sigmas)
    fig, axes = plt.subplots(n_examples, n_cols, figsize=(2.6 * n_cols, 2.6 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for r, x_gt in enumerate(images):
        # Column 0: ground truth
        gt_np = postprocess(x_gt).squeeze(0).cpu().permute(1, 2, 0).numpy()
        axes[r, 0].imshow(gt_np)
        axes[r, 0].axis("off")
        if r == 0:
            axes[r, 0].set_title("Ground truth", fontsize=10)

        for ci, s in enumerate(sigmas):
            torch.manual_seed(SEED + r)
            y = gaussian_blur_fft(x_gt, s) + torch.randn_like(x_gt) * eta

            torch.manual_seed(SEED + r * 100)
            x_oracle = pnp_flow_reconstruct(model, y=y, sigma_blur=s, num_steps=pnp_steps)

            obs_np = postprocess(y).squeeze(0).cpu().permute(1, 2, 0).numpy()
            ora_np = postprocess(x_oracle).squeeze(0).cpu().permute(1, 2, 0).numpy()

            obs_col = 1 + 2 * ci
            ora_col = obs_col + 1
            axes[r, obs_col].imshow(obs_np)
            axes[r, obs_col].axis("off")
            axes[r, ora_col].imshow(ora_np)
            axes[r, ora_col].axis("off")

            if r == 0:
                axes[r, obs_col].set_title(rf"$y$ ($\sigma$={s})", fontsize=10)
                axes[r, ora_col].set_title(rf"oracle ($\sigma$={s})", fontsize=10)

    plt.suptitle(f"Oracle reconstructions across sigma at eta={eta} (CelebA)", fontsize=12)
    plt.tight_layout()
    os.makedirs(fig_dir, exist_ok=True)
    out_path = os.path.join(fig_dir, "fig1_oracle_qualitative.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate oracle figures for subsection 1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--pnp-steps", type=int, default=100)
    p.add_argument("--results-root", default=None)
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--skip-qualitative", action="store_true",
                   help="Skip Figure 1 generation (faster, only renders heatmap)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()
    results_root = args.results_root or os.path.join(repo_root, "results", "blind")
    fig_dir = args.fig_dir or os.path.join(results_root, "figures")

    print(f"Reading from {results_root}")
    print(f"Writing to {fig_dir}")

    render_oracle_heatmap(results_root, fig_dir)
    if not args.skip_qualitative:
        render_qualitative_grid(
            results_root, fig_dir, args.device,
            n_examples=args.n_examples, pnp_steps=args.pnp_steps,
        )


if __name__ == "__main__":
    main()
