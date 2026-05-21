"""Composes publication-ready grid figures from saved per-image reconstructions.

Reads PNG outputs and per_image.json files from the result directories
produced by the blind deconvolution experiments, and composes them into
multi-method, multi-row grid figures with metric annotations under each
panel.

Sources of images and metrics
-----------------------------
- results/blind/krishnan_baseline/<dataset>/sigma_<sigma>_noise_<noise>/qualitative/
    clean, observed, krishnan, pnp_with_krishnan_sigma
- results/blind/extended_evaluation/qualitative/<dataset>/
    clean, observed, blind, oracle  (BSD68, Set12 only)
- results/blind/multi_image/qualitative/B=<B>/
    clean, observed, blind, oracle  (one configuration: sigma=1.5, noise=0.05)
- results/blind/naive_joint_estimation/iteration_progression/
    iteration snapshots (clean, observed, outer000..outer029)

Per-image metric records are loaded from the matching per_image.json files.

The compositor handles missing methods gracefully: any panel for which no
image exists renders as an "N/A" placeholder.

Usage
-----
    python -m pnpflow.methods.blind.experiments.figures all

or selectively:

    python -m pnpflow.methods.blind.experiments.figures headline_celeba
    python -m pnpflow.methods.blind.experiments.figures ood
    python -m pnpflow.methods.blind.experiments.figures multi_obs
    python -m pnpflow.methods.blind.experiments.figures joint_failure
    python -m pnpflow.methods.blind.experiments.figures sigma_sweep

Output PDFs and PNGs are written to results/blind/figures/.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from .._helpers import find_repo_root


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MethodSpec:
    """Defines how a method's image and metric are located."""
    key: str
    label: str
    file_suffix: str
    metric_keys: Sequence[str] = field(default_factory=tuple)


METHODS: Dict[str, MethodSpec] = {
    "clean":        MethodSpec("clean",        "Clean",                 "_clean",        ()),
    "observed":     MethodSpec("observed",     "Degraded",              "_observed",     ()),
    "blind":        MethodSpec("blind",        "Ours (blind)",          "_blind",        ("psnr_mean", "psnr_blind", "psnr")),
    "oracle":       MethodSpec("oracle",       "Oracle",                "_oracle",       ("oracle_psnr", "psnr_oracle")),
    "krishnan":     MethodSpec("krishnan",     "Krishnan",              "_krishnan",     ("psnr_classical",)),
    "krishnan_pnp": MethodSpec("krishnan_pnp", r"Krishnan-$\hat{\sigma}$ + PnP-Flow", "_pnp_with_krishnan_sigma", ("psnr_pnp_with_krishnan_sigma",)),
}


# ---------------------------------------------------------------------------
# Image and metric resolution
# ---------------------------------------------------------------------------

def _find_image(idx: int, suffix: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    """Look for `<idx:04d><suffix>.png` across the listed directories."""
    fname = f"{idx:04d}{suffix}.png"
    for d in search_dirs:
        candidate = d / fname
        if candidate.is_file():
            return candidate
    return None


def _load_per_image_records(per_image_path: Path) -> List[Dict]:
    if not per_image_path.is_file():
        return []
    with open(per_image_path) as f:
        return json.load(f)


def _extract_metric(records: List[Dict], idx: int, candidate_keys: Sequence[str]) -> Optional[float]:
    """Find image record `index == idx` and return the first matching metric key."""
    for rec in records:
        if rec.get("index") == idx:
            for k in candidate_keys:
                if k in rec:
                    return float(rec[k])
            return None
    return None


# ---------------------------------------------------------------------------
# Grid composer
# ---------------------------------------------------------------------------

@dataclass
class GridRow:
    label: Optional[str]
    images: List[Optional[Path]]
    metrics: List[Optional[float]]


def compose_grid(
    rows: List[GridRow],
    column_labels: List[str],
    output_path: Path,
    *,
    metric_unit: str = "PSNR",
    metric_format: str = "{:.2f}",
    fig_width_per_col: float = 1.7,
    fig_height_per_row: float = 1.9,
    title: Optional[str] = None,
) -> None:
    """Render a grid of images with column headers, optional row labels, and
    metric annotations beneath each panel."""
    n_rows = len(rows)
    n_cols = len(column_labels)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width_per_col * n_cols, fig_height_per_row * n_rows),
        squeeze=False,
    )

    for r, row in enumerate(rows):
        for c, (img_path, metric_val) in enumerate(zip(row.images, row.metrics)):
            ax = axes[r, c]
            if img_path is not None and img_path.is_file():
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        fontsize=14, color="#888888", transform=ax.transAxes)
                ax.set_facecolor("#f5f5f5")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if r == 0:
                ax.set_title(column_labels[c], fontsize=10, pad=4)

            if metric_val is not None:
                ax.set_xlabel(f"{metric_unit}: {metric_format.format(metric_val)}",
                              fontsize=8)

        if row.label:
            axes[r, 0].set_ylabel(row.label, fontsize=9, rotation=90, labelpad=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if output_path.suffix.lower() == ".pdf":
        fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Path helpers tied to the result-directory conventions
# ---------------------------------------------------------------------------

def _krishnan_qual_dir(repo_root: Path, dataset: str, sigma: float, noise: float) -> Path:
    return repo_root / "results" / "blind" / "krishnan_baseline" / dataset / \
        f"sigma_{sigma:.2f}_noise_{noise:.2f}" / "qualitative"


def _krishnan_per_image(repo_root: Path, dataset: str, sigma: float, noise: float) -> Path:
    return repo_root / "results" / "blind" / "krishnan_baseline" / dataset / \
        f"sigma_{sigma:.2f}_noise_{noise:.2f}" / "per_image.json"


def _extended_qual_dir(repo_root: Path, dataset: str) -> Path:
    return repo_root / "results" / "blind" / "extended_evaluation" / "qualitative" / dataset


def _extended_per_image(repo_root: Path) -> Path:
    # extended_evaluation stores everything in a single results.json keyed by
    # "{dataset}/sigma={sigma}/noise={noise}"; that layout is loaded separately.
    return repo_root / "results" / "blind" / "extended_evaluation" / "results.json"


def _multi_image_qual_dir(repo_root: Path, B: int) -> Path:
    return repo_root / "results" / "blind" / "multi_image" / "qualitative" / f"B={B}"


def _multi_image_records(repo_root: Path) -> Path:
    return repo_root / "results" / "blind" / "multi_image" / "reconstruction.json"


def _joint_progression_dir(repo_root: Path) -> Path:
    return repo_root / "results" / "blind" / "naive_joint_estimation" / "iteration_progression"


def _joint_records(repo_root: Path) -> Path:
    return repo_root / "results" / "blind" / "naive_joint_estimation" / "per_image.json"


def _figures_out(repo_root: Path) -> Path:
    return repo_root / "results" / "blind" / "figures"


# ---------------------------------------------------------------------------
# Metric extraction across the various JSON layouts
# ---------------------------------------------------------------------------

def _load_extended_records(repo_root: Path, dataset: str, sigma: float, noise: float) -> List[Dict]:
    """Fetch the per_image list from extended_evaluation results.json for one config."""
    path = _extended_per_image(repo_root)
    if not path.is_file():
        return []
    with open(path) as f:
        data = json.load(f)
    key = f"{dataset}/sigma={sigma}/noise={noise}"
    cfg = data.get(key)
    return cfg.get("per_image", []) if cfg else []


def _load_multi_records(repo_root: Path, B: int) -> List[Dict]:
    """Multi-image reconstruction.json records per-B aggregates, not per-image.
    For per-image metrics at a given B, the only persisted source is the qualitative
    images themselves (no per-image JSON exists). Return empty so panels show no metric.
    """
    return []


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _resolve_method_panel(
    method_key: str,
    idx: int,
    search_dirs: Sequence[Path],
    metric_record_sources: Sequence[List[Dict]],
) -> tuple[Optional[Path], Optional[float]]:
    """Locate the image file and metric value for a single (method, image)."""
    spec = METHODS[method_key]
    img = _find_image(idx, spec.file_suffix, search_dirs)
    metric: Optional[float] = None
    for records in metric_record_sources:
        metric = _extract_metric(records, idx, spec.metric_keys)
        if metric is not None:
            break
    return img, metric


def build_headline(
    repo_root: Path,
    dataset: str,
    sigma: float,
    noise: float,
    image_indices: Sequence[int],
    *,
    output_name: Optional[str] = None,
) -> Path:
    """Six-column comparison: Clean | Degraded | Krishnan | Krishnan-PnP | Blind | Oracle."""
    methods = ["clean", "observed", "krishnan", "krishnan_pnp", "blind", "oracle"]
    column_labels = [METHODS[m].label for m in methods]

    krishnan_qual = _krishnan_qual_dir(repo_root, dataset, sigma, noise)
    extended_qual = _extended_qual_dir(repo_root, dataset)
    multi_qual = _multi_image_qual_dir(repo_root, B=1)  # fallback for blind/oracle on CelebA

    search_dirs = [krishnan_qual, extended_qual, multi_qual]
    metric_sources = [
        _load_per_image_records(_krishnan_per_image(repo_root, dataset, sigma, noise)),
        _load_extended_records(repo_root, dataset, sigma, noise),
    ]

    rows = []
    for idx in image_indices:
        images = []
        metrics = []
        for m in methods:
            img, met = _resolve_method_panel(m, idx, search_dirs, metric_sources)
            images.append(img)
            metrics.append(met)
        rows.append(GridRow(label=None, images=images, metrics=metrics))

    out_name = output_name or f"headline_{dataset}_sigma_{sigma:.2f}_noise_{noise:.2f}.pdf"
    out_path = _figures_out(repo_root) / out_name
    compose_grid(rows, column_labels, out_path)
    return out_path


def build_ood(
    repo_root: Path,
    sigma: float,
    noise: float,
    bsd68_indices: Sequence[int] = (0, 1, 2),
    set12_indices: Sequence[int] = (0, 1, 2),
    *,
    output_name: str = "ood_comparison.pdf",
) -> Path:
    """OOD generalisation: same six-column structure, mixed BSD68 + Set12 rows."""
    methods = ["clean", "observed", "krishnan", "krishnan_pnp", "blind", "oracle"]
    column_labels = [METHODS[m].label for m in methods]

    rows: List[GridRow] = []
    for dataset, indices, label_prefix in (
        ("BSD68", bsd68_indices, "BSD68"),
        ("Set12", set12_indices, "Set12"),
    ):
        krishnan_qual = _krishnan_qual_dir(repo_root, dataset, sigma, noise)
        extended_qual = _extended_qual_dir(repo_root, dataset)
        search_dirs = [krishnan_qual, extended_qual]
        metric_sources = [
            _load_per_image_records(_krishnan_per_image(repo_root, dataset, sigma, noise)),
            _load_extended_records(repo_root, dataset, sigma, noise),
        ]
        for i, idx in enumerate(indices):
            images = []
            metrics = []
            for m in methods:
                img, met = _resolve_method_panel(m, idx, search_dirs, metric_sources)
                images.append(img)
                metrics.append(met)
            label = label_prefix if i == 0 else None
            rows.append(GridRow(label=label, images=images, metrics=metrics))

    out_path = _figures_out(repo_root) / output_name
    compose_grid(rows, column_labels, out_path)
    return out_path


def build_multi_observation(
    repo_root: Path,
    image_indices: Sequence[int] = (0, 1),
    B_values: Sequence[int] = (1, 64),
    *,
    output_name: str = "multi_observation_progression.pdf",
) -> Path:
    """Multi-observation extension: Clean | Degraded | B=B1 | ... | B=Bk | Oracle.

    Note: multi_image.py currently saves qualitative outputs only at B=1 and
    B=max. For richer B coverage in this figure, extend multi_image.py to save
    all B values and re-run with --b-values 1 4 16 64.
    """
    column_labels = ["Clean", "Degraded"] + [f"$B = {B}$" for B in B_values] + ["Oracle"]

    rows: List[GridRow] = []
    for idx in image_indices:
        images: List[Optional[Path]] = []
        metrics: List[Optional[float]] = []

        # Clean and observed from the first available B
        first_dir = _multi_image_qual_dir(repo_root, B=B_values[0])
        images.append(_find_image(idx, "_clean", [first_dir]))
        metrics.append(None)
        images.append(_find_image(idx, "_observed", [first_dir]))
        metrics.append(None)

        # Blind reconstruction at each B
        for B in B_values:
            qd = _multi_image_qual_dir(repo_root, B=B)
            images.append(_find_image(idx, "_blind", [qd]))
            metrics.append(None)  # per-image B metrics not persisted; would require code change

        # Oracle: take from any available B (oracle is independent of B)
        oracle_dir_candidates = [_multi_image_qual_dir(repo_root, B=B) for B in B_values]
        oracle_img = None
        for d in oracle_dir_candidates:
            oracle_img = _find_image(idx, "_oracle", [d])
            if oracle_img is not None:
                break
        images.append(oracle_img)
        metrics.append(None)

        rows.append(GridRow(label=None, images=images, metrics=metrics))

    out_path = _figures_out(repo_root) / output_name
    compose_grid(rows, column_labels, out_path)
    return out_path


def build_joint_failure(
    repo_root: Path,
    image_indices: Sequence[int] = (0, 1, 2),
    *,
    output_name: str = "joint_estimation_failure.pdf",
    outer_snapshot: int = 29,
) -> Path:
    """Failure of joint estimation: Clean | Degraded | Joint (final) | Our blind | Oracle.

    Joint estimation images come from naive_joint_estimation's iteration progression
    (with outer iteration index `outer_snapshot`, default 29 = final iteration).
    Our blind and oracle come from the extended_evaluation or krishnan_baseline
    qualitative directories at the matching configuration.
    """
    methods_after_joint = ["blind", "oracle"]
    column_labels = ["Clean", "Degraded", f"Joint (outer={outer_snapshot})"] + \
                    [METHODS[m].label for m in methods_after_joint]

    joint_dir = _joint_progression_dir(repo_root)
    # Joint scripts use img<idx>_... naming
    rows: List[GridRow] = []
    for idx in image_indices:
        images: List[Optional[Path]] = []
        metrics: List[Optional[float]] = []

        # Clean and observed from joint progression
        clean_p = joint_dir / f"img{idx}_clean.png"
        obs_p = joint_dir / f"img{idx}_observed.png"
        joint_p = joint_dir / f"img{idx}_outer{outer_snapshot:03d}.png"
        images.extend([
            clean_p if clean_p.is_file() else None,
            obs_p if obs_p.is_file() else None,
            joint_p if joint_p.is_file() else None,
        ])
        metrics.extend([None, None, None])

        # Our blind and oracle from the matching configuration
        # The naive joint experiment runs at the script default sigma=1.5, noise=0.05.
        # We fetch images for the same idx from the extended/krishnan qualitative dirs.
        # Note: image indices may not align across experiments because of differing
        # image selections. For thesis use, prefer to re-run with matched indices.
        search = [
            _extended_qual_dir(repo_root, "CelebA"),
            _krishnan_qual_dir(repo_root, "CelebA", sigma=1.5, noise=0.05),
        ]
        for m in methods_after_joint:
            img, _ = _resolve_method_panel(m, idx, search, [])
            images.append(img)
            metrics.append(None)

        rows.append(GridRow(label=None, images=images, metrics=metrics))

    out_path = _figures_out(repo_root) / output_name
    compose_grid(rows, column_labels, out_path)
    return out_path


def build_sigma_sweep(
    repo_root: Path,
    dataset: str,
    noise: float,
    sigmas: Sequence[float] = (0.5, 1.5, 3.0, 4.0),
    image_index: int = 0,
    *,
    output_name: Optional[str] = None,
) -> Path:
    """Per-row sigma sweep: rows are different blur levels, columns show
    Clean | Degraded | Blind | Oracle for the same scene at each sigma."""
    methods = ["clean", "observed", "blind", "oracle"]
    column_labels = [METHODS[m].label for m in methods]

    rows: List[GridRow] = []
    for sigma in sigmas:
        krishnan_qual = _krishnan_qual_dir(repo_root, dataset, sigma, noise)
        extended_qual = _extended_qual_dir(repo_root, dataset)
        search_dirs = [krishnan_qual, extended_qual]
        metric_sources = [
            _load_per_image_records(_krishnan_per_image(repo_root, dataset, sigma, noise)),
            _load_extended_records(repo_root, dataset, sigma, noise),
        ]
        images, metrics = [], []
        for m in methods:
            img, met = _resolve_method_panel(m, image_index, search_dirs, metric_sources)
            images.append(img)
            metrics.append(met)
        rows.append(GridRow(label=fr"$\sigma = {sigma}$", images=images, metrics=metrics))

    out_name = output_name or f"sigma_sweep_{dataset}_idx{image_index}.pdf"
    out_path = _figures_out(repo_root) / out_name
    compose_grid(rows, column_labels, out_path)
    return out_path


def build_trajectory(
    repo_root: Path,
    image_index: int = 0,
    *,
    output_name: str = "trajectory_progression.pdf",
) -> Path:
    """Joint-estimation trajectory progression as a stand-in for a PnP-Flow
    trajectory figure. Rows: one image. Columns: outer iterations.

    To render the PnP-Flow internal trajectory across t = 0.0 ... 1.0 instead,
    extend reconstruction.py with a snapshots-along-trajectory output and add
    a corresponding loader here.
    """
    joint_dir = _joint_progression_dir(repo_root)
    outers = [0, 5, 10, 15, 20, 25, 29]
    column_labels = ["Clean", "Degraded"] + [f"outer={o}" for o in outers]

    images: List[Optional[Path]] = []
    metrics: List[Optional[float]] = []
    clean_p = joint_dir / f"img{image_index}_clean.png"
    obs_p = joint_dir / f"img{image_index}_observed.png"
    images.append(clean_p if clean_p.is_file() else None)
    metrics.append(None)
    images.append(obs_p if obs_p.is_file() else None)
    metrics.append(None)
    for o in outers:
        p = joint_dir / f"img{image_index}_outer{o:03d}.png"
        images.append(p if p.is_file() else None)
        metrics.append(None)

    rows = [GridRow(label=None, images=images, metrics=metrics)]
    out_path = _figures_out(repo_root) / output_name
    compose_grid(rows, column_labels, out_path, fig_width_per_col=1.4, fig_height_per_row=1.6)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

FIGURE_BUILDERS = {
    "headline_celeba": lambda r: build_headline(r, "CelebA", 1.5, 0.05, image_indices=[0, 1, 2, 3]),
    "headline_bsd68":  lambda r: build_headline(r, "BSD68",  1.5, 0.05, image_indices=[0, 1, 2, 3]),
    "headline_set12":  lambda r: build_headline(r, "Set12",  1.5, 0.05, image_indices=[0, 1, 2, 3]),
    "ood":             lambda r: build_ood(r, sigma=1.5, noise=0.05),
    "multi_obs":       lambda r: build_multi_observation(r),
    "joint_failure":   lambda r: build_joint_failure(r),
    "sigma_sweep":     lambda r: build_sigma_sweep(r, "CelebA", noise=0.05),
    "trajectory":      lambda r: build_trajectory(r),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose publication figures from saved reconstructions.")
    p.add_argument(
        "figures",
        nargs="*",
        choices=list(FIGURE_BUILDERS.keys()) + ["all"],
        default=["all"],
        help="Which figures to build. Default: all.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(find_repo_root())

    targets = list(FIGURE_BUILDERS.keys()) if "all" in args.figures else args.figures

    for name in targets:
        print(f"Building {name} ...")
        try:
            out = FIGURE_BUILDERS[name](repo_root)
            print(f"  -> {out}")
        except Exception as e:
            print(f"  ! skipped due to: {e}")


if __name__ == "__main__":
    main()
