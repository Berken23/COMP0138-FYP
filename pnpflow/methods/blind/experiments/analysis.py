"""Analysis pipeline for the thesis Results and Discussion chapter.

Loads every JSON output produced by the experiment scripts and emits

    - LaTeX tables ready to paste into the dissertation
    - a Markdown report with all key aggregate statistics
    - a single JSON dump for further ad hoc analysis

Outputs land in results/blind/analysis/.

The script is read only with respect to the experiment results, no JSON in
the rest of results/blind/ is modified. Run multiple times safely.

Coverage of the produced tables and statistics, organised by thesis
section.

    Headline tables (per dataset)
        sigma error            (oracle - blind) at each (sigma, eta) cell
        PSNR gap               (oracle - blind) at each (sigma, eta) cell
        full metrics           PSNR, SSIM, LPIPS for blind and oracle

    External baseline comparison (where Krishnan data exists)
        sigma error vs Krishnan
        PSNR comparison vs Krishnan classical and Krishnan + PnP-Flow
        speed comparison summary

    Convergence diagnostics
        naive joint estimation: objective, x L2 error, sigma error per outer iter
        Krishnan: objective, x L2 error, sigma error, sigma gradient norm
        PnP-Flow trajectory: PSNR, x L2 error, objective per step

    Failed approaches summary
        sigma error per approach
        gradient or residual diagnostic per approach where applicable

    Aggregate statistics
        mean and std of sigma error per dataset
        mean and std of PSNR gap per dataset
        percentage of configurations with gap below 0.1, 0.5, 1.0 dB

The script does not interpret. It only extracts and formats.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

from .._helpers import NOISE_LEVELS, SIGMA_VALUES, find_repo_root


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> Optional[object]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, content: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def _signed(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{nd}f}"


def _table_header(noise_levels: List[float]) -> str:
    cols = "c" + "c" * len(noise_levels)
    head_row = "$\\sigma_{\\text{true}}$"
    for n in noise_levels:
        head_row += f" & $\\eta = {n}$"
    return cols, head_row


def _wrap_table(body: str, caption: str, label: str, cols: str, header_row: str) -> str:
    return (
        "\\begin{table}[h]\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{cols}}}\n"
        "\\toprule\n"
        f"{header_row} \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------
# Headline tables: blur-SURE on CelebA, BSD68, Set12
# ---------------------------------------------------------------------------

def _blur_sure_dataset_results(results_root: str) -> Dict[str, Dict]:
    """Return a flat dict of {(dataset, sigma, eta): cell_dict}."""
    out: Dict[str, Dict] = {}
    bsf = _load(os.path.join(results_root, "blur_sure_full", "results.json")) or {}
    for key, cell in bsf.items():
        m = re.match(r"sigma=([\d.]+)/noise=([\d.]+)", key)
        if not m:
            continue
        s = float(m.group(1))
        n = float(m.group(2))
        out[("CelebA", s, n)] = cell
    ext = _load(os.path.join(results_root, "extended_evaluation", "results.json")) or {}
    for key, cell in ext.items():
        m = re.match(r"(\w+)/sigma=([\d.]+)/noise=([\d.]+)", key)
        if not m:
            continue
        d = m.group(1)
        s = float(m.group(2))
        n = float(m.group(3))
        out[(d, s, n)] = cell
    return out


def headline_tables(results_root: str, output_dir: str) -> None:
    cells = _blur_sure_dataset_results(results_root)
    if not cells:
        return

    datasets = sorted({k[0] for k in cells.keys()})
    cols, header_row = _table_header(NOISE_LEVELS)

    for dname in datasets:
        # Sigma error table
        body = ""
        for s in SIGMA_VALUES:
            row = f"{s:.1f}"
            for n in NOISE_LEVELS:
                cell = cells.get((dname, s, n))
                if cell is None:
                    row += " & --"
                    continue
                row += f" & ${_fmt(cell.get('sigma_error_mean'), 4)} \\pm {_fmt(cell.get('sigma_error_std', 0.0), 4)}$"
            body += row + " \\\\\n"
        tex = _wrap_table(
            body,
            caption=f"{dname}: blur-SURE sigma estimation error (mean $\\pm$ std).",
            label=f"tab:sigma_error_{dname.lower()}",
            cols=cols,
            header_row=header_row,
        )
        _write_text(os.path.join(output_dir, "tables", f"sigma_error_{dname}.tex"), tex)

        # PSNR gap table
        body = ""
        for s in SIGMA_VALUES:
            row = f"{s:.1f}"
            for n in NOISE_LEVELS:
                cell = cells.get((dname, s, n))
                if cell is None:
                    row += " & --"
                    continue
                row += f" & {_signed(cell.get('psnr_gap'))}"
            body += row + " \\\\\n"
        tex = _wrap_table(
            body,
            caption=f"{dname}: PSNR gap (oracle minus blind) in dB.",
            label=f"tab:psnr_gap_{dname.lower()}",
            cols=cols,
            header_row=header_row,
        )
        _write_text(os.path.join(output_dir, "tables", f"psnr_gap_{dname}.tex"), tex)

        # Full metrics table
        body = ""
        for s in SIGMA_VALUES:
            for n in NOISE_LEVELS:
                cell = cells.get((dname, s, n))
                if cell is None:
                    continue
                row = (
                    f"{s:.1f} & {n:.2f} & "
                    f"{_fmt(cell.get('psnr_mean'), 2)} & {_fmt(cell.get('ssim_mean'), 4)} & {_fmt(cell.get('lpips_mean'), 4)} & "
                    f"{_fmt(cell.get('oracle_psnr_mean'), 2)} & {_fmt(cell.get('oracle_ssim_mean'), 4)} & {_fmt(cell.get('oracle_lpips_mean'), 4)} & "
                    f"{_signed(cell.get('psnr_gap'))}"
                )
                body += row + " \\\\\n"
        tex = (
            "\\begin{table}[h]\n\\centering\n\\small\n"
            "\\begin{tabular}{ccccccccc}\n\\toprule\n"
            "$\\sigma$ & $\\eta$ & PSNR & SSIM & LPIPS & oPSNR & oSSIM & oLPIPS & gap \\\\\n"
            "\\midrule\n"
            f"{body}"
            "\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{{dname}: full metrics (blur-SURE blind vs non blind oracle).}}\n"
            f"\\label{{tab:full_metrics_{dname.lower()}}}\n"
            "\\end{table}\n"
        )
        _write_text(os.path.join(output_dir, "tables", f"full_metrics_{dname}.tex"), tex)


# ---------------------------------------------------------------------------
# Krishnan comparison
# ---------------------------------------------------------------------------

def _krishnan_results(results_root: str) -> Dict:
    """Return {(dataset, sigma, noise): summary} or fall back to legacy
    layout with no noise in path. Reports which layout was found.
    """
    base = os.path.join(results_root, "krishnan_baseline")
    if not os.path.isdir(base):
        return {"layout": "missing", "cells": {}}

    cells: Dict = {}
    layout = "unknown"
    for dataset in sorted(os.listdir(base)):
        ds_dir = os.path.join(base, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for cfg in sorted(os.listdir(ds_dir)):
            cfg_dir = os.path.join(ds_dir, cfg)
            if not os.path.isdir(cfg_dir):
                continue
            summary = _load(os.path.join(cfg_dir, "summary.json"))
            if summary is None:
                continue
            m_full = re.match(r"sigma_([\d.]+)_noise_([\d.]+)", cfg)
            m_legacy = re.match(r"sigma_([\d.]+)$", cfg)
            if m_full:
                s = float(m_full.group(1))
                n = float(m_full.group(2))
                layout = "full"
            elif m_legacy:
                s = float(m_legacy.group(1))
                # Legacy bug: the eta in summary.json is the last noise we
                # ran (eta=0.1) since the inner loop overwrote.
                n = float(summary.get("noise_std", 0.1))
                layout = "legacy_eta_only"
            else:
                continue
            cells[(dataset, s, n)] = summary

    return {"layout": layout, "cells": cells}


def krishnan_comparison(results_root: str, output_dir: str) -> Dict:
    bsf_cells = _blur_sure_dataset_results(results_root)
    krishnan = _krishnan_results(results_root)
    layout = krishnan["layout"]
    krish_cells = krishnan["cells"]

    if not krish_cells:
        return {"layout": layout, "cells_present": 0}

    datasets = sorted({k[0] for k in krish_cells.keys()})

    # Per dataset comparison table (sigma error and PSNR gap)
    for dname in datasets:
        body = ""
        rows_present = 0
        for s in SIGMA_VALUES:
            for n in NOISE_LEVELS:
                bsf = bsf_cells.get((dname, s, n))
                kr = krish_cells.get((dname, s, n))
                if bsf is None or kr is None:
                    continue
                rows_present += 1
                row = (
                    f"{s:.1f} & {n:.2f} & "
                    f"{_fmt(bsf.get('sigma_error_mean'), 4)} & "
                    f"{_fmt(kr.get('sigma_error_mean'), 4)} & "
                    f"{_signed(bsf.get('psnr_gap'))} & "
                    f"{_signed(kr.get('psnr_gap_pnp_mean'))} & "
                    f"{_fmt(kr.get('psnr_classical_mean'), 2)} & "
                    f"{_fmt(bsf.get('oracle_psnr_mean'), 2)}"
                )
                body += row + " \\\\\n"

        if rows_present == 0:
            continue
        tex = (
            "\\begin{table}[h]\n\\centering\n\\small\n"
            "\\begin{tabular}{cccccccc}\n\\toprule\n"
            "$\\sigma$ & $\\eta$ & "
            "blur-SURE err & Krishnan err & "
            "blur-SURE gap & Krishnan PnP gap & Krishnan classical PSNR & oracle PSNR \\\\\n"
            "\\midrule\n"
            f"{body}"
            "\\bottomrule\n\\end{tabular}\n"
            f"\\caption{{{dname}: blur-SURE versus Krishnan classical baseline. "
            f"Layout {layout}.}}\n"
            f"\\label{{tab:krishnan_compare_{dname.lower()}}}\n"
            "\\end{table}\n"
        )
        _write_text(os.path.join(output_dir, "tables", f"krishnan_compare_{dname}.tex"), tex)

    return {"layout": layout, "cells_present": len(krish_cells)}


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------

def convergence_summary(results_root: str, output_dir: str) -> Dict:
    summary: Dict = {"naive_joint": None, "krishnan": None, "pnp_trajectory": None}

    naive = _load(os.path.join(results_root, "naive_joint_estimation", "per_image.json"))
    if naive:
        per_image = naive
        objs = [r.get("objective_history", []) for r in per_image]
        sigs = [r.get("sigma_error_history", []) for r in per_image]
        l2s = [r.get("x_l2_error_history", []) for r in per_image]
        if objs and any(objs):
            obj_arr = np.array([o for o in objs if o])
            sig_arr = np.array([s for s in sigs if s])
            l2_arr = np.array([l for l in l2s if l])
            summary["naive_joint"] = {
                "n_images": len(per_image),
                "objective_first_mean": float(obj_arr[:, 0].mean()),
                "objective_last_mean": float(obj_arr[:, -1].mean()),
                "sigma_error_first_mean": float(sig_arr[:, 0].mean()),
                "sigma_error_last_mean": float(sig_arr[:, -1].mean()),
                "sigma_error_min_mean": float(sig_arr.min(axis=1).mean()),
                "x_l2_error_last_mean": float(l2_arr[:, -1].mean()),
                "outer_iters": int(obj_arr.shape[1]),
            }

    # Krishnan: scan all cells, average per cell
    krishnan = _krishnan_results(results_root)
    krish_summaries: List[Dict] = []
    base = os.path.join(results_root, "krishnan_baseline")
    if os.path.isdir(base):
        for dataset in os.listdir(base):
            ds_dir = os.path.join(base, dataset)
            if not os.path.isdir(ds_dir):
                continue
            for cfg in os.listdir(ds_dir):
                per = _load(os.path.join(ds_dir, cfg, "per_image.json"))
                if not per:
                    continue
                obj_curves = [r.get("convergence", {}).get("objective_history", []) for r in per]
                sig_curves = [r.get("convergence", {}).get("sigma_error_history", []) for r in per]
                grad_curves = [r.get("convergence", {}).get("sigma_grad_norm_history", []) for r in per]
                obj_curves = [c for c in obj_curves if c]
                sig_curves = [c for c in sig_curves if c]
                grad_curves = [c for c in grad_curves if c]
                if not obj_curves:
                    continue
                obj = np.array(obj_curves)
                sig = np.array(sig_curves) if sig_curves else None
                grd = np.array(grad_curves) if grad_curves else None
                krish_summaries.append({
                    "cell": f"{dataset}/{cfg}",
                    "objective_first_mean": float(obj[:, 0].mean()),
                    "objective_last_mean": float(obj[:, -1].mean()),
                    "objective_ratio": float(obj[:, -1].mean() / max(obj[:, 0].mean(), 1e-12)),
                    "sigma_error_first_mean": float(sig[:, 0].mean()) if sig is not None else None,
                    "sigma_error_last_mean": float(sig[:, -1].mean()) if sig is not None else None,
                    "grad_norm_last_mean": float(grd[:, -1].mean()) if grd is not None else None,
                    "outer_iters": int(obj.shape[1]),
                })
    summary["krishnan"] = krish_summaries

    # PnP-Flow trajectory convergence (blur-SURE)
    pnp_conv = _load(os.path.join(results_root, "blur_sure_full", "convergence.json"))
    if pnp_conv:
        blind = [c for c in pnp_conv if c.get("kind") == "blind"]
        oracle = [c for c in pnp_conv if c.get("kind") == "oracle"]

        def avg(items: List[Dict], field: str):
            curves = [it.get("history", {}).get(field, []) for it in items]
            curves = [c for c in curves if c]
            if not curves:
                return None
            m = max(len(c) for c in curves)
            curves = [c + [c[-1]] * (m - len(c)) for c in curves]
            return np.mean(np.array(curves), axis=0)

        psnr_blind = avg(blind, "psnr_history")
        psnr_oracle = avg(oracle, "psnr_history")
        l2_blind = avg(blind, "x_l2_error_history")
        l2_oracle = avg(oracle, "x_l2_error_history")

        summary["pnp_trajectory"] = {
            "blind_psnr_first_mean": float(psnr_blind[0]) if psnr_blind is not None else None,
            "blind_psnr_last_mean": float(psnr_blind[-1]) if psnr_blind is not None else None,
            "oracle_psnr_first_mean": float(psnr_oracle[0]) if psnr_oracle is not None else None,
            "oracle_psnr_last_mean": float(psnr_oracle[-1]) if psnr_oracle is not None else None,
            "blind_l2_first_mean": float(l2_blind[0]) if l2_blind is not None else None,
            "blind_l2_last_mean": float(l2_blind[-1]) if l2_blind is not None else None,
            "n_steps": int(len(psnr_blind)) if psnr_blind is not None else None,
            "n_blind_images": len(blind),
            "n_oracle_images": len(oracle),
        }

    _ensure_dir(output_dir)
    _write_text(
        os.path.join(output_dir, "convergence_summary.json"),
        json.dumps(summary, indent=2),
    )
    return summary


# ---------------------------------------------------------------------------
# Failed approaches summary
# ---------------------------------------------------------------------------

def failed_approaches_summary(results_root: str, output_dir: str) -> Optional[Dict]:
    base = os.path.join(results_root, "failed_approaches")
    if not os.path.isdir(base):
        return None

    rows: List[Dict] = []
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".json") or fname == "summary.json":
            continue
        data = _load(os.path.join(base, fname))
        if not data or "per_image" not in data:
            continue
        per = data["per_image"]
        sigma_errs = [r.get("sigma_error", float("nan")) for r in per]
        sigma_errs = [e for e in sigma_errs if not np.isnan(e)]

        best_per_image: List[float] = []
        for r in per:
            hist = r.get("sigma_history") or r.get("sigma_history_outer") or []
            if hist:
                # Approximate best = closest to 1.5 (the headline true sigma)
                best = min(abs(s - 1.5) for s in hist) if hist else float("nan")
                best_per_image.append(best)

        rows.append({
            "approach": fname.replace(".json", ""),
            "sigma_error_mean": float(np.mean(sigma_errs)) if sigma_errs else None,
            "sigma_error_std": float(np.std(sigma_errs)) if sigma_errs else None,
            "best_sigma_error_mean": float(np.mean(best_per_image)) if best_per_image else None,
            "psnr_mean": data.get("psnr_mean"),
            "n_images": len(per),
        })

    body = ""
    for r in rows:
        body += (
            f"\\texttt{{{r['approach']}}} & "
            f"{_fmt(r['sigma_error_mean'], 4)} & "
            f"{_fmt(r['best_sigma_error_mean'], 4)} & "
            f"{_fmt(r['psnr_mean'], 2)} \\\\\n"
        )
    tex = (
        "\\begin{table}[h]\n\\centering\n\\small\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Approach & Final $\\sigma$ error & Best $\\sigma$ error & PSNR (dB) \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n\\end{tabular}\n"
        "\\caption{Failed approaches summary at $\\sigma_{\\text{true}}=1.5$, $\\eta=0.05$. "
        "Best $\\sigma$ error is the closest the method came to the true value during its iterations.}\n"
        "\\label{tab:failed_approaches}\n\\end{table}\n"
    )
    _write_text(os.path.join(output_dir, "tables", "failed_approaches.tex"), tex)
    _write_text(
        os.path.join(output_dir, "failed_approaches_summary.json"),
        json.dumps(rows, indent=2),
    )
    return {"approaches": rows}


# ---------------------------------------------------------------------------
# Multi image analysis
# ---------------------------------------------------------------------------

def multi_image_summary(results_root: str, output_dir: str) -> Optional[Dict]:
    """Summarise the multi image (B sweep) experiment outputs into LaTeX
    tables and a JSON dump. Reproduces the headline Table 7 from the
    paper plus a per sigma breakdown of estimation accuracy.
    """
    est = _load(os.path.join(results_root, "multi_image", "estimation.json"))
    recon = _load(os.path.join(results_root, "multi_image", "reconstruction.json"))
    if est is None and recon is None:
        return None

    summary: Dict = {}

    # Estimation table: sigma error vs B averaged over all (sigma, eta).
    if est:
        b_values = sorted({int(k.split("B=")[1]) for k in est.keys()})
        rows: List[Dict] = []
        body = ""
        for B in b_values:
            errors_all: List[float] = []
            for sigma in SIGMA_VALUES:
                for noise in NOISE_LEVELS:
                    key = f"sigma={sigma}/noise={noise}/B={B}"
                    cell = est.get(key)
                    if cell:
                        errors_all.extend(cell.get("sigma_errors", []))
            row_stats = {
                "B": B,
                "mean_error": float(np.mean(errors_all)) if errors_all else None,
                "std_error": float(np.std(errors_all)) if errors_all else None,
                "max_error": float(np.max(errors_all)) if errors_all else None,
                "n_samples": len(errors_all),
            }
            rows.append(row_stats)
            body += (
                f"{B} & "
                f"{_fmt(row_stats['mean_error'], 4)} & "
                f"{_fmt(row_stats['std_error'], 4)} & "
                f"{_fmt(row_stats['max_error'], 4)} \\\\\n"
            )
        tex = (
            "\\begin{table}[h]\n\\centering\n"
            "\\begin{tabular}{cccc}\n\\toprule\n"
            "$B$ & Mean error & Std error & Max error \\\\\n"
            "\\midrule\n"
            f"{body}"
            "\\bottomrule\n\\end{tabular}\n"
            "\\caption{Multi image blur-SURE sigma estimation error as a function of $B$, "
            "averaged over all $(\\sigma_{\\text{true}}, \\eta)$ configurations.}\n"
            "\\label{tab:multi_image_estimation}\n\\end{table}\n"
        )
        _write_text(os.path.join(output_dir, "tables", "multi_image_estimation.tex"), tex)
        summary["estimation_vs_B"] = rows

        # Per sigma breakdown at noise=0.05.
        per_sigma_rows: List[Dict] = []
        per_sigma_body = ""
        for B in b_values:
            row = f"{B}"
            cell_for_row: Dict[float, float] = {}
            for sigma in SIGMA_VALUES:
                key = f"sigma={sigma}/noise=0.05/B={B}"
                cell = est.get(key)
                err = cell.get("mean_error") if cell else None
                cell_for_row[sigma] = err
                row += f" & {_fmt(err, 4)}"
            per_sigma_body += row + " \\\\\n"
            per_sigma_rows.append({"B": B, **{f"sigma_{s}": cell_for_row[s] for s in SIGMA_VALUES}})
        sigma_header = "$B$" + "".join(f" & $\\sigma_{{{s}}}$" for s in SIGMA_VALUES)
        col_count = "c" + "c" * len(SIGMA_VALUES)
        tex = (
            "\\begin{table}[h]\n\\centering\n"
            f"\\begin{{tabular}}{{{col_count}}}\n\\toprule\n"
            f"{sigma_header} \\\\\n"
            "\\midrule\n"
            f"{per_sigma_body}"
            "\\bottomrule\n\\end{tabular}\n"
            "\\caption{Multi image blur-SURE sigma estimation error per $\\sigma_{\\text{true}}$ "
            "at $\\eta = 0.05$.}\n"
            "\\label{tab:multi_image_per_sigma}\n\\end{table}\n"
        )
        _write_text(os.path.join(output_dir, "tables", "multi_image_per_sigma.tex"), tex)
        summary["estimation_per_sigma_at_noise_0p05"] = per_sigma_rows

    # Reconstruction table: PSNR/SSIM/LPIPS/gap vs B at sigma=1.5, noise=0.05.
    if recon:
        b_values_r = sorted(int(b) for b in recon.keys())
        recon_rows: List[Dict] = []
        body = ""
        for B in b_values_r:
            r = recon[str(B)]
            recon_rows.append({
                "B": B,
                "sigma_est": r.get("sigma_est"),
                "sigma_error": r.get("sigma_error"),
                "psnr_mean": r.get("psnr_mean"),
                "ssim_mean": r.get("ssim_mean"),
                "lpips_mean": r.get("lpips_mean"),
                "psnr_gap": r.get("psnr_gap"),
            })
            body += (
                f"{B} & "
                f"{_fmt(r.get('sigma_est'), 4)} & "
                f"{_fmt(r.get('sigma_error'), 4)} & "
                f"{_fmt(r.get('psnr_mean'), 2)} & "
                f"{_fmt(r.get('ssim_mean'), 4)} & "
                f"{_fmt(r.get('lpips_mean'), 4)} & "
                f"{_signed(r.get('psnr_gap'))} \\\\\n"
            )
        tex = (
            "\\begin{table}[h]\n\\centering\n\\small\n"
            "\\begin{tabular}{ccccccc}\n\\toprule\n"
            "$B$ & $\\hat{\\sigma}$ & $|\\hat{\\sigma} - \\sigma|$ & PSNR & SSIM & LPIPS & gap \\\\\n"
            "\\midrule\n"
            f"{body}"
            "\\bottomrule\n\\end{tabular}\n"
            "\\caption{Multi image reconstruction quality versus $B$ at "
            "$\\sigma_{\\text{true}} = 1.5$, $\\eta = 0.05$. Gap is oracle minus blind PSNR.}\n"
            "\\label{tab:multi_image_reconstruction}\n\\end{table}\n"
        )
        _write_text(os.path.join(output_dir, "tables", "multi_image_reconstruction.tex"), tex)
        summary["reconstruction_vs_B"] = recon_rows

    _write_text(
        os.path.join(output_dir, "multi_image_summary.json"),
        json.dumps(summary, indent=2),
    )
    return summary


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_statistics(results_root: str, output_dir: str) -> Dict:
    cells = _blur_sure_dataset_results(results_root)
    if not cells:
        return {}

    datasets = sorted({k[0] for k in cells.keys()})
    out: Dict = {}

    for dname in datasets:
        sigma_errs = []
        gaps = []
        psnrs = []
        for s in SIGMA_VALUES:
            for n in NOISE_LEVELS:
                cell = cells.get((dname, s, n))
                if not cell:
                    continue
                sigma_errs.append(cell.get("sigma_error_mean"))
                gaps.append(cell.get("psnr_gap"))
                psnrs.append(cell.get("psnr_mean"))
        sigma_errs = [e for e in sigma_errs if e is not None]
        gaps = [g for g in gaps if g is not None]
        psnrs = [p for p in psnrs if p is not None]

        n_total = len(gaps)
        out[dname] = {
            "sigma_error_mean": float(np.mean(sigma_errs)) if sigma_errs else None,
            "sigma_error_std": float(np.std(sigma_errs)) if sigma_errs else None,
            "sigma_error_max": float(np.max(sigma_errs)) if sigma_errs else None,
            "psnr_gap_mean": float(np.mean(gaps)) if gaps else None,
            "psnr_gap_std": float(np.std(gaps)) if gaps else None,
            "psnr_gap_max": float(np.max(gaps)) if gaps else None,
            "psnr_mean": float(np.mean(psnrs)) if psnrs else None,
            "frac_gap_below_0p1": float(sum(1 for g in gaps if abs(g) < 0.1) / n_total) if n_total else None,
            "frac_gap_below_0p5": float(sum(1 for g in gaps if abs(g) < 0.5) / n_total) if n_total else None,
            "frac_gap_below_1p0": float(sum(1 for g in gaps if abs(g) < 1.0) / n_total) if n_total else None,
            "n_configurations": n_total,
        }

    _write_text(
        os.path.join(output_dir, "aggregate_statistics.json"),
        json.dumps(out, indent=2),
    )
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    results_root: str,
    output_dir: str,
    aggregate: Dict,
    convergence: Dict,
    krishnan_info: Dict,
    failed: Optional[Dict],
    multi: Optional[Dict] = None,
) -> None:
    lines: List[str] = []
    lines.append("# Analysis Report")
    lines.append("")
    lines.append("Generated from the JSON outputs of the experiment scripts.")
    lines.append("This report contains aggregate statistics only. Inferences and")
    lines.append("interpretations are left to the dissertation prose.")
    lines.append("")

    lines.append("## Headline aggregate statistics")
    lines.append("")
    lines.append("| Dataset | Sigma err mean | Sigma err std | Gap mean | Gap max | gap < 0.1 dB | gap < 0.5 dB | gap < 1.0 dB |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for dname, stats in aggregate.items():
        lines.append(
            f"| {dname} | "
            f"{_fmt(stats['sigma_error_mean'], 4)} | "
            f"{_fmt(stats['sigma_error_std'], 4)} | "
            f"{_signed(stats['psnr_gap_mean'])} | "
            f"{_signed(stats['psnr_gap_max'])} | "
            f"{_fmt(stats['frac_gap_below_0p1'] * 100, 1)}% | "
            f"{_fmt(stats['frac_gap_below_0p5'] * 100, 1)}% | "
            f"{_fmt(stats['frac_gap_below_1p0'] * 100, 1)}% |"
        )
    lines.append("")

    lines.append("## Convergence diagnostics")
    lines.append("")
    naive = convergence.get("naive_joint")
    if naive:
        lines.append("### Naive joint estimation")
        lines.append(f"- Outer iterations recorded: {naive['outer_iters']}")
        lines.append(f"- Initial sigma error mean: {_fmt(naive['sigma_error_first_mean'], 4)}")
        lines.append(f"- Final sigma error mean: {_fmt(naive['sigma_error_last_mean'], 4)}")
        lines.append(f"- Best sigma error reached: {_fmt(naive['sigma_error_min_mean'], 4)}")
        lines.append(f"- Objective change: {_fmt(naive['objective_first_mean'], 6)} -> {_fmt(naive['objective_last_mean'], 6)}")
        lines.append("")
    krish = convergence.get("krishnan", [])
    if krish:
        lines.append("### Krishnan baseline (per cell)")
        lines.append("")
        lines.append("| Cell | Outer iters | Sigma err first | Sigma err last | Objective ratio | Grad norm last |")
        lines.append("|---|---|---|---|---|---|")
        for r in krish:
            lines.append(
                f"| {r['cell']} | {r['outer_iters']} | "
                f"{_fmt(r['sigma_error_first_mean'], 4)} | {_fmt(r['sigma_error_last_mean'], 4)} | "
                f"{_fmt(r['objective_ratio'], 4)} | {_fmt(r['grad_norm_last_mean'], 6)} |"
            )
        lines.append("")
    pnp = convergence.get("pnp_trajectory")
    if pnp:
        lines.append("### PnP-Flow reconstruction trajectory (blur-SURE blind vs oracle)")
        lines.append(f"- Steps: {pnp['n_steps']}")
        lines.append(f"- Images recorded blind: {pnp['n_blind_images']}, oracle: {pnp['n_oracle_images']}")
        lines.append(f"- Blind PSNR: {_fmt(pnp['blind_psnr_first_mean'], 2)} -> {_fmt(pnp['blind_psnr_last_mean'], 2)}")
        lines.append(f"- Oracle PSNR: {_fmt(pnp['oracle_psnr_first_mean'], 2)} -> {_fmt(pnp['oracle_psnr_last_mean'], 2)}")
        lines.append("")

    lines.append("## Krishnan baseline data coverage")
    lines.append("")
    lines.append(f"- Layout detected: `{krishnan_info.get('layout')}`")
    lines.append(f"- Cells present: {krishnan_info.get('cells_present', 0)}")
    if krishnan_info.get("layout") == "legacy_eta_only":
        lines.append("")
        lines.append("**Note**: the legacy directory layout means each (dataset, sigma) ")
        lines.append("only retains one noise level (eta=0.1, the last in the inner loop). ")
        lines.append("Re-run with the patched code to recover all three noise levels.")
    lines.append("")

    if multi:
        lines.append("## Multi image (B sweep)")
        lines.append("")
        rows = multi.get("estimation_vs_B")
        if rows:
            lines.append("Sigma estimation error vs B (averaged over all sigma and noise)")
            lines.append("")
            lines.append("| B | Mean error | Std error | Max error | n samples |")
            lines.append("|---|---|---|---|---|")
            for r in rows:
                lines.append(
                    f"| {r['B']} | "
                    f"{_fmt(r['mean_error'], 4)} | "
                    f"{_fmt(r['std_error'], 4)} | "
                    f"{_fmt(r['max_error'], 4)} | "
                    f"{r['n_samples']} |"
                )
            lines.append("")
        recon = multi.get("reconstruction_vs_B")
        if recon:
            lines.append("Reconstruction quality vs B at sigma=1.5, eta=0.05")
            lines.append("")
            lines.append("| B | sigma_est | sigma error | PSNR | SSIM | LPIPS | gap |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in recon:
                lines.append(
                    f"| {r['B']} | "
                    f"{_fmt(r['sigma_est'], 4)} | "
                    f"{_fmt(r['sigma_error'], 4)} | "
                    f"{_fmt(r['psnr_mean'], 2)} | "
                    f"{_fmt(r['ssim_mean'], 4)} | "
                    f"{_fmt(r['lpips_mean'], 4)} | "
                    f"{_signed(r['psnr_gap'])} |"
                )
            lines.append("")

    if failed:
        lines.append("## Failed approaches")
        lines.append("")
        lines.append("| Approach | Final sigma err | Best sigma err | PSNR | n images |")
        lines.append("|---|---|---|---|---|")
        for r in failed["approaches"]:
            lines.append(
                f"| `{r['approach']}` | "
                f"{_fmt(r['sigma_error_mean'], 4)} | "
                f"{_fmt(r['best_sigma_error_mean'], 4)} | "
                f"{_fmt(r['psnr_mean'], 2)} | "
                f"{r['n_images']} |"
            )
        lines.append("")

    lines.append("## File index")
    lines.append("")
    lines.append("LaTeX tables under `tables/` ready to `\\input{...}` into the dissertation.")
    lines.append("")
    lines.append("Numerical summaries under the analysis directory as JSON.")
    lines.append("")
    _write_text(os.path.join(output_dir, "report.md"), "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Analyse experiment results for the thesis")
    p.add_argument("--results-root", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    repo_root = find_repo_root()
    results_root = args.results_root or os.path.join(repo_root, "results", "blind")
    output_dir = args.output_dir or os.path.join(results_root, "analysis")

    print(f"Reading from {results_root}")
    print(f"Writing to {output_dir}")

    headline_tables(results_root, output_dir)
    krishnan_info = krishnan_comparison(results_root, output_dir)
    convergence = convergence_summary(results_root, output_dir)
    aggregate = aggregate_statistics(results_root, output_dir)
    failed = failed_approaches_summary(results_root, output_dir)
    multi = multi_image_summary(results_root, output_dir)
    write_report(
        results_root, output_dir, aggregate, convergence, krishnan_info, failed, multi,
    )

    print()
    print(f"Wrote LaTeX tables to {os.path.join(output_dir, 'tables')}")
    print(f"Wrote summary JSONs and report.md to {output_dir}")


if __name__ == "__main__":
    main()
