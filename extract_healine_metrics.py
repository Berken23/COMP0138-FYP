"""Extract reconstruction metrics at the headline configuration.

Pulls SSIM_blind, SSIM_oracle, LPIPS_blind, LPIPS_oracle for CelebA, BSD68,
and Set12 at the central configuration sigma=1.5, nu=0.05. Prints a neat
table and the LaTeX-paste values for Table 5.4.
"""
import json
import numpy as np

ext = json.load(open('results/blind/extended_evaluation/results.json'))
mi = json.load(open('results/blind/multi_image/reconstruction.json'))

print()
print("=" * 72)
print("  Headline reconstruction quality at sigma=1.5, nu=0.05")
print("=" * 72)
print()
print(f"  {'Dataset':<10} {'PSNR':>17} {'SSIM':>17} {'LPIPS':>17}")
print(f"  {'':<10} {'blind / oracle':>17} {'blind / oracle':>17} {'blind / oracle':>17}")
print("-" * 72)

# CelebA: pull from multi_image B=1 (single-observation, the headline case)
b1 = mi['1']
psnr_str = f"{b1['psnr_mean']:.2f} / {b1['orc_psnr_mean']:.2f}"
ssim_str = f"{b1['ssim_mean']:.3f} / {b1['orc_ssim_mean']:.3f}"
lpips_str = f"{b1['lpips_mean']:.3f} / {b1['orc_lpips_mean']:.3f}"
print(f"  {'CelebA':<10} {psnr_str:>17} {ssim_str:>17} {lpips_str:>17}")

# BSD68 and Set12: pull from extended_evaluation per-image arrays
for d in ['BSD68', 'Set12']:
    per = ext[f'{d}/sigma=1.5/noise=0.05']['per_image']
    psnr_b = np.mean([p['psnr_mean'] for p in per])
    psnr_o = np.mean([p['oracle_psnr'] for p in per])
    ssim_b = np.mean([p['ssim_mean'] for p in per])
    ssim_o = np.mean([p['oracle_ssim'] for p in per])
    lpips_b = np.mean([p['lpips_mean'] for p in per])
    lpips_o = np.mean([p['oracle_lpips'] for p in per])

    psnr_str = f"{psnr_b:.2f} / {psnr_o:.2f}"
    ssim_str = f"{ssim_b:.3f} / {ssim_o:.3f}"
    lpips_str = f"{lpips_b:.3f} / {lpips_o:.3f}"
    print(f"  {d:<10} {psnr_str:>17} {ssim_str:>17} {lpips_str:>17}")

print()
print("=" * 72)
print("  Values to paste into Table 5.4 (overwriting the TBD cells)")
print("=" * 72)
print()

print(f"  CelebA  SSIM_oracle  = {b1['orc_ssim_mean']:.3f}")
print(f"  CelebA  LPIPS_oracle = {b1['orc_lpips_mean']:.3f}")
for d in ['BSD68', 'Set12']:
    per = ext[f'{d}/sigma=1.5/noise=0.05']['per_image']
    ssim_o = np.mean([p['oracle_ssim'] for p in per])
    lpips_o = np.mean([p['oracle_lpips'] for p in per])
    print(f"  {d}   SSIM_oracle  = {ssim_o:.3f}")
    print(f"  {d}   LPIPS_oracle = {lpips_o:.3f}")

print()
