# Analysis Report

Generated from the JSON outputs of the experiment scripts.
This report contains aggregate statistics only. Inferences and
interpretations are left to the dissertation prose.

## Headline aggregate statistics

| Dataset | Sigma err mean | Sigma err std | Gap mean | Gap max | gap < 0.1 dB | gap < 0.5 dB | gap < 1.0 dB |
|---|---|---|---|---|---|---|---|
| BSD68 | 0.3929 | 0.3667 | +0.11 | +0.38 | 66.7% | 100.0% | 100.0% |
| CelebA | 0.4423 | 0.4197 | +0.23 | +0.99 | 44.4% | 88.9% | 100.0% |
| Set12 | 0.4170 | 0.4101 | +0.13 | +0.60 | 61.1% | 83.3% | 100.0% |

## Convergence diagnostics

### Naive joint estimation
- Outer iterations recorded: 30
- Initial sigma error mean: 1.2321
- Final sigma error mean: 0.5628
- Best sigma error reached: 0.0219
- Objective change: 0.008763 -> 0.004780

### Krishnan baseline (per cell)

| Cell | Outer iters | Sigma err first | Sigma err last | Objective ratio | Grad norm last |
|---|---|---|---|---|---|
| CelebA/sigma_1.50_noise_0.05 | 5 | 0.8171 | 0.2095 | 0.4814 | 0.000041 |
| Set12/sigma_1.50_noise_0.05 | 5 | 0.6324 | 0.1297 | 0.3701 | 0.000069 |
| BSD68/sigma_1.50_noise_0.05 | 5 | 0.6525 | 0.3080 | 0.3842 | 0.000044 |

### PnP-Flow reconstruction trajectory (blur-SURE blind vs oracle)
- Steps: 101
- Images recorded blind: 8, oracle: 8
- Blind PSNR: 25.68 -> 26.47
- Oracle PSNR: 25.68 -> 26.50

## Krishnan baseline data coverage

- Layout detected: `full`
- Cells present: 3

## Multi image (B sweep)

Sigma estimation error vs B (averaged over all sigma and noise)

| B | Mean error | Std error | Max error | n samples |
|---|---|---|---|---|
| 1 | 0.4352 | 0.4278 | 1.8933 | 90 |
| 2 | 0.4341 | 0.4287 | 1.8280 | 90 |
| 4 | 0.4374 | 0.4250 | 1.8059 | 90 |
| 8 | 0.4389 | 0.4374 | 1.8424 | 90 |
| 16 | 0.4397 | 0.4357 | 1.8181 | 90 |
| 32 | 0.4411 | 0.4349 | 1.8045 | 90 |
| 64 | 0.4409 | 0.4351 | 1.7868 | 90 |

Reconstruction quality vs B at sigma=1.5, eta=0.05

| B | sigma_est | sigma error | PSNR | SSIM | LPIPS | gap |
|---|---|---|---|---|---|---|
| 1 | 1.4839 | 0.0161 | 26.53 | 0.7787 | 0.2022 | -0.03 |
| 2 | 1.5349 | 0.0349 | 26.53 | 0.7783 | 0.2012 | -0.03 |
| 4 | 1.5712 | 0.0712 | 26.53 | 0.7780 | 0.2006 | -0.02 |
| 8 | 1.5843 | 0.0843 | 26.53 | 0.7779 | 0.2004 | -0.02 |
| 16 | 1.6025 | 0.1025 | 26.52 | 0.7777 | 0.2002 | -0.02 |
| 32 | 1.5820 | 0.0820 | 26.53 | 0.7779 | 0.2005 | -0.02 |
| 64 | 1.5858 | 0.0858 | 26.53 | 0.7778 | 0.2004 | -0.02 |

## Failed approaches

| Approach | Final sigma err | Best sigma err | PSNR | n images |
|---|---|---|---|---|
| `a14` | 0.5628 | 0.0219 | 25.88 | 10 |
| `convergence` | 0.5482 | 0.0219 | -- | 10 |
| `em_damped` | 0.2928 | 0.0079 | 26.36 | 10 |
| `em_tweedie` | 0.4194 | 0.1374 | 26.27 | 10 |
| `grid_fresh` | 0.6394 | 0.1214 | -- | 10 |
| `grid_warm` | 0.6394 | 0.1214 | -- | 10 |
| `morozov` | 0.2468 | 0.0079 | 26.39 | 10 |
| `sgd` | 1.4320 | 1.4320 | 24.68 | 10 |
| `time_lr_aggressive` | 1.3672 | 1.3584 | 24.80 | 10 |
| `time_lr_moderate` | 0.9868 | 0.9456 | 25.37 | 10 |
| `type2` | 3.5000 | 1.5000 | -- | 10 |
| `whiteness` | 1.4000 | 0.1034 | -- | 10 |

## File index

LaTeX tables under `tables/` ready to `\input{...}` into the dissertation.

Numerical summaries under the analysis directory as JSON.
