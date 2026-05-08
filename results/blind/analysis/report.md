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
| BSD68/sigma_0.50 | 5 | 1.4707 | 0.4591 | 0.3203 | 0.000111 |
| BSD68/sigma_1.00 | 5 | 1.0425 | 0.3915 | 0.4158 | 0.000083 |
| BSD68/sigma_1.50 | 5 | 0.6414 | 0.5078 | 0.5384 | 0.000046 |
| BSD68/sigma_2.00 | 5 | 0.2820 | 0.8931 | 0.6206 | 0.000023 |
| BSD68/sigma_3.00 | 5 | 0.5681 | 1.8872 | 0.6453 | 0.000019 |
| BSD68/sigma_4.00 | 5 | 1.4739 | 3.1796 | 0.5330 | 0.000045 |
| CelebA/sigma_0.50 | 5 | 1.6616 | 0.6698 | 0.4713 | 0.000093 |
| CelebA/sigma_1.00 | 5 | 1.2235 | 0.3756 | 0.5799 | 0.000071 |
| CelebA/sigma_1.50 | 5 | 0.8085 | 0.2360 | 0.7095 | 0.000038 |
| CelebA/sigma_2.00 | 5 | 0.4049 | 0.4350 | 0.8010 | 0.000030 |
| CelebA/sigma_3.00 | 5 | 0.4025 | 1.3176 | 0.8636 | 0.000015 |
| CelebA/sigma_4.00 | 5 | 1.3160 | 2.6206 | 0.8089 | 0.000015 |
| Set12/sigma_0.50 | 5 | 1.5235 | 0.5431 | 0.3557 | 0.000109 |
| Set12/sigma_1.00 | 5 | 1.0667 | 0.3034 | 0.4669 | 0.000089 |
| Set12/sigma_1.50 | 5 | 0.6293 | 0.2703 | 0.6132 | 0.000039 |
| Set12/sigma_2.00 | 5 | 0.2046 | 0.7535 | 0.7208 | 0.000012 |
| Set12/sigma_3.00 | 5 | 0.6206 | 1.9755 | 0.7490 | 0.000016 |
| Set12/sigma_4.00 | 5 | 1.5128 | 3.3693 | 0.5089 | 0.000017 |

### PnP-Flow reconstruction trajectory (blur-SURE blind vs oracle)
- Steps: 101
- Images recorded blind: 8, oracle: 8
- Blind PSNR: 25.68 -> 26.47
- Oracle PSNR: 25.68 -> 26.50

## Krishnan baseline data coverage

- Layout detected: `legacy_eta_only`
- Cells present: 18

**Note**: the legacy directory layout means each (dataset, sigma) 
only retains one noise level (eta=0.1, the last in the inner loop). 
Re-run with the patched code to recover all three noise levels.

## Multi image (B sweep)

Sigma estimation error vs B (averaged over all sigma and noise)

| B | Mean error | Std error | Max error | n samples |
|---|---|---|---|---|
| 1 | 0.4352 | 0.4279 | 1.8961 | 90 |
| 2 | 0.4342 | 0.4288 | 1.8263 | 90 |
| 4 | 0.4373 | 0.4250 | 1.8043 | 90 |
| 8 | 0.4390 | 0.4375 | 1.8425 | 90 |
| 16 | 0.4397 | 0.4358 | 1.8183 | 90 |
| 32 | 0.4411 | 0.4349 | 1.8040 | 90 |
| 64 | 0.4409 | 0.4351 | 1.7870 | 90 |

Reconstruction quality vs B at sigma=1.5, eta=0.05

| B | sigma_est | sigma error | PSNR | SSIM | LPIPS | gap |
|---|---|---|---|---|---|---|
| 1 | 1.4835 | 0.0165 | 26.53 | 0.7786 | 0.2023 | -0.03 |
| 2 | 1.5349 | 0.0349 | 26.53 | 0.7783 | 0.2013 | -0.03 |
| 4 | 1.5704 | 0.0704 | 26.53 | 0.7779 | 0.2007 | -0.03 |
| 8 | 1.5843 | 0.0843 | 26.52 | 0.7778 | 0.2005 | -0.02 |
| 16 | 1.6025 | 0.1025 | 26.52 | 0.7776 | 0.2002 | -0.02 |
| 32 | 1.5820 | 0.0820 | 26.53 | 0.7778 | 0.2005 | -0.02 |
| 64 | 1.5859 | 0.0859 | 26.52 | 0.7778 | 0.2004 | -0.02 |

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
