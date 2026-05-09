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
| BSD68/sigma_1.50_noise_0.01 | 5 | 0.6564 | 0.1851 | 0.0567 | 0.000042 |
| BSD68/sigma_1.00_noise_0.05 | 5 | 1.0511 | 0.3610 | 0.2508 | 0.000063 |
| BSD68/sigma_1.50_noise_0.05 | 5 | 0.6525 | 0.3080 | 0.3842 | 0.000044 |
| BSD68/sigma_2.00 | 5 | 0.2820 | 0.8931 | 0.6206 | 0.000023 |
| BSD68/sigma_3.00_noise_0.05 | 5 | 0.5295 | 1.2680 | 0.7379 | 0.000014 |
| BSD68/sigma_4.00_noise_0.01 | 5 | 1.3748 | 1.6988 | 0.2992 | 0.000005 |
| BSD68/sigma_1.50 | 5 | 0.6414 | 0.5078 | 0.5384 | 0.000046 |
| BSD68/sigma_0.50 | 5 | 1.4707 | 0.4591 | 0.3203 | 0.000111 |
| BSD68/sigma_0.50_noise_0.01 | 5 | 1.4782 | 0.5322 | 0.1388 | 0.000088 |
| BSD68/sigma_1.00_noise_0.01 | 5 | 1.0540 | 0.3276 | 0.0718 | 0.000055 |
| BSD68/sigma_2.00_noise_0.05 | 5 | 0.2910 | 0.5136 | 0.5456 | 0.000026 |
| BSD68/sigma_2.00_noise_0.01 | 5 | 0.2940 | 0.2953 | 0.0772 | 0.000021 |
| BSD68/sigma_1.00 | 5 | 1.0425 | 0.3915 | 0.4158 | 0.000083 |
| BSD68/sigma_3.00 | 5 | 0.5681 | 1.8872 | 0.6453 | 0.000019 |
| BSD68/sigma_4.00 | 5 | 1.4739 | 3.1796 | 0.5330 | 0.000045 |
| BSD68/sigma_0.50_noise_0.05 | 5 | 1.4760 | 0.5110 | 0.2130 | 0.000110 |
| BSD68/sigma_4.00_noise_0.05 | 5 | 1.3994 | 2.3028 | 0.7620 | 0.000010 |
| BSD68/sigma_3.00_noise_0.01 | 5 | 0.5184 | 0.9030 | 0.1930 | 0.000011 |
| CelebA/sigma_1.50_noise_0.01 | 5 | 0.8199 | 0.2415 | 0.0747 | 0.000046 |
| CelebA/sigma_1.00_noise_0.05 | 5 | 1.2299 | 0.4729 | 0.3362 | 0.000070 |
| CelebA/sigma_1.50_noise_0.05 | 5 | 0.8171 | 0.2095 | 0.4814 | 0.000041 |
| CelebA/sigma_2.00 | 5 | 0.4049 | 0.4350 | 0.8010 | 0.000030 |
| CelebA/sigma_3.00_noise_0.05 | 5 | 0.3748 | 0.7868 | 0.8371 | 0.000016 |
| CelebA/sigma_4.00_noise_0.01 | 5 | 1.2469 | 1.4533 | 0.3139 | 0.000010 |
| CelebA/sigma_1.50 | 5 | 0.8085 | 0.2360 | 0.7095 | 0.000038 |
| CelebA/sigma_0.50 | 5 | 1.6616 | 0.6698 | 0.4713 | 0.000093 |
| CelebA/sigma_0.50_noise_0.01 | 5 | 1.6670 | 0.8033 | 0.1615 | 0.000083 |
| CelebA/sigma_1.00_noise_0.01 | 5 | 1.2318 | 0.5136 | 0.0907 | 0.000067 |
| CelebA/sigma_2.00_noise_0.05 | 5 | 0.4170 | 0.1588 | 0.6484 | 0.000026 |
| CelebA/sigma_2.00_noise_0.01 | 5 | 0.4210 | 0.0985 | 0.0976 | 0.000021 |
| CelebA/sigma_1.00 | 5 | 1.2235 | 0.3756 | 0.5799 | 0.000071 |
| CelebA/sigma_3.00 | 5 | 0.4025 | 1.3176 | 0.8636 | 0.000015 |
| CelebA/sigma_4.00 | 5 | 1.3160 | 2.6206 | 0.8089 | 0.000015 |
| CelebA/sigma_0.50_noise_0.05 | 5 | 1.6658 | 0.7702 | 0.2879 | 0.000086 |
| CelebA/sigma_4.00_noise_0.05 | 5 | 1.2641 | 1.7278 | 0.8822 | 0.000008 |
| CelebA/sigma_3.00_noise_0.01 | 5 | 0.3663 | 0.6441 | 0.2210 | 0.000018 |
| Set12/sigma_1.50_noise_0.01 | 5 | 0.6322 | 0.1214 | 0.0450 | 0.000067 |
| Set12/sigma_1.00_noise_0.05 | 5 | 1.0672 | 0.3262 | 0.2386 | 0.000066 |
| Set12/sigma_1.50_noise_0.05 | 5 | 0.6324 | 0.1297 | 0.3701 | 0.000069 |
| Set12/sigma_2.00 | 5 | 0.2046 | 0.7535 | 0.7208 | 0.000012 |
| Set12/sigma_3.00_noise_0.05 | 5 | 0.5904 | 1.1719 | 0.7833 | 0.000007 |
| Set12/sigma_4.00_noise_0.01 | 5 | 1.4329 | 1.7522 | 0.2760 | 0.000003 |
| Set12/sigma_1.50 | 5 | 0.6293 | 0.2703 | 0.6132 | 0.000039 |
| Set12/sigma_0.50 | 5 | 1.5235 | 0.5431 | 0.3557 | 0.000109 |
| Set12/sigma_0.50_noise_0.01 | 5 | 1.5214 | 0.6540 | 0.1257 | 0.000109 |
| Set12/sigma_1.00_noise_0.01 | 5 | 1.0665 | 0.3484 | 0.0622 | 0.000046 |
| Set12/sigma_2.00_noise_0.05 | 5 | 0.2120 | 0.3716 | 0.5488 | 0.000015 |
| Set12/sigma_2.00_noise_0.01 | 5 | 0.2138 | 0.2799 | 0.0610 | 0.000021 |
| Set12/sigma_1.00 | 5 | 1.0667 | 0.3034 | 0.4669 | 0.000089 |
| Set12/sigma_3.00 | 5 | 0.6206 | 1.9755 | 0.7490 | 0.000016 |
| Set12/sigma_4.00 | 5 | 1.5128 | 3.3693 | 0.5089 | 0.000017 |
| Set12/sigma_0.50_noise_0.05 | 5 | 1.5227 | 0.6286 | 0.2079 | 0.000086 |
| Set12/sigma_4.00_noise_0.05 | 5 | 1.4513 | 2.2652 | 0.8292 | 0.000004 |
| Set12/sigma_3.00_noise_0.01 | 5 | 0.5833 | 0.9525 | 0.1695 | 0.000009 |

### PnP-Flow reconstruction trajectory (blur-SURE blind vs oracle)
- Steps: 101
- Images recorded blind: 8, oracle: 8
- Blind PSNR: 25.68 -> 26.47
- Oracle PSNR: 25.68 -> 26.50

## Krishnan baseline data coverage

- Layout detected: `full`
- Cells present: 54

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
