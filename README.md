# Blind PnP-Flow Deconvolution

This repository contains the code for the COMP0138 Final Year Project
"Blind Inverse Problems with Flow-Based Plug-and-Play Priors" (UCL,
2025/26).

**This work is an extension of the PnP-Flow framework of Martin et al.**
([ICLR 2025 paper](https://arxiv.org/abs/2410.02423), upstream repository
[github.com/annegnx/PnP-Flow](https://github.com/annegnx/PnP-Flow)). The
upstream codebase is preserved unmodified and the non-blind PnP-Flow
entry points remain functional; all files under `pnpflow/methods/blind/`
are the contribution of the present thesis and extend the upstream
framework to the blind Gaussian deconvolution setting, in which the blur
parameter is unknown and must be inferred from the degraded observation
alone.

## Setup

```
pip install -e .
```

The CelebA dataset and pretrained OT flow-matching network are
downloaded via the upstream scripts:

```
bash download.sh celeba-dataset
bash download.sh pretrained-network-celeba
```

BSD68 and Set12 must be staged manually as PNG files at `data/BSD68/`
and `data/Set12/`. The canonical sources are the DnCNN benchmark
distributions:

- BSD68: [github.com/cszn/DnCNN/tree/master/testsets/BSD68](https://github.com/cszn/DnCNN/tree/master/testsets/BSD68)
- Set12: [github.com/cszn/DnCNN/tree/master/testsets/Set12](https://github.com/cszn/DnCNN/tree/master/testsets/Set12)

Download the PNG files from the linked directories and place them
directly into `data/BSD68/` (68 PNGs) and `data/Set12/` (12 PNGs).

## Layout

```
pnpflow/methods/blind/
    __init__.py             public API
    forward.py              FFT Gaussian blur (self adjoint)
    sigma_estimation.py     blur-SURE single and multi observation
    reconstruction.py       PnP-Flow reconstruction at fixed sigma
    evaluation.py           PSNR, SSIM, LPIPS, non blind oracle
    data.py                 CelebA, BSD68, Set12 loaders
    main.py                 simple per image entry point
    _helpers.py             shared utilities for the experiments
    experiments/            scripts that reproduce every thesis result
```

## Reproducing the thesis

All experiments write their outputs under `results/blind/<experiment>/`
relative to the repository root. JSON files contain the full numerical
results, including per image records.

Run from the repository root.

### 1. Non blind hyperparameter ablation

Establishes the oracle ceiling and isolates the effect of each PnP-Flow
hyperparameter (gamma schedule, number of steps, learning rate, denoiser
samples, outer iterations, warm start).

```
python -m pnpflow.methods.blind.experiments.non_blind_ablation
```

### 2. Trajectory straightness analysis

Reports the RMSE deviation of the PnP-Flow trajectory from the straight
line between x_0 and x_T at every flow time, across datasets, sigma
values, and gamma schedules.

```
python -m pnpflow.methods.blind.experiments.trajectory_straightness
```

### 3. Naive joint estimation (per step Adam from Martin et al. appendix A.14)

The naive scheme that updates sigma via per step Adam against
||y - H_sigma(x)||^2. Demonstrates the over sharpening bias that
motivates the proposed pipeline.

```
python -m pnpflow.methods.blind.experiments.naive_joint_estimation
```

### 3b. Full failed approaches catalogue

Reproduces every failed sigma estimation approach reported in the thesis,
each producing the specific numerical values referenced in the text. The
eleven approaches are A.14 Adam, plain SGD, time restricted updates with
LR decay, convergence detection, decoupled grid search (warm and fresh
start), EM with Tweedie estimates, EM with damping (eta=0.1), Morozov
discrepancy stopping, residual whiteness, and Type II marginal likelihood
with operator mismatch. Each writes its sigma history, residual history,
gradient signals, and final error to a JSON file under
`results/blind/failed_approaches/`.

```
python -m pnpflow.methods.blind.experiments.failed_approaches
```

Run a single approach with `--approaches em_damped`. The full list is
`a14 sgd time_lr convergence grid_warm grid_fresh em_tweedie em_damped
morozov whiteness type2`.

### 4. Full Blur-SURE evaluation on CelebA

The main contribution. Calibrates the lambda multiplier on synthetic data
(stage 1) then sweeps the full sigma and noise grid on 100 CelebA images
with three reconstruction seeds per image (stage 2). Pass
`--skip-calibration` to use the calibrated value (10*eta^2) directly.

```
python -m pnpflow.methods.blind.experiments.blur_sure_full
```

### 5. Multi image blur-SURE

Demonstrates the 1/B variance reduction obtained by averaging blur-SURE
curves over B observations. Part 1 sweeps estimation accuracy across the
full sigma and noise grid for B in {1, 2, 4, 8, 16, 32, 64}. Part 2
reports reconstruction quality at sigma=1.5, noise=0.05 for each B.

```
python -m pnpflow.methods.blind.experiments.multi_image
```

### 6. Extended evaluation on BSD68 and Set12

Mirrors the CelebA protocol on the standard image restoration benchmarks.
Lambda is fixed at 10*eta^2. Three reconstruction seeds per image, all
three metrics, oracle gap reported.

```
python -m pnpflow.methods.blind.experiments.extended_evaluation
```

### 6b. Krishnan classical baseline

External baseline based on the principles of Krishnan, Tay, and Fergus (2011).
The original method is adapted here to the parametric Gaussian setting and uses
plain L1 image gradient sparsity rather than the original normalised L1 over L2
prior. Alternates a Krishnan style ADMM image update with grid search on sigma.
Produces both a classical reconstruction and a PnP-Flow reconstruction using
the estimated sigma so the failure mode (estimation versus prior) can be
isolated. Defaults run on 50 CelebA images at sigma=1.5 and eta=0.05.

```
python -m pnpflow.methods.blind.experiments.krishnan_baseline
```

To expand to other configurations:

```
python -m pnpflow.methods.blind.experiments.krishnan_baseline --dataset BSD68 --sigma 3.0
```

### 7. Render every paper figure from the JSON outputs

Loads the JSON results from each experiment directory and produces the
figures used in the thesis (deviation curves, sigma error and PSNR gap
heatmaps, SURE curves, per image scatters and errorbars, multi image
sweep panels, ablation bar charts, A.14 sigma trajectory). Skips any
experiment whose JSON output is not present.

```
python -m pnpflow.methods.blind.experiments.plots
```

Figures are written to `results/blind/figures/`.

## Qualitative reconstruction images

Three of the experiments save clean / observed / blind / oracle PNG
quadruplets for the first eight images at sigma=1.5, noise=0.05.

- `blur_sure_full` writes to `results/blind/blur_sure_full/qualitative/`
- `multi_image` writes to `results/blind/multi_image/qualitative/B=1/` and `B=64/`
- `extended_evaluation` writes to `results/blind/extended_evaluation/qualitative/<dataset>/`

## Smoke run

A short single image sanity check using `main.py`.

```
python -m pnpflow.methods.blind.main --dataset CelebA --sigma 1.5 --num-images 2 --pnp-steps 20 --device cpu
```

## Datasets

- CelebA is loaded via the upstream `DataLoaders` helper from `pnpflow/dataloaders.py`. The data must be staged at `data/celeba/` exactly as the upstream framework expects.
- BSD68 should be staged as PNG files at `data/BSD68/`.
- Set12 should be staged as PNG files at `data/Set12/`.

## Model

All experiments use the pretrained CelebA flow matching model from the
upstream framework, located at `model/celeba/ot/model_final.pt`.

## Output format

Each experiment writes one or more JSON files under
`results/blind/<experiment>/`. Numerical results are stored as plain
floats and lists. The `main.py` entry point additionally saves PNG
reconstructions per image.

## Citation and acknowledgements

If you use this code, please cite both the present thesis and the
upstream PnP-Flow paper:

- Martin et al., *PnP-Flow* (ICLR 2025). https://arxiv.org/abs/2410.02423

The PnP-Flow framework, the OT flow-matching architecture and
checkpoints, the upstream training pipeline, and the CelebA dataset
loader are taken unchanged from
[github.com/annegnx/PnP-Flow](https://github.com/annegnx/PnP-Flow). I gratefully
acknowledge the upstream authors, whose work this thesis
builds upon. License terms of the upstream code (BSD 3-Clause) apply to
all unmodified upstream files.
