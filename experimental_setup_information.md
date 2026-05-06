# Experimental Setup — Information Pack

This document contains every piece of information needed to write the
Experimental Setup chapter of the dissertation. It is organised under the
eight categories specified in `experimental_setup_requirements.md`. All
values reflect the actual implementation in `pnpflow/methods/blind/`.

---

## 1. Datasets

Three datasets, all preprocessed to 128 by 128 pixels and normalised to the range minus one to one.

### CelebA

- A large scale face attribute dataset of celebrity images.
- 250 images randomly selected from the test split using a fixed seed of 42 for reproducibility.
- Native resolution roughly 178 by 218 RGB. Resized then centre cropped to 128 by 128 RGB. Normalised channel wise to minus one to one.
- Source: the standard CelebA test split exposed by the upstream PnP-Flow `DataLoaders` helper.
- Chosen because the OT flow matching prior used in PnP-Flow was trained on CelebA. Evaluation here measures the method under conditions that match the training distribution of the prior, isolating sigma estimation as the primary source of error.

### BSD68

- Berkeley Segmentation Dataset 68 image test set, the standard benchmark for image denoising and restoration on natural scenes.
- All 68 images.
- Originally grayscale at varying resolutions. Each image is resized then centre cropped to 128 by 128, the single channel is replicated to three channels for compatibility with the RGB flow prior, then normalised to minus one to one.
- Source: the `clausmichele/CBSD68-dataset` GitHub repository (`CBSD68/original_png/`).
- Chosen because BSD68 contains diverse natural scenes that lie outside the face distribution of the flow matching prior. Performance here probes the generalisation of the proposed pipeline beyond its training distribution.

### Set12

- A collection of 12 classical grayscale test images widely used in image restoration evaluation.
- All 12 images.
- Native resolution varies. Each image is resized then centre cropped to 128 by 128, the single channel is replicated to three channels, then normalised to minus one to one.
- Source: the `cszn/KAIR` GitHub repository (`testsets/set12/`).
- Chosen because Set12 contains classical test images such as Lena, Cameraman, and Barbara that bear little resemblance to faces. Performance here is the strongest test of out of distribution generalisation.

---

## 2. Degradation Conditions

Identical degradation protocol across all three datasets. Each clean image x is degraded as

```
y = H_sigma(x) + eta * epsilon,    epsilon ~ N(0, I)
```

where H_sigma denotes isotropic Gaussian blur with standard deviation sigma applied in the FFT domain with circular boundary conditions, and eta is the additive Gaussian noise standard deviation.

- **Blur parameter sigma**: six values, namely 0.5, 1.0, 1.5, 2.0, 3.0, 4.0. The same blur model is used during generation and during sigma estimation, satisfying the operator consistency condition derived in the methodology.
- **Noise standard deviation eta**: three values, namely 0.01, 0.05, 0.1.
- **Total combinations**: 6 sigma × 3 eta equals 18 distinct configurations evaluated on every image of every dataset.
- **Reproducibility**: the noise epsilon is sampled with a per image seed (`torch.manual_seed(SEED + idx)`) so that all methods see the exact same observation y for a given (image, sigma, eta) tuple. This eliminates noise variability from the comparison between methods.

---

## 3. Baselines

The methodology compares against three classes of reference, namely an established failed approach from the literature, a controlled non blind upper bound, and a catalogue of nine alternative blind estimators implemented and rejected during development. No external blind deconvolution baselines such as Krishnan et al. or DeblurGAN are included in this evaluation.

### Naive joint estimation (A.14)

- The default approach to blind inverse problems described by Martin et al. (the PnP-Flow authors). At each step of the PnP-Flow trajectory, the operator parameter sigma is updated via Adam against the data fit residual `||H_sigma(x) - y||^2`.
- Implementation: ours, faithful to the algorithm described in the PnP-Flow paper. Source at `pnpflow/methods/blind/_helpers.py::pnp_flow_trajectory_blind`.
- Configuration: sigma initialised at 3.0, sigma maximum at 5.0, 30 outer iterations, 100 PnP-Flow steps per outer iteration, Adam learning rate 1e-3, gradient norm clipped at 1.0.
- Establishes the failure mode that motivates the proposed solution.

### Non blind oracle (upper bound)

- A controlled upper bound that runs PnP-Flow reconstruction with the true sigma supplied directly. No estimation involved.
- Implementation: identical to the proposed pipeline's reconstruction stage, with sigma fixed at the ground truth.
- Reports the best reconstruction quality achievable under the same flow matching prior. The PSNR gap between any blind method and this oracle is the primary measure of estimation quality reported throughout the results.

### Catalogue of nine failed alternative blind estimators

Implemented and reported as part of the failure mode analysis to demonstrate the systematic nature of the over sharpening bias. All produce sigma estimates that diverge from the true value. Listed for completeness.

1. Plain SGD on sigma in place of Adam.
2. Time restricted Adam updates with exponential learning rate decay across outer iterations.
3. Convergence detection by freezing sigma when its change falls below a threshold for several consecutive iterations.
4. Decoupled alternating minimisation with grid search over sigma, with both warm and fresh restart variants.
5. Expectation Maximisation with Tweedie estimates collected from the PnP-Flow denoiser.
6. EM with damping at eta = 0.1.
7. Morozov discrepancy stopping rule applied on top of EM with damping.
8. Residual whiteness imbalance using a low / high frequency power decomposition.
9. Type II marginal likelihood with analytical marginalisation under a diagonal Gaussian image prior.

All nine are implemented in `pnpflow/methods/blind/experiments/failed_approaches.py`.

---

## 4. Evaluation Metrics

Three standard metrics used throughout, all computed in the post normalisation range zero to one after clipping the reconstructions from minus one to one.

- **PSNR** (Peak Signal to Noise Ratio) in decibels. Higher is better. Reflects mean squared error in the pixel domain.
- **SSIM** (Structural Similarity Index). Range zero to one, higher is better. Computed via `skimage.metrics.structural_similarity` with `data_range=1.0` and `channel_axis=2`.
- **LPIPS** (Learned Perceptual Image Patch Similarity) using the AlexNet backbone. Lower is better. Captures perceptual similarity that PSNR does not.

The primary headline metric for the blind versus oracle comparison is the **PSNR gap**, defined as `psnr_oracle - psnr_blind`. A gap near zero indicates the blind method achieves the oracle ceiling.

---

## 5. Hyperparameter Values

Final configuration for the proposed pipeline. Values were either fixed by the design or selected via the ablation study described in section 8.

### Sigma estimation (Stage 1, blur-SURE)

- Search bounds: sigma_min = 0.3, sigma_max = 8.0 (covers the full tested grid with margin).
- Coarse grid size: M = 80 candidate sigmas.
- Refinement: bounded scalar minimisation (`scipy.optimize.minimize_scalar`, `method='bounded'`) within plus or minus two grid spacings around the coarse minimum, tolerance 1e-4, maximum 50 iterations.
- Regularisation parameter: lambda = 10 * eta^2 (universally optimal across all 18 (sigma, eta) configurations, see calibration below).
- The estimator depends only on the observation y and the noise variance eta^2. No reconstruction, no denoiser call, no model parameters are involved.

### Lambda calibration (one off, on synthetic data)

- Multiplier sweep: 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0.
- Calibration set: 10 synthetic noise realisations of uniform noise images at each of 6 sigmas and 3 eta values, total 1800 evaluations per multiplier.
- Selection criterion: minimum mean absolute sigma error.
- Result: lambda = 10 * eta^2 wins for all three eta levels.

### Reconstruction (Stage 2, PnP-Flow)

- Number of trajectory steps: T = 100.
- Step size schedule: gamma(t) = 1.0 (constant). Selected via the non blind ablation in section 8 because the constant schedule outperforms `1_minus_t` by +4.2 to +5.2 dB across all datasets.
- Data fit step size: 1.0.
- Number of Monte Carlo denoiser samples per step: S = 1 (the ablation found higher values gave no measurable improvement).
- Outer iterations: 1 (single pass, the ablation found higher values gave no measurable improvement).
- Warm start: not used (the ablation found no measurable difference).
- Initialisation: x = y (the blurred observation).
- Stochastic interpolation: `z_tilde = t * z + (1 - t) * epsilon` with `epsilon ~ N(0, I)`.

### Pretrained denoiser

- The OT flow matching network from Martin et al., loaded from `model/celeba/ot/model_final.pt` (132 MB).
- Architecture: U-Net for OT flow matching at 128 by 128 resolution, three channels, approximately 30 million parameters.
- Used in evaluation mode throughout, no fine tuning.

### Multi image variant

- Batch sizes evaluated: B in 1, 2, 4, 8, 16, 32, 64.
- Trials per (sigma, eta, B): 5.
- Reconstructions per B in the reconstruction comparison: 8 images, 3 seeds each.
- All other hyperparameters identical to the single image variant.

### Failed baseline (A.14)

- Sigma initial: 3.0.
- Sigma maximum: 5.0.
- Outer iterations: 30.
- PnP-Flow steps per outer iteration: 100.
- Adam learning rate on sigma: 1e-3.
- Gradient norm clipping: 1.0.
- Padding for `LearnableGaussianBlur`: reflect.

---

## 6. Implementation Details

### Software

- Python 3.10.
- PyTorch with CUDA 12.1 backend.
- Key dependencies: `torch`, `torchvision`, `numpy<2`, `scipy`, `scikit-image`, `lpips`, `Pillow`, `matplotlib`, `pytorch-ignite`, `deepinv`, `gdown`.
- The proposed method, all baselines, and the experiment scripts are part of a single Python package at `pnpflow/methods/blind/` of the project repository.

### Hardware

- All large scale experiments executed on a single NVIDIA H100 PCIe GPU (80 GB HBM3) rented through Lambda Cloud.
- Local development and smoke testing performed on CPU.

### Runtime

- Sigma estimation via blur-SURE: approximately 0.014 seconds per image on H100.
- A single PnP-Flow reconstruction trajectory of 100 steps: approximately 1.3 seconds per image on H100.
- A single naive A.14 outer iteration: approximately 4 seconds per image on H100, total 131.7 seconds per image with 30 outer iterations on the Colab GPU referenced in the original failure analysis.

### Reproducibility

- Code is publicly available at `https://github.com/Berken23/COMP0138-FYP`, branch `submission-refactor`.
- Random seed: base seed 42, with deterministic offsets per image (`SEED + idx`) and per reconstruction seed (`SEED + idx * 100 + seed_offset`).
- All datasets, model checkpoints, configuration values, and command line invocations required to reproduce every result are documented in `pnpflow/methods/blind/README.md`.
- Each experiment writes per image and aggregate results to `results/blind/<experiment>/` as JSON, alongside qualitative reconstruction PNGs at the (sigma, eta) configuration of the headline figures.

---

## 7. Experimental Procedure

### Single image evaluation on CelebA (`blur_sure_full`)

For each of the 18 (sigma, eta) configurations and each of the 250 CelebA images.

1. Generate the observation y with the seeded noise sample.
2. Estimate sigma via blur-SURE (single shot, no iteration).
3. Reconstruct three times with three different denoiser seeds. Record mean and standard deviation of PSNR, SSIM, LPIPS across seeds.
4. Run the non blind oracle once with the true sigma.
5. Compute the per image PSNR gap.

Aggregated as mean across the 250 images. Total reconstructions: 250 × 18 × 4 equals 18000.

### Extended evaluation on BSD68 and Set12 (`extended_evaluation`)

Identical protocol to the CelebA evaluation, applied to all 68 BSD68 images and all 12 Set12 images. Total reconstructions: (68 + 12) × 18 × 4 equals 5760.

### Multi image evaluation (`multi_image`)

Two parts.

- **Part 1, sigma estimation accuracy versus B**. For each (sigma, eta, B) and 5 trials, draw B observations of distinct images blurred with the same sigma and independent noise, then estimate sigma by averaging the B blur-SURE curves. Report mean and standard deviation of the absolute sigma error across trials.
- **Part 2, reconstruction comparison versus B**. At fixed sigma equal to 1.5 and eta equal to 0.05, estimate sigma using B observations, then reconstruct 8 held out images with the resulting sigma. Three reconstruction seeds per image. Run the non blind oracle for direct gap comparison.

Total in Part 1: 6 × 3 × 7 × 5 SURE evaluations equals 630. Total in Part 2: 7 × 8 × 4 trajectories equals 224.

### Non blind ablation (`non_blind_ablation`)

Six one parameter ablations evaluated at sigma = 1.5 and eta = 0.05, on 50 images per dataset.

- Gamma schedule (3 settings): `1_minus_t`, `sqrt_1_minus_t`, `constant`.
- Number of trajectory steps (4 settings): 25, 50, 100, 200.
- Data fit learning rate (3 settings): 0.5, 1.0, 2.0.
- Number of denoiser samples (3 settings): 1, 3, 5.
- Outer iterations (5 settings): 1, 5, 10, 20, 30.
- Warm start versus fresh start (2 settings).

Total trajectories approximately 3000. Each ablation isolates one hyperparameter while holding the others at the default.

### Trajectory straightness analysis (`trajectory_straightness`)

For each of three datasets, six sigma values, three gamma schedules, and 30 images per condition, the full PnP-Flow trajectory is recorded at every step. The RMSE deviation from the straight line interpolation between the initial and final iterates is reported per flow time t.

### Failed baseline (`a14_baseline`)

10 CelebA images at sigma = 1.5, eta = 0.05, sigma_init = 3.0, 30 outer iterations of A.14 per step Adam on sigma.

### Failed approaches catalogue (`failed_approaches`)

10 CelebA images at sigma = 1.5, eta = 0.05, with each of the nine alternative estimators run for the iteration count appropriate to its design (typically 30 outer iterations for the alternating methods, 40 EM iterations for the EM variants, 20 iterations for the spectral methods). Each estimator's sigma history, residual history, gradient signals, and final error are recorded.

### Aggregation conventions

- All scalar metrics are reported as mean across images at each (dataset, sigma, eta) cell.
- Where stochastic variation is reported, it is the standard deviation across the three reconstruction seeds.
- The PSNR gap is the cell mean of the per image (oracle PSNR minus blind PSNR).

---

## 8. Ablation Studies

Four families of controlled experiments isolate the contribution of individual design choices.

### Non blind hyperparameter ablation

Tests whether the proposed pipeline's reconstruction stage is sensitive to its hyperparameters when sigma is correct. Each parameter is varied while others are held at the defaults listed in section 5. Hypothesis: the reconstruction stage should be robust to most hyperparameters, with the gamma schedule being the dominant factor (because attenuating the data fit gradient late in the trajectory discards measurement consistency at the most refined timesteps). Conditions: sigma = 1.5, eta = 0.05, all three datasets, 50 images per dataset.

### Trajectory straightness analysis

Tests whether the rectified flow trajectory is approximately straight in practice, validating the linear interpolation assumption used in the methodology. Hypothesis: deviations should be small and largely independent of sigma and the gamma schedule. Conditions: 6 sigma values, 3 gamma schedules, 3 datasets, 30 images per cell.

### Multi image variance reduction

Tests whether averaging blur-SURE curves over B independent observations reduces estimation error. Hypothesis: the dominant source of error is image content dependent bias, not stochastic noise, so averaging should reduce variance but not bias. Conditions: B in 1, 2, 4, 8, 16, 32, 64; full sigma and eta grid; 5 trials per cell.

### Lambda calibration

Tests which value of the Wiener regularisation constant lambda minimises sigma estimation error across the full sigma and eta grid. Hypothesis: a single multiplier of eta^2 should generalise across all noise levels, simplifying deployment. Conditions: 10 lambda multipliers spanning 5 orders of magnitude, 6 sigma values, 3 eta values, 10 calibration trials per cell.

---

## Notes for the Writer

- Cite the methodology chapter when referring to the FFT Gaussian operator, the two necessary conditions for unbiased sigma estimation, the over sharpening bias, and the blur-SURE derivation.
- Cross reference the results chapter for the specific tables of PSNR gaps, sigma errors, and metric breakdowns.
- The repository link, the README path, and the JSON output paths are stable and may be referenced in an appendix on reproducibility.
- The thesis style avoids colons, em dashes, and proper names where citations alone suffice. The values above are written in plain prose where possible to make this easier.
