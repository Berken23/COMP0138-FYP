# Interim Report: Blind Plug-and-Play Flow Matching for Image Restoration

## Work Completed

The project extends PnP-Flow (Gagneux et al., ICLR 2025), a method combining Plug-and-Play priors with Flow Matching generative models for solving image inverse problems, to the *blind* setting where the degradation operator is unknown. The original PnP-Flow framework assumes the forward operator H is known exactly; our contribution relaxes this assumption by jointly estimating the blur kernel width sigma alongside the image reconstruction. The codebase builds on the official PnP-Flow repository with pre-trained Optimal Transport Flow Matching models on CelebA (64x64 face images), and all development has been conducted in a single Colab-compatible notebook (`blind_pnp_flow_colab.ipynb`).

Four of the six project aims have been implemented and tested. **Aim 3** (non-blind deconvolution) established baseline reconstruction quality by comparing four gamma scheduling strategies for the PnP-Flow data-fit step weighting. **Aim 4** (single-image blind deconvolution) is the core contribution, implementing a per-step Adam baseline (Algorithm A.14) and a novel search-initialised approach using grid search with parabolic refinement at intermediate flow time. **Aim 5** (multi-image blind inference) demonstrated that sharing a learnable sigma across B images reduces estimation variance, with batch sizes B in {1, 2, 4, 8}. **Aim 6** (optional trajectory analysis) verified the near-linearity of OT flow trajectories and fitted a power-law scaling relationship for sigma error vs batch size.

A key technical finding was the *identity trap*: when sigma is initialised below the true value, the blur operator approaches the identity, causing sigma to collapse toward zero. We also discovered that the data-fit loss landscape is well-posed for sigma estimation only at intermediate flow times (around t = 0.2), shifting its minimum toward zero at later times as the denoiser removes blur information. These findings motivated the initialisation-from-above strategy and the search-based approach.

All code is implemented in `blind_pnp_flow_colab.ipynb`. The following has been completed:

- Implemented `LearnableGaussianBlur` operator parameterised by log_sigma with differentiable kernel generation and gradient support
- Implemented `BlindGaussianBlurProblem` dataclass encapsulating ground truth, observation, true sigma, and operator configuration
- Implemented `blind_alternating_descent` — pure data-fit alternating gradient descent baseline (no generative prior)
- Implemented `_pnp_flow_trajectory_blind` — PnP-Flow trajectory with learnable operator, supporting per-step sigma updates via Adam
- Implemented `blind_pnp_flow_single` — per-step Adam blind PnP-Flow following Algorithm A.14 from the paper
- Implemented `blind_pnp_flow_robust` — novel search-initialised method using multi-pass median estimation with Adam refinement
- Implemented `blind_pnp_flow_multi` — multi-image blind PnP-Flow with shared sigma across B observations
- Implemented Aim 3 schedule comparison (`constant`, `1_minus_t`, `sqrt_1_minus_t`, `alpha_1_minus_t`) with PSNR bar charts and image grids
- Implemented Aim 4 ablation studies for sigma_freeze_iters and gamma_style with convergence plots
- Implemented Aim 6 trajectory linearity analysis and power-law scaling study (sigma error vs B across multiple seeds)

## Work To Be Done

| Milestone | Description | Target Date |
|-----------|-------------|-------------|
| **Novel method refinement** | Finalise the robust search-initialised blind PnP-Flow (multi-pass median estimation with Adam refinement). Run full evaluation on all 8 test images and compare against per-step Adam baseline. Address Image 4 outlier. | Week 1 (by 16 Mar) |
| **Module integration** | Port final working methods from the notebook into `blind_pnp_flow.py` as clean, tested Python functions. Ensure consistency with existing test suite. | Week 2 (by 23 Mar) |
| **Extended evaluation** | Test on additional degradation levels (varying sigma_true, noise levels), potentially on AFHQ-CAT dataset. Compute PSNR, SSIM, and LPIPS metrics systematically. | Week 3 (by 30 Mar) |
| **Report writing** | Draft the final report covering all aims, including methodology, experimental results, ablation studies, and discussion of the identity trap and loss landscape findings. | Weeks 4-5 (by 13 Apr) |
| **Final submission** | Polish report, ensure reproducibility of all experiments, clean up repository. | Week 6 (by 20 Apr) |

The main remaining technical challenge is improving the robustness of the novel search-initialised method. The current implementation uses 5 estimation passes with independent noise realisations and takes the median as a robust sigma estimate, followed by 10 Adam refinement iterations. This has been implemented but not yet fully evaluated. If the median-based approach proves insufficiently robust, fallback strategies include increasing the number of passes, ensembling multiple search times, or reverting to the per-step Adam baseline (which is itself a valid contribution as a blind extension of PnP-Flow).

## Code Summary

The implementation is organised around three main components. First, `blind_degradations.py` defines `LearnableGaussianBlur`, a differentiable Gaussian blur operator parameterised by `log_sigma` (ensuring positivity via exponentiation), which dynamically generates convolution kernels and supports gradient-based optimisation. Second, `blind_data.py` provides the `BlindGaussianBlurProblem` dataclass that encapsulates a test instance (ground truth image, observation, true sigma, and operator configuration). Third, the solver methods are implemented in `blind_pnp_flow.py` (clean module) and extensively developed in the Colab notebook.

The notebook (`blind_pnp_flow_colab.ipynb`, ~48 cells) serves as the primary development and experimentation environment. It is structured into sections corresponding to each project aim: setup and data loading, helper functions including the PnP-Flow trajectory implementation, Aim 3 schedule analysis, Aim 4 blind deconvolution with ablation studies, Aim 5 multi-image experiments, and Aim 6 scaling analysis. Each section includes both the experimental code and markdown findings cells for recording results.

The test suite in `pnpflow/tests/` contains 12+ unit and integration tests covering the learnable operator (kernel generation, parameter clamping), the sigma recovery pipeline (fixed x, recover sigma), and the alternating descent loop (stability, directionality, convergence). These tests validate the core blind estimation machinery and were developed alongside the implementation to catch regressions.

## Literature Review

The literature review for this project is complete. It covers the foundational PnP framework (Venkatakrishnan et al., 2013; Romano et al., 2017), flow matching and score-based generative models (Lipman et al., 2023; Liu et al., 2023; Tong et al., 2023), the PnP-Flow paper itself (Gagneux et al., ICLR 2025), blind inverse problem methods including alternating minimisation and expectation-maximisation approaches, and related work on diffusion-based blind deconvolution. Key references also include DiffPIR (Zhu et al., 2023), D-Flow (Ben-Hamu et al., 2024), and the DeepInverse library (Tachella et al.) which provides baseline implementations. The review situates our contribution — extending PnP-Flow to the blind setting with a novel search-initialised sigma estimation strategy — within the broader landscape of learning-based image restoration methods.
