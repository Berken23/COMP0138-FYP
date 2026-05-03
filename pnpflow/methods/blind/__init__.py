"""Blind PnP-Flow deconvolution.

This package implements the decoupled two stage pipeline developed for the
COMP0138 Final Year Project. It extends Martin et al.'s PnP-Flow framework
to the blind setting, in which the Gaussian blur parameter is unknown and
must be inferred jointly with the underlying image.

Public surface:

    estimate_sigma_blur_sure   sigma estimation from y alone
    pnp_flow_reconstruct       reconstruction at a fixed sigma
    non_blind_oracle           upper bound run with the true sigma
    evaluate                   PSNR, SSIM, LPIPS in one call
    load_dataset               CelebA, BSD68, Set12 loaders
    gaussian_blur_fft          forward operator
"""
from .forward import gaussian_blur_adjoint, gaussian_blur_fft
from .sigma_estimation import (
    compute_blur_sure,
    estimate_sigma_blur_sure,
    estimate_sigma_multi_sure,
)
from .reconstruction import pnp_flow_reconstruct
from .evaluation import (
    compute_lpips,
    compute_ssim,
    evaluate,
    non_blind_oracle,
    postprocess,
    psnr_db,
)
from .data import load_bsd68, load_celeba, load_dataset, load_set12
from ._helpers import (
    NOISE_LEVELS,
    SEED,
    SIGMA_VALUES,
    TimingTracker,
    pnp_flow_trajectory_blind,
    pnp_flow_trajectory_tracked,
)

__all__ = [
    "gaussian_blur_fft",
    "gaussian_blur_adjoint",
    "compute_blur_sure",
    "estimate_sigma_blur_sure",
    "estimate_sigma_multi_sure",
    "pnp_flow_reconstruct",
    "non_blind_oracle",
    "evaluate",
    "psnr_db",
    "compute_ssim",
    "compute_lpips",
    "postprocess",
    "load_dataset",
    "load_celeba",
    "load_bsd68",
    "load_set12",
    "TimingTracker",
    "pnp_flow_trajectory_blind",
    "pnp_flow_trajectory_tracked",
    "SIGMA_VALUES",
    "NOISE_LEVELS",
    "SEED",
]
