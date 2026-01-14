"""
Blind PnP-Flow: Solves inverse problems where the forward operator is unknown.

Alternates between:
1) Image update using PnP-Flow (data gradient + interpolation + FM denoiser)
2) Operator parameter update via gradient descent

This implementation fixes the main correctness pitfalls:
- Uses the correct image-gradient w.r.t. x via autograd (implicitly includes H_phi^T)
- Never disables grad globally during the image-gradient computation
- Does NOT mutate args.lr_pnp in-place (avoids compounding across runs)
- Keeps the FM model frozen and inference-only
"""

import os
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils


class BlindPnPFlow:
    def __init__(
        self,
        model: nn.Module,
        learnable_operator: nn.Module,
        device: torch.device,
        args: Any,
        operator_lr: float = 1e-3,
        operator_update_freq: int = 1,
        operator_reg_weight: float = 0.01,
    ):
        self.device = device
        self.args = args

        # --- Frozen Flow Matching model (prior) ---
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # --- Learnable forward operator H_phi ---
        self.learnable_operator = learnable_operator.to(device)
        self.learnable_operator.eval()  # keep deterministic if possible

        self.operator_optimizer = torch.optim.Adam(
            self.learnable_operator.parameters(),
            lr=operator_lr,
        )
        self.operator_update_freq = max(1, int(operator_update_freq))
        self.operator_reg_weight = float(operator_reg_weight)

        self.operator_history = []
        self.H_adj = None  # only used for legacy metric funcs in your utils

    # -------------------------
    # Flow Matching pieces
    # -------------------------

    def model_forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return velocity v_theta(t, x)."""
        if self.args.model == "ot":
            return self.model(x, t)
        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            return model_fn(x.type(torch.float), t * 999)
        raise ValueError(f"Unknown model type: {self.args.model}")

    def denoiser(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        D_t(x) = x + (1 - t) * v_theta(t, x)
        where t is shape (B,) and we broadcast to (B,1,1,1).
        """
        with torch.no_grad():
            v = self.model_forward(x, t)
            t4 = t.view(-1, 1, 1, 1)
            return x + (1.0 - t4) * v

    def interpolation_step(self, z: torch.Tensor, t4: torch.Tensor) -> torch.Tensor:
        """
        z_tilde = (1 - t) * eps + t * z
        Caller provides t4 with shape (B,1,1,1).
        """
        return t4 * z + (1.0 - t4) * torch.randn_like(z)

    # -------------------------
    # Step-size schedule
    # -------------------------

    def learning_rate_strat(self, base_lr: float, t: torch.Tensor) -> torch.Tensor:
        """
        Return gamma(t) with broadcasting shape (B,1,1,1).
        """
        t4 = t.view(-1, 1, 1, 1)
        style = getattr(self.args, "gamma_style", "alpha_1_minus_t")

        if style == "1_minus_t":
            return base_lr * (1.0 - t4)
        if style == "sqrt_1_minus_t":
            return base_lr * torch.sqrt(torch.clamp(1.0 - t4, min=0.0))
        if style == "constant":
            return torch.full_like(t4, float(base_lr))
        # default: alpha_1_minus_t
        alpha = float(getattr(self.args, "alpha", 1.0))
        return base_lr * (1.0 - t4) ** alpha

    def _time_at_iter(self, iteration: int, steps: int, device: torch.device, batch_size: int) -> torch.Tensor:
        """
        Produces t in [0,1]. Using (steps-1) makes final step hit t=1 exactly (if steps>1).
        """
        if steps <= 1:
            val = 0.0
        else:
            val = iteration / (steps - 1)
        return torch.full((batch_size,), float(val), device=device)

    # -------------------------
    # Correct data fidelity gradient wrt x
    # -------------------------

    def grad_datafit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Correct gradient w.r.t x using autograd:
        Gaussian: F = (1/(2*sigma^2)) * ||H_phi(x)-y||^2
        Laplace:  F = (1/sigma) * ||H_phi(x)-y||_1  (subgradient)
        """
        sigma = float(self.args.sigma_noise)

        # Robust even if caller is in no_grad by accident:
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            if "add_noise" in self.learnable_operator.forward.__code__.co_varnames:
                Hx = self.learnable_operator(x_req, add_noise=False)
            else:
                Hx = self.learnable_operator(x_req)

            residual = Hx - y

            if self.args.noise_type == "gaussian":
                loss = 0.5 * residual.pow(2).sum() / (sigma ** 2)
            elif self.args.noise_type == "laplace":
                loss = residual.abs().sum() / sigma
            else:
                raise ValueError("Noise type not supported")

            grad = torch.autograd.grad(loss, x_req, create_graph=False, retain_graph=False)[0]

        return grad.detach()

    # -------------------------
    # Operator update
    # -------------------------

    def operator_regularization(self) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        for p in self.learnable_operator.parameters():
            reg = reg + p.pow(2).sum()
        return reg

    def _project_operator_params(self) -> None:
        """Project operator parameters to valid ranges (customize per operator family)."""
        with torch.no_grad():
            if hasattr(self.learnable_operator, "log_sigma"):
                self.learnable_operator.log_sigma.clamp_(-5, 5)
            if hasattr(self.learnable_operator, "log_length"):
                self.learnable_operator.log_length.clamp_(-2, 5)
            if hasattr(self.learnable_operator, "angle"):
                angle = self.learnable_operator.angle
                self.learnable_operator.angle.data = torch.atan2(torch.sin(angle), torch.cos(angle))

    def update_operator(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Minimize over phi: 0.5||H_phi(x)-y||^2 + lambda||phi||^2
        x is detached to block gradients into the image.
        """
        with torch.enable_grad():
            self.operator_optimizer.zero_grad(set_to_none=True)

            x_det = x.detach()
            if "add_noise" in self.learnable_operator.forward.__code__.co_varnames:
                Hx = self.learnable_operator(x_det, add_noise=False)
            else:
                Hx = self.learnable_operator(x_det)
            loss = 0.5 * (Hx - y).pow(2).sum()

            if self.operator_reg_weight > 0:
                loss = loss + self.operator_reg_weight * self.operator_regularization()

            loss.backward()
            self.operator_optimizer.step()
            self._project_operator_params()

        return float(loss.item())

    def get_operator_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if hasattr(self.learnable_operator, "get_sigma"):
            params["sigma"] = float(self.learnable_operator.get_sigma())
        if hasattr(self.learnable_operator, "get_params"):
            extra = self.learnable_operator.get_params()
            # Ensure python floats where possible
            for k, v in extra.items():
                try:
                    params[k] = float(v)
                except Exception:
                    params[k] = v
        if hasattr(self.learnable_operator, "get_masked_ratio"):
            params["mask_ratio"] = float(self.learnable_operator.get_masked_ratio())
        return params

    # -------------------------
    # Main blind solver
    # -------------------------

    def solve_blind_ip(self, test_loader, sigma_noise: float, true_degradation=None) -> None:
        self.args.sigma_noise = float(sigma_noise)

        steps = int(self.args.steps_pnp)
        num_samples = int(self.args.num_samples)
        max_batch = int(self.args.max_batch)

        # IMPORTANT: do NOT mutate args.lr_pnp in-place
        base_lr = float(self.args.lr_pnp)
        if self.args.noise_type == "gaussian":
            lr = (sigma_noise ** 2) * base_lr
        elif self.args.noise_type == "laplace":
            lr = sigma_noise * base_lr
        else:
            raise ValueError("Noise type not supported")

        loader = iter(test_loader)

        for batch in range(max_batch):
            clean_img, labels = next(loader)
            self.args.batch = batch

            print(f"\n{'='*60}")
            print(f"Processing batch {batch + 1}/{max_batch}")
            print(f"Image shape: {clean_img.shape}")
            print(f"{'='*60}")

            # Create observation y using true degradation (only for synthetic evaluation)
            if true_degradation is None:
                raise ValueError("Need true_degradation to create observation for evaluation.")
            H_true = true_degradation.H
            y = H_true(clean_img.clone().to(self.device))

            # Add noise
            if self.args.noise_type == "gaussian":
                torch.manual_seed(batch)
                y = y + torch.randn_like(y) * sigma_noise
            else:  # laplace
                noise = torch.distributions.laplace.Laplace(
                    torch.zeros_like(y),
                    sigma_noise * torch.ones_like(y),
                ).sample().to(self.device)
                y = y + noise

            clean_cpu = clean_img.to("cpu")
            y = y.to(self.device)

            # Initialize x (simple, but valid)
            x = y.clone()

            # Reset operator history
            self.operator_history = []

            # Timing/memory
            if getattr(self.args, "compute_time", False):
                torch.cuda.synchronize()
                time_per_batch = 0.0
            else:
                time_per_batch = None

            if getattr(self.args, "compute_memory", False):
                torch.cuda.reset_max_memory_allocated(self.device)

            # Main alternating loop
            for iteration in range(steps):
                if time_per_batch is not None:
                    t1 = perf_counter()

                t = self._time_at_iter(iteration, steps, self.device, batch_size=len(x))
                lr_t = self.learning_rate_strat(lr, t)  # (B,1,1,1)

                # ---- IMAGE UPDATE (PnP-Flow) ----
                grad_F = self.grad_datafit(x, y)  # correct grad wrt x

                with torch.no_grad():
                    z = x - lr_t * grad_F
                    x_acc = torch.zeros_like(x)
                    t4 = t.view(-1, 1, 1, 1)
                    for _ in range(num_samples):
                        z_tilde = self.interpolation_step(z, t4)
                        x_acc = x_acc + self.denoiser(z_tilde, t)
                    x = x_acc / max(1, num_samples)

                # ---- OPERATOR UPDATE ----
                op_loss = None
                if iteration % self.operator_update_freq == 0:
                    op_loss = self.update_operator(x, y)

                # Logging operator params
                if iteration % 10 == 0:
                    params = self.get_operator_params()
                    self.operator_history.append(
                        {"iteration": iteration, "params": params, "op_loss": op_loss}
                    )
                    print(f"Iter {iteration}/{steps}: ", end="")
                    for k, v in params.items():
                        if isinstance(v, (float, int, np.number)):
                            print(f"{k}={float(v):.4f} ", end="")
                        else:
                            print(f"{k}={v} ", end="")
                    if op_loss is not None:
                        print(f"op_loss={op_loss:.4e}", end="")
                    print()

                # Timing
                if time_per_batch is not None:
                    torch.cuda.synchronize()
                    t2 = perf_counter()
                    time_per_batch += (t2 - t1)

                # Save intermediate metrics
                if getattr(self.args, "save_results", False):
                    if iteration % 50 == 0 or self.should_save_image(iteration, steps):
                        restored = x.detach().clone()
                        utils.compute_psnr(clean_cpu, y, restored, self.args, self.H_adj, iter=iteration)
                        utils.compute_ssim(clean_cpu, y, restored, self.args, self.H_adj, iter=iteration)
                        utils.compute_lpips(clean_cpu, y, restored, self.args, self.H_adj, iter=iteration)

            # Final save
            if getattr(self.args, "save_results", False):
                restored = x.detach().clone()
                utils.save_images(clean_cpu, y, restored, self.args, self.H_adj, iter="final")
                utils.compute_psnr(clean_cpu, y, restored, self.args, self.H_adj, iter="final")
                utils.compute_ssim(clean_cpu, y, restored, self.args, self.H_adj, iter="final")
                utils.compute_lpips(clean_cpu, y, restored, self.args, self.H_adj, iter="final")
                self.save_operator_history(batch)

            # Memory/time summaries per batch
            if getattr(self.args, "compute_memory", False):
                utils.save_memory_use(
                    {"batch": batch, "max_allocated": torch.cuda.max_memory_allocated(self.device)},
                    self.args,
                )
            if time_per_batch is not None:
                utils.save_time_use({"batch": batch, "time_per_batch": time_per_batch}, self.args)

            # Print final operator params
            print(f"\n{'='*60}")
            print("Final operator parameters:")
            final_params = self.get_operator_params()
            for k, v in final_params.items():
                if isinstance(v, (float, int, np.number)):
                    print(f"  {k}: {float(v):.6f}")
                else:
                    print(f"  {k}: {v}")
            print(f"{'='*60}\n")

        # Averages across batches
        if getattr(self.args, "save_results", False):
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if getattr(self.args, "compute_memory", False):
            utils.compute_average_memory(self.args)
        if getattr(self.args, "compute_time", False):
            utils.compute_average_time(self.args)

    # -------------------------
    # IO / helpers
    # -------------------------

    def save_operator_history(self, batch: int) -> None:
        save_path = Path(self.args.save_path_ip) / f"operator_history_batch_{batch}.json"
        with open(save_path, "w") as f:
            json.dump(self.operator_history, f, indent=2)
        print(f"Saved operator history to {save_path}")

    def should_save_image(self, iteration: int, steps: int) -> bool:
        stride = max(1, steps // 10)
        return iteration % stride == 0

    def run_method(self, data_loaders, degradation, sigma_noise: float) -> None:
        folder = f"blind_{self.args.method}_{self.args.dataset}"
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)
        os.makedirs(self.args.save_path_ip, exist_ok=True)

        # For your metric code compatibility
        self.H_adj = degradation.H_adj if hasattr(degradation, "H_adj") else (lambda x: x)

        print(f"\n{'='*80}")
        print("BLIND PNP-FLOW")
        print(f"{'='*80}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Learnable operator: {type(self.learnable_operator).__name__}")
        print(f"Noise: {self.args.noise_type} (sigma={sigma_noise})")
        print(f"Operator LR: {self.operator_optimizer.param_groups[0]['lr']}")
        print(f"Save path: {self.args.save_path_ip}")
        print(f"{'='*80}\n")

        self.solve_blind_ip(
            data_loaders[self.args.eval_split],
            sigma_noise,
            true_degradation=degradation,
        )
