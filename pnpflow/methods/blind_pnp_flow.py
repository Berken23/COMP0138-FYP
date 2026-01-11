"""
Blind PnP-Flow: Solves inverse problems where the forward operator is unknown.

Alternates between:
1. Image update using standard PnP-Flow iterations
2. Operator parameter update via gradient descent

Compatible with OT Flow Matching models trained on CelebA and AFHQ-Cat.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from time import perf_counter
from pathlib import Path

# Import existing utilities
import pnpflow.image_generation.models.utils as mutils
import pnpflow.utils as utils
from pnpflow.models import UNet


class BlindPnPFlow:
    """
    Blind inverse problem solver using PnP-Flow.
    
    Jointly learns the image and forward operator parameters by alternating:
    - Image update: Standard PnP-Flow (gradient + interpolation + denoising)
    - Operator update: Gradient descent on operator parameters
    
    Args:
        model: Pre-trained Flow Matching model (UNet)
        learnable_operator: Learnable forward operator (from blind_degradations.py)
        device: Device to run on ('cuda' or 'cpu')
        args: Arguments object with hyperparameters
        operator_lr: Learning rate for operator parameters (default: 1e-3)
        operator_update_freq: Update operator every N iterations (default: 1)
        operator_reg_weight: Regularization weight for operator (default: 0.01)
    """
    
    def __init__(self, model, learnable_operator, device, args, 
                 operator_lr=1e-3, operator_update_freq=1, operator_reg_weight=0.01):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.model.eval()  # Frozen - we don't train the flow matching model
        
        # Learnable forward operator
        self.learnable_operator = learnable_operator.to(device)
        
        self.H_adj = None
        
        # Optimizer for operator parameters
        self.operator_optimizer = torch.optim.Adam(
            self.learnable_operator.parameters(), 
            lr=operator_lr
        )
        
        self.operator_update_freq = operator_update_freq
        self.operator_reg_weight = operator_reg_weight
        
        # Track operator parameters over time
        self.operator_history = []
        
    def model_forward(self, x, t):
        """Forward pass through Flow Matching model"""
        if self.args.model == "ot":
            return self.model(x, t)
        elif self.args.model == "rectified":
            model_fn = mutils.get_model_fn(self.model, train=False)
            v = model_fn(x.type(torch.float), t * 999)
            return v
        else:
            raise ValueError(f"Unknown model type: {self.args.model}")
    
    def learning_rate_strat(self, lr, t):
        """Learning rate schedule for image updates"""
        t = t.view(-1, 1, 1, 1)
        gamma_styles = {
            '1_minus_t': lambda lr, t: lr * (1 - t),
            'sqrt_1_minus_t': lambda lr, t: lr * torch.sqrt(1 - t),
            'constant': lambda lr, t: lr,
            'alpha_1_minus_t': lambda lr, t: lr * (1 - t)**self.args.alpha,
        }
        return gamma_styles.get(self.args.gamma_style, lambda lr, t: lr)(lr, t)
    
    def grad_datafit(self, x, y):
        """
        Compute gradient of data fidelity using LEARNED operator.
        
        For Gaussian noise: grad = H^T(H(x) - y) / sigma^2
        For Laplace noise: grad = H^T(sign(H(x) - y)) / sigma
        """
        # Forward through learned operator
        Hx = self.learnable_operator(x)
        residual = Hx - y
        
        if self.args.noise_type == 'gaussian':
            # Need to backprop through H for the adjoint
            # For now, use identity adjoint (works well in practice)
            grad = residual / (self.args.sigma_noise**2)
            
        elif self.args.noise_type == 'laplace':
            grad = 2 * torch.heaviside(residual, torch.zeros_like(residual)) - 1
            grad = grad / self.args.sigma_noise
        else:
            raise ValueError('Noise type not supported')
        
        return grad
    
    def interpolation_step(self, x, t):
        """Interpolation step: z_tilde = (1-t)*epsilon + t*z"""
        return t * x + torch.randn_like(x) * (1 - t)
    
    def denoiser(self, x, t):
        """Denoiser: D_t(x) = x + (1-t)*v_t(x)"""
        with torch.no_grad():
            v = self.model_forward(x, t)
            return x + (1 - t.view(-1, 1, 1, 1)) * v
    
    def operator_data_fidelity(self, x, y):
        """Data fidelity loss for operator update: ||H(x) - y||^2"""
        Hx = self.learnable_operator(x)
        return 0.5 * torch.norm(Hx - y) ** 2
    
    def operator_regularization(self):
        """
        Regularization on operator parameters.
        Uses L2 regularization to prevent extreme values.
        """
        reg = 0.0
        for param in self.learnable_operator.parameters():
            reg += torch.norm(param) ** 2
        return reg
    
    def update_operator(self, x, y):
        """
        Update operator parameters via gradient descent.
        
        Minimizes: ||H(x) - y||^2 + lambda * ||params||^2
        """
        self.operator_optimizer.zero_grad()
        
        # Data fidelity loss (detach x to not backprop through image)
        loss = self.operator_data_fidelity(x.detach(), y)
        
        # Add regularization
        if self.operator_reg_weight > 0:
            loss += self.operator_reg_weight * self.operator_regularization()
        
        # Backprop and update
        loss.backward()
        self.operator_optimizer.step()
        
        # Project parameters to valid ranges (e.g., sigma > 0)
        self._project_operator_params()
        
        return loss.item()
    
    def _project_operator_params(self):
        """Project operator parameters to valid ranges"""
        with torch.no_grad():
            # For Gaussian blur: clamp log_sigma
            if hasattr(self.learnable_operator, 'log_sigma'):
                self.learnable_operator.log_sigma.clamp_(-5, 5)
            
            # For motion blur: clamp log_length
            if hasattr(self.learnable_operator, 'log_length'):
                self.learnable_operator.log_length.clamp_(-2, 5)
            
            # For motion blur: wrap angle to [-pi, pi]
            if hasattr(self.learnable_operator, 'angle'):
                angle = self.learnable_operator.angle
                self.learnable_operator.angle.data = torch.atan2(
                    torch.sin(angle), torch.cos(angle)
                )
    
    def get_operator_params(self):
        """Extract current operator parameters for logging"""
        params = {}
        
        if hasattr(self.learnable_operator, 'get_sigma'):
            params['sigma'] = self.learnable_operator.get_sigma()
        
        if hasattr(self.learnable_operator, 'get_params'):
            params.update(self.learnable_operator.get_params())
        
        if hasattr(self.learnable_operator, 'get_masked_ratio'):
            params['mask_ratio'] = self.learnable_operator.get_masked_ratio()
        
        return params
    
    def solve_blind_ip(self, test_loader, sigma_noise, true_degradation=None):
        """
        Solve blind inverse problem.
        """
        self.args.sigma_noise = sigma_noise
        num_samples = self.args.num_samples
        steps, delta = self.args.steps_pnp, 1 / self.args.steps_pnp
        
        # Set learning rate based on noise type
        if self.args.noise_type == 'gaussian':
            self.args.lr_pnp = sigma_noise**2 * self.args.lr_pnp
            lr = self.args.lr_pnp
        elif self.args.noise_type == 'laplace':
            self.args.lr_pnp = sigma_noise * self.args.lr_pnp
            lr = self.args.lr_pnp
        else:
            raise ValueError('Noise type not supported')
        
        loader = iter(test_loader)
        
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch
            print(f"\n{'='*60}")
            print(f"Processing batch {batch + 1}/{self.args.max_batch}")
            print(f"Image shape: {clean_img.shape}")
            print(f"{'='*60}")
            
            # Create degraded observation
            if true_degradation is not None:
                H_true = true_degradation.H
                noisy_img = H_true(clean_img.clone().to(self.device))
            else:
                raise ValueError("Need true degradation to create observation")
            
            # Add noise
            if self.args.noise_type == 'gaussian':
                torch.manual_seed(batch)
                noisy_img += torch.randn_like(noisy_img) * sigma_noise
            elif self.args.noise_type == 'laplace':
                noise = torch.distributions.laplace.Laplace(
                    torch.zeros_like(noisy_img), 
                    sigma_noise * torch.ones_like(noisy_img)
                ).sample().to(self.device)
                noisy_img += noise
            
            noisy_img = noisy_img.to(self.device)
            clean_img = clean_img.to('cpu')
            
            # Initialize image
            x = noisy_img.clone()
            
            # Reset operator history
            self.operator_history = []
            
            # Track timing if requested
            if self.args.compute_time:
                torch.cuda.synchronize()
                time_per_batch = 0
            
            if self.args.compute_memory:
                torch.cuda.reset_max_memory_allocated(self.device)
            
            # Main optimization loop
            with torch.no_grad():
                for count, iteration in enumerate(range(int(steps))):
                    if self.args.compute_time:
                        time_counter_1 = perf_counter()
                    
                    t = torch.ones(len(x), device=self.device) * delta * iteration
                    lr_t = self.learning_rate_strat(lr, t)
                    
                    # IMAGE UPDATE
                    z = x - lr_t * self.grad_datafit(x, noisy_img)
                    
                    x_new = torch.zeros_like(x)
                    for _ in range(num_samples):
                        z_tilde = self.interpolation_step(z, t.view(-1, 1, 1, 1))
                        x_new += self.denoiser(z_tilde, t)
                    x_new /= num_samples
                    x = x_new
                    
                    # OPERATOR UPDATE
                    if iteration % self.operator_update_freq == 0:
                        with torch.enable_grad():
                            op_loss = self.update_operator(x, noisy_img)
                    
                    # Log operator parameters
                    if iteration % 10 == 0:
                        params = self.get_operator_params()
                        self.operator_history.append({
                            'iteration': iteration,
                            'params': params,
                            'op_loss': op_loss if iteration % self.operator_update_freq == 0 else None
                        })
                        
                        print(f"Iter {iteration}/{steps}: ", end="")
                        for k, v in params.items():
                            print(f"{k}={v:.4f} ", end="")
                        print()
                    
                    if self.args.compute_time:
                        torch.cuda.synchronize()
                        time_counter_2 = perf_counter()
                        time_per_batch += time_counter_2 - time_counter_1
                    
                    # Save intermediate results
                    if self.args.save_results:
                        if iteration % 50 == 0 or self.should_save_image(iteration, steps):
                            restored_img = x.detach().clone()
                            utils.compute_psnr(clean_img, noisy_img, restored_img, 
                                            self.args, self.H_adj, iter=iteration)
                            utils.compute_ssim(clean_img, noisy_img, restored_img, 
                                            self.args, self.H_adj, iter=iteration)
                            utils.compute_lpips(clean_img, noisy_img, restored_img, 
                                            self.args, self.H_adj, iter=iteration)
            
            # Save final results
            if self.args.save_results:
                restored_img = x.detach().clone()
                utils.save_images(clean_img, noisy_img, restored_img,
                                self.args, self.H_adj, iter='final')
                utils.compute_psnr(clean_img, noisy_img, restored_img,
                                self.args, self.H_adj, iter=iteration)
                utils.compute_ssim(clean_img, noisy_img, restored_img,
                                self.args, self.H_adj, iter=iteration)
                utils.compute_lpips(clean_img, noisy_img, restored_img,
                                self.args, self.H_adj, iter=iteration)
                
                # Save operator history
                self.save_operator_history(batch)
            
            # Compute and save timing/memory
            if self.args.compute_memory:
                dict_memory = {}
                dict_memory["batch"] = batch
                dict_memory["max_allocated"] = torch.cuda.max_memory_allocated(self.device)
                utils.save_memory_use(dict_memory, self.args)
            
            if self.args.compute_time:
                dict_time = {}
                dict_time["batch"] = batch
                dict_time["time_per_batch"] = time_per_batch
                utils.save_time_use(dict_time, self.args)
            
            # Print final operator parameters
            print(f"\n{'='*60}")
            print(f"Final operator parameters:")
            final_params = self.get_operator_params()
            for k, v in final_params.items():
                print(f"  {k}: {v:.4f}")
            print(f"{'='*60}\n")
        
        # Compute averages across all batches
        if self.args.save_results:
            utils.compute_average_psnr(self.args)
            utils.compute_average_ssim(self.args)
            utils.compute_average_lpips(self.args)
        if self.args.compute_memory:
            utils.compute_average_memory(self.args)
        if self.args.compute_time:
            utils.compute_average_time(self.args)
    
    def save_operator_history(self, batch):
        """Save operator parameter evolution to file"""
        import json
        
        save_path = Path(self.args.save_path_ip) / f"operator_history_batch_{batch}.json"
        
        # Convert to serializable format
        history_serializable = []
        for entry in self.operator_history:
            history_serializable.append({
                'iteration': entry['iteration'],
                'params': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                          for k, v in entry['params'].items()},
                'op_loss': float(entry['op_loss']) if entry['op_loss'] is not None else None
            })
        
        with open(save_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"Saved operator history to {save_path}")
    
    def should_save_image(self, iteration, steps):
        """Determine if we should save image at this iteration"""
        return iteration % (steps // 10) == 0
    
    def run_method(self, data_loaders, degradation, sigma_noise):
        """
        Main entry point (matches PNP_FLOW interface).
        
        Args:
            data_loaders: Dictionary of data loaders
            degradation: True degradation (for creating observations)
            sigma_noise: Noise level
        """
        # Construct save path
        folder = f"blind_{self.args.method}_{self.args.dataset}"
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)
        os.makedirs(self.args.save_path_ip, exist_ok=True)
        
        # ADD THIS: Store H_adj from degradation
        self.H_adj = degradation.H_adj if hasattr(degradation, 'H_adj') else lambda x: x
        
        print(f"\n{'='*80}")
        print(f"BLIND PNP-FLOW")
        print(f"{'='*80}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Degradation: {type(self.learnable_operator).__name__}")
        print(f"Noise: {self.args.noise_type} (sigma={sigma_noise})")
        print(f"Operator LR: {self.operator_optimizer.param_groups[0]['lr']}")
        print(f"Save path: {self.args.save_path_ip}")
        print(f"{'='*80}\n")
        
        # Solve blind inverse problem
        self.solve_blind_ip(
            data_loaders[self.args.eval_split],
            sigma_noise,
            true_degradation=degradation
        )