"""
Blind PnP-Flow with Langevin Dynamics for Posterior Sampling

Extends deterministic blind reconstruction to sample from p(x, θ_H | y).
Generates multiple plausible solutions with uncertainty quantification.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from time import perf_counter

from pnpflow.methods.blind_pnp_flow import BlindPnPFlow
import pnpflow.utils as utils


class BlindPnPFlowLangevin(BlindPnPFlow):
    """
    Posterior sampling variant of Blind PnP-Flow using Langevin dynamics.
    
    Instead of finding a single (x*, θ*), generates samples from p(x, θ | y).
    
    Args:
        model: Flow Matching model
        learnable_operator: Learnable degradation operator
        device: 'cuda' or 'cpu'
        args: Configuration arguments
        operator_lr: Learning rate for operator
        operator_update_freq: Update operator every N iterations
        operator_reg_weight: Regularization weight
        langevin_temp_image: Temperature for image sampling (default: 0.01)
        langevin_temp_operator: Temperature for operator sampling (default: 0.001)
    """
    
    def __init__(self, model, learnable_operator, device, args,
                 operator_lr=1e-3, operator_update_freq=1, operator_reg_weight=0.01,
                 langevin_temp_image=0.01, langevin_temp_operator=0.001):
        
        # Initialize parent class
        super().__init__(model, learnable_operator, device, args,
                        operator_lr, operator_update_freq, operator_reg_weight)
        
        # Langevin parameters
        self.langevin_temp_image = langevin_temp_image
        self.langevin_temp_operator = langevin_temp_operator
        
        # Storage for samples
        self.image_samples = []
        self.operator_samples = []
    
    def langevin_noise(self, shape, temperature, device):
        """
        Generate Langevin noise: √(2η) · ε where ε ~ N(0, I)
        
        Args:
            shape: Shape of noise tensor
            temperature: Langevin temperature η
            device: Device to create noise on
            
        Returns:
            Noise tensor of given shape
        """
        noise_scale = np.sqrt(2 * temperature)
        return noise_scale * torch.randn(shape, device=device)
    
    def denoiser_with_langevin(self, x, t):
        """
        Denoiser with Langevin noise: D_t(x) + √(2η) · ε
        
        This turns the deterministic denoiser into a stochastic sampler.
        """
        # Standard denoising
        x_denoised = self.denoiser(x, t)
        
        # Add Langevin noise
        langevin_noise = self.langevin_noise(
            x_denoised.shape, 
            self.langevin_temp_image, 
            self.device
        )
        
        return x_denoised + langevin_noise
    
    def update_operator_with_langevin(self, x, y):
        """
        Update operator with Langevin dynamics.
        
        Standard update: θ = θ - lr·∇L(θ)
        Langevin update: θ = θ - lr·∇L(θ) + √(2η)·ε
        """
        # Standard gradient update
        self.operator_optimizer.zero_grad()
        
        loss = self.operator_data_fidelity(x.detach(), y)
        if self.operator_reg_weight > 0:
            loss += self.operator_reg_weight * self.operator_regularization()
        
        loss.backward()
        
        # BEFORE optimizer step: add Langevin noise to gradients
        with torch.no_grad():
            for param in self.learnable_operator.parameters():
                if param.grad is not None:
                    # Add Langevin noise
                    noise = self.langevin_noise(
                        param.grad.shape,
                        self.langevin_temp_operator,
                        self.device
                    )
                    param.grad += noise
        
        self.operator_optimizer.step()
        self._project_operator_params()
        
        return loss.item()
    
    def sample_posterior(self, y_obs, num_samples=10, burn_in=50, thinning=5):
        """
        Generate posterior samples p(x, θ | y).
        
        Args:
            y_obs: Observed degraded image
            num_samples: Number of samples to generate
            burn_in: Number of iterations to discard (burn-in period)
            thinning: Save every N-th sample (reduces correlation)
            
        Returns:
            image_samples: List of reconstructed images
            operator_samples: List of operator parameters
        """
        total_iterations = burn_in + num_samples * thinning
        
        print(f"\n{'='*80}")
        print(f"POSTERIOR SAMPLING")
        print(f"{'='*80}")
        print(f"Samples: {num_samples}")
        print(f"Burn-in: {burn_in}")
        print(f"Thinning: {thinning}")
        print(f"Total iterations: {total_iterations}")
        print(f"{'='*80}\n")
        
        # Initialize
        x = y_obs.clone()
        samples_collected = 0
        
        image_samples = []
        operator_samples = []
        
        # Sampling loop
        with torch.no_grad():
            for iteration in range(total_iterations):
                t = torch.ones(len(x), device=self.device) * (iteration / total_iterations)
                lr_t = self.learning_rate_strat(self.args.lr_pnp, t)
                
                # ============================================
                # IMAGE UPDATE WITH LANGEVIN
                # ============================================
                z = x - lr_t * self.grad_datafit(x, y_obs)
                
                # Interpolation + Langevin denoising
                x_new = torch.zeros_like(x)
                for _ in range(self.args.num_samples):
                    z_tilde = self.interpolation_step(z, t.view(-1, 1, 1, 1))
                    x_new += self.denoiser_with_langevin(z_tilde, t)
                x_new /= self.args.num_samples
                x = x_new
                
                # ============================================
                # OPERATOR UPDATE WITH LANGEVIN
                # ============================================
                if iteration % self.operator_update_freq == 0:
                    with torch.enable_grad():
                        self.update_operator_with_langevin(x, y_obs)
                
                # ============================================
                # COLLECT SAMPLES (after burn-in, with thinning)
                # ============================================
                if iteration >= burn_in and (iteration - burn_in) % thinning == 0:
                    # Save sample
                    image_samples.append(x.detach().clone().cpu())
                    operator_samples.append(self.get_operator_params().copy())
                    samples_collected += 1
                    
                    if samples_collected % 5 == 0:
                        print(f"Collected {samples_collected}/{num_samples} samples")
        
        print(f"\n✓ Sampling complete: {len(image_samples)} samples collected\n")
        
        return image_samples, operator_samples
    
    def solve_blind_ip_sampling(self, test_loader, sigma_noise, 
                                num_samples=10, burn_in=50, thinning=5):
        """
        Solve blind inverse problem with posterior sampling.
        
        Generates multiple solutions instead of single point estimate.
        """
        self.args.sigma_noise = sigma_noise
        
        # Set learning rate
        if self.args.noise_type == 'gaussian':
            self.args.lr_pnp = sigma_noise**2 * self.args.lr_pnp
        elif self.args.noise_type == 'laplace':
            self.args.lr_pnp = sigma_noise * self.args.lr_pnp
        
        loader = iter(test_loader)
        
        for batch in range(self.args.max_batch):
            (clean_img, labels) = next(loader)
            self.args.batch = batch
            
            print(f"\n{'='*60}")
            print(f"Processing batch {batch + 1}/{self.args.max_batch}")
            print(f"{'='*60}")
            
            # Create observation (for synthetic validation)
            # In real blind problems, you'd just have y_obs
            if hasattr(self, 'true_degradation'):
                y_obs = self.true_degradation.H(clean_img.clone().to(self.device))
                if self.args.noise_type == 'gaussian':
                    torch.manual_seed(batch)
                    y_obs += torch.randn_like(y_obs) * sigma_noise
            else:
                raise ValueError("Need true degradation for synthetic validation")
            
            y_obs = y_obs.to(self.device)
            clean_img = clean_img.to('cpu')
            
            # Generate posterior samples
            image_samples, operator_samples = self.sample_posterior(
                y_obs, num_samples, burn_in, thinning
            )
            
            # Save samples
            self.save_posterior_samples(
                batch, clean_img, y_obs, 
                image_samples, operator_samples
            )
            
            # Compute uncertainty metrics
            self.compute_uncertainty_metrics(
                batch, clean_img, 
                image_samples, operator_samples
            )
    
    def save_posterior_samples(self, batch, clean_img, y_obs, 
                              image_samples, operator_samples):
        """Save posterior samples to disk"""
        save_dir = Path(self.args.save_path_ip) / f"batch_{batch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images
        for i, img_sample in enumerate(image_samples):
            img_path = save_dir / f"sample_{i:03d}.pt"
            torch.save(img_sample, img_path)
        
        # Save operator parameters
        import json
        op_path = save_dir / "operator_samples.json"
        with open(op_path, 'w') as f:
            json.dump(operator_samples, f, indent=2)
        
        print(f"✓ Saved {len(image_samples)} samples to {save_dir}")
    
    def compute_uncertainty_metrics(self, batch, clean_img, 
                                   image_samples, operator_samples):
        """
        Compute uncertainty quantification metrics.
        
        Returns:
            metrics: Dictionary with:
                - pixel_variance: Per-pixel variance across samples
                - operator_mean: Mean operator parameters
                - operator_std: Std of operator parameters
                - coverage: Calibration metric
        """
        # Stack samples
        images_tensor = torch.stack(image_samples)  # [N, B, C, H, W]
        
        # Compute pixel-wise statistics
        mean_image = images_tensor.mean(dim=0)
        variance_map = images_tensor.var(dim=0)
        std_map = variance_map.sqrt()
        
        # Compute operator statistics
        operator_means = {}
        operator_stds = {}
        
        for key in operator_samples[0].keys():
            values = [sample[key] for sample in operator_samples]
            operator_means[key] = np.mean(values)
            operator_stds[key] = np.std(values)
        
        # Save uncertainty visualizations
        save_dir = Path(self.args.save_path_ip) / f"batch_{batch}"
        
        # Save variance map
        variance_path = save_dir / "pixel_variance.pt"
        torch.save(variance_map, variance_path)
        
        # Save operator statistics
        import json
        stats_path = save_dir / "operator_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump({
                'means': {k: float(v) for k, v in operator_means.items()},
                'stds': {k: float(v) for k, v in operator_stds.items()},
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print("UNCERTAINTY QUANTIFICATION")
        print(f"{'='*60}")
        print(f"Mean pixel std: {std_map.mean():.6f}")
        print(f"Max pixel std: {std_map.max():.6f}")
        print(f"\nOperator Statistics:")
        for key in operator_means:
            print(f"  {key}: {operator_means[key]:.4f} ± {operator_stds[key]:.4f}")
        print(f"{'='*60}\n")
        
        return {
            'pixel_variance': variance_map,
            'operator_mean': operator_means,
            'operator_std': operator_stds,
        }
    
    def run_method(self, data_loaders, degradation, sigma_noise, 
                   num_samples=10, burn_in=50, thinning=5):
        """
        Main entry point for posterior sampling.
        """
        # Store true degradation for creating observations
        self.true_degradation = degradation
        
        # Set H_adj
        self.H_adj = degradation.H_adj if hasattr(degradation, 'H_adj') else lambda x: x
        
        # Setup save path
        folder = f"blind_langevin_{self.args.method}_{self.args.dataset}"
        self.args.save_path_ip = os.path.join(self.args.save_path, folder)
        os.makedirs(self.args.save_path_ip, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"BLIND PNP-FLOW LANGEVIN SAMPLING")
        print(f"{'='*80}")
        print(f"Dataset: {self.args.dataset}")
        print(f"Degradation: {type(self.learnable_operator).__name__}")
        print(f"Noise: {self.args.noise_type} (sigma={sigma_noise})")
        print(f"Image temperature: {self.langevin_temp_image}")
        print(f"Operator temperature: {self.langevin_temp_operator}")
        print(f"{'='*80}\n")
        
        # Run posterior sampling
        self.solve_blind_ip_sampling(
            data_loaders[self.args.eval_split],
            sigma_noise,
            num_samples=num_samples,
            burn_in=burn_in,
            thinning=thinning
        )