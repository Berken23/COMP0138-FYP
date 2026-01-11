"""
Blind PnP-Flow with Langevin Dynamics for Posterior Sampling

Extends deterministic blind reconstruction to sample from p(x, θ_H | y).
Generates multiple plausible solutions with uncertainty quantification and calibration.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from time import perf_counter
import matplotlib.pyplot as plt
import json

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
    
    def compute_enhanced_uncertainty_metrics(self, batch, clean_img, 
                                            image_samples, operator_samples):
        """
        Enhanced uncertainty computation with full statistics.
        
        Returns comprehensive uncertainty metrics including:
        - Pixel-wise statistics (mean, variance, std, entropy)
        - Confidence intervals (50%, 90%, 95%)
        - Sample diversity
        - Operator statistics
        """
        # Stack samples
        images_tensor = torch.stack(image_samples)  # [N, B, C, H, W]
        
        # ============================================
        # 1. PIXEL-WISE STATISTICS
        # ============================================
        mean_image = images_tensor.mean(dim=0)
        variance_map = images_tensor.var(dim=0)
        std_map = variance_map.sqrt()
        
        # ============================================
        # 2. CONFIDENCE INTERVALS
        # ============================================
        # 50% CI (25th and 75th percentiles)
        ci_50_lower = torch.quantile(images_tensor, 0.25, dim=0)
        ci_50_upper = torch.quantile(images_tensor, 0.75, dim=0)
        ci_50_width = ci_50_upper - ci_50_lower
        
        # 90% CI (5th and 95th percentiles)
        ci_90_lower = torch.quantile(images_tensor, 0.05, dim=0)
        ci_90_upper = torch.quantile(images_tensor, 0.95, dim=0)
        ci_90_width = ci_90_upper - ci_90_lower
        
        # 95% CI (2.5th and 97.5th percentiles)
        ci_95_lower = torch.quantile(images_tensor, 0.025, dim=0)
        ci_95_upper = torch.quantile(images_tensor, 0.975, dim=0)
        ci_95_width = ci_95_upper - ci_95_lower
        
        # ============================================
        # 3. PREDICTIVE ENTROPY (per pixel)
        # ============================================
        # Approximate with Gaussian assumption: H = 0.5 * log(2πe * σ²)
        entropy_map = 0.5 * torch.log(2 * np.pi * np.e * (variance_map + 1e-8))
        
        # ============================================
        # 4. SAMPLE DIVERSITY
        # ============================================
        # Pairwise L2 distance between samples
        pairwise_dists = []
        for i in range(len(image_samples)):
            for j in range(i+1, len(image_samples)):
                dist = torch.norm(image_samples[i] - image_samples[j])
                pairwise_dists.append(dist.item())
        mean_diversity = np.mean(pairwise_dists) if pairwise_dists else 0.0
        
        # ============================================
        # 5. OPERATOR STATISTICS
        # ============================================
        operator_means = {}
        operator_stds = {}
        operator_ci_95 = {}
        
        for key in operator_samples[0].keys():
            values = np.array([sample[key] for sample in operator_samples])
            operator_means[key] = np.mean(values)
            operator_stds[key] = np.std(values)
            operator_ci_95[key] = (
                np.percentile(values, 2.5),
                np.percentile(values, 97.5)
            )
        
        # ============================================
        # 6. SAVE UNCERTAINTY DATA
        # ============================================
        save_dir = Path(self.args.save_path_ip) / f"batch_{batch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all uncertainty maps
        torch.save({
            'mean_image': mean_image,
            'variance_map': variance_map,
            'std_map': std_map,
            'entropy_map': entropy_map,
            'ci_50_lower': ci_50_lower,
            'ci_50_upper': ci_50_upper,
            'ci_90_lower': ci_90_lower,
            'ci_90_upper': ci_90_upper,
            'ci_95_lower': ci_95_lower,
            'ci_95_upper': ci_95_upper,
        }, save_dir / "uncertainty_maps.pt")
        
        # Save operator statistics
        with open(save_dir / "operator_statistics.json", 'w') as f:
            json.dump({
                'means': {k: float(v) for k, v in operator_means.items()},
                'stds': {k: float(v) for k, v in operator_stds.items()},
                'ci_95': {k: [float(v[0]), float(v[1])] for k, v in operator_ci_95.items()},
                'sample_diversity': float(mean_diversity),
            }, f, indent=2)
        
        # ============================================
        # 7. PRINT SUMMARY
        # ============================================
        print(f"\n{'='*60}")
        print("UNCERTAINTY QUANTIFICATION")
        print(f"{'='*60}")
        print(f"Pixel-wise uncertainty:")
        print(f"  Mean std: {std_map.mean():.6f}")
        print(f"  Max std: {std_map.max():.6f}")
        print(f"  Mean entropy: {entropy_map.mean():.6f}")
        print(f"\nConfidence interval widths (mean):")
        print(f"  50% CI: {ci_50_width.mean():.6f}")
        print(f"  90% CI: {ci_90_width.mean():.6f}")
        print(f"  95% CI: {ci_95_width.mean():.6f}")
        print(f"\nSample diversity:")
        print(f"  Mean pairwise L2: {mean_diversity:.6f}")
        print(f"\nOperator Statistics:")
        for key in operator_means:
            ci_low, ci_high = operator_ci_95[key]
            print(f"  {key}: {operator_means[key]:.4f} ± {operator_stds[key]:.4f}")
            print(f"         95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        print(f"{'='*60}\n")
        
        return {
            'mean_image': mean_image,
            'variance_map': variance_map,
            'std_map': std_map,
            'entropy_map': entropy_map,
            'ci_50_width': ci_50_width.mean().item(),
            'ci_90_width': ci_90_width.mean().item(),
            'ci_95_width': ci_95_width.mean().item(),
            'sample_diversity': mean_diversity,
            'operator_mean': operator_means,
            'operator_std': operator_stds,
            'operator_ci_95': operator_ci_95,
        }
    
    def compute_calibration_metrics(self, batch, clean_img, image_samples):
        """
        Compute calibration metrics to validate uncertainty estimates.
        
        Checks if stated confidence matches actual coverage.
        """
        if clean_img is None:
            print("No ground truth available, skipping calibration")
            return None
        
        images_tensor = torch.stack(image_samples).to(self.device)
        ground_truth = clean_img.to(self.device)
        
        # ============================================
        # 1. COVERAGE AT DIFFERENT CONFIDENCE LEVELS
        # ============================================
        confidence_levels = [0.50, 0.68, 0.90, 0.95, 0.99]
        coverages = {}
        
        for conf in confidence_levels:
            alpha = (1 - conf) / 2
            ci_lower = torch.quantile(images_tensor, alpha, dim=0)
            ci_upper = torch.quantile(images_tensor, 1 - alpha, dim=0)
            
            in_interval = (ground_truth >= ci_lower) & (ground_truth <= ci_upper)
            coverage = in_interval.float().mean().item()
            coverages[f'{int(conf*100)}%'] = coverage
        
        # ============================================
        # 2. EXPECTED CALIBRATION ERROR (ECE)
        # ============================================
        # ECE = mean absolute deviation from perfect calibration
        ece = 0
        for conf in confidence_levels:
            expected = conf
            actual = coverages[f'{int(conf*100)}%']
            ece += abs(expected - actual)
        ece /= len(confidence_levels)
        
        # ============================================
        # 3. SHARPNESS
        # ============================================
        # Average width of 95% CI
        ci_95_lower = torch.quantile(images_tensor, 0.025, dim=0)
        ci_95_upper = torch.quantile(images_tensor, 0.975, dim=0)
        sharpness = (ci_95_upper - ci_95_lower).mean().item()
        
        # ============================================
        # 4. CALIBRATION CURVE
        # ============================================
        num_bins = 10
        conf_levels = np.linspace(0.1, 0.99, num_bins)
        predicted_conf = []
        actual_coverage = []
        
        for conf in conf_levels:
            alpha = (1 - conf) / 2
            ci_lower = torch.quantile(images_tensor, alpha, dim=0)
            ci_upper = torch.quantile(images_tensor, 1 - alpha, dim=0)
            in_interval = (ground_truth >= ci_lower) & (ground_truth <= ci_upper)
            coverage = in_interval.float().mean().item()
            
            predicted_conf.append(conf)
            actual_coverage.append(coverage)
        
        # ============================================
        # 5. SAVE CALIBRATION DATA
        # ============================================
        save_dir = Path(self.args.save_path_ip) / f"batch_{batch}"
        
        calibration_data = {
            'coverages': coverages,
            'ece': float(ece),
            'sharpness': float(sharpness),
            'calibration_curve': {
                'predicted': predicted_conf,
                'actual': actual_coverage,
            }
        }
        
        with open(save_dir / "calibration_metrics.json", 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # ============================================
        # 6. PLOT CALIBRATION CURVE
        # ============================================
        self.plot_calibration_curve(
            predicted_conf, actual_coverage, ece,
            save_dir / "calibration_curve.png"
        )
        
        # ============================================
        # 7. PRINT SUMMARY
        # ============================================
        print(f"\n{'='*60}")
        print("CALIBRATION METRICS")
        print(f"{'='*60}")
        print("Coverage (expected → actual):")
        for level, cov in coverages.items():
            expected = float(level.rstrip('%')) / 100
            status = "✓" if abs(expected - cov) < 0.05 else "↓"
            print(f"  {level}: {cov:.3f} (expected: {expected:.2f}) {status}")
        print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
        print(f"  {'✓ Well-calibrated' if ece < 0.05 else '↓  Needs improvement'}")
        print(f"\nSharpness (95% CI width): {sharpness:.4f}")
        print(f"  (Lower is better, but must maintain calibration)")
        print(f"{'='*60}\n")
        
        return calibration_data
    
    def plot_calibration_curve(self, predicted_conf, actual_coverage, ece, save_path):
        """Plot calibration curve"""
        plt.figure(figsize=(8, 8))
        
        # Perfect calibration (diagonal)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        
        # Actual calibration
        plt.plot(predicted_conf, actual_coverage, 'bo-', 
                label='Model Calibration', linewidth=2, markersize=8)
        
        # Fill area (calibration error)
        plt.fill_between(predicted_conf, predicted_conf, actual_coverage, 
                        alpha=0.3, color='red', 
                        label=f'ECE = {ece:.3f}')
        
        plt.xlabel('Predicted Confidence', fontsize=14)
        plt.ylabel('Actual Coverage', fontsize=14)
        plt.title('Calibration Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved calibration curve to {save_path}")
    
    def visualize_uncertainty(self, batch, image_samples, uncertainty_metrics):
        """Create uncertainty visualizations"""
        save_dir = Path(self.args.save_path_ip) / f"batch_{batch}"
        
        mean_image = uncertainty_metrics['mean_image'][0]  # [C, H, W]
        std_map = uncertainty_metrics['std_map'][0]  # [C, H, W]
        
        # ============================================
        # 1. UNCERTAINTY HEATMAP
        # ============================================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Mean reconstruction
        axes[0].imshow(mean_image.permute(1, 2, 0).cpu().numpy().clip(0, 1))
        axes[0].set_title('Mean Reconstruction', fontsize=14)
        axes[0].axis('off')
        
        # Uncertainty (std map)
        im = axes[1].imshow(std_map.mean(dim=0).cpu().numpy(), cmap='hot')
        axes[1].set_title('Pixel-wise Uncertainty (Std)', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(mean_image.permute(1, 2, 0).cpu().numpy().clip(0, 1), alpha=0.7)
        im2 = axes[2].imshow(std_map.mean(dim=0).cpu().numpy(), cmap='hot', alpha=0.5)
        axes[2].set_title('Reconstruction + Uncertainty', fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / "uncertainty_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved uncertainty heatmap")
        
        # ============================================
        # 2. SAMPLE GRID
        # ============================================
        num_show = min(9, len(image_samples))
        nrows = int(np.sqrt(num_show))
        ncols = int(np.ceil(num_show / nrows))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 3*nrows))
        if num_show == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, sample in enumerate(image_samples[:num_show]):
            axes[i].imshow(sample[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axes[i].set_title(f'Sample {i+1}', fontsize=10)
            axes[i].axis('off')
        
        # Hide extra subplots
        for i in range(num_show, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / "samples_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved samples grid")
    
    def visualize_operator_distribution(self, batch, operator_samples, true_params=None):
        """Plot operator parameter distributions"""
        save_dir = Path(self.args.save_path_ip) / f"batch_{batch}"
        
        for param_name in operator_samples[0].keys():
            values = [sample[param_name] for sample in operator_samples]
            
            plt.figure(figsize=(10, 6))
            
            # Histogram
            plt.hist(values, bins=30, density=True, alpha=0.7, 
                    color='blue', edgecolor='black', label='Posterior Samples')
            
            # Mean and std
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')
            plt.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2)
            plt.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2,
                       label=f'±1 Std: {std_val:.3f}')
            
            # True value (if known)
            if true_params and param_name in true_params:
                true_val = true_params[param_name]
                plt.axvline(true_val, color='green', linestyle='-', linewidth=2,
                           label=f'True: {true_val:.3f}')
            
            plt.xlabel(f'{param_name}', fontsize=14)
            plt.ylabel('Density', fontsize=14)
            plt.title(f'Posterior Distribution of {param_name}', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.savefig(save_dir / f"operator_{param_name}_distribution.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved operator distributions")
    
    def solve_blind_ip_sampling(self, test_loader, sigma_noise, 
                                num_samples=10, burn_in=50, thinning=5,
                                true_operator_params=None):
        """
        Solve blind inverse problem with posterior sampling and full analysis.
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
            
            # Create observation
            if hasattr(self, 'true_degradation'):
                y_obs = self.true_degradation.H(clean_img.clone().to(self.device))
                if self.args.noise_type == 'gaussian':
                    torch.manual_seed(batch)
                    y_obs += torch.randn_like(y_obs) * sigma_noise
            else:
                raise ValueError("Need true degradation for synthetic validation")
            
            y_obs = y_obs.to(self.device)
            clean_img_device = clean_img.to(self.device)
            
            # Generate posterior samples
            image_samples, operator_samples = self.sample_posterior(
                y_obs, num_samples, burn_in, thinning
            )
            
            # Save samples
            self.save_posterior_samples(
                batch, clean_img, y_obs, 
                image_samples, operator_samples
            )
            
            # Compute enhanced uncertainty metrics
            uncertainty_metrics = self.compute_enhanced_uncertainty_metrics(
                batch, clean_img, 
                image_samples, operator_samples
            )
            
            # Compute calibration (if ground truth available)
            calibration_metrics = self.compute_calibration_metrics(
                batch, clean_img_device, image_samples
            )
            
            # Visualizations
            self.visualize_uncertainty(batch, image_samples, uncertainty_metrics)
            self.visualize_operator_distribution(batch, operator_samples, true_operator_params)
    
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
        op_path = save_dir / "operator_samples.json"
        with open(op_path, 'w') as f:
            json.dump(operator_samples, f, indent=2)
        
        print(f"✓ Saved {len(image_samples)} samples to {save_dir}")
    
    def run_method(self, data_loaders, degradation, sigma_noise, 
                   num_samples=10, burn_in=50, thinning=5,
                   true_operator_params=None):
        """
        Main entry point for posterior sampling.
        
        Args:
            true_operator_params: Dict of true operator parameters (for validation)
                                 e.g., {'sigma': 2.5}
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
        
        # Run posterior sampling with full analysis
        self.solve_blind_ip_sampling(
            data_loaders[self.args.eval_split],
            sigma_noise,
            num_samples=num_samples,
            burn_in=burn_in,
            thinning=thinning,
            true_operator_params=true_operator_params
        )