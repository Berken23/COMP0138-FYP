"""
Test Langevin with optimal parameters for publication-quality results
"""

import torch
import sys
from pathlib import Path
import time

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from pnpflow.utils import define_model, load_model
from pnpflow.methods.blind_pnp_flow_langevin import BlindPnPFlowLangevin
from pnpflow.blind_degradations import LearnableGaussianBlur
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def find_model_path(dataset='celeba'):
    """Find model checkpoint"""
    script_dir = Path(__file__).resolve().parent
    possible_paths = [
        script_dir.parent.parent / 'model' / dataset / 'ot' / 'model_final.pt',
        script_dir.parent / 'model' / dataset / 'ot' / 'model_final.pt',
        Path('model') / dataset / 'ot' / 'model_final.pt',
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None


def create_test_args():
    """Args for optimal test"""
    class Args:
        def __init__(self):
            self.dataset = 'celeba'
            self.dim_image = 128
            self.num_channels = 3
            self.model = 'ot'
            self.root = ''
            self.problem = 'blind_gaussian_deblurring'
            self.method = 'blind_pnp_flow'
            self.noise_type = 'gaussian'
            self.steps_pnp = 100
            self.lr_pnp = 1.0
            self.alpha = 0.5
            self.gamma_style = 'alpha_1_minus_t'
            self.num_samples = 3
            self.eval_split = 'test'
            self.max_batch = 1
            self.batch_size_ip = 1
            self.save_results = True
            self.save_path = 'results/test_langevin_optimal'
            self.compute_time = False
            self.compute_memory = False
            self.train = False
            self.eval = True
            self.dict_cfg_method = {
                'steps_pnp': self.steps_pnp,
                'lr_pnp': self.lr_pnp,
            }
    return Args()


class TrueDegradation:
    def __init__(self, sigma, device):
        self.sigma = sigma
        self.device = device
    
    def H(self, x):
        kernel_size = 61
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=self.device)
        yy, xx = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        pad = kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='circular')
        return F.conv2d(x_padded, kernel, groups=3)
    
    def H_adj(self, y):
        return y


def test_langevin_optimal():
    """Test with optimal parameters for publication-quality results"""
    print("\n" + "="*80)
    print("LANGEVIN SAMPLING WITH OPTIMAL PARAMETERS")
    print("="*80)
    print("\nConfiguration:")
    print("  • Burn-in: 1000 iterations")
    print("  • Samples: 20")
    print("  • Thinning: 50")
    print("  • Total: 2000 iterations")
    print("  • Temperature: 0.1 (image), 0.01 (operator)")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cpu':
        print("  WARNING: This will take ~30-45 minutes on CPU")
        print("   Consider using GPU for faster results")
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return False
    
    args = create_test_args()
    
    # Load model
    print("\n1. Loading model...")
    model, state = define_model(args)
    model_path = find_model_path('celeba')
    
    if model_path is None:
        print("ERROR: Model not found")
        return False
    
    load_model('ot', model, state, download=False,
              checkpoint_path=model_path, dataset=None, device=device)
    model.eval()
    print("✓ Model loaded")
    
    # Create synthetic problem
    print("\n2. Creating synthetic problem...")
    clean_img = torch.randn(1, 3, 128, 128, device=device)
    true_sigma = 2.5
    
    learnable_blur = LearnableGaussianBlur(
        kernel_size=61,
        num_channels=3,
        init_sigma=1.0,
        device=device
    )
    print("✓ Created learnable operator")
    print(f"  Initial σ: 1.0 (misspecified)")
    print(f"  True σ: {true_sigma}")
    
    # Create Langevin sampler
    print("\n3. Creating Langevin sampler...")
    sampler = BlindPnPFlowLangevin(
        model=model,
        learnable_operator=learnable_blur,
        device=device,
        args=args,
        operator_lr=1e-3,
        langevin_temp_image=0.1,
        langevin_temp_operator=0.01
    )
    print("✓ Sampler created")
    
    # Create degradation
    true_degradation = TrueDegradation(true_sigma, device)
    
    # Create dataloader
    dataset = TensorDataset(clean_img.cpu(), torch.zeros(1))
    test_loader = DataLoader(dataset, batch_size=1)
    data_loaders = {'test': test_loader}
    
    # Run sampling
    print("\n4. Running optimal sampling (this may take a while)...")
    print("="*80)
    
    start_time = time.time()
    
    try:
        sampler.run_method(
            data_loaders=data_loaders,
            degradation=true_degradation,
            sigma_noise=0.05,
            num_samples=20,              # 20 samples
            burn_in=1000,                # 1000 burn-in
            thinning=50,                 # Save every 50th
            true_operator_params={'sigma': true_sigma}
        )
        
        elapsed_time = time.time() - start_time
        
        print("="*80)
        print(f"\n OPTIMAL LANGEVIN TEST COMPLETED!")
        print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
        
        # Analyze results
        import json
        calib_file = Path('results/test_langevin_optimal/blind_langevin_blind_pnp_flow_celeba/batch_0/calibration_metrics.json')
        op_file = Path('results/test_langevin_optimal/blind_langevin_blind_pnp_flow_celeba/batch_0/operator_statistics.json')
        
        print("\n" + "="*80)
        print("FINAL RESULTS ANALYSIS")
        print("="*80)
        
        if calib_file.exists() and op_file.exists():
            with open(calib_file) as f:
                calib = json.load(f)
            with open(op_file) as f:
                op_stats = json.load(f)
            
            ece = calib['ece']
            coverage_95 = calib['coverages']['95%']
            learned_sigma = op_stats['means']['sigma']
            sigma_std = op_stats['stds']['sigma']
            
            print(f"\n Calibration Quality:")
            print(f"  ECE: {ece:.4f}")
            if ece < 0.1:
                print("   Excellent calibration!")
            elif ece < 0.2:
                print("   Good calibration")
            elif ece < 0.4:
                print("    Moderate calibration")
            else:
                print("    Poor calibration (consider more iterations)")
            
            print(f"\n  95% Coverage: {coverage_95:.3f} (target: 0.95)")
            if abs(coverage_95 - 0.95) < 0.05:
                print("  ✅ Well-calibrated!")
            
            print(f"\nOperator Learning:")
            print(f"  Learned σ: {learned_sigma:.3f} ± {sigma_std:.3f}")
            print(f"  True σ: {true_sigma}")
            error = abs(learned_sigma - true_sigma)
            print(f"  Error: {error:.3f} ({error/true_sigma*100:.1f}%)")
            
            if error < 0.3:
                print("   Operator learned successfully!")
            
            print("\n" + "="*80)
            print("COMPARISON TO SHORT RUN:")
            print("="*80)
            print("  200 iterations → σ ≈ 1.14, ECE ≈ 0.63")
            print(f"  2000 iterations → σ ≈ {learned_sigma:.2f}, ECE ≈ {ece:.2f}")
            print(f"  Improvement: {(0.63-ece)/0.63*100:.0f}% better calibration")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_langevin_optimal()
    sys.exit(0 if success else 1)