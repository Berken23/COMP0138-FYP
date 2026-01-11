"""
Test Langevin posterior sampling
"""

import torch
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from pnpflow.utils import define_model, load_model
from pnpflow.methods.blind_pnp_flow_langevin import BlindPnPFlowLangevin
from pnpflow.blind_degradations import LearnableGaussianBlur
from torch.utils.data import TensorDataset, DataLoader


def find_model_path(dataset='celeba'):
    """Find model checkpoint in various possible locations"""
    # Get the directory where this script is located
    script_dir = Path(__file__).resolve().parent
    
    possible_paths = [
        # From pnpflow/tests/ -> ../../model/
        script_dir.parent.parent / 'model' / dataset / 'ot' / 'model_final.pt',
        # From pnpflow/ -> ../model/
        script_dir.parent / 'model' / dataset / 'ot' / 'model_final.pt',
        # Absolute fallback
        Path('model') / dataset / 'ot' / 'model_final.pt',
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def create_test_args():
    """Minimal args for testing"""
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
            self.save_path = 'results/test_langevin'
            self.compute_time = False
            self.compute_memory = False
            self.train = False
            self.eval = True
            self.dict_cfg_method = {
                'steps_pnp': self.steps_pnp,
                'lr_pnp': self.lr_pnp,
            }
    return Args()


def test_langevin_sampling():
    """Test posterior sampling"""
    print("\n" + "="*80)
    print("TESTING LANGEVIN POSTERIOR SAMPLING")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = create_test_args()
    
    # Load model
    print("\n1. Loading model...")
    model, state = define_model(args)
    
    # Find model path
    model_path = find_model_path('celeba')
    if model_path is None:
        print("ERROR: Could not find model at any expected location:")
        print("  - COMP0138-FYP/model/celeba/ot/model_final.pt")
        print("  - pnpflow/model/celeba/ot/model_final.pt")
        return False
    
    print(f"Found model at: {model_path}")
    
    try:
        load_model('ot', model, state, download=False,
                  checkpoint_path=model_path, dataset=None, device=device)
        model.eval()
        print("✓ Model loaded")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    # Create synthetic problem
    print("\n2. Creating synthetic problem...")
    clean_img = torch.randn(1, 3, 128, 128, device=device)
    true_sigma = 2.5
    
    # Create learnable operator
    learnable_blur = LearnableGaussianBlur(
        kernel_size=61,
        num_channels=3,
        init_sigma=1.0,
        device=device
    )
    print("✓ Created learnable operator (init sigma=1.0, true sigma=2.5)")
    
    # Create Langevin sampler
    print("\n3. Creating Langevin sampler...")
    sampler = BlindPnPFlowLangevin(
        model=model,
        learnable_operator=learnable_blur,
        device=device,
        args=args,
        operator_lr=1e-3,
        langevin_temp_image=0.01,
        langevin_temp_operator=0.001
    )
    print("✓ Sampler created")
    print(f"  Image temperature: 0.01")
    print(f"  Operator temperature: 0.001")
    
    # Create degradation
    import torch.nn.functional as F
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
    
    true_degradation = TrueDegradation(true_sigma, device)
    print("✓ Created true degradation")
    
    # Create dataloader
    dataset = TensorDataset(clean_img.cpu(), torch.zeros(1))
    test_loader = DataLoader(dataset, batch_size=1)
    data_loaders = {'test': test_loader}
    
    # Run sampling
    print("\n4. Running posterior sampling...")
    print("="*80)
    
    try:
        sampler.run_method(
            data_loaders=data_loaders,
            degradation=true_degradation,
            sigma_noise=0.05,
            num_samples=5,  # Generate 5 samples
            burn_in=20,     # 20 burn-in iterations
            thinning=2      # Save every 2nd sample
        )
        
        print("="*80)
        print("\n✅ LANGEVIN SAMPLING TEST PASSED!")
        print("\nCheck results in: results/test_langevin/")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR during sampling: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_langevin_sampling()
    sys.exit(0 if success else 1)