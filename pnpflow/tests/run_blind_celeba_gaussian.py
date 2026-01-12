"""
Run Blind PnP-Flow on CelebA dataset for Gaussian deblurring
Uses existing downloaded CelebA data
"""

import argparse
from types import SimpleNamespace
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import from your project structure
from pnpflow.methods.blind_pnp_flow import BlindPnPFlow
from pnpflow.utils import define_model, load_model
from pnpflow.blind_degradations import LearnableGaussianBlur
from pnpflow.degradations import GaussianDeblurring
from pnpflow.dataloaders import DataLoaders


def main():
    parser = argparse.ArgumentParser(description="Blind deblurring on CelebA")
    
    # Paths - USE ABSOLUTE PATH
    parser.add_argument("--data_root", type=str, 
                       default=r"C:\Users\gokce\UCL\Year 4\COMP0138 - Final Year Project\COMP0138-FYP\data",
                       help="Root directory for datasets")
    parser.add_argument("--model_path", type=str, default="./model/celeba/ot/model_final.pt",
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--save_path", type=str, default="./results/blind_celeba_gaussian",
                       help="Directory to save results")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    # Problem setup
    parser.add_argument("--true_sigma", type=float, default=2.0,
                       help="True blur sigma (ground truth)")
    parser.add_argument("--init_sigma", type=float, default=1.0,
                       help="Initial sigma for learnable operator")
    parser.add_argument("--sigma_noise", type=float, default=0.05,
                       help="Noise level")
    
    # Algorithm parameters
    parser.add_argument("--steps_pnp", type=int, default=100,
                       help="Number of PnP iterations")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples for denoising")
    parser.add_argument("--lr_pnp", type=float, default=1.0,
                       help="Learning rate for image (scaled by noise)")
    parser.add_argument("--operator_lr", type=float, default=1e-3,
                       help="Learning rate for operator")
    parser.add_argument("--operator_reg_weight", type=float, default=0.01,
                       help="Regularization weight for operator")
    
    # Dataset parameters
    parser.add_argument("--max_batch", type=int, default=10,
                       help="Number of images to process")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size (keep at 1 for blind IP)")
    parser.add_argument("--num_workers", type=int, default=0,  # Changed to 0 for Windows
                       help="Number of dataloader workers")
    
    args_cli = parser.parse_args()
    
    # Setup device
    device = torch.device(args_cli.device if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Check if data exists
    data_path = Path(args_cli.data_root)
    print(f"\nChecking data directory: {data_path}")
    
    if not data_path.exists():
        print(f"✗ ERROR: Data directory not found: {data_path}")
        print(f"  Please check the path!")
        return
    
    # List what's in the data directory
    print(f"\nContents of data directory:")
    for item in data_path.iterdir():
        print(f"  - {item.name}")
    
    # Check for celeba subdirectory
    celeba_path = data_path / "celeba"
    if celeba_path.exists():
        print(f"\n✓ Found CelebA directory: {celeba_path}")
        print(f"  Contents:")
        for item in celeba_path.iterdir():
            if item.is_dir():
                num_files = len(list(item.glob("*")))
                print(f"    - {item.name}/ ({num_files} files)")
            else:
                print(f"    - {item.name}")
    else:
        print(f"\n⚠️  No 'celeba' subdirectory found")
        print(f"   Looking for data directly in: {data_path}")
    
    # =========================================================================
    # 1. CREATE ARGS OBJECT (matching your project structure)
    # =========================================================================
    
    args = SimpleNamespace(
        # Dataset config
        dataset="celeba",
        data_root=args_cli.data_root,  # Use the absolute path
        eval_split="test",
        dim_image=128,
        num_channels=3,
        
        # Model config
        model="ot",
        root='',
        
        # Problem config
        problem="blind_gaussian_deblurring",
        method="blind_pnp_flow",
        noise_type="gaussian",
        
        # PnP-Flow settings
        steps_pnp=args_cli.steps_pnp,
        lr_pnp=args_cli.lr_pnp,
        alpha=0.5,
        gamma_style="alpha_1_minus_t",
        num_samples=args_cli.num_samples,
        
        # Evaluation
        max_batch=args_cli.max_batch,
        batch_size_ip=args_cli.batch_size,
        
        # Saving
        save_results=True,
        save_path=args_cli.save_path,
        compute_time=True,
        compute_memory=False,
        
        # Flags
        train=False,
        eval=True,
        
        # Method config dict
        dict_cfg_method={
            'steps_pnp': args_cli.steps_pnp,
            'lr_pnp': args_cli.lr_pnp,
            'alpha': 0.5,
            'gamma_style': 'alpha_1_minus_t',
            'num_samples': args_cli.num_samples,
        }
    )
    
    print("\n" + "="*80)
    print("BLIND GAUSSIAN DEBLURRING ON CELEBA")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Image size: {args.dim_image}x{args.dim_image}")
    print(f"True σ: {args_cli.true_sigma}")
    print(f"Init σ: {args_cli.init_sigma}")
    print(f"Noise σ: {args_cli.sigma_noise}")
    print(f"PnP steps: {args.steps_pnp}")
    print(f"Operator LR: {args_cli.operator_lr}")
    print(f"Processing: {args.max_batch} images")
    print("="*80)
    
    # =========================================================================
    # 2. LOAD PRETRAINED MODEL
    # =========================================================================
    
    print("\nLoading pretrained Flow Matching model...")
    
    model, state = define_model(args)
    
    try:
        load_model(
            args.model, 
            model, 
            state, 
            download=False,
            checkpoint_path=args_cli.model_path,
            dataset=None,
            device=device
        )
        model.eval()
        print(f"✓ Model loaded from {args_cli.model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # =========================================================================
    # 3. CREATE DATALOADERS
    # =========================================================================
    
    print("\nCreating CelebA dataloaders...")
    print("(This should be quick if data is already downloaded)")
    
    try:
        # Create dataloaders with explicit paths
        data_loader_obj = DataLoaders(
            args,
            batch_size_train=args_cli.batch_size,
            batch_size_test=args_cli.batch_size
        )
        data_loaders = data_loader_obj.get_dataloaders()
        
        # Check if we got data
        test_loader = data_loaders.get('test')
        if test_loader is None:
            print("✗ ERROR: No test dataloader created!")
            return
        
        num_images = len(test_loader.dataset)
        print(f"✓ Loaded {num_images} test images")
        
        if num_images == 0:
            print("✗ ERROR: Dataset is empty!")
            print("  Check that CelebA data is properly organized")
            return
            
    except Exception as e:
        print(f"✗ Error creating dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # =========================================================================
    # 4. CREATE DEGRADATION OPERATORS
    # =========================================================================
    
    print("\nSetting up degradation operators...")
    
    true_degradation = GaussianDeblurring(
        kernel_size=61,
        num_channels=3,
        sigma=args_cli.true_sigma,
        device=device
    )
    print(f"✓ True degradation: Gaussian blur with σ = {args_cli.true_sigma}")
    
    learnable_operator = LearnableGaussianBlur(
        kernel_size=61,
        num_channels=3,
        init_sigma=args_cli.init_sigma,
        device=device
    )
    print(f"✓ Learnable operator: Initialized with σ = {args_cli.init_sigma}")
    print(f"  (Misspecified by {abs(args_cli.true_sigma - args_cli.init_sigma):.1f})")
    
    # =========================================================================
    # 5. CREATE BLIND PNP-FLOW SOLVER
    # =========================================================================
    
    print("\nInitializing Blind PnP-Flow solver...")
    
    solver = BlindPnPFlow(
        model=model,
        learnable_operator=learnable_operator,
        device=device,
        args=args,
        operator_lr=args_cli.operator_lr,
        operator_update_freq=1,
        operator_reg_weight=args_cli.operator_reg_weight
    )
    
    print("✓ Solver initialized")
    
    # =========================================================================
    # 6. RUN BLIND RECONSTRUCTION
    # =========================================================================
    
    print("\n" + "="*80)
    print("STARTING BLIND RECONSTRUCTION")
    print("="*80)
    print(f"Processing {args.max_batch} images...")
    print("Progress will be shown for each batch")
    print("="*80 + "\n")
    
    try:
        solver.run_method(
            data_loaders=data_loaders,
            degradation=true_degradation,
            sigma_noise=args_cli.sigma_noise
        )
        
        print("\n" + "="*80)
        print("✅ BLIND RECONSTRUCTION COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {args.save_path}")
        print("\nGenerated files:")
        print("  • Reconstructed images")
        print("  • PSNR/SSIM/LPIPS metrics")
        print("  • Operator parameter history (operator_history_batch_*.json)")
        print("  • Summary statistics")
        
        # Print final learned parameters
        final_sigma = learnable_operator.get_sigma()
        error = abs(final_sigma - args_cli.true_sigma)
        error_pct = (error / args_cli.true_sigma) * 100
        
        print("\n" + "="*80)
        print("OPERATOR LEARNING RESULTS")
        print("="*80)
        print(f"True σ:        {args_cli.true_sigma:.3f}")
        print(f"Initial σ:     {args_cli.init_sigma:.3f}")
        print(f"Learned σ:     {final_sigma:.3f}")
        print(f"Error:         {error:.3f} ({error_pct:.1f}%)")
        
        if error < 0.3:
            print("✅ Operator learned successfully!")
        elif error < 0.5:
            print("⚠️  Moderate learning (consider more iterations)")
        else:
            print("⚠️  Poor learning (try higher operator LR or more iterations)")
        
    except Exception as e:
        print(f"\n✗ ERROR during reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()