import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pnpflow.dataloaders import DataLoaders
from pnpflow.degradations import GaussianDeblurring
from pnpflow.methods.blind_pnp_flow import BlindPnPFlow
from pnpflow.blind_degradations import LearnableGaussianBlur
from pnpflow.utils import define_model, load_model


def parse_args():
    p = argparse.ArgumentParser("Blind PnP-Flow (no YAML)")

    # Core
    p.add_argument("--root", type=str, default=".", help="Repo root (default: .)")
    p.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "afhq_cat"])
    p.add_argument("--model", type=str, default="ot", choices=["ot", "rectified", "gradient_step", "diffusion"])
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    
    # Blind problem (start with Gaussian deblur)
    p.add_argument("--problem", type=str, default="blind_gaussian_deblurring",
                   choices=["blind_gaussian_deblurring"])
    p.add_argument("--dim_image", type=int, default=128, choices=[128, 256])
    p.add_argument("--num_channels", type=int, default=3)
    p.add_argument("--kernel_size", type=int, default=61)

    # True degradation + noise
    p.add_argument("--sigma_blur_true", type=float, default=2.5)
    p.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "laplace"])
    p.add_argument("--sigma_noise", type=float, default=0.05)

    # PnP-Flow params
    p.add_argument("--steps_pnp", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--lr_pnp", type=float, default=1.0)
    p.add_argument("--gamma_style", type=str, default="alpha_1_minus_t",
                   choices=["1_minus_t", "sqrt_1_minus_t", "constant", "alpha_1_minus_t"])
    p.add_argument("--alpha", type=float, default=1.0)

    # Blind operator learning
    p.add_argument("--init_sigma", type=float, default=0.8)
    p.add_argument("--operator_lr", type=float, default=1e-3)
    p.add_argument("--operator_update_freq", type=int, default=1)
    p.add_argument("--operator_reg_weight", type=float, default=1e-2)

    # Data + logging
    p.add_argument("--batch_size_ip", type=int, default=1)
    p.add_argument("--max_batch", type=int, default=3)
    p.add_argument("--save_results", action="store_true", default=True)
    p.add_argument("--no_save_results", action="store_true", help="Disable saving")
    p.add_argument("--compute_time", action="store_true", default=False)
    p.add_argument("--compute_memory", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()

    if args.no_save_results:
        args.save_results = False

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device", device)

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

    # define/load model using your existing utilities
    # define_model expects args.root etc to exist
    args.method = "blind_pnp_flow"  # for consistent logging inside BlindPnPFlow
    args.is_blind = True

    (model, state) = define_model(args)

    # load checkpoint path based on your directory layout
    if args.model == "ot":
        ckpt = os.path.join(args.root, "model", args.dataset, "ot", "model_final.pt")
        load_model("ot", model, state, download=False, checkpoint_path=ckpt, dataset=None, device=device)
    elif args.model == "rectified":
        ckpt = os.path.join(args.root, "model", args.dataset, "rectified", "model_final.pth")
        load_model("rectified", model, state, download=False, checkpoint_path=ckpt, dataset=None, device=device)
    elif args.model == "gradient_step":
        ckpt = os.path.join(args.root, "model", args.dataset, "gradient_step", "model_final.pt")
        load_model("gradient_step", model, state, download=False, checkpoint_path=ckpt, dataset=None, device=device)
    else:
        # diffusion etc: assume already handled in your codebase
        pass

    model.eval()

    # Data loaders
    data_loaders = DataLoaders(args.dataset, args.batch_size_ip, args.batch_size_ip).load_data()

    # Save path (matches your existing pattern)
    base_results = "results_laplace" if args.noise_type == "laplace" else "results"
    args.save_path = os.path.join(args.root, base_results, args.dataset, args.model,
                                  args.problem, args.method, args.eval_split)
    os.makedirs(args.save_path, exist_ok=True)

    # True degradation + learnable operator
    if args.problem == "blind_gaussian_deblurring":
        degradation = GaussianDeblurring(
            args.sigma_blur_true, args.kernel_size, "fft",
            args.num_channels, args.dim_image, device
        )
        learnable_operator = LearnableGaussianBlur(
            kernel_size=args.kernel_size,
            num_channels=args.num_channels,
            init_sigma=args.init_sigma,
            device=device,
        )
        args.true_operator_params = {"sigma": args.sigma_blur_true}
    else:
        raise ValueError("Unsupported problem")

    # Construct method
    method = BlindPnPFlow(
        model=model,
        learnable_operator=learnable_operator,
        device=device,
        args=args,
        operator_lr=args.operator_lr,
        operator_update_freq=args.operator_update_freq,
        operator_reg_weight=args.operator_reg_weight,
    )

    # Run
    method.run_method(data_loaders, degradation, args.sigma_noise)

    # Print final
    final_params = method.get_operator_params()
    print("\nFINAL LEARNED OPERATOR PARAMETERS:")
    for k, v in final_params.items():
        tv = args.true_operator_params.get(k, None)
        if tv is None:
            print(f"  {k}: learned={v:.4f}")
        else:
            print(f"  {k}: learned={v:.4f}, true={tv}")
    print("\nSaved to:", args.save_path)


if __name__ == "__main__":
    main()
