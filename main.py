# main.py
import os
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from pnpflow.utils import load_cfg_from_cfg_file, merge_cfg_from_list
from pnpflow.degradations import *
from pnpflow.dataloaders import DataLoaders
from pnpflow.train_flow_matching import FLOW_MATCHING
from pnpflow.train_denoiser import GRADIENT_STEP_DENOISER
from pnpflow.compute_metric import ComputeMetric
from pnpflow.methods.pnp_flow import PNP_FLOW
from pnpflow.methods.d_flow import D_FLOW
from pnpflow.methods.ot_ode import OT_ODE
from pnpflow.methods.flow_priors import FLOW_PRIORS
from pnpflow.methods.pnp_gs import PROX_PNP
from pnpflow.methods.pnp_diff import PNP_DIFF
from pnpflow.methods.blind_pnp_flow import BlindPnPFlow  # NEW

from pnpflow.blind_degradations import (  # NEW
    LearnableGaussianBlur,
    LearnableMotionBlur,
    LearnableMask,
    LearnableDownsampling,
)

from pnpflow.utils import gaussian_blur, define_model, load_model
import warnings
warnings.filterwarnings("ignore", module="matplotlib\\..*")

torch.cuda.empty_cache()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Main')
    cfg = load_cfg_from_cfg_file('./' + 'config/main_config.yaml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    dataset_config = cfg.root + 'config/dataset_config/{}.yaml'.format(cfg.dataset)
    cfg.update(load_cfg_from_cfg_file(dataset_config))

    method_config_file = cfg.root + 'config/method_config/{}.yaml'.format(cfg.method)
    cfg.update(load_cfg_from_cfg_file(method_config_file))

    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    method_cfg = load_cfg_from_cfg_file(method_config_file)
    cfg.dict_cfg_method = {}
    for key in method_cfg.keys():
        cfg.dict_cfg_method[key] = cfg[key]
    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    (model, state) = define_model(args)

    if args.train:
        args.batch_size = args.batch_size_train
        print('Training...')
        data_loaders = DataLoaders(args.dataset, args.batch_size_train, args.batch_size_train).load_data()
        if args.model == "ot":
            generative_method = FLOW_MATCHING(model, device, args)
        elif args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            raise ValueError("Model not implemented yet: choose between 'ot' and 'gradient_step'")
        generative_method.train(data_loaders)
        print('Training done!')

    if args.eval:

        if args.model == "ot" or args.model == "gradient_step":
            model_path = args.root + 'model/{}/{}/model_final.pt'.format(args.dataset, args.model)
            load_model(args.model, model, state, download=False,
                       checkpoint_path=model_path, dataset=None,  device=device)
            model.eval()

        elif args.model == "rectified":
            model_path = args.root + 'model/{}/{}/model_final.pth'.format(args.dataset, args.model)
            load_model(args.model, model, state, download=False,
                       checkpoint_path=model_path, dataset=None, device=device)
            model.eval()

        elif args.model == "diffusion":
            model.eval()

        if args.model == "gradient_step":
            generative_method = GRADIENT_STEP_DENOISER(model, device, args)
        else:
            generative_method = FLOW_MATCHING(model, device, args)

        if args.compute_metrics:
            print('Computing metrics...')
            data_loaders = DataLoaders(args.dataset, 5000, 5000).load_data()
            metric = ComputeMetric(data_loaders, generative_method, device, args)
            metric.compute_metrics(5000)
            print('Computing metrics done!')

        # ====================================================================
        # STANDARD (NON-BLIND) INVERSE PROBLEMS
        # ====================================================================

        if args.problem == "denoising":
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.2
            degradation = Denoising()

        elif args.problem == "inpainting":
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            half_size_mask = 20 if args.dim_image == 128 else 40
            degradation = BoxInpainting(half_size_mask)

        elif args.problem == "paintbrush_inpainting":
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            degradation = PaintbrushInpainting()

        elif args.problem == "random_inpainting":
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.01
            p = 0.7
            degradation = RandomInpainting(p)

        elif args.problem == "superresolution":
            sf = 2 if args.dim_image == 128 else 4
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            degradation = Superresolution(sf, args.dim_image)

        elif args.problem == "gaussian_deblurring_FFT":
            sigma_blur = 1.0 if args.dim_image == 128 else 3.0
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            kernel_size = 61
            degradation = GaussianDeblurring(
                sigma_blur, kernel_size, "fft", args.num_channels, args.dim_image, device
            )

        # ====================================================================
        # BLIND INVERSE PROBLEMS (NEW)
        # ====================================================================

        elif args.problem == "blind_gaussian_deblurring":
            if args.dim_image == 128:
                sigma_blur_true = 2.5
            else:
                sigma_blur_true = 3.0

            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            kernel_size = 61

            # True degradation for observation generation (non-learnable)
            degradation = GaussianDeblurring(
                sigma_blur_true, kernel_size, "fft",
                args.num_channels, args.dim_image, device
            )

            # Learnable operator (initialized wrong), Option B: constructor accepts device=
            init_sigma = getattr(args, 'init_sigma', 1.0)
            learnable_operator = LearnableGaussianBlur(
                kernel_size=kernel_size,
                num_channels=args.num_channels,
                init_sigma=init_sigma,
                device=device,  # ✅ now supported by updated blind_degradations.py
            )

            print(f"Blind Gaussian Deblurring:")
            print(f"  True sigma: {sigma_blur_true}")
            print(f"  Init sigma: {init_sigma} (learnable)")

            args.is_blind = True
            args.true_operator_params = {'sigma': sigma_blur_true}

        elif args.problem == "blind_motion_deblurring":
            true_length = getattr(args, 'true_length', 10.0)
            true_angle = getattr(args, 'true_angle', 0.5)

            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            kernel_size = 61

            # Create observation operator (frozen learnable operator set to true params)
            true_operator = LearnableMotionBlur(
                kernel_size=kernel_size,
                num_channels=args.num_channels,
                init_length=true_length,
                init_angle=true_angle,
                device=device,
            )
            true_operator.eval()

            class MotionBlurDegradation:
                def __init__(self, operator):
                    self.operator = operator
                def H(self, x):
                    return self.operator(x)
                def H_adj(self, y):
                    return y  # identity adjoint placeholder

            degradation = MotionBlurDegradation(true_operator)

            init_length = getattr(args, 'init_length', 5.0)
            init_angle = getattr(args, 'init_angle', 0.0)
            learnable_operator = LearnableMotionBlur(
                kernel_size=kernel_size,
                num_channels=args.num_channels,
                init_length=init_length,
                init_angle=init_angle,
                device=device,
            )

            print(f"Blind Motion Deblurring:")
            print(f"  True: length={true_length}, angle={true_angle}")
            print(f"  Init: length={init_length}, angle={init_angle} (learnable)")

            args.is_blind = True
            args.true_operator_params = {'length': true_length, 'angle': true_angle}

        elif args.problem == "blind_box_inpainting":
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05
            half_size_mask = 20 if args.dim_image == 128 else 40

            degradation = BoxInpainting(half_size_mask)

            init_mask_ratio = getattr(args, 'init_mask_ratio', 0.3)
            learnable_operator = LearnableMask(
                image_shape=(args.num_channels, args.dim_image, args.dim_image),
                init_ratio=init_mask_ratio,
                device=device,
            )

            print(f"Blind Box Inpainting:")
            print(f"  True mask: {half_size_mask}x{half_size_mask} center box")
            print(f"  Init: random mask with ratio {init_mask_ratio}")

            args.is_blind = True
            args.true_operator_params = {'mask_size': half_size_mask}

        elif args.problem == "blind_random_inpainting":
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.01
            p = getattr(args, 'p', 0.7)

            degradation = RandomInpainting(p)

            init_mask_ratio = getattr(args, 'init_mask_ratio', 0.5)
            learnable_operator = LearnableMask(
                image_shape=(args.num_channels, args.dim_image, args.dim_image),
                init_ratio=init_mask_ratio,
                device=device,
            )

            print(f"Blind Random Inpainting:")
            print(f"  True: random mask with p={p}")
            print(f"  Init: random mask with ratio {init_mask_ratio}")

            args.is_blind = True
            args.true_operator_params = {'p': p}

        elif args.problem == "blind_superresolution":
            sf = 2 if args.dim_image == 128 else 4
            sigma_noise = 0.3 if args.noise_type == 'laplace' else 0.05

            degradation = Superresolution(sf, args.dim_image)

            scale_factor = getattr(args, 'scale_factor', sf)
            learnable_operator = LearnableDownsampling(
                scale_factor=scale_factor,
                num_channels=args.num_channels,
                device=device,
            )

            print(f"Blind Super-Resolution:")
            print(f"  Scale factor: {sf}")
            print(f"  Learning downsampling kernel")

            args.is_blind = True
            args.true_operator_params = {'scale_factor': sf}

        else:
            raise ValueError(f"Unknown problem: {args.problem}")

        # ====================================================================
        # RUN METHOD
        # ====================================================================

        is_blind = getattr(args, 'is_blind', False)

        data_loaders = DataLoaders(args.dataset, args.batch_size_ip, args.batch_size_ip).load_data()

        if args.noise_type == 'laplace':
            args.save_path = os.path.join(
                args.root, 'results_laplace', args.dataset, args.model,
                args.problem, args.method, args.eval_split
            )
        else:
            args.save_path = os.path.join(
                args.root, 'results', args.dataset, args.model,
                args.problem, args.method, args.eval_split
            )
        os.makedirs(args.save_path, exist_ok=True)

        if not is_blind:
            print('Solving the {} inverse problem with the method {}...'.format(args.problem, args.method))
            print('sigma_noise', sigma_noise)

            if args.method == 'pnp_flow':
                method = PNP_FLOW(model, device, args)
            elif args.method == 'd_flow':
                method = D_FLOW(model, device, args)
            elif args.method == 'ot_ode':
                method = OT_ODE(model, device, args)
            elif args.method == 'flow_priors':
                method = FLOW_PRIORS(model, device, args)
            elif args.method == 'pnp_gs':
                method = PROX_PNP(generative_method, device, args)
            elif args.method == 'pnp_diff':
                method = PNP_DIFF(model, device, args)
            else:
                raise ValueError("The method you entered does not exist")

            method.run_method(data_loaders, degradation, sigma_noise)

        else:
            print('Solving BLIND {} inverse problem with the method {}...'.format(args.problem, args.method))
            print('sigma_noise', sigma_noise)

            if args.method != 'blind_pnp_flow':
                raise ValueError(f"Method '{args.method}' not supported for blind problems. Use 'blind_pnp_flow'.")

            operator_lr = getattr(args, 'operator_lr', 1e-3)
            operator_update_freq = getattr(args, 'operator_update_freq', 1)
            operator_reg_weight = getattr(args, 'operator_reg_weight', 0.01)

            method = BlindPnPFlow(
                model=model,
                learnable_operator=learnable_operator,
                device=device,
                args=args,
                operator_lr=operator_lr,
                operator_update_freq=operator_update_freq,
                operator_reg_weight=operator_reg_weight
            )

            print(f"\nBlind PnP-Flow Configuration:")
            print(f"  Operator LR: {operator_lr}")
            print(f"  Update freq: {operator_update_freq}")
            print(f"  Reg weight: {operator_reg_weight}")

            method.run_method(data_loaders, degradation, sigma_noise)

            print(f"\n{'='*80}")
            print("FINAL LEARNED OPERATOR PARAMETERS:")
            final_params = method.get_operator_params()
            for key, value in final_params.items():
                true_value = args.true_operator_params.get(key, 'N/A')
                print(f"  {key}: learned={value:.4f}, true={true_value}")
            print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
