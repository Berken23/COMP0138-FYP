# ============================================================================
# RUN BASELINE CODE DIRECTLY
# ============================================================================

from pnpflow.methods.pnp_flow import PNP_FLOW as PNP_FLOW_BASELINE
from pnpflow.degradations import GaussianDeblurring
import argparse

print("="*60)
print("RUNNING BASELINE CODE DIRECTLY")
print("="*60)

# Create baseline args (mimicking their config)
args_baseline = argparse.Namespace(
    model="ot",
    noise_type="gaussian",
    lr_pnp=0.05,  # Base value, will be scaled
    steps_pnp=100,
    num_samples=1,
    gamma_style="alpha_1_minus_t",
    alpha=1.0,
    sigma_noise=0.05,
    method="pnp_flow",
    num_channels=3,
    dim_image=128,
    save_results=False,
    compute_time=False,
    compute_memory=False,
)

# Scale lr by sigma_noise^2 (as baseline does)
args_baseline.lr_pnp = args_baseline.sigma_noise**2 * args_baseline.lr_pnp
print(f"✓ Baseline lr_pnp: {args_baseline.lr_pnp:.6f}")

# Create baseline PnP instance
pnp_baseline = PNP_FLOW_BASELINE(model, device, args_baseline)

# Create test problem
x_gt = get_image(0, normalize=True)
sigma_blur = 1.5
sigma_noise = 0.05

# Use baseline's GaussianDeblurring degradation
degradation_baseline = GaussianDeblurring(
    sigma_blur, 
    kernel_size=61, 
    operation_mode="fft",
    num_channels=3,
    dim_image=128,
    device=device
)

# Create observation
y = degradation_baseline.H(x_gt) + torch.randn_like(x_gt) * sigma_noise
obs_psnr = compute_psnr(y, x_gt, max_val=2.0)
print(f"✓ Observation PSNR: {obs_psnr:.2f} dB")

# Initialize with H_adj(1) (as baseline does)
x = degradation_baseline.H_adj(torch.ones_like(y))
print(f"✓ Initialized with H_adj(1)")

# Run baseline algorithm
print(f"\nRunning BASELINE PnP-Flow...")
H = degradation_baseline.H
H_adj = degradation_baseline.H_adj

steps = args_baseline.steps_pnp
delta = 1.0 / steps
lr = args_baseline.lr_pnp
num_samples = args_baseline.num_samples

with torch.no_grad():
    for iteration in range(steps):
        t = torch.ones(len(x), device=device) * delta * iteration
        lr_t = pnp_baseline.learning_rate_strat(lr, t)
        
        z = x - lr_t * pnp_baseline.grad_datafit(x, y, H, H_adj)
        
        x_new = torch.zeros_like(x)
        for _ in range(num_samples):
            z_tilde = pnp_baseline.interpolation_step(z, t.view(-1, 1, 1, 1))
            x_new += pnp_baseline.denoiser(z_tilde, t)
        x_new /= num_samples
        x = x_new
        
        if iteration % 25 == 0:
            psnr = compute_psnr(x, x_gt, max_val=2.0)
            print(f"  Iter {iteration}: PSNR = {psnr:.2f} dB")

final_psnr = compute_psnr(x, x_gt, max_val=2.0)
print(f"\nFinal PSNR: {final_psnr:.2f} dB")

if final_psnr > obs_psnr:
    print(f"✅ BASELINE WORKS! Improved by {final_psnr - obs_psnr:.2f} dB")
else:
    print(f"❌ BASELINE ALSO BROKEN! Degraded by {obs_psnr - final_psnr:.2f} dB")
    print("\nThis means the checkpoint file itself has issues!")