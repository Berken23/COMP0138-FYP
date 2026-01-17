"""
TODO:
Step 0 - Pick the minimal blind problem to validate the pipeline
Step 1 - Build the learnable forward operator (cleanly)
Step 2 - Build a "true operator" using the same class
Step 3 - Define the blind objective and the alternating loop
Step 4 - Add the PnP-Flow Matching image update
Step 5 - Choose an alternating schedule that won't implode
Step 6 - Logging and outputs (stop breaking metric parsers)
Step 7 - Make it runnable with one terminal command
Step 8 - Verify correctness (not just "it runs")    
"""

import torch
from pnpflow.blind_degradations import LearnableGaussianBlur
from pnpflow.blind_data import BlindGaussianBlurProblem
from pnpflow.blind_data import make_blind_gaussian_blur_problem


def blind_alternating_descent(
    prob: BlindGaussianBlurProblem,
    *,
    sigma_init: float,
    sigma_max: float,
    num_iters: int = 200,
    operator_lr: float = 5e-2,
    image_lr: float = 1e-1,
    operator_update_freq: int = 5,
):
    """
    Minimal blind alternating optimisation:
    - No PnP
    - No Flow
    - Pure data-consistency descent
    """

    device = prob.x_gt.device
    x = torch.randn_like(prob.x_gt, requires_grad=True)

    op = LearnableGaussianBlur(
        init_sigma=sigma_init,
        sigma_max=sigma_max,
        padding="reflect",
    ).to(device)

    op_opt = torch.optim.Adam(op.parameters(), lr=operator_lr)

    sigma_history = []
    loss_history = []

    for it in range(num_iters):
        # --- Image step ---
        x.requires_grad_(True)
        pred = op(x)
        loss_x = torch.mean((pred - prob.y) ** 2)

        grad_x = torch.autograd.grad(loss_x, x)[0]
        with torch.no_grad():
            x -= image_lr * grad_x

        x = x.detach()

        # --- Operator step (every k iterations) ---
        if it % operator_update_freq == 0:
            op_opt.zero_grad(set_to_none=True)

            pred = op(x.detach())
            loss_op = torch.mean((pred - prob.y) ** 2)
            loss_op.backward()
            op_opt.step()

        sigma_history.append(op.sigma().item())
        loss_history.append(loss_x.item())

    return {
        "x": x,
        "sigma_history": sigma_history,
        "loss_history": loss_history,
        "sigma_final": op.sigma().item(),
    }
    
    
def _run(
    *,
    seed: int,
    sigma_true: float,
    sigma_init: float,
    sigma_max: float,
    num_iters: int,
    operator_lr: float,
    image_lr: float,
    operator_update_freq: int,
    noise: float = 0.0,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    x_gt = torch.randn(1, 1, 64, 64, device=device)

    prob = make_blind_gaussian_blur_problem(
        x_gt=x_gt,
        sigma_true=sigma_true,
        sigma_noise=noise,
        sigma_max=sigma_max,
        padding="reflect",
        seed=seed,
    )

    out = blind_alternating_descent(
        prob,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=num_iters,
        operator_lr=operator_lr,
        image_lr=image_lr,
        operator_update_freq=operator_update_freq,
    )
    return out

def _print_history_summary(sigma_history, loss_history, k=20):
    n = len(sigma_history)

    print("\n=== Blind Alternating Descent Summary ===")
    print(f"Total iterations: {n}")

    if n <= 2 * k:
        print("\nSigma history:")
        print(sigma_history)
        print("\nLoss history:")
        print(loss_history)
    else:
        print("\nSigma history (first {}):".format(k))
        print(sigma_history[:k])
        print("\nSigma history (last {}):".format(k))
        print(sigma_history[-k:])

        print("\nLoss history (first {}):".format(k))
        print(loss_history[:k])
        print("\nLoss history (last {}):".format(k))
        print(loss_history[-k:])

if __name__ == "__main__":
    sigma_true = 2.5
    sigma_init = 0.4
    sigma_max = 6.0

    out = _run(
        seed=0,
        sigma_true=sigma_true,
        sigma_init=sigma_init,
        sigma_max=sigma_max,
        num_iters=300,
        operator_lr=5e-2,
        image_lr=1e-1,
        operator_update_freq=5,
        noise=0.0,
    )
    
    _print_history_summary(
        out["sigma_history"],
        out["loss_history"],
        k=20,
    )
    