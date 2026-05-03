"""Dataset loaders for the blind deconvolution evaluation.

Loads CelebA, BSD68, and Set12 from local directories and returns lists of
preprocessed tensors of shape (1, 3, H, W) in [-1, 1]. The Colab download
logic from the development notebook is intentionally omitted, datasets must
be staged on disk before invoking these loaders.
"""
from __future__ import annotations

import os
import random
from typing import List

import torch
from PIL import Image
from torchvision import transforms


def _build_transform(img_size: int, force_gray: bool) -> transforms.Compose:
    steps = [
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
    ]
    if force_gray:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps += [
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
    return transforms.Compose(steps)


def _load_image_dir(
    directory: str,
    img_size: int,
    device: str,
    extensions=(".png", ".jpg", ".bmp", ".jpeg"),
) -> List[torch.Tensor]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    transform_rgb = _build_transform(img_size, force_gray=False)
    transform_gray = _build_transform(img_size, force_gray=True)

    files = sorted(f for f in os.listdir(directory) if f.lower().endswith(extensions))
    images: List[torch.Tensor] = []
    for f in files:
        img = Image.open(os.path.join(directory, f))
        t = transform_gray(img) if img.mode == "L" else transform_rgb(img)
        images.append(t.unsqueeze(0).to(device))
    return images


def load_celeba(
    repo_root: str,
    device: str,
    num_images: int = 100,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Load a deterministic random subset of the CelebA test split via the
    upstream DataLoaders helper.
    """
    import sys

    pnpflow_dir = os.path.join(repo_root, "pnpflow")
    if pnpflow_dir not in sys.path:
        sys.path.insert(0, pnpflow_dir)

    from dataloaders import DataLoaders, CelebADataset

    original_init = CelebADataset.__init__

    def absolute_init(self, img_dir, partition_csv, partition, transform=None):
        original_init(
            self,
            os.path.abspath(img_dir),
            os.path.abspath(partition_csv),
            partition,
            transform,
        )

    CelebADataset.__init__ = absolute_init
    prev_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        loaders = DataLoaders("celeba", batch_size_train=1, batch_size_test=1)
        test_loader = loaders.load_data()["test"]
    finally:
        os.chdir(prev_cwd)
        CelebADataset.__init__ = original_init

    pool: List[torch.Tensor] = []
    for i, (x, _) in enumerate(test_loader):
        pool.append(x.to(device).float())
        if i >= 999:
            break

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pool)), min(num_images, len(pool))))
    return [pool[i] for i in indices]


def load_bsd68(repo_root: str, device: str, img_size: int = 128) -> List[torch.Tensor]:
    return _load_image_dir(os.path.join(repo_root, "data", "BSD68"), img_size, device)


def load_set12(repo_root: str, device: str, img_size: int = 128) -> List[torch.Tensor]:
    return _load_image_dir(os.path.join(repo_root, "data", "Set12"), img_size, device)


def load_dataset(
    name: str,
    repo_root: str,
    device: str,
    img_size: int = 128,
    num_celeba: int = 100,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Dispatch by dataset name. Returns a list of (1, 3, H, W) tensors."""
    if name == "CelebA":
        return load_celeba(repo_root, device, num_images=num_celeba, seed=seed)
    if name == "BSD68":
        return load_bsd68(repo_root, device, img_size=img_size)
    if name == "Set12":
        return load_set12(repo_root, device, img_size=img_size)
    raise ValueError(f"Unknown dataset '{name}'. Choose CelebA, BSD68, or Set12.")
