"""
PyTorch DataLoaders for scalogram image classification.

Uses torchvision.datasets.ImageFolder with stratified train/val/test splits
and on-the-fly data augmentation for the training set.

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import (
    DATA_SCALOGRAMS, IMG_SIZE,
    BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED,
)

# ──────────────────────────────────────────────
# ImageNet normalization (required for GoogLeNet)
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _get_transforms():
    """Return (train_transform, eval_transform)."""

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset with a custom transform (overriding the parent's)."""

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataloaders(
    data_dir=DATA_SCALOGRAMS,
    batch_size=BATCH_SIZE,
    seed=RANDOM_SEED,
    num_workers=0,
):
    """Build train / val / test DataLoaders with stratified splits.

    Parameters
    ----------
    data_dir : Path
        Root directory containing class sub-folders.
    batch_size : int
    seed : int
    num_workers : int

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    train_tf, eval_tf = _get_transforms()

    # Load full dataset WITHOUT transforms (we apply per-split later)
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    class_names = full_dataset.classes
    targets = np.array(full_dataset.targets)

    total = len(full_dataset)
    indices = np.arange(total)

    # --- Stratified split: Train vs (Val + Test) ---
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1 - TRAIN_RATIO),
        stratify=targets[indices],
        random_state=seed,
    )

    # --- Stratified split: Val vs Test ---
    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_ratio,
        stratify=targets[temp_idx],
        random_state=seed,
    )

    print(f"  Dataset splits  -> Train: {len(train_idx):,}  |  "
          f"Val: {len(val_idx):,}  |  Test: {len(test_idx):,}")

    # Wrap subsets with appropriate transforms
    train_set = _TransformSubset(Subset(full_dataset, train_idx), train_tf)
    val_set = _TransformSubset(Subset(full_dataset, val_idx), eval_tf)
    test_set = _TransformSubset(Subset(full_dataset, test_idx), eval_tf)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
