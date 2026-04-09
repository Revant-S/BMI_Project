"""
Model architectures for ECG scalogram classification.

Contains:
    - SmallNet : Custom lightweight CNN (~5K params)
    - get_googlenet : GoogLeNet with modified head for binary classification

Usage:
    from src.model import SmallNet, get_googlenet
    model = SmallNet(num_classes=2)
    model = get_googlenet(num_classes=2, freeze_base=False)
"""

import torch
import torch.nn as nn
from torchvision import models


# ──────────────────────────────────────────────────────────────
# SmallNet — Custom lightweight CNN
# ──────────────────────────────────────────────────────────────
#
# Architecture (layer count):
#   Block 1:  Conv2d → BatchNorm2d → ReLU → MaxPool2d        (4)
#   Block 2:  Conv2d → BatchNorm2d → ReLU → MaxPool2d        (4)
#   Block 3:  Conv2d → BatchNorm2d → ReLU → MaxPool2d        (4)
#   Block 4:  Conv2d → BatchNorm2d → ReLU → MaxPool2d        (4)
#   Head:     AdaptiveAvgPool2d → Flatten → Linear → ReLU
#             → Dropout → Linear                              (5*)
#   ──────────────────────────────────────────────────────
#   Total named layers:  ~21
#
#   *Flatten is a reshape op, counted here for consistency
#    with the "21-layer" reference in the project brief.
# ──────────────────────────────────────────────────────────────


class _ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, pool_size=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class SmallNet(nn.Module):
    """Lightweight 21-layer CNN for scalogram classification.

    Filter progression per block: 16 → 4 → 8 → 16
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            _ConvBlock(3, 16),    # Block 1:  3 → 16  | 224→112
            _ConvBlock(16, 4),    # Block 2: 16 →  4  | 112→56
            _ConvBlock(4, 8),     # Block 3:  4 →  8  | 56→28
            _ConvBlock(8, 16),    # Block 4:  8 → 16  | 28→14
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),      # 14×14 → 1×1
            nn.Flatten(),                 # 16
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x


# ──────────────────────────────────────────────────────────────
# GoogLeNet — Transfer learning
# ──────────────────────────────────────────────────────────────


def get_googlenet(num_classes: int = 2, freeze_base: bool = False):
    """Return a GoogLeNet with modified FC head for binary classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (default 2).
    freeze_base : bool
        If True, freeze all pre-trained parameters (only train the new heads).

    Returns
    -------
    model : nn.Module
    """
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)

    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Replace main classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Replace auxiliary classifier heads (used during training)
    if model.aux1 is not None:
        aux1_in = model.aux1.fc2.in_features
        model.aux1.fc2 = nn.Linear(aux1_in, num_classes)

    if model.aux2 is not None:
        aux2_in = model.aux2.fc2.in_features
        model.aux2.fc2 = nn.Linear(aux2_in, num_classes)

    return model


# ──────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Factory function to create a model by name.

    Parameters
    ----------
    name : str
        'smallnet' or 'googlenet'.
    num_classes : int
    **kwargs :
        Passed to the model constructor / factory.

    Returns
    -------
    model : nn.Module
    """
    name = name.lower().strip()
    if name == 'smallnet':
        return SmallNet(num_classes=num_classes)
    elif name == 'googlenet':
        return get_googlenet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name!r}. Choose 'smallnet' or 'googlenet'.")


if __name__ == "__main__":
    # Quick sanity check
    for name in ['smallnet', 'googlenet']:
        m = get_model(name)
        dummy = torch.randn(2, 3, 224, 224)
        m.eval()
        out = m(dummy)
        print(f"{name:>10s}  |  params: {count_parameters(m):>10,}  |  "
              f"output shape: {tuple(out.shape)}")
