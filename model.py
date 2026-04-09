"""
model.py
--------
Small 1D-CNN architecture for ECG arrhythmia classification,
closely following the design described in:

  "Deep-Learning-Based Arrhythmia Detection Using ECG Signals:
   A Comparative Study and Performance Evaluation"
  Diagnostics 2023, 13(24), 3605

Architecture (1D version of the paper's Small CNN):
  Conv1(1→32, k=11) → BN → ReLU → MaxPool(2)
  Conv2(32→64, k=9) → BN → ReLU → MaxPool(2)
  Conv3(64→128, k=7) → BN → ReLU → MaxPool(2)
  Flatten → Dropout(0.5) → FC(num_classes)

Input:  (batch, 1, 200)   – one heartbeat segment (200 samples, 1 channel)
Output: (batch, num_classes) – raw logits
"""

import torch
import torch.nn as nn


class SmallCNN1D(nn.Module):
    def __init__(self, num_classes: int = 3, input_length: int = 200):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),           # 200 → 100

            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),           # 100 → 50

            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),           # 50 → 25
        )

        # Compute the flattened size dynamically so we aren't hard-coding it
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            flat_size = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(flat_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)         # (B, 128, 25)
        x = x.view(x.size(0), -1)    # flatten
        return self.classifier(x)    # (B, num_classes)


if __name__ == "__main__":
    model = SmallCNN1D(num_classes=3)
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")
    dummy_input = torch.randn(8, 1, 200)
    out = model(dummy_input)
    print(f"Output shape: {out.shape}")   # should be (8, 3)
