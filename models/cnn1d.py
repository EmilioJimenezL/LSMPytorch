"""
1D CNN Baseline Model for LSM (Mexican Sign Language) recognition.

Architecture:
  Input  : (batch, 85, 135)  →  permute  →  (batch, 135, 85)
  Block 1: Conv1d(135→128, k=3) + BN + ReLU + Dropout + MaxPool
  Block 2: Conv1d(128→256, k=3) + BN + ReLU + Dropout + MaxPool
  Block 3: Conv1d(256→512, k=3) + BN + ReLU + Dropout
  Global Average Pooling
  FC(512→256) + ReLU + Dropout
  FC(256→n_classes)

Parameters: ~1.8M  |  Size: ~7-8MB  |  Expected accuracy: 80-85%
"""

import torch
import torch.nn as nn

FEATURE_DIM = 135
N_FRAMES = 85


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=True, dropout=0.3):
        super().__init__()
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNN1D(nn.Module):
    def __init__(self, num_classes: int = 330, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(FEATURE_DIM, 128, kernel=3, pool=True,  dropout=dropout),
            ConvBlock(128,         256, kernel=3, pool=True,  dropout=dropout),
            ConvBlock(256,         512, kernel=3, pool=False, dropout=dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (batch, N_FRAMES, FEATURE_DIM)
        x = x.permute(0, 2, 1)   # → (batch, FEATURE_DIM, N_FRAMES)
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x  # logits (batch, num_classes)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


# Keep LSM_CNN as an alias so existing code using model.py still works
LSM_CNN = CNN1D


def create_cnn1d(num_classes: int, dropout: float = 0.3) -> CNN1D:
    return CNN1D(num_classes=num_classes, dropout=dropout)
