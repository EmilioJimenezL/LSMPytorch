"""
1D CNN Baseline Model for LSM (Mexican Sign Language) recognition.

Architecture:
  Input  : (batch, 85, input_dim)  →  permute  →  (batch, input_dim, 85)
  Block 1: Conv1d → BN → ReLU → Dropout → MaxPool
  Block 2: Conv1d → BN → ReLU → Dropout → MaxPool
  Block 3: Conv1d → BN → ReLU → Dropout
  Pool   : AdaptiveAvgPool1d  OR  attentive temporal pool
  FC → num_classes
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class AttentivePool1d(nn.Module):
    """Linear score → softmax over time → weighted sum. Core ML-safe."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, C, T)
        x_t = x.transpose(1, 2)          # (B, T, C)
        weights = F.softmax(self.score(x_t).squeeze(-1), dim=1)  # (B, T)
        return (x_t * weights.unsqueeze(-1)).sum(dim=1)          # (B, C)


class CNN1D(nn.Module):
    def __init__(
        self,
        num_classes: int = 330,
        dropout: float = 0.3,
        input_dim: int = FEATURE_DIM,
        pool: str = "avg",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.pool_type = pool

        self.features = nn.Sequential(
            ConvBlock(input_dim, 128, kernel=3, pool=True,  dropout=dropout),
            ConvBlock(128,       256, kernel=3, pool=True,  dropout=dropout),
            ConvBlock(256,       512, kernel=3, pool=False, dropout=dropout),
        )

        if pool == "attn":
            self.pool = AttentivePool1d(512)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: (batch, N_FRAMES, input_dim)
        x = x.permute(0, 2, 1).contiguous()
        x = self.features(x)
        if self.pool_type == "attn":
            x = self.pool(x)            # (B, 512)
            x = x.unsqueeze(-1)         # match Flatten expectation via classifier
        else:
            x = self.pool(x)            # (B, 512, 1)
        return self.classifier(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)


LSM_CNN = CNN1D


def create_cnn1d(
    num_classes: int,
    dropout: float = 0.3,
    input_dim: int = FEATURE_DIM,
    pool: str = "avg",
    **kwargs,
) -> CNN1D:
    return CNN1D(
        num_classes=num_classes,
        dropout=dropout,
        input_dim=input_dim,
        pool=pool,
    )
