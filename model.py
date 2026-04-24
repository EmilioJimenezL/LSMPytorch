"""
LSM — Modelo 1D CNN para clasificación de señas
=================================================
Arquitectura:
  Input  : (batch, N_FRAMES, 135)  →  permute  →  (batch, 135, N_FRAMES)
  Block 1: Conv1d(135→128, k=3) + BN + ReLU + Dropout + MaxPool
  Block 2: Conv1d(128→256, k=3) + BN + ReLU + Dropout + MaxPool
  Block 3: Conv1d(256→512, k=3) + BN + ReLU + Dropout
  Global Average Pooling
  FC(512→256) + ReLU + Dropout
  FC(256→n_classes) + Softmax
"""

import torch
import torch.nn as nn
from dataset import FEATURE_DIM, N_FRAMES


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


class LSM_CNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(FEATURE_DIM, 128, kernel=3, pool=True,  dropout=dropout),
            ConvBlock(128,         256, kernel=3, pool=True,  dropout=dropout),
            ConvBlock(256,         512, kernel=3, pool=False, dropout=dropout),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling → (batch, 512, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (batch, N_FRAMES, FEATURE_DIM)
        x = x.permute(0, 2, 1)   # → (batch, FEATURE_DIM, N_FRAMES)
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x                  # logits (batch, n_classes)

    def predict_proba(self, x):
        """Retorna probabilidades softmax — usado en inferencia."""
        return torch.softmax(self.forward(x), dim=-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSM_CNN(n_classes=800)
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    dummy = torch.randn(4, N_FRAMES, FEATURE_DIM)
    out = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
