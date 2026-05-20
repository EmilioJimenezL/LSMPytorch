"""
LSM — Modelo 3D CNN para clasificación de señas
=================================================
Red convolucional 3D con mecanismo de atención temporal.

Arquitectura:
  Input  : (batch, N_FRAMES, FEATURE_DIM)
           → view → (batch, 1, N_FRAMES, FEATURE_DIM, 1)
  Block 1: Conv3d(1→32,  k=3×3×1, stride=1×1×1) + BN3d + ReLU + Dropout3d
  Block 2: Conv3d(32→64, k=3×3×1, stride=2×1×1) + BN3d + ReLU + Dropout3d  ← downsample tiempo 2×
  Block 3: Conv3d(64→128,k=3×3×1, stride=2×1×1) + BN3d + ReLU + Dropout3d  ← downsample tiempo 2×
  Atención temporal: Conv3d(128→32→1) + Sigmoid → pondera el mapa de características
  AdaptiveAvgPool3d(1,1,1) → FC(128→256) + BN1d + ReLU + Dropout → FC(256→n_classes)

Nota: el reshape usa (batch, 1, N_FRAMES, FEATURE_DIM, 1) — preserva todas las 135
features; la dimensión D (temporal) es la que recibe el stride descendente.
El esquema (batch, 1, 17, 85, 1) del spec original tiene inconsistencia de elementos
(1445 ≠ 11475) y no es aplicable directamente.

Campo receptivo: ~11 frames (temporal), todas las features (espacial)
Parámetros: ~0.2M
"""

import torch
import torch.nn as nn

from dataset import FEATURE_DIM, N_FRAMES


class LSM_3DCNN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()

        # Bloque 1 — mantiene resolución temporal completa
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout * 0.5),
        )

        # Bloque 2 — downsample temporal 2× (~85 → 43 frames)
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), stride=(2, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout * 0.5),
        )

        # Bloque 3 — downsample temporal 2× (~43 → 22 frames)
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), stride=(2, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
        )

        # Atención temporal aprendida por canal
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 1)),
            nn.Sigmoid(),
        )

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        self._init_weights()

    def forward(self, x):
        # x: (batch, N_FRAMES, FEATURE_DIM)
        batch = x.size(0)

        # Reshape a volumen 3D: D=tiempo, H=features, W=1
        x = x.view(batch, 1, N_FRAMES, FEATURE_DIM, 1)
        # → (batch, 1, 85, 135, 1)

        x = self.conv1(x)   # → (batch, 32,  85, 135, 1)
        x = self.conv2(x)   # → (batch, 64,  43, 135, 1)
        x = self.conv3(x)   # → (batch, 128, 22, 135, 1)

        # Atención: broadcast (batch, 1, 22, 135, 1) sobre 128 canales
        attn = self.temporal_attention(x)
        x    = x * attn

        x = self.gap(x)             # → (batch, 128, 1, 1, 1)
        return self.classifier(x)   # logits (batch, n_classes)

    def predict_proba(self, x):
        """Retorna probabilidades softmax — usado en inferencia."""
        return torch.softmax(self.forward(x), dim=-1)

    def receptive_field(self) -> str:
        return "~11 frames (temporal), full joints (spatial)"

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSM_3DCNN(n_classes=330)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print(f"Campo receptivo:        {model.receptive_field()}")

    dummy = torch.randn(4, N_FRAMES, FEATURE_DIM)
    out   = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
