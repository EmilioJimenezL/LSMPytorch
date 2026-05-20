"""
LSM — Modelo TCN para clasificación de señas
=============================================
Temporal Convolutional Network con convoluciones causales dilatadas.

Arquitectura:
  Input  : (batch, N_FRAMES, FEATURE_DIM)  →  permute  →  (batch, FEATURE_DIM, N_FRAMES)
  Proyección 1×1: FEATURE_DIM → 64
  4 bloques residuales con dilatación 1, 2, 4, 8
  Global Average Pooling
  FC(256→128) + ReLU + Dropout
  FC(128→n_classes)

Campo receptivo: 1 + Σ(k-1)×2^i para i∈[0,3] = 61 frames (71.8%)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import FEATURE_DIM, N_FRAMES


class _ResidualBlock(nn.Module):
    """Bloque residual TCN con convolución causal dilatada."""

    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation

        # Sin padding en Conv1d — se aplica pad izquierdo manual para causalidad
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop2 = nn.Dropout(dropout)

        # Proyección residual si los canales difieren
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu     = nn.ReLU()

    def forward(self, x):
        # Pad izquierdo → conv → BN → ReLU → dropout  (longitud preservada: T → T)
        out = self.relu(self.bn1(self.conv1(F.pad(x, (self.left_pad, 0)))))
        out = self.drop1(out)
        out = self.bn2(self.conv2(F.pad(out, (self.left_pad, 0))))
        out = self.drop2(out)
        res = x if self.res_conv is None else self.res_conv(x)
        return self.relu(out + res)


class LSM_TCN(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()

        num_channels     = [64, 128, 256, 256]
        kernel_size      = 5
        self._kernel_size = kernel_size
        self._num_layers  = len(num_channels)

        # Proyección de entrada
        self.input_proj = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, num_channels[0], kernel_size=1),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
        )

        # Bloques TCN con dilatación exponencial
        self.tcn_blocks = nn.ModuleList()
        for i in range(self._num_layers):
            in_ch  = num_channels[i - 1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            self.tcn_blocks.append(
                _ResidualBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_classes),
        )

        self._init_weights()

    def forward(self, x):
        # x: (batch, N_FRAMES, FEATURE_DIM)
        x = x.permute(0, 2, 1)     # → (batch, FEATURE_DIM, N_FRAMES)
        x = self.input_proj(x)     # → (batch, 64, N_FRAMES)
        for block in self.tcn_blocks:
            x = block(x)           # longitud N_FRAMES preservada
        x = self.gap(x)
        return self.classifier(x)  # logits (batch, n_classes)

    def predict_proba(self, x):
        """Retorna probabilidades softmax — usado en inferencia."""
        return torch.softmax(self.forward(x), dim=-1)

    def receptive_field(self) -> int:
        """Campo receptivo total en frames."""
        return 1 + sum((self._kernel_size - 1) * (2 ** i)
                       for i in range(self._num_layers))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = LSM_TCN(n_classes=330)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print(f"Campo receptivo:        {model.receptive_field()} frames")

    dummy = torch.randn(4, N_FRAMES, FEATURE_DIM)
    out   = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")
