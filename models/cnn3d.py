"""
3D CNN with Temporal Attention for LSM skeleton-based recognition.

Input:  (batch, 85, 135)
Output: (batch, num_classes)

Reshape strategy: (batch, 85, 135) → (batch, 1, 85, 135, 1)
  - D-dimension (first spatial): frames (85) — strided for temporal downsampling
  - H-dimension (second spatial): features (135) — preserved throughout
  - W-dimension: singleton (1) — acts as a squeeze dimension

Architecture:
  Conv3d block 1: 1 → 32,  stride=(1,1,1) — maintain temporal resolution
  Conv3d block 2: 32 → 64, stride=(2,1,1) — downsample frames 2× (~85 → 43)
  Conv3d block 3: 64 → 128,stride=(2,1,1) — downsample frames 2× (~43 → 22)
  Temporal attention: channel-wise weighting via 128→32→1 Conv3d + Sigmoid
  AdaptiveAvgPool3d(1,1,1) → FC(256 → 128 → num_classes)

Parameters: ~0.2M  |  Size: ~1MB  |  Expected accuracy: 80-85%

Note on template reshape: the original template used (batch, 1, 17, 85, 1) which
mismatches element counts (1445 ≠ 11475). Corrected to (batch, 1, 85, 135, 1).
"""

import torch
import torch.nn as nn


class LSM3DCNNModel(nn.Module):
    def __init__(
        self,
        num_classes: int = 330,
        num_frames: int = 85,
        feature_dim: int = 135,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_frames  = num_frames
        self.feature_dim = feature_dim

        # Block 1: maintain temporal resolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 0.5),
        )

        # Block 2: downsample frames ~2×
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), stride=(2, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 0.5),
        )

        # Block 3: downsample frames ~2×
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), stride=(2, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
        )

        # Temporal attention: learns per-position importance weights
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=(1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 1)),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        # x: (batch, 85, 135) → (batch, 1, 85, 135, 1) — unsqueeze traces better for Core ML
        x = x.unsqueeze(1).unsqueeze(-1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        attn = self.temporal_attention(x)
        x = x * attn

        # Mean pool over D,H,W — Core ML compatible (AdaptiveAvgPool3d not supported)
        x = x.mean(dim=(2, 3, 4))
        return self.classifier(x)

    def receptive_field(self) -> str:
        return "~11 frames (temporal), full joints (spatial)"

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def create_3dcnn(num_classes: int, dropout: float = 0.3) -> LSM3DCNNModel:
    return LSM3DCNNModel(num_classes=num_classes, dropout_rate=dropout)
