"""
Temporal Convolutional Network (TCN) for LSM skeleton-based recognition.

Input:  (batch, 85, 135)
Output: (batch, num_classes)

Architecture:
  Input projection: Conv1d 135 → 64
  4 residual blocks with exponential dilation (1, 2, 4, 8)
  Causal (left-only) padding: output length == input length at every block
  Global average pooling → FC(256 → 128 → num_classes)

Receptive field: 1 + Σ (k-1)*2^i for i in [0,3] = 1+4+8+16+32 = 61 frames (72% of 85)
Parameters: ~1.8M  |  Size: ~7-10MB  |  Expected accuracy: 85-87%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """TCN residual block with causal (left-only) dilated convolutions."""

    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation

        # No padding in Conv1d — we pad manually before each conv for causality
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        # 1×1 conv to match channels when they differ
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        # Left-pad → conv → BN → ReLU → dropout  (length preserved: T → T)
        out = self.relu(self.bn1(self.conv1(F.pad(x, (self.left_pad, 0)))))
        out = self.drop1(out)

        out = self.bn2(self.conv2(F.pad(out, (self.left_pad, 0))))
        out = self.drop2(out)

        res = x if self.res_conv is None else self.res_conv(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    TCN for skeleton action recognition.

    num_channels=[64, 128, 256, 256] with 4 layers produces:
      Block 0: 64  → 64  , dilation=1
      Block 1: 64  → 128 , dilation=2
      Block 2: 128 → 256 , dilation=4
      Block 3: 256 → 256 , dilation=8
    """

    def __init__(
        self,
        num_classes: int = 330,
        num_frames: int = 85,
        input_dim: int = 135,
        num_channels=None,
        kernel_size: int = 5,
        dropout: float = 0.3,
        num_layers: int = 4,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256, 256]

        self.num_frames   = num_frames
        self._kernel_size = kernel_size
        self._num_layers  = num_layers

        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, num_channels[0], kernel_size=1),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU(),
        )

        self.tcn_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch  = num_channels[i - 1] if i > 0 else num_channels[0]
            out_ch = num_channels[i]
            self.tcn_blocks.append(
                ResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )

        final_ch = num_channels[-1]
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_ch, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Linear(128, num_classes)
        self._init_weights()

    def forward(self, x):
        # x: (batch, 85, 135)
        x = x.permute(0, 2, 1)       # → (batch, 135, 85)
        x = self.input_projection(x)  # → (batch, 64, 85)
        for block in self.tcn_blocks:
            x = block(x)             # length preserved at 85
        x = self.fc_layers(x)
        return self.classifier(x)    # → (batch, num_classes)

    def receptive_field(self) -> int:
        """Total receptive field in frames."""
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


def create_tcn(num_classes: int, dropout: float = 0.3) -> TemporalConvNet:
    return TemporalConvNet(
        num_classes=num_classes,
        dropout=dropout,
        num_channels=[64, 128, 256, 256],
        num_layers=4,
    )
