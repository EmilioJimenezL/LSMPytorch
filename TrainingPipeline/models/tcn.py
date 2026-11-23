"""
Temporal Convolutional Network (TCN) for LSM skeleton-based recognition.

Supports causal (legacy) and non-causal (symmetric) padding, configurable
input_dim, and avg / attentive temporal pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """TCN residual block with dilated convolutions (causal or symmetric)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        dilation=1,
        dropout=0.3,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.left_pad = (kernel_size - 1) * dilation
        self.pad_total = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2   = nn.BatchNorm1d(out_channels)
        self.drop2 = nn.Dropout(dropout)

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def _pad(self, x):
        if self.causal:
            return F.pad(x, (self.left_pad, 0))
        # Symmetric: split pad between left and right.
        left = self.pad_total // 2
        right = self.pad_total - left
        return F.pad(x, (left, right))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(self._pad(x))))
        out = self.drop1(out)
        out = self.bn2(self.conv2(self._pad(out)))
        out = self.drop2(out)
        res = x if self.res_conv is None else self.res_conv(x)
        return self.relu(out + res)


class AttentivePool1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x):
        # x: (B, C, T)
        x_t = x.transpose(1, 2)
        weights = F.softmax(self.score(x_t).squeeze(-1), dim=1)
        return (x_t * weights.unsqueeze(-1)).sum(dim=1)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 330,
        num_frames: int = 85,
        input_dim: int = 135,
        num_channels=None,
        kernel_size: int = 5,
        dropout: float = 0.3,
        num_layers: int = 4,
        causal: bool = True,
        pool: str = "avg",
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [64, 128, 256, 256]

        self.num_frames = num_frames
        self.input_dim = input_dim
        self._kernel_size = kernel_size
        self._num_layers = num_layers
        self.causal = causal
        self.pool_type = pool

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
                    causal=causal,
                )
            )

        final_ch = num_channels[-1]
        if pool == "attn":
            self.temporal_pool = AttentivePool1d(final_ch)
            self.fc_layers = nn.Sequential(
                nn.Linear(final_ch, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
        else:
            self.temporal_pool = None
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
        x = x.permute(0, 2, 1).contiguous()
        x = self.input_projection(x)
        for block in self.tcn_blocks:
            x = block(x)
        if self.temporal_pool is not None:
            x = self.temporal_pool(x)
            x = self.fc_layers(x)
        else:
            x = self.fc_layers(x)
        return self.classifier(x)

    def receptive_field(self) -> int:
        """Total receptive field in frames (2 convs per residual block)."""
        return 1 + sum(
            2 * (self._kernel_size - 1) * (2 ** i)
            for i in range(self._num_layers)
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def create_tcn(
    num_classes: int,
    dropout: float = 0.3,
    input_dim: int = 135,
    causal: bool = True,
    pool: str = "avg",
    **kwargs,
) -> TemporalConvNet:
    return TemporalConvNet(
        num_classes=num_classes,
        dropout=dropout,
        input_dim=input_dim,
        num_channels=[64, 128, 256, 256],
        num_layers=4,
        causal=causal,
        pool=pool,
    )
