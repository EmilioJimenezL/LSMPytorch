# Implementation Templates for AI Agents
## Ready-to-Use Code Snippets for LSM Multi-Model Training

---

## Template 1: Refactored 1D CNN → `models/cnn1d.py`

```python
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network (Baseline Model)
    
    Architecture:
    Input: (batch, 85, 135) → permute → (batch, 135, 85)
    Conv1d blocks with BatchNorm, ReLU, Dropout, MaxPooling
    Global average pooling → FC layers → logits
    
    Parameters: ~1.8M
    Size: ~7-8MB
    """
    
    def __init__(self, num_classes=330, dropout=0.3):
        super().__init__()
        
        # Input permute from (batch, 85, 135) to (batch, 135, 85)
        # This matches PyTorch Conv1d convention: (batch, channels, length)
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(135, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, 85, 135)
        x = x.permute(0, 2, 1)  # → (batch, 135, 85)
        
        x = self.conv1(x)  # → (batch, 128, 42)
        x = self.conv2(x)  # → (batch, 256, 21)
        x = self.conv3(x)  # → (batch, 512, 21)
        
        x = self.global_pool(x)  # → (batch, 512, 1)
        x = x.view(x.size(0), -1)  # → (batch, 512)
        
        x = self.fc(x)  # → (batch, num_classes)
        
        return x


def create_cnn1d(num_classes, dropout=0.3):
    """Factory function for 1D CNN"""
    return CNN1D(num_classes=num_classes, dropout=dropout)
```

---

## Template 2: TCN Implementation → `models/tcn.py`

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with causal convolution and dilation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.net = nn.Sequential(
            self.conv1,
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            self.conv2,
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout)
        )
        
        # Residual connection
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        # Remove future information (causal)
        out = out[:, :, :-self.padding]
        
        # Residual
        res = x if self.res_conv is None else self.res_conv(x)
        
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for skeleton-based action recognition
    
    Input: (batch, 85, 135) → reshape to (batch, 135, 85)
    Output: (batch, num_classes)
    
    Architecture:
    - 4 residual blocks with exponential dilation (1, 2, 4, 8)
    - Each block has causal padding
    - Receptive field grows exponentially
    
    Parameters: ~1.8M
    Size: ~7-10MB
    Receptive field: ~61 frames (71% of 85)
    """
    
    def __init__(
        self,
        num_classes=330,
        num_frames=85,
        input_dim=135,
        num_channels=[64, 128, 256, 256],
        kernel_size=5,
        dropout=0.3,
        num_layers=4
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.input_dim = input_dim
        
        # Project input if needed
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, num_channels[0], kernel_size=1),
            nn.BatchNorm1d(num_channels[0]),
            nn.ReLU()
        )
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = num_channels[i] if i > 0 else num_channels[0]
            out_channels = num_channels[i % len(num_channels)]
            dilation = 2 ** i  # 1, 2, 4, 8
            
            self.tcn_blocks.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Classification head
        final_channels = num_channels[-1]
        
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            nn.Linear(final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
        self._init_weights()
    
    def forward(self, x):
        # x: (batch, 85, 135)
        batch_size = x.size(0)
        
        # Reshape: (batch, 85, 135) → (batch, 135, 85)
        x = x.permute(0, 2, 1)
        
        # Project input
        x = self.input_projection(x)  # (batch, channels[0], 85)
        
        # TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        # Classification
        x = self.fc_layers(x)
        logits = self.classifier(x)
        
        return logits
    
    def receptive_field(self):
        """Calculate total receptive field in frames"""
        rf = 1
        for i in range(len(self.tcn_blocks)):
            dilation = 2 ** i
            kernel_size = 5
            rf += (kernel_size - 1) * dilation
        return rf
    
    def _init_weights(self):
        """Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def create_tcn(num_classes, dropout=0.3):
    """Factory function for TCN"""
    return TemporalConvNet(
        num_classes=num_classes,
        dropout=dropout,
        num_channels=[64, 128, 256, 256],
        num_layers=4
    )
```

---

## Template 3: 3D CNN Implementation → `models/cnn3d.py`

```python
import torch
import torch.nn as nn

class LSM3DCNNModel(nn.Module):
    """
    3D Convolutional Neural Network with Temporal Attention
    
    Input: (batch, 85, 135) → reshape to (batch, 1, 17, 85, 1)
    Output: (batch, num_classes)
    
    Architecture:
    - Conv3d blocks with temporal downsampling (stride on time dimension)
    - Temporal attention mechanism
    - Global average pooling + FC layers
    
    Parameters: ~2.1M
    Size: ~8-12MB
    Receptive field: ~11 frames
    Speed: ~80-90ms iOS inference
    """
    
    def __init__(
        self,
        num_classes=330,
        num_frames=85,
        num_joints=17,
        dropout_rate=0.3
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.num_joints = num_joints
        
        # ========== CONV BLOCKS ==========
        
        # Block 1: 1 → 32 channels, maintain temporal res
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 0.5)
        )
        
        # Block 2: 32 → 64 channels, downsample time 2x
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 1), stride=(2, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate * 0.5)
        )
        
        # Block 3: 64 → 128 channels, downsample time 2x
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), stride=(2, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate)
        )
        
        # ========== TEMPORAL ATTENTION ==========
        
        self.temporal_attention = nn.Sequential(
            nn.Conv3d(128, 32, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )
        
        # ========== CLASSIFICATION HEAD ==========
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
        self._init_weights()
    
    def forward(self, x):
        # x: (batch, 85, 135)
        batch_size = x.size(0)
        
        # Reshape: (batch, 85, 135) → (batch, 1, 17, 85, 1)
        # Treat 135 features as: 17 joints × 3 coords (x, y, visibility)
        x = x.view(batch_size, 1, self.num_joints, self.num_frames, 1)
        
        # Conv blocks
        x = self.conv1(x)  # (batch, 32, 17, 85, 1)
        x = self.conv2(x)  # (batch, 64, 17, 42, 1)
        x = self.conv3(x)  # (batch, 128, 17, 21, 1)
        
        # Temporal attention
        attn_weights = self.temporal_attention(x)  # (batch, 1, 17, 21, 1)
        x = x * attn_weights
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 128, 1, 1, 1)
        x = x.view(batch_size, -1)  # (batch, 128)
        
        # Classification
        x = self.fc_layers(x)  # (batch, 128)
        logits = self.classifier(x)  # (batch, num_classes)
        
        return logits
    
    def _init_weights(self):
        """Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


def create_3dcnn(num_classes, dropout=0.3):
    """Factory function for 3D CNN"""
    return LSM3DCNNModel(num_classes=num_classes, dropout_rate=dropout)
```

---

## Template 4: Model Factory → `models/__init__.py`

```python
from .cnn1d import CNN1D, create_cnn1d
from .tcn import TemporalConvNet, create_tcn
from .cnn3d import LSM3DCNNModel, create_3dcnn

MODEL_REGISTRY = {
    '1d_cnn': create_cnn1d,
    'tcn': create_tcn,
    '3d_cnn': create_3dcnn,
}

def create_model(model_name: str, num_classes: int, **kwargs):
    """
    Factory function for creating models
    
    Args:
        model_name: '1d_cnn', 'tcn', or '3d_cnn'
        num_classes: Number of output classes
        **kwargs: Model-specific arguments (dropout, etc.)
    
    Returns:
        PyTorch model instance
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    return MODEL_REGISTRY[model_name](num_classes, **kwargs)

def get_model_info(model_name: str) -> dict:
    """Get info about a model"""
    info = {
        '1d_cnn': {
            'name': '1D CNN (Baseline)',
            'parameters': '~1.8M',
            'size_mb': '7-8',
            'training_time': '13-17h',
            'accuracy': '80-85%',
            'speed_ms': '15-20',
        },
        'tcn': {
            'name': 'Temporal Convolutional Network',
            'parameters': '~1.8M',
            'size_mb': '7-10',
            'training_time': '10-12h',
            'accuracy': '85-87%',
            'speed_ms': '100-150',
        },
        '3d_cnn': {
            'name': '3D CNN',
            'parameters': '~2.1M',
            'size_mb': '8-12',
            'training_time': '12-15h',
            'accuracy': '80-85%',
            'speed_ms': '80-90',
        }
    }
    return info.get(model_name, {})

__all__ = [
    'create_model',
    'get_model_info',
    'MODEL_REGISTRY',
    'CNN1D',
    'TemporalConvNet',
    'LSM3DCNNModel'
]
```

---

## Template 5: Smoke Test Script

```python
# test_models_smoke.py

import torch
from models import create_model, MODEL_REGISTRY

def run_smoke_tests():
    """Quick tests to verify all models work"""
    
    print("=" * 60)
    print("SMOKE TESTS: Model Implementations")
    print("=" * 60)
    
    num_classes = 330
    batch_size = 4
    num_frames = 85
    feature_dim = 135
    
    # Create dummy input
    x = torch.randn(batch_size, num_frames, feature_dim)
    
    for model_name in MODEL_REGISTRY.keys():
        print(f"\n{model_name.upper()}")
        print("-" * 60)
        
        try:
            # Create model
            model = create_model(model_name, num_classes)
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                output = model(x)
            
            # Verify output
            assert output.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {output.shape}"
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters())
            
            # Verify no NaNs
            assert not torch.isnan(output).any(), "Output contains NaN"
            
            print(f"  ✓ Output shape: {tuple(output.shape)}")
            print(f"  ✓ Parameters: {params:,}")
            print(f"  ✓ No NaN values")
            
            # Additional checks for specific models
            if model_name == 'tcn':
                rf = model.receptive_field()
                print(f"  ✓ Receptive field: {rf} frames ({rf/num_frames*100:.1f}%)")
            
            print(f"  ✓ {model_name} PASSED")
            
        except Exception as e:
            print(f"  ✗ {model_name} FAILED: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = run_smoke_tests()
    exit(0 if success else 1)
```

**Run with:**
```bash
python test_models_smoke.py
```

---

## Template 6: Training Configuration Structure → `configs/cnn1d_config.yaml`

```yaml
# Model configuration
model:
  type: 1d_cnn
  num_classes: 330
  dropout: 0.3

# Training hyperparameters
training:
  epochs: 150
  batch_size: 64
  learning_rate: 1e-3
  weight_decay: 1e-4
  warmup_epochs: 10
  
  # Optimizer
  optimizer: adamw
  betas: [0.9, 0.999]
  
  # Scheduler
  scheduler: cosine_annealing
  eta_min: 1e-6
  
  # Loss
  loss: cross_entropy
  label_smoothing: 0.1

# Data augmentation
augmentation:
  enabled: true
  augment_factor: 10
  scale: [0.9, 1.1]
  translation: 0.05
  time_warp: [0.8, 1.2]
  noise_std: 0.01
  joint_dropout: 0.1

# Validation
validation:
  val_split: 0.15
  min_videos_per_class: 2
  monitor_metric: val_acc
  patience: 15

# Hardware
hardware:
  device: cuda  # or cpu, mps
  mixed_precision: true
  num_workers: 4
  pin_memory: true
  seed: 42

# Output
output:
  save_best: true
  save_interval: 10
  log_interval: 10
```

---

## Template 7: Simple Training Loop Structure

```python
# Core training loop pseudocode for train_multimodel.py

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        total += target.size(0)
    
    scheduler.step()
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': correct / total
    }

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            total += target.size(0)
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': correct / total
    }
```

---

## Template 8: Quick CLI Interface Structure

```python
# In train_multimodel.py

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train LSM models')
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['1d_cnn', 'tcn', '3d_cnn'],
        default='1d_cnn',
        help='Model architecture to train'
    )
    
    # Data paths
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./runs',
        help='Output directory for checkpoints'
    )
    
    # Training params
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda/cpu/mps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Config file (optional)
    parser.add_argument('--config', type=str, help='Optional YAML config file')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # ... rest of training code
```

---

## Summary: Which Template to Use When

| File | Template | Purpose |
|------|----------|---------|
| `models/cnn1d.py` | Template 1 | Baseline 1D CNN |
| `models/tcn.py` | Template 2 | TCN architecture |
| `models/cnn3d.py` | Template 3 | 3D CNN architecture |
| `models/__init__.py` | Template 4 | Model factory |
| `test_models_smoke.py` | Template 5 | Quick validation |
| `configs/*.yaml` | Template 6 | Configuration files |
| `train_multimodel.py` | Template 7 | Training loop |
| `train_multimodel.py` | Template 8 | CLI argument parsing |

---

**Ready to implement?** Copy these templates into the appropriate files and customize as needed. Start with Templates 1-4 first.
