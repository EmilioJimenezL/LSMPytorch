# AI Agent Prompt: LSM Model Implementation & Comparison
## Environment Setup + TCN + 3D CNN + Testing vs 1D CNN

---

## 📋 Context & Overview

You are working on the **LSMPytorch** repository (Mexican Sign Language Recognition). The project currently has:
- **1D CNN model** (`model.py`) — baseline using Conv1d on (batch, 135, 85) skeleton sequences
- **Training pipeline** (`train.py`) — standard PyTorch training with augmentation, validation, checkpoints
- **Dataset** — Vision Framework extracted landmarks in JSON format (21 left hand + 21 right hand + 17 pose keypoints = 135 features per frame, 85 frames fixed)

Your task is to:
1. ✓ Set up a multi-model training environment
2. ✓ Implement TCN (Temporal Convolutional Network) architecture
3. ✓ Implement 3D CNN architecture
4. ✓ Create a fair comparison framework testing all three models
5. ✓ Generate comparison reports (accuracy, speed, memory, training time)

**Repository:** https://github.com/EmilioJimenezL/LSMPytorch

---

## Part 1: Environment Preparation

### 1.1 Directory Structure

Create the following structure in the repository:

```
LSMPytorch/
├── model.py                    # EXISTING: 1D CNN (LSM_CNN)
├── models/                     # NEW: Multi-model directory
│   ├── __init__.py
│   ├── cnn1d.py               # Refactor existing 1D CNN here
│   ├── tcn.py                 # NEW: TCN implementation
│   ├── cnn3d.py               # NEW: 3D CNN implementation
│   └── base.py                # Optional: common base class
│
├── train.py                    # EXISTING: Keep original
├── train_multimodel.py         # NEW: Enhanced trainer supporting model selection
├── compare_models.py           # NEW: Comparison framework
├── inference_benchmark.py      # NEW: Speed/memory benchmarking
│
├── configs/                    # NEW: Model configs
│   ├── cnn1d_config.yaml
│   ├── tcn_config.yaml
│   └── cnn3d_config.yaml
│
├── results/                    # NEW: Comparison outputs
│   ├── accuracy_comparison.csv
│   ├── speed_comparison.json
│   ├── training_curves_all_models.png
│   └── model_sizes.txt
│
└── requirements.txt            # UPDATE: Add new dependencies
```

### 1.2 Update Requirements

Add the following to `requirements.txt`:

```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
coremltools>=7.0
pyyaml>=6.0
tensorboard>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

### 1.3 Virtual Environment Setup Script

Create `setup_env.sh`:

```bash
#!/bin/bash

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support (update CUDA version as needed)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

echo "✓ Environment setup complete"
echo "Run: source venv/bin/activate"
```

---

## Part 2: Model Implementation

### 2.1 Refactor Existing 1D CNN → `models/cnn1d.py`

**Goal:** Move the existing `LSM_CNN` from `model.py` to `models/cnn1d.py` with minimal changes.

**Steps:**
1. Extract `LSM_CNN` class from `model.py`
2. Save to `models/cnn1d.py`
3. Add docstring explaining it's the baseline 1D convolutional model
4. Keep exact same architecture:
   - Conv1d(135→128, k=3) + BatchNorm + ReLU + Dropout
   - Conv1d(128→256, k=3) + BatchNorm + ReLU + Dropout
   - Conv1d(256→512, k=3) + BatchNorm + ReLU + Dropout
   - GlobalAvgPool + FC(512→256) + FC(256→n_classes)
5. Add utility function: `def create_cnn1d(num_classes, dropout=0.3):`
6. **Do NOT change the architecture** — this is our baseline

**Expected output:**
```python
# models/cnn1d.py
class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network (baseline model)
    Input: (batch, 85, 135)
    Output: (batch, num_classes)
    """
    # ... existing code ...

def create_cnn1d(num_classes, dropout=0.3):
    return CNN1D(num_classes, dropout)
```

---

### 2.2 Implement TCN → `models/tcn.py`

**Reference:** See TCN_vs_3DCNN_Detailed_Analysis.md (provided separately)

**Requirements:**
1. **Architecture:**
   - Input shape: (batch, 85, 135) → reshape to (batch, 135, 85) for 1D convolutions
   - 4 residual blocks with dilations: 1, 2, 4, 8
   - Causal padding to maintain temporal ordering
   - Global average pooling + classification head

2. **Implementation details:**
   - Use `ResidualBlock` class with Conv1d + BatchNorm + ReLU + Dropout
   - Causal convolution (padding=dilation*(kernel_size-1))
   - Remove future padding after each conv: `out = out[:, :, :-self.padding]`
   - Residual connections with 1×1 conv for dimension matching

3. **Class structure:**
   ```python
   class ResidualBlock(nn.Module):
       def __init__(self, in_channels, out_channels, kernel_size=5, dilation=1, dropout=0.3)
       
   class TemporalConvNet(nn.Module):
       def __init__(self, num_classes=330, num_frames=85, input_dim=51, ...)
       def forward(self, x)
       def receptive_field(self) -> int
   
   def create_tcn(num_classes, dropout=0.3):
       return TemporalConvNet(num_classes=num_classes, dropout=dropout)
   ```

4. **Key features:**
   - Parameters should be ~1.8M (similar to 1D CNN)
   - Receptive field calculation method
   - Initialization with Kaiming normal
   - Support for variable num_classes

5. **Testing:**
   - Verify output shape: input (8, 85, 135) → output (8, num_classes)
   - Print receptive field: should be ~61 frames out of 85
   - Count parameters: ~1.8M

---

### 2.3 Implement 3D CNN → `models/cnn3d.py`

**Reference:** See LSM_Architecture_Overhaul.md, Part 2.2

**Requirements:**
1. **Architecture:**
   - Input: (batch, 85, 135) → reshape to (batch, 1, 17, 85, 1) for 3D convolutions
   - 3 Conv3d blocks with temporal downsampling
   - Temporal attention mechanism
   - Global average pooling + classification head

2. **Implementation details:**
   - Conv3d(1→32, kernel=3×3×1, stride=1) with BatchNorm + ReLU
   - Conv3d(32→64, kernel=3×3×1, stride=2 temporal) — downsample time 2x
   - Conv3d(64→128, kernel=3×3×1, stride=2 temporal) — downsample time 2x
   - Temporal attention: Conv3d→ReLU→Conv3d→Sigmoid on temporal dimension
   - FC(128→256→num_classes)

3. **Class structure:**
   ```python
   class LSM3DCNNModel(nn.Module):
       def __init__(self, num_classes=330, num_frames=85, num_joints=17, dropout_rate=0.3)
       def forward(self, x)
   
   def create_3dcnn(num_classes, dropout=0.3):
       return LSM3DCNNModel(num_classes=num_classes, dropout_rate=dropout)
   ```

4. **Key features:**
   - Parameters: ~2.1M
   - Receptive field: ~11 frames
   - Input reshape logic: (batch, 85, 135) → (batch, 1, 17, 85, 1)
   - Temporal attention weights the output by learned importance scores

5. **Testing:**
   - Verify output shape and parameter count
   - Ensure attention mechanism weights correctly

---

### 2.4 Create `models/__init__.py`

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
    """Factory function for creating models"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](num_classes, **kwargs)

__all__ = ['create_model', 'MODEL_REGISTRY', 'CNN1D', 'TemporalConvNet', 'LSM3DCNNModel']
```

---

## Part 3: Training Infrastructure

### 3.1 Enhanced Trainer → `train_multimodel.py`

**Goal:** Create a unified trainer that works with any model architecture

**Requirements:**
1. **CLI arguments:**
   ```
   --model {1d_cnn, tcn, 3d_cnn}  # Model selection
   --config config.yaml            # Optional: override with config file
   --dataset ./data                # Dataset path
   --output ./runs                 # Output directory
   --epochs 150
   --batch 64
   --lr 1e-3
   --dropout 0.3
   --device {cuda, cpu, mps}
   --seed 42
   ```

2. **Key features:**
   - Load dataset (reuse existing `dataset.py` logic)
   - Create model via `models.create_model()`
   - Train with validation monitoring
   - Save best checkpoint + periodic checkpoints
   - Log metrics to TensorBoard (optional)
   - Generate training curves
   - Save model metadata (architecture, params, config)

3. **Training loop:**
   - Use same augmentation strategy as original
   - AdamW optimizer + CosineAnnealingLR
   - Mixed precision training (torch.cuda.amp)
   - Gradient clipping
   - Early stopping (optional patience parameter)

4. **Output structure:**
   ```
   runs/{model_name}_{timestamp}/
   ├── best_model.pt
   ├── checkpoint_epoch10.pt
   ├── classes.json
   ├── config.json
   ├── training_curves.png
   ├── metrics.json
   └── model_info.txt
   ```

5. **Model info tracking:**
   - Model architecture name
   - Parameter count
   - Training time
   - Best validation accuracy
   - Training/validation loss history

---

### 3.2 Configuration Files → `configs/`

Create YAML configs for each model:

**`configs/cnn1d_config.yaml`:**
```yaml
model:
  type: 1d_cnn
  dropout: 0.3

training:
  epochs: 150
  batch_size: 64
  learning_rate: 1e-3
  weight_decay: 1e-4
  warmup_epochs: 10
  
augmentation:
  scale: [0.9, 1.1]
  translation: 0.05
  time_warp: [0.8, 1.2]
  noise: 0.01
  augment_factor: 10
  
validation:
  split: 0.15
  min_videos: 2
```

**`configs/tcn_config.yaml`:**
```yaml
model:
  type: tcn
  num_channels: [64, 128, 256, 256]
  kernel_size: 5
  num_layers: 4
  dropout: 0.3

training:
  epochs: 120
  batch_size: 16
  learning_rate: 1e-3
  weight_decay: 1e-4
  warmup_epochs: 10
  
augmentation:
  # Same as 1D CNN
```

**`configs/cnn3d_config.yaml`:**
```yaml
model:
  type: 3d_cnn
  dropout: 0.3

training:
  epochs: 100
  batch_size: 12
  learning_rate: 1e-3
  weight_decay: 1e-4
  warmup_epochs: 10
  
augmentation:
  # Same as 1D CNN
```

---

## Part 4: Comparison Framework

### 4.1 Comparison Script → `compare_models.py`

**Goal:** Train all three models on the same data and generate comparison report

**Usage:**
```bash
python compare_models.py \
    --dataset ./SalidasLSMSkeletization \
    --output ./results \
    --models 1d_cnn tcn 3d_cnn \
    --epochs 150 \
    --seed 42
```

**Features:**
1. **Train all models sequentially:**
   - Use same train/val/test split for all
   - Log training time for each model
   - Save best checkpoint and metrics

2. **Evaluation on test set:**
   - Top-1 accuracy
   - Top-5 accuracy
   - Per-class accuracy (show best/worst)
   - Confusion matrix (save as image)
   - Loss distribution

3. **Memory profiling:**
   - Peak GPU memory usage during training
   - Model size on disk
   - Core ML export size (if applicable)

4. **Speed benchmarking:**
   - Inference time per sample (CPU & GPU)
   - Throughput (samples/sec)
   - Latency statistics (min/max/mean/std)

5. **Output files:**
   ```
   results/
   ├── comparison_summary.csv
   │   # Columns: model | params | train_time | val_acc | test_acc | test_top5 | memory_peak | inference_ms
   │
   ├── model_details.json
   │   # Detailed metrics per model
   │
   ├── training_curves_comparison.png
   │   # All three models' loss/accuracy curves overlaid
   │
   ├── accuracy_per_class.png
   │   # Bar chart: which classes each model excels at
   │
   ├── confusion_matrices/
   │   ├── 1d_cnn_confusion.png
   │   ├── tcn_confusion.png
   │   └── 3d_cnn_confusion.png
   │
   ├── inference_speed.png
   │   # Latency comparison chart
   │
   └── final_report.txt
       # Markdown summary with conclusions
   ```

6. **Report content:**
   - Model comparison table
   - Winner for each metric (accuracy, speed, size)
   - Training time comparison
   - Recommendation based on use case (mobile vs accuracy)
   - Per-class performance analysis

---

### 4.2 Inference Benchmark → `inference_benchmark.py`

**Goal:** Profile inference speed and memory for each model

**Usage:**
```bash
python inference_benchmark.py \
    --models ./runs/1d_cnn_best/best_model.pt \
              ./runs/tcn_best/best_model.pt \
              ./runs/3d_cnn_best/best_model.pt \
    --output ./results/speed_benchmark.json \
    --num_iterations 1000
```

**Measurements:**
1. **Latency:**
   - Per-sample inference time (single forward pass)
   - Batch inference (batch_size=8, 16, 32)
   - Min/max/mean/std latencies
   - P95, P99 percentiles

2. **Throughput:**
   - Samples per second
   - For different batch sizes

3. **Memory:**
   - Peak memory during inference
   - Model size in memory (weights + buffers)

4. **Output format:**
   ```json
   {
     "1d_cnn": {
       "latency_ms": {"mean": 15.3, "std": 2.1, "p95": 22.5},
       "throughput_samples_per_sec": 65.4,
       "memory_mb": 12.5,
       "model_size_mb": 8.2
     },
     "tcn": {...},
     "3d_cnn": {...}
   }
   ```

---

## Part 5: Testing & Validation

### 5.1 Unit Tests → `tests/test_models.py`

Create test file to verify:

```python
# Test for each model:
def test_cnn1d_forward():
    # Input shape: (8, 85, 135)
    # Output shape should be: (8, num_classes)
    
def test_tcn_forward():
    # Same input/output
    # Verify receptive field > 50
    
def test_3dcnn_forward():
    # Same input/output
    # Verify temporal attention works
    
def test_model_registry():
    # Verify all models can be created via factory
    
def test_gradient_flow():
    # Ensure gradients flow through all models
    
def test_device_compatibility():
    # Test on CPU, CUDA (if available), and MPS (if macOS)
```

### 5.2 Smoke Tests

Before full training, run quick smoke tests:

```bash
python -c "
from models import create_model
import torch

for model_name in ['1d_cnn', 'tcn', '3d_cnn']:
    print(f'Testing {model_name}...')
    model = create_model(model_name, 330)
    x = torch.randn(4, 85, 135)
    y = model(x)
    assert y.shape == (4, 330), f'Expected (4, 330), got {y.shape}'
    params = sum(p.numel() for p in model.parameters())
    print(f'  ✓ {model_name}: {params:,} parameters')
"
```

---

## Part 6: AI Agent Implementation Checklist

### Phase 1: Setup & Refactoring (🎯 Priority: HIGH)

- [ ] Create `models/` directory structure
- [ ] Update `requirements.txt` with new dependencies
- [ ] Create `setup_env.sh` script
- [ ] Refactor existing 1D CNN to `models/cnn1d.py`
- [ ] Create `models/__init__.py` with factory function
- [ ] Run smoke tests on refactored 1D CNN

**Estimated time:** 1-2 hours  
**Success criteria:** Smoke test passes, 1D CNN produces same results as original

---

### Phase 2: Model Implementation (🎯 Priority: HIGH)

- [ ] Implement TCN in `models/tcn.py`
  - [ ] ResidualBlock class with causal convolutions
  - [ ] TemporalConvNet main class
  - [ ] Receptive field calculation
  - [ ] Test with dummy input
  
- [ ] Implement 3D CNN in `models/cnn3d.py`
  - [ ] Conv3d blocks with temporal downsampling
  - [ ] Temporal attention mechanism
  - [ ] Input reshape logic
  - [ ] Test with dummy input

- [ ] Create model configs in `configs/`
- [ ] Run smoke tests on all three models

**Estimated time:** 4-6 hours  
**Success criteria:** All models accept (batch, 85, 135) input, output (batch, num_classes), parameters match expectations

---

### Phase 3: Training Infrastructure (🎯 Priority: MEDIUM)

- [ ] Create `train_multimodel.py`
  - [ ] Parse CLI arguments for model selection
  - [ ] Load dataset using existing `dataset.py`
  - [ ] Implement training loop with mixed precision
  - [ ] Save checkpoints and metrics
  - [ ] Generate training curves

- [ ] Create config loading mechanism
- [ ] Test training a 1D CNN model from scratch

**Estimated time:** 3-4 hours  
**Success criteria:** Successfully train 1D CNN model on sample dataset, produce expected outputs

---

### Phase 4: Comparison & Benchmarking (🎯 Priority: MEDIUM)

- [ ] Implement `compare_models.py`
  - [ ] Sequential training of all models on same split
  - [ ] Evaluation on test set
  - [ ] Metric collection (accuracy, time, memory)
  - [ ] Visualization generation

- [ ] Implement `inference_benchmark.py`
  - [ ] Latency measurement
  - [ ] Throughput calculation
  - [ ] Memory profiling
  - [ ] JSON output with results

- [ ] Generate final comparison report

**Estimated time:** 3-4 hours  
**Success criteria:** Comparison script runs without errors, produces readable report and visualizations

---

### Phase 5: Testing & Documentation (🎯 Priority: LOW)

- [ ] Create unit tests in `tests/test_models.py`
- [ ] Create README update explaining multi-model setup
- [ ] Create TESTING.md guide
- [ ] Verify all models work on sample data

**Estimated time:** 2-3 hours  
**Success criteria:** All tests pass, documentation is clear

---

## Part 7: Execution Order

**Recommended for AI agents:**

1. **Start with Phase 1** — Setup and refactoring (establish clean foundation)
2. **Then Phase 2** — Implement new models (core work)
3. **Then Phase 3** — Training infrastructure (test on small dataset first)
4. **Then Phase 4** — Comparison (run full comparison on real data)
5. **Finally Phase 5** — Testing and docs (polish)

**Each phase should be completed before moving to next.** After each phase, run validation tests.

---

## Part 8: Expected Results

After completing all phases, you should have:

1. ✓ Three separate model implementations (1D CNN, TCN, 3D CNN)
2. ✓ Unified training script supporting all models
3. ✓ Fair comparison showing:
   - **1D CNN baseline** ~80-85% accuracy, 13-17h training
   - **TCN** ~85-87% accuracy (+5-10%), 10-12h training
   - **3D CNN** ~80-85% accuracy, similar to 1D CNN
4. ✓ Speed benchmarks showing inference latency on CPU/GPU
5. ✓ Comparison report recommending best model for deployment

---

## Part 9: Debugging Tips for AI Agents

**If training fails:**
- Check dataset loading: `python -c "from dataset import load_raw_dataset; data = load_raw_dataset('./data')"`
- Verify model output shape with dummy input
- Check learning rate (try 1e-4 to 1e-2 range)
- Ensure batch size fits in VRAM

**If comparison script fails:**
- Run individual model training first
- Check saved checkpoint paths
- Verify test set has samples from all classes

**If inference is slow:**
- Profile with `torch.profiler`
- Check if model is on correct device (GPU/CPU)
- For 3D CNN, verify padding/reshape operations are correct

---

## Summary

This prompt provides AI agents with:
✓ Clear directory structure to create  
✓ Exact requirements for each model  
✓ Specific implementation details  
✓ Testing methodology  
✓ Execution sequence  
✓ Success criteria for each phase  

The agent should follow this systematically, testing after each phase before proceeding.

---

**Ready to implement?** The AI agent should start with Phase 1, then proceed sequentially. Ask for clarification if any requirement is ambiguous.
