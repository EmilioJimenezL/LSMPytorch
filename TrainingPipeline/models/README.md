# models

Three sequence classifiers behind one factory. All take a tensor `(batch, 85, input_dim)` and return logits `(batch, num_classes)`. Parameter counts below are for 518 classes (the lab cap-5 vocabulary).

| Registry key | `train.py --model` | Class | File | Params (v1 / v2 input) |
|---|---|---|---|---|
| `1d_cnn` | `cnn` | `CNN1D` (alias `LSM_CNN`) | `cnn1d.py` | 0.81M / 0.82M |
| `tcn` | `tcn` | `TemporalConvNet` | `tcn.py` | 1.53M / 1.53M |
| `3d_cnn` | `3dcnn` | `LSM3DCNNModel` | `cnn3d.py` | 0.26M / 0.26M |

`__init__.py` exposes `create_model(name, num_classes, **kwargs)`, `MODEL_REGISTRY`, `TRAIN_MODEL_MAP` (CLI name → registry key), `count_parameters`, `get_model_info`. Kwargs the factories do not use are ignored. `MODEL_INFO` holds early planning estimates (accuracy, latency) that are **not measured results**; use `Runs/*/metrics.json` instead.

## `input_dim` and layouts

`input_dim` is the per-frame feature size, taken from the layout: **135** for `v1_135`, **162** for `v2_162` (see [`../README.md`](../README.md#feature-layouts)). `train.py` reads it from the data and saves it in `model_config`; `CoreMLConversion` rebuilds the model from that.

| Model | Where `input_dim` is used |
|---|---|
| `CNN1D` | In-channels of the first `Conv1d` |
| `TCN` | In-channels of the 1×1 input projection |
| `3D CNN` | **Ignored.** Convolutions slide over the feature axis, so any width runs, but it has only been trained and benchmarked on `v1_135`. `pool` and `causal` are also ignored |

Old checkpoints without `model_config` load as v1, `avg` pool, causal TCN (covered by a test).

## Temporal pooling: `pool="avg" | "attn"`

Applies to the CNN and TCN. Attention pooling is a learned weighted average over time: `Linear(C→1)` scores each frame, softmax over time, weighted sum of the frame features `(B, C, T) → (B, C)`. It lets the model focus on the frames where the sign happens instead of weighting rest frames equally. It uses only linear/softmax ops, so it converts to Core ML cleanly. `avg` is `AdaptiveAvgPool1d(1)`. All `lab_cap5*` full runs use `attn`.

## 1D CNN (`cnn1d.py`)

```
(B,85,D) → permute → (B,D,85)
ConvBlock  D→128,   k=3, BN, ReLU, Dropout, MaxPool/2    85 → 42
ConvBlock  128→256, k=3, BN, ReLU, Dropout, MaxPool/2    42 → 21
ConvBlock  256→512, k=3, BN, ReLU, Dropout               21
Pool (avg | attn) → 512
Linear 512→256, ReLU, Dropout → Linear 256→classes
```

Receptive field about 18 frames (three k=3 convs with two stride-2 pools). Fast baseline.

## TCN (`tcn.py`)

```
(B,85,D) → permute → (B,D,85)
Conv1d 1×1 D→64, BN, ReLU                       (input projection)
4 × ResidualBlock, channels [64,128,256,256], kernel 5, dilation 1,2,4,8
      block: pad → Conv → BN → ReLU → Dropout → pad → Conv → BN → Dropout, + skip (1×1 if channels change), ReLU
Pool (avg | attn) → 256
Linear 256→256, BN, ReLU, Dropout → Linear 256→128, BN, ReLU, Dropout(p/2) → Linear 128→classes
```

- **Receptive field:** `1 + Σ 2·(k−1)·2^i` over the 4 blocks (two convs per block) = `1 + 8·(1+2+4+8)` = **121 frames**, larger than the 85-frame input, so every output position sees the whole clip. `receptive_field()` returns it.
- **`causal`:** `True` (factory default, legacy) left-pads only so no position sees the future; `False` pads symmetrically. Sequences are fully buffered before classification, so non-causal is used for v2 (`--causal 0`). Output length stays 85 either way.
- Kaiming init for convs; BatchNorm in the head.

## 3D CNN (`cnn3d.py`)

```
(B,85,D) → (B,1,85,D,1)          # depth = frames, height = features, width = 1
Conv3d 1→32,   k(3,3,1), stride (1,1,1)   85
Conv3d 32→64,  k(3,3,1), stride (2,1,1)   → 43
Conv3d 64→128, k(3,3,1), stride (2,1,1)   → 22
   (each: BN, ReLU, Dropout3d)
Attention gate: Conv3d 128→32 → ReLU → Conv3d 32→1 → Sigmoid; features × gate
Mean over (T, D, 1) → 128
Linear 128→256, BN, ReLU, Dropout → Linear 256→classes
```

Temporal receptive field is 9 frames by calculation (the `receptive_field()` string says "~11"). The gate is a sigmoid weight per (frame, feature) position, not a softmax; the final reduction is a plain `mean` because `AdaptiveAvgPool3d` does not convert to Core ML. Smallest model, but it underperformed in earlier cycles (≈4–18% val top-1 across the earlier uncapped runs; see [`Legacy/`](../../Legacy/README.md)) and has no run in `Runs/`.

## Observed results

Lab cap-5, v2_162, attention pool: CNN 36.68% val top-1, TCN 33.20% (see [`Runs/`](../../Runs/README.md)).
