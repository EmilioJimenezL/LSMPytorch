# configs

Reference hyperparameter files, one per architecture. They are **documentation only**: `train.py` does not read YAML (a loader is not wired in). Real settings come from CLI flags / `lsm train`. The files predate the v2 layout and the cap-5 cycle.

| File | `model.type` | epochs | batch | Notes |
|---|---|---|---|---|
| `cnn1d_config.yaml` | `1d_cnn` | 150 | 64 | |
| `tcn_config.yaml` | `tcn` | 120 | 16 | `num_channels [64,128,256,256]`, `kernel_size 5`, `num_layers 4` |
| `cnn3d_config.yaml` | `3d_cnn` | 100 | 12 | |

Shared values in all three: `num_classes: 307`, dropout 0.3, AdamW (lr 1e-3, weight decay 1e-4, betas 0.9/0.999), 10 warmup epochs, cosine annealing to 1e-6, cross-entropy with label smoothing 0.1, augmentation factor 10 (scale 0.9–1.1, translation 0.05, time warp 0.8–1.2, noise σ 0.01, joint dropout 0.1), val split 0.15, `min_videos_per_class: 2`, patience 15, checkpoint every 10 epochs.

How they differ from what actually ran:

- `num_classes` is 307 here; the current lab runs have 518 and the value is derived from the data.
- No `layout`, `pool`, `causal`, `max_videos_per_class`, or `legacy_augment` keys. The cap-5 runs used `--layout v2_162 --pool attn --augment 10 --joint_dropout 0.1` (TCN also `--causal 0`).
- `hardware.mixed_precision: true` (AMP) is not implemented in `train.py`; the device is auto-selected (CUDA, MPS, CPU).
- `train.py` does implement warmup, early stopping, and joint dropout (as flags), even though older docs listed them as unimplemented.

For the exact settings of a run, read its `Runs/<name>/metrics.json` (`model_config`) and README.
