# Runs

One folder per training run (one `train.py` invocation, or a resumed series of them). Runs are the record of *how* a model was trained and *how well* it did. The deployable weights are copied out to [`../Models/`](../Models/README.md). Older cycles live in [`../Legacy/Runs`](../Legacy/Runs/README.md).

## Folder convention

Name: `<dataset>_<cap or min>_<model>[_variant]`, e.g. `lab_cap5_cnn`, `lab_cap5_ablation_fixed`, `colores_smoke_tcn`. The same stem is used for `Models/Pytorch/<name>.pt` and `Models/CoreML/<name>.mlpackage`.

Each run folder contains:

| File | Tracked in git | Content |
|---|---|---|
| `README.md` | yes | Goal, data, configuration, results, how to reproduce, caveats. `lsm train` writes a skeleton from `metrics.json`; fill in the rest |
| `metrics.json` | yes | model, layout, `model_config`, `n_classes`, best val top-1, `stopped_early`, `epochs_run`, `val` and `extra_test` top-1/top-5 |
| `history.json` | yes | Per-epoch `train_loss`, `train_acc`, `val_loss`, `val_acc` |
| `classes.json` | yes | Ordered class names (index = model output) |
| `split_manifest.json` | yes | Which file went to `train`, `val` or `extra_test` |
| `training_curves.png` | yes | Loss and accuracy curves |
| `best_model.pt` | no (copied to `Models/Pytorch/<name>.pt`, which uses Git LFS) | Best val top-1 checkpoint |
| `last.pt`, `checkpoint_epoch{N}.pt` | no (gitignored) | Resume points and periodic snapshots |

Write a README for every new run. Report: val top-1/top-5 and, if the run used a cap, `extra_test` top-1/top-5; epochs run and whether early stopping fired; layout, pool, augmentation and any non-default flag; the exact command; the prepared cache (name and `settings.hash`). Reproduce the numbers from `metrics.json`, not from memory.

## Cycle folders

A folder without weights, such as [`lab_cap5/`](./lab_cap5/README.md), summarizes a whole experiment cycle: `REPORT.md` and a `reproduce.sh` that lists the commands.

## Index

All runs below use the lab dataset (3,774 videos extracted, ≥5 valid videos per class → 518 classes) with a cap of 5 videos per class (train 2,072 / val 518 / extra_test 1,029), except the smoke runs.

| Run | Model | Layout | Epochs | Val top-1 | Val top-5 | Extra top-1 | Extra top-5 |
|---|---|---|---|---|---|---|---|
| [`lab_cap5_cnn`](./lab_cap5_cnn/README.md) | CNN, attn pool | v2_162 | 87 (early stop) | **36.68%** | 57.72% | **26.63%** | 52.96% |
| [`lab_cap5_tcn`](./lab_cap5_tcn/README.md) | TCN non-causal, attn pool | v2_162 | 114 (early stop) | 33.20% | 53.09% | 23.81% | 46.45% |
| [`lab_cap5_ablation_fixed`](./lab_cap5_ablation_fixed/README.md) | CNN, attn pool, GPU aug | v2_162 | 20 | 17.18% | 39.00% | 13.80% | 31.88% |
| [`lab_cap5_ablation_legacy`](./lab_cap5_ablation_legacy/README.md) | CNN, avg pool, static aug | v1_135 | 20 | 24.71% | 43.82% | 15.06% | 36.83% |
| [`colores_smoke_cnn`](./colores_smoke_cnn/README.md) | CNN, attn pool | v2_162 | 2 | 33.33% | 100% | 25.00% | 100% |
| [`colores_smoke_tcn`](./colores_smoke_tcn/README.md) | TCN non-causal, attn pool | v2_162 | 2 | 33.33% | 100% | 25.00% | 100% |

Cycle summary: [`lab_cap5/REPORT.md`](./lab_cap5/REPORT.md).
