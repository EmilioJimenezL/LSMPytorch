# lab_cap5_cnn

**Goal:** best 1D CNN on the lab vocabulary with the fixed v2 pipeline (balanced cap-5 data, GPU augmentation, attention pooling, warmup, early stopping). Part of the [lab_cap5 cycle](../lab_cap5/README.md); weights in `Models/Pytorch/lab_cap5_cnn.pt`, Core ML in `Models/CoreML/lab_cap5_cnn.mlpackage`.

## Data

| Item | Value |
|---|---|
| Cache | `Workspace/prepared/lab_v2_cap5.npz` (hash `1849851f3acc5842`) |
| Layout | `v2_162`, 85 frames |
| Classes | 518 (≥5 valid videos each; `Rosa` split by category) |
| Split (seed 42) | train 2,072 / val 518 / **extra_test 1,029** (videos beyond the cap of 5 per class; never trained on) |

## Configuration

`model_config` from `metrics.json`: `input_dim 162`, `layout v2_162`, `pool attn`, `dropout 0.3` (`causal` is recorded but unused by the CNN).

Training recipe: `--model cnn --epochs 150 --batch 64 --augment 10 --warmup 10 --patience 15 --pool attn --joint_dropout 0.1`, AdamW lr 1e-3, label smoothing 0.1, Apple MPS. Epoch limit, batch and warmup come from the cycle script; they are not stored in `metrics.json`.

## Results (`metrics.json`)

| Split | n | Top-1 | Top-5 |
|---|---|---|---|
| Val | 518 | **36.68%** (190) | 57.72% (299) |
| Extra test | 1,029 | **26.63%** (274) | 52.96% (545) |

- 87 epochs run, early stopping fired; best val epoch 72 (patience 15).
- Final-epoch train accuracy 98.3% (measured on augmented batches), train loss 1.29 (label smoothing 0.1 keeps the floor above 0). Val is far lower: the model overfits 4 training videos per class.
- Best model across the cycle: beats the 20-epoch legacy ablation (24.71% val / 15.06% extra) and the TCN.
- Core ML parity gate: max |Δlogit| 0.017, top-1 agreement 100% (256 val sequences), PASS.

## Files

`metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`. Local only (gitignored): `best_model.pt`, `last.pt`, `checkpoint_epoch{10..80}.pt`.

## Reproduce

```bash
./lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_cnn \
  --epochs 150 --batch 64 --augment 10 --warmup 10 --patience 15 --pool attn
```

See [`../lab_cap5/reproduce.sh`](../lab_cap5/reproduce.sh).

## Caveats

Random split: the same signer can appear in train and val, so accuracy for a new signer is probably lower. Val was used for checkpoint selection and early stopping; `extra_test` was not.
