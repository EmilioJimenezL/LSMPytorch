# lab_cap5_ablation_legacy

**Goal:** the "legacy" baseline of the 20-epoch CNN ablation: v1 layout, static CPU augmentation, average pooling, i.e. the pre-fix pipeline on the same capped data. Compared against [`lab_cap5_ablation_fixed`](../lab_cap5_ablation_fixed/README.md). Part of the [lab_cap5 cycle](../lab_cap5/README.md). Weights: `Models/Pytorch/lab_cap5_ablation_legacy.pt`.

## Data

Cache `Workspace/prepared/lab_v1_cap5.npz` (`v1_135`, hash `7f7f0815c22fde2c`): 518 classes, train 2,072 / val 518 / extra_test 1,029. Same videos and split as the v2 cache (the split manifests are identical).

## Configuration

`model_config`: `input_dim 135`, `layout v1_135`, `pool avg`, `dropout 0.3`. Recipe: `--model cnn --epochs 20 --batch 64 --augment 10 --legacy-augment --warmup 0 --patience 0 --pool avg`. `--legacy-augment` expands the train set 10× once on the CPU (`dataset.apply_augmentation`) instead of augmenting each batch on the GPU. Flags beyond `metrics.json` are from the cycle script.

## Results (`metrics.json`)

| Split | n | Top-1 | Top-5 |
|---|---|---|---|
| Val | 518 | **24.71%** (128) | 43.82% (227) |
| Extra test | 1,029 | 15.06% (155) | 36.83% (379) |

20 epochs run, best val at epoch 18. Final train accuracy 99.6%, train loss 1.24 (memorized the train set by about epoch 10).

## Reading it

Best of the two 20-epoch ablations on val and extra, but the gap to the fixed path closes with training: the fixed CNN at early stopping ([`lab_cap5_cnn`](../lab_cap5_cnn/README.md)) gets 36.68% val / 26.63% extra. Static augmentation lets the model fit quickly; it does not generalize further.

## Files

`metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`. Local only: `best_model.pt`, `last.pt`, `checkpoint_epoch{10,20}.pt`.

## Reproduce

```bash
./lsm train --model cnn --cache Workspace/prepared/lab_v1_cap5.npz --name lab_cap5_ablation_legacy \
  --epochs 20 --batch 64 --augment 10 --legacy-augment --warmup 0 --patience 0 --pool avg
```
