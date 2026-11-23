# lab_cap5_ablation_fixed

**Goal:** the "fixed" side of the 20-epoch CNN ablation: v2 layout, GPU augmentation, attention pooling. Compared against [`lab_cap5_ablation_legacy`](../lab_cap5_ablation_legacy/README.md). Part of the [lab_cap5 cycle](../lab_cap5/README.md). Weights: `Models/Pytorch/lab_cap5_ablation_fixed.pt`.

## Data

Cache `Workspace/prepared/lab_v2_cap5.npz` (`v2_162`, hash `1849851f3acc5842`): 518 classes, train 2,072 / val 518 / extra_test 1,029, same split as every lab_cap5 run.

## Configuration

`model_config`: `input_dim 162`, `layout v2_162`, `pool attn`, `dropout 0.3`. Recipe: `--model cnn --epochs 20 --batch 64 --augment 10 --warmup 5 --patience 0 --pool attn --joint_dropout 0.1`. Fresh GPU augmentation every epoch; no early stopping. Warmup, patience and epoch count are from the cycle script, not `metrics.json`.

## Results (`metrics.json`)

| Split | n | Top-1 | Top-5 |
|---|---|---|---|
| Val | 518 | 17.18% (89) | 39.00% (202) |
| Extra test | 1,029 | 13.80% (142) | 31.88% (328) |

20 epochs run, best val at epoch 18. Final train accuracy 42.6% (augmented batches), train loss 3.15: still under-trained, not converged.

## Reading it

At a matched 20-epoch budget this path loses to the legacy path (24.71% val) because fresh augmentation and warmup make memorizing the small train set slower. It is not a final result: [`lab_cap5_cnn`](../lab_cap5_cnn/README.md) is the same configuration trained to early stopping (87 epochs) and reaches 36.68% val / 26.63% extra.

## Files

`metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`. Local only: `best_model.pt`, `last.pt`, `checkpoint_epoch{10,20}.pt`.

## Reproduce

```bash
./lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_ablation_fixed \
  --epochs 20 --batch 64 --augment 10 --warmup 5 --patience 0 --pool attn
```
