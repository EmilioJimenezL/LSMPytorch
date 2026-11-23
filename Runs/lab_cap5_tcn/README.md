# lab_cap5_tcn

**Goal:** TCN counterpart of [`lab_cap5_cnn`](../lab_cap5_cnn/README.md) on the same data and split, using the fixed pipeline with a **non-causal** TCN and attention pooling. Part of the [lab_cap5 cycle](../lab_cap5/README.md); weights in `Models/Pytorch/lab_cap5_tcn.pt`, Core ML in `Models/CoreML/lab_cap5_tcn.mlpackage`.

## Data

Identical to `lab_cap5_cnn`: cache `Workspace/prepared/lab_v2_cap5.npz` (hash `1849851f3acc5842`), `v2_162`, 518 classes, train 2,072 / val 518 / extra_test 1,029 (seed 42, same `split_manifest.json`).

## Configuration

`model_config` from `metrics.json`: `input_dim 162`, `layout v2_162`, `causal false`, `pool attn`, `dropout 0.3`. Architecture: 4 residual blocks, channels 64/128/256/256, kernel 5, dilations 1/2/4/8, receptive field 121 frames, ~1.53M parameters (see [`TrainingPipeline/models`](../../TrainingPipeline/models/README.md)).

Training recipe: `--model tcn --epochs 120 --batch 16 --augment 10 --warmup 10 --patience 15 --pool attn --causal 0 --joint_dropout 0.1`, AdamW lr 1e-3, label smoothing 0.1, Apple MPS. Epoch limit, batch and warmup come from the cycle script; they are not stored in `metrics.json`.

## Results (`metrics.json`)

| Split | n | Top-1 | Top-5 |
|---|---|---|---|
| Val | 518 | 33.20% (172) | 53.09% (275) |
| Extra test | 1,029 | 23.81% (245) | 46.45% (478) |

- 114 epochs run, early stopping fired; best val epoch 99 (patience 15).
- Final-epoch train accuracy 99.5% (augmented batches), train loss 1.20.
- 3.5 points below the CNN on val and 2.8 on extra_test. The previous uncapped cycle had TCN slightly ahead of CNN (48.57% vs 47.89%); with only 4 videos per class the larger TCN overfits more.
- Core ML parity gate: max |Δlogit| 0.018, top-1 agreement 100% (256 val sequences), PASS.

## Files

`metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`. Local only (gitignored): `best_model.pt`, `last.pt`, `checkpoint_epoch{10..110}.pt`.

## Reproduce

```bash
./lsm train --model tcn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_tcn \
  --epochs 120 --batch 16 --augment 10 --warmup 10 --patience 15 --pool attn --causal 0
```

See [`../lab_cap5/reproduce.sh`](../lab_cap5/reproduce.sh).

## Caveats

Same as the CNN run: random split (possible signer overlap), val used for early stopping, `extra_test` held out.
