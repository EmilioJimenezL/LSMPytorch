# colores_smoke_tcn

**Goal:** smoke test, not a model. Same purpose as [`colores_smoke_cnn`](../colores_smoke_cnn/README.md) for the TCN path (non-causal, attention pool): confirm prepare → train → artifacts works in seconds. The accuracy numbers are meaningless.

## Data

Cache `Workspace/prepared/colores_smoke.npz` (hash `5a3af5ad2e958a7c`): `v2_162`, 19 videos, 3 classes (`A`, `Abuela`, `Amarillo`), `min_videos 2`, cap 5, seed 42. Split train 12 / val 3 / extra_test 4. Built from the small local sample extraction (`LandmarkExtraction/LSMExtractorCLI/build/sample_v2_out`, not in git).

## Configuration

`model_config`: `input_dim 162`, `layout v2_162`, `causal false`, `pool attn`, `dropout 0.3`; 2 epochs.

## Results (`metrics.json`)

| Split | n | Top-1 | Top-5 |
|---|---|---|---|
| Val | 3 | 33.33% (1/3) | 100% |
| Extra test | 4 | 25.00% (1/4) | 100% |

2 epochs run, best at epoch 1; final train accuracy 66.7%, train loss 0.89. Top-5 is 100% because there are only 3 classes.

## Files

`metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`. Local only: `best_model.pt`, `last.pt`. No weights are kept in `Models/`.

## Reproduce

```bash
./lsm train --model tcn --cache Workspace/prepared/colores_smoke.npz --name colores_smoke_tcn \
  --epochs 2 --pool attn --causal 0
```
