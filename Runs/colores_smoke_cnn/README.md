# colores_smoke_cnn

**Goal:** smoke test, not a model. Verify the v2 pipeline end to end in seconds (prepare cache → train CNN → metrics/curves, and optionally convert + parity). It is the model for the 2-epoch smoke check of `lsm pipeline`. The accuracy numbers are meaningless.

## Data

Cache `Workspace/prepared/colores_smoke.npz` (hash `5a3af5ad2e958a7c`), built from the small v2 sample extraction (`LandmarkExtraction/LSMExtractorCLI/build/sample_v2_out`, not in git): `v2_162`, `min_videos 2`, cap 5, seed 42.

| Item | Value |
|---|---|
| Videos | 19 |
| Classes | 3 (`A`, `Abuela`, `Amarillo`; the sample is a few classes from Abecedario/Familia/Colores, despite the name) |
| Split | train 12 / val 3 / extra_test 4 |

## Configuration

`model_config`: `input_dim 162`, `layout v2_162`, `pool attn`, `dropout 0.3`; 2 epochs.

## Results (`metrics.json`)

| Split | n | Top-1 | Top-5 |
|---|---|---|---|
| Val | 3 | 33.33% (1/3) | 100% |
| Extra test | 4 | 25.00% (1/4) | 100% |

2 epochs run, best at epoch 1; final train accuracy 58.3%, train loss 0.88. Top-5 is 100% only because there are 3 classes. Pass criterion: the run completes and writes `metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`.

## Files

`metrics.json`, `history.json`, `classes.json`, `split_manifest.json`, `training_curves.png`. Local only: `best_model.pt`, `last.pt`. No weights are kept in `Models/`.

## Reproduce

```bash
./lsm prepare --landmarks <sample landmarks dir> --layout v2_162 --min-videos 2 --cap 5 --name colores_smoke
./lsm train --model cnn --cache Workspace/prepared/colores_smoke.npz --name colores_smoke_cnn --epochs 2 --pool attn
```

Sibling: [`colores_smoke_tcn`](../colores_smoke_tcn/README.md).
