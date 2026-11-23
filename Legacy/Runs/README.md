# Legacy/Runs

Run folders from the cycles before cap-5/v2. Kept for provenance (which classes, which split, how the curves looked). They contain only `classes.json`, `split_manifest.json` and (for most) `training_curves.png`; `metrics.json` and `history.json` were not kept, and checkpoints (`*.pt`) are local and gitignored. Use the numbers below, taken from the earlier project README and the lab cap-5 report. Not maintained; new runs go to [`../../Runs`](../../Runs/README.md).

| Folder | Vocabulary / split | Layout | CNN val top-1 | TCN | 3D CNN |
|---|---|---|---|---|---|
| `m4_min12_{cnn,tcn,3dcnn}` | ≥12 videos: 28 classes, 373 videos (train 317 / val 56) | v1_135 | 48.21% | 35.71% | 3.57% |
| `m4_min7_{cnn,tcn,3dcnn}` | ≥7 videos: 54 classes, 568 videos (486 / 82), includes the ≥12 words | v1_135 | 52.44% | 39.02% | 15.85% |
| `m4_min5_{cnn,tcn,3dcnn}` | ≥5 videos: 286 classes, 1,864 videos (1,550 / 314), includes the ≥7 words | v1_135 | 47.45% | 47.45% | 13.69% |
| `lab_min5_{cnn,tcn,3dcnn}` | Lab dataset ≥5 valid videos, **uncapped**: 517 classes, 3,619 videos (train 3,026 / val 593) | v1_135 | 47.89% (150 ep) | 48.57% (120 ep) | ~18.4%, aborted at ~74/100 epochs |
| `m4_smoke` | 1-epoch CNN on `Colores`: 2 classes, 12 videos (10 / 2) | v1_135 | – | – | – |

The 307-class LSMOutput runs (CNN 47.16%, TCN 44.78%, 3D CNN 15.52%) are not stored here. All use stratified splits with augmentation applied after the split, no `extra_test`. Tier runs used `train_m4_tiers.sh` ([`../Scripts`](../Scripts/README.md)); the 3D CNN performed poorly in every cycle.

Compare with the capped, balanced v2 cycle: [`Runs/lab_cap5`](../../Runs/lab_cap5/README.md). Cap-5 numbers are lower (CNN 36.68% val) because each class has 4 training videos, and are not comparable with these.
