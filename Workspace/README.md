# Workspace

Local, regenerable working files. **Everything here except this README is gitignored** (`Workspace/prepared/`, `Workspace/exports/`, `Workspace/logs/`, `Workspace/scratch/`). Safe to delete; the pipeline recreates it.

| Folder | Content | Written by |
|---|---|---|
| `prepared/` | Prepared dataset caches `<name>.npz` and split manifests `<name>.split.json` (parsed landmarks, `X (N,85,D)`, classes, train/val/extra indices, settings hash) | `lsm prepare` / `TrainingPipeline/prepare_dataset.py` |
| `exports/` | Scratch Core ML exports and fixtures before they are promoted to `Models/CoreML` | `convert_to_coreml.py` when run by hand |
| `logs/` | One log per `lsm` command: `<timestamp>_<command>.log` | `lsm` |
| `scratch/` | Anything temporary (experiments, test output, reports) | you |

## Current prepared caches

| Cache | Layout | Classes | Videos | Train / Val / Extra | Hash |
|---|---|---|---|---|---|
| `lab_v2_cap5.npz` | v2_162 | 518 | 3,619 | 2,072 / 518 / 1,029 | `1849851f3acc5842` |
| `lab_v1_cap5.npz` | v1_135 | 518 | 3,619 | 2,072 / 518 / 1,029 | `7f7f0815c22fde2c` |
| `colores_smoke.npz` | v2_162 | 3 | 19 | 12 / 3 / 4 | `5a3af5ad2e958a7c` |

The cache `settings` (layout, `min_videos`, cap, seed, preprocess version) must match what a model is trained with: `train.py` takes its layout from the cache. If landmarks or preprocessing change, re-run `lsm prepare` (the hash changes). A pre-reorganization copy of these files also exists at the repo root (`prepared/`, `exports/`), which is gitignored too.
