# TrainingPipeline

Python package that turns landmark JSON into trained PyTorch classifiers. It is the import root for the repo (`pytest.ini` sets `pythonpath = .`; other areas add it via a small `_paths.py`). Normally driven by `./lsm prepare` and `./lsm train` ([Orchestrator](../Orchestrator/README.md)); the scripts also run directly.

```
*_landmarks.json ──► dataset.py + features.py ──► prepare_dataset.py ──► Workspace/prepared/<name>.npz (+ .split.json)
                                                                      └─► train.py ──► Runs/<name>/ ──► Models/Pytorch/<name>.pt
```

## Files

| File / dir | Role |
|---|---|
| `train.py` | Training entry point (CNN / TCN / 3D CNN), metrics, curves |
| `prepare_dataset.py` | Parse JSON once → shared `.npz` cache + split, so every model sees the same data |
| `dataset.py` | Dataset discovery/loading (`load_raw_dataset`), class keys, legacy CPU augmentation |
| `features.py` | Feature layouts v1/v2, per-frame vectorizers, `prepare_sequence_*`, constants |
| `augment.py` | GPU batch augmentation (layout-aware) |
| `split_utils.py` | Stratified split, per-class cap + `extra_test`, `split_manifest.json` writer |
| `_repo.py` | `REPO_ROOT` / `TRAINING_PIPELINE` helpers |
| [`models/`](./models/README.md) | 1D CNN, TCN, 3D CNN + factory |
| [`configs/`](./configs/README.md) | Reference YAML hyperparameters (documentation, not loaded) |
| [`evaluation/`](./evaluation/README.md) | Eval, comparison, error analysis, report scripts |
| [`tests/`](./tests/README.md) | pytest suite |
| `requirements.txt` | torch, torchvision, numpy, scipy, matplotlib, coremltools, pyyaml, tensorboard, scikit-learn, pandas, rich, pytest |
| `pytest.ini` | `pythonpath = .`, `testpaths = tests` |

Python 3.11+. Install with `./lsm setup`, or `pip install -r TrainingPipeline/requirements.txt` in a venv (on macOS the default torch wheel uses MPS; on Linux/Windows pick a CUDA or CPU wheel).

## Feature layouts

Every sequence is resampled to **85 frames** (`N_FRAMES`). A layout fixes the per-frame vector. The layout is stored in the cache settings, the checkpoint (`model_config`) and the Core ML metadata (`feature_layout`; missing means `v1_135`).

| Layout | Dim | Vector | Missing joint | Preprocessing | Used by |
|---|---|---|---|---|---|
| `v1_135` | 135 | L hand 21×(x,y) = 42, R hand 42, pose 17×(x,y,visibility) = 51 (MediaPipe-style slots) | x=y=0 | Shoulder-midpoint centering, valid-frame filter, linear interpolation; pose visibility binarized at 0.5 | Fase 1, `lab_min5` and older models, legacy ablation |
| `v2_162` | 162 | L hand 21×(x,y,c) = 63, R hand 63, pose 12×(x,y,c) = 36 | x=y=c=0 | Shoulder-midpoint centering **and** scaling by median shoulder width, per-joint interpolation (gaps are not blended), confidence ≥0.5 set to 1 | All `lab_cap5*` runs; new models only |

Details:

- **Valid frame:** v1 needs a visible shoulder and (a hand or ≥3 pose points); v2 needs a shoulder with confidence ≥0.1. A video needs ≥10 valid frames or it is dropped.
- **v2 pose ids:** 0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear, 5 neck, 6 L shoulder, 7 R shoulder, 8 L elbow, 9 R elbow, 10 L wrist, 11 R wrist.
- v1 can be built from v2 JSON (names are mapped into the 17 slots). Both JSON schemas load; see [`LandmarkExtraction`](../LandmarkExtraction/README.md).
- The v1 layout is the contract with the Swift app's `SequencePreprocessor` and Fase 1; v2 matches `SequencePreprocessorV2`. Keep Python and Swift in sync.

## Dataset and class rules

- Layout on disk: nested `Category/Word/*_landmarks.json` or flat `Word/*_landmarks.json`. Top-level folders starting with `_` or `.` (for example `_invalidos`) are ignored.
- Class key = word folder. A word that appears in 2+ categories becomes `Word (Category)` (e.g. `Rosa (Colores)`).
- `--min-videos N` counts **valid** sequences after preprocessing, not files. Excluded classes are listed at load time.
- `classes.json` order defines output indices; the iOS app and Core ML metadata use the same list.

## Split (`split_utils.py`)

Always done **before** augmentation, seeded (default 42), per class:

1. If `--max-videos-per-class K` (cap): shuffle, keep K, the remainder becomes **`extra_test`** (held out, never trained on).
2. Of the kept videos: 2 → 1 train / 1 val; ≥3 → `max(1, round(n × val_split))` val (default 0.15), the rest train. With cap 5 this is 4 train / 1 val.

Without a cap there is no `extra_test`. The split is written as `split_manifest.json` (per-file `train|val|extra_test`). The lab cap-5 cache has 3,619 videos / 518 classes: train 2,072, val 518, extra_test 1,029.

## prepare_dataset.py

```bash
python TrainingPipeline/prepare_dataset.py --dataset <landmarks dir> \
  --output Workspace/prepared/lab_v2_cap5.npz --layout v2_162 --min-videos 5 --max-videos-per-class 5
```

| Flag | Default |
|---|---|
| `--dataset` (required), `--output` (required, `.npz`) | – |
| `--layout` | `v2_162` (`v1_135`) |
| `--min-videos` | 5 |
| `--max-videos-per-class` | 5 |
| `--val-split` | 0.15 |
| `--seed` | 42 |
| `--no-shoulder-norm` | off (v1 legacy only) |

Writes `<name>.npz` with `X (N,85,D)`, `y`, `classes`, `paths`, `categories`, `class_names`, `idx_train`, `idx_val`, `idx_extra`, and `settings` (JSON incl. layout, preprocess version, counts, and a 16-hex `hash`), plus `<name>.split.json` next to it.

## train.py

```bash
python TrainingPipeline/train.py --model tcn --cache Workspace/prepared/lab_v2_cap5.npz \
  --layout v2_162 --pool attn --causal 0 --epochs 120 --batch 16 --output Runs/lab_cap5_tcn
```

Data comes from `--cache` (preferred; layout is read from the cache) or `--dataset` (parses JSON on the fly; applies `--min_videos`, `--max_videos_per_class`, `--val_split`, seed 42).

| Flag | Default | Notes |
|---|---|---|
| `--model` | `cnn` | `cnn`, `tcn`, `3dcnn` |
| `--dataset` / `--cache` | – | one required |
| `--output` | `./runs` | The orchestrator passes `Runs/<name>` |
| `--layout` | `v1_135` | `v1_135` / `v2_162`; overridden by the cache |
| `--epochs` | 100 | |
| `--batch` | 64 | |
| `--lr` | 1e-3 | AdamW, weight decay 1e-4 |
| `--dropout` | 0.3 | |
| `--pool` | `avg` | `avg` or `attn` (CNN, TCN) |
| `--causal` | 1 | TCN only; `0` = symmetric padding |
| `--augment` | 10 | Augmented copies per sample (GPU, per batch) or static factor with `--legacy-augment` |
| `--joint_dropout` | 0.1 | Zero a whole joint across time with this probability |
| `--legacy-augment` | off | Static 10× CPU augmentation from the old pipeline (v1 ablation baseline) |
| `--warmup` | 10 | Linear warmup epochs, then cosine to 1e-6 |
| `--patience` | 15 | Early stopping on val top-1; `0` disables |
| `--val_split`, `--min_videos`, `--max_videos_per_class` | 0.15, 2, none | `--dataset` mode only |
| `--no-shoulder-norm` | off | `--dataset` v1 only |
| `--resume` | – | Continue from a `last.pt` (restores epoch, optimizer, scheduler, history) |
| `--workers` | auto | 4 on CUDA, else 0 |

Recipe: cross-entropy with label smoothing 0.1, gradient clipping at 1.0, device auto-selected (CUDA, then MPS, then CPU). **GPU augmentation** (`augment.py`) multiplies each batch by `1 + augment` with fresh randomness each epoch: per-axis scale 0.9–1.1 (p 0.8), translation ±0.05 (p 0.8), time warp 0.8–1.2× (p 0.7), Gaussian noise σ 0.01 on x/y (p 0.6), joint dropout. Confidence/visibility channels and missing joints (zeros) are never altered. The CLI flags use mixed styles (`--val_split` vs `--val-split`); the orchestrator exposes kebab-case.

### Outputs (in `--output`)

| File | Content |
|---|---|
| `best_model.pt` | Best val top-1 checkpoint |
| `last.pt` | Latest epoch (for `--resume`); gitignored |
| `checkpoint_epoch{N}.pt` | Every 10 epochs; gitignored |
| `classes.json` | Ordered class names |
| `split_manifest.json` | Per-file split assignment |
| `history.json` | `train_loss`, `train_acc`, `val_loss`, `val_acc` per epoch |
| `metrics.json` | model, layout, `model_config`, `n_classes`, `best_val_acc`, `stopped_early`, `epochs_run`, `val {top1, top5, n}`, `extra_test {n, top1, top5, ...}` |
| `training_curves.png` | Loss and accuracy curves |

Checkpoint payload: `epoch, model, optimizer, scheduler, best_val_acc, history, classes, n_frames, feature_dim, model_type, model_config`. `model_config` holds `input_dim, layout, causal, pool, dropout`, which `CoreMLConversion` uses to rebuild the network.

## Reproducing the lab cap-5 cycle

See [`Runs/lab_cap5/REPORT.md`](../Runs/lab_cap5/REPORT.md) and `Runs/lab_cap5/reproduce.sh`.
