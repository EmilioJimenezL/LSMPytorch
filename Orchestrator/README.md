# Orchestrator: the `lsm` command

One cross-platform CLI that drives the whole pipeline by calling the existing scripts (it never re-implements them). Implemented in `Orchestrator/lsm_cli/` (stdlib + optional `rich`), launched through thin wrappers in the repo root:

| Platform | Launcher |
|---|---|
| macOS, Linux | `./lsm` (bash) |
| Windows | `lsm.ps1` (PowerShell) or `lsm.cmd` |

The wrappers use `.venv`'s Python when it exists, otherwise the system Python (enough for `setup` and `doctor`).

> **Status:** this README is the command specification. `Orchestrator/lsm_cli/` is still being written, so a command may not exist yet. The underlying scripts it wraps are documented in their own READMEs and run today. Flags marked **†** are small additions the wrapped scripts need (not in the original sketch). Defaults not fixed by an underlying script are marked *planned*.

```
videos ─extract (macOS)─► landmarks JSON ─prepare─► Workspace/prepared/<name>.npz
   ─train─► Runs/<name>/ + Models/Pytorch/<name>.pt ─convert─► Models/CoreML/<name>.mlpackage ─deploy─► TestingTools app
                         └─evaluate / report
```

## Quick start

```bash
./lsm setup                                   # venv + dependencies for this platform
./lsm doctor                                  # check device, Core ML, Xcode, git-lfs
./lsm pipeline --videos ~/LSMVideos --name lab_cap5     # everything (macOS)
./lsm pipeline --landmarks ~/landmarks --models cnn,tcn --no-deploy   # Linux/Windows: training + conversion inputs
```

Plain `./lsm` prints the command list and a "start here" hint. `./lsm <command> --help` shows purpose, grouped flags with defaults, examples, and outputs (the same text as below).

## Global flags (every command)

| Flag | Default | Meaning |
|---|---|---|
| `-v, --verbose` | off | Stream full output of the underlying scripts; show Python tracebacks |
| `-q, --quiet` | off | Show only stage results |
| `--dry-run` | off | Print the resolved plan and the exact underlying commands, then stop |
| `--yes` | off | Do not ask for confirmation (overwrite a run, re-extract). Non-interactive runs, such as CI, never prompt |
| `--workspace DIR` | `Workspace` | Root for prepared caches, logs, scratch |
| `-h, --help` | | Command help |

Behavior shared by all commands:

- **Plan first:** each command prints resolved inputs, output paths, device (cuda, mps, cpu), layout, and estimated work (videos found, classes, epochs) before running. `--dry-run` stops there.
- **Guided input:** a missing required argument is prompted for on an interactive terminal (and validated immediately); in non-interactive use it fails with a clear message.
- **Progress:** numbered stage banners (`[2/5] Preparing dataset`), progress bars with ETA for extraction (videos) and training (epochs, with loss/accuracy, best val, patience counter), elapsed time per stage. Colors and bars use `rich` when installed, with a plain-text fallback; color is off when output is not a terminal or `NO_COLOR` is set.
- **Actionable errors:** known failures are explained with a fix, for example non-macOS extraction (points to the Linux/Windows docs), missing `.venv` (`run lsm setup`), cache/model layout mismatch, no CUDA, missing coremltools, parity gate failure (prints the diff numbers), missing Xcode or git-lfs.
- **Final summary:** what was produced (paths), key metrics (val and extra_test top-1), and the suggested next command.
- **Exit codes:** `0` success, `1` user or input error, `2` a step failed.
- **Logs:** every run writes `Workspace/logs/<timestamp>_<command>.log`.

## Commands

### `lsm setup`

Create `.venv` and install dependencies. Works before the venv exists.

| Flag | Default | Meaning |
|---|---|---|
| `--cpu` | | Force the CPU torch build |
| `--cuda VERSION` | | Install the CUDA torch build, e.g. `--cuda 12.4` |

Default (neither flag): macOS installs the standard wheel (Apple MPS); Linux/Windows use CUDA if an NVIDIA GPU is detected, otherwise CPU *(planned)*. Installs `TrainingPipeline/requirements.txt` (torch, torchvision, numpy, scipy, matplotlib, coremltools, pyyaml, tensorboard, scikit-learn, pandas, rich, pytest). Requires Python 3.11+.
**Output:** `.venv/`. **Next:** `lsm doctor`.

```bash
./lsm setup               # macOS
./lsm setup --cpu         # CI / no GPU
./lsm setup --cuda 12.4   # NVIDIA, Linux or Windows
```

### `lsm doctor`

Read-only environment report: Python version, torch version, selected device (cuda, mps, cpu), whether `coremltools` can convert and whether it can `predict` (macOS only), whether Xcode (`xcodebuild`) and `git-lfs` are available. No flags besides the global ones. Exit `0` if the environment can run the requested stages, `1` if a hard requirement (Python, torch) is missing.

```bash
./lsm doctor
```

### `lsm extract` (macOS only)

Extract landmarks from videos with `LSMExtractorCLI` (builds it with `xcodebuild` if the binary is missing). Details of formats: [`LandmarkExtraction`](../LandmarkExtraction/README.md).

| Flag | Default | Meaning |
|---|---|---|
| `--videos DIR` | required | Root with `category/word/*.{mp4,mov,m4v}` |
| `--output DIR` | `<videos>_landmarks` next to the videos *(planned)* | Landmark output root |
| `--overlay` | off | Also write `*_overlay.mp4` skeleton videos |
| `--skip-existing` | off | Skip videos that already have `*_landmarks.json` |
| `--manifest FILE` | `<output>/dataset_manifest.json` | Existing manifest to update |

**Output:** `<output>/{category}/{word}/*_landmarks.json` (schema v2), `dataset_manifest.json`, `_batch_report.json`. On Linux/Windows the command stops with a pointer to get landmarks from a Mac.

```bash
./lsm extract --videos ~/LSMVideos --output ~/LSMVideosOutput --skip-existing
```

### `lsm prepare`

Parse landmark JSON once into a shared cache and split (wraps `TrainingPipeline/prepare_dataset.py`), so every model trains and is compared on identical data.

| Flag | Default | Meaning |
|---|---|---|
| `--landmarks DIR` | required | Landmark dataset root (nested or flat) |
| `--layout` | `v2_162` | `v2_162` or `v1_135` |
| `--min-videos N` | 5 | Minimum **valid** sequences per class; others are excluded |
| `--cap N` | 5 | Max videos per class kept for train/val; the rest become `extra_test` |
| `--val-split F` | 0.15 | Validation fraction (per class, minimum 1) |
| `--seed N` | 42 | Split seed |
| `--name NAME` | derived from folder, layout and cap *(planned)* | Cache name |

**Output:** `Workspace/prepared/<name>.npz` and `<name>.split.json`; prints class count, train/val/extra counts, settings hash. **Next:** `lsm train --cache Workspace/prepared/<name>.npz`.

```bash
./lsm prepare --landmarks ~/landmarks --layout v2_162 --min-videos 5 --cap 5 --name lab_v2_cap5
```

### `lsm train`

Train one model (wraps `TrainingPipeline/train.py`). Details of the recipe: [`TrainingPipeline`](../TrainingPipeline/README.md), architectures: [`models`](../TrainingPipeline/models/README.md).

| Flag | Default | Meaning |
|---|---|---|
| `--model` | required | `cnn`, `tcn`, `3dcnn` |
| `--cache FILE` | one of the two | Prepared `.npz` (preferred; layout comes from it) |
| `--landmarks DIR` | one of the two | Parse landmarks directly (`train.py --dataset`); split computed on the fly. Honors `--layout`, `--min-videos`, `--cap`, `--val-split` † |
| `--name NAME` | derived from model and input *(planned)* | Run name: `Runs/<name>/`, `Models/Pytorch/<name>.pt` |
| `--epochs N` | 100 | Maximum epochs |
| `--batch N` | 64 | Batch size |
| `--lr F` | 1e-3 | AdamW learning rate |
| `--pool` | `avg` | `avg` or `attn` (CNN, TCN) |
| `--causal 0\|1` | 1 | TCN only; `0` = symmetric padding |
| `--warmup N` | 10 | Linear warmup epochs, then cosine decay |
| `--patience N` | 15 | Early stopping on val top-1; `0` disables |
| `--resume` | off | Continue from `Runs/<name>/last.pt` |
| `--legacy-augment` | off | Static CPU 10× augmentation (ablation baseline, v1) |
| `-- <args>` † | | Anything after `--` goes to `train.py` (e.g. `--augment 5 --joint_dropout 0.2`) |

Defaults are `train.py`'s. Recipes used for the lab cap-5 runs: CNN `--epochs 150 --batch 64 --pool attn`, TCN `--epochs 120 --batch 16 --pool attn --causal 0`.
If `Runs/<name>/` exists and `--resume` is not given, you are asked before overwriting (`--yes` skips the prompt).

**Output:** `Runs/<name>/` with `best_model.pt`, `last.pt`, `checkpoint_epoch{N}.pt` (every 10 epochs), `classes.json`, `split_manifest.json`, `history.json`, `metrics.json`, `training_curves.png`, and a generated `README.md` skeleton filled from `metrics.json` (add goal and notes by hand). Then copies `best_model.pt` to `Models/Pytorch/<name>.pt`. Summary shows val and extra_test top-1/top-5. **Next:** `lsm convert --run <name>`.

```bash
./lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_cnn --epochs 150 --pool attn
./lsm train --model tcn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_tcn --epochs 120 --batch 16 --pool attn --causal 0
./lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_cnn --resume
```

### `lsm convert`

Convert a trained model to Core ML and run the **parity gate** (wraps `CoreMLConversion/convert_to_coreml.py`). Details: [`CoreMLConversion`](../CoreMLConversion/README.md).

| Flag | Default | Meaning |
|---|---|---|
| `--run NAME` | one of the two | Use `Runs/<name>/best_model.pt` (output name = run name) |
| `--checkpoint FILE` | one of the two | Convert any checkpoint (needs `--name` for the output name †) |
| `--cache FILE` † | the cache the run was trained on | Validation tensors for parity and the fixture |
| `--no-parity` | off | Skip the parity gate |
| `--fixture FILE` | `Models/CoreML/<name>_fixture.json` | Where to write the golden fixture |

Parity: first up to 256 validation sequences, **max |Δlogit| ≤ 0.05 and top-1 agreement ≥ 99%** or the command exits `2` and prints the numbers. On platforms without Core ML `predict` (Linux, Windows) parity is skipped automatically with a warning; run it on a Mac before deploying.
**Output:** `Models/CoreML/<name>.mlpackage`, `<name>_classes.json` (copy of the run's `classes.json`), `<name>_fixture.json`. **Next:** `lsm deploy --run <name>`.

```bash
./lsm convert --run lab_cap5_cnn
./lsm convert --run lab_cap5_tcn --no-parity      # Linux/Windows (WSL2 for coremltools on Windows)
```

### `lsm deploy`

Copy a converted model into the iOS test app (wraps `CoreMLConversion/export_to_app.py`).

| Flag | Default | Meaning |
|---|---|---|
| `--run NAME` | required | Model from `Models/CoreML/<name>*` |
| `--sign-set` | `labCap5` | App `SignSet` case |
| `--display-name TEXT` | run name | Name shown in the app picker |
| `--family` † | from the run's model (`cnn`, `tcn`, `cnn3d`) | App model family |

**Output:** into `TestingTools/LSMMobileModelTesting/LSMMobileModelTesting/`: the `.mlpackage`, `<name>_classes.json`, `<name>_phase2_fixture.json`, and a printed Swift `LSMModel(...)` entry that you paste into `AVAILABLE_MODELS` (the command does not edit Swift code). **Next:** build the app in Xcode and check the parity lines. See [`TestingTools`](../TestingTools/README.md).

```bash
./lsm deploy --run lab_cap5_cnn --sign-set labCap5 --display-name "CNN — Lab cap5 (v2)"
```

### `lsm evaluate`

Evaluate a run (wraps `TrainingPipeline/evaluation/`).

| Flag | Default | Meaning |
|---|---|---|
| `--run NAME` | required | Run to evaluate |
| `--dataset DIR` † | | Also run full-dataset inference with `evaluate_nn.py` (v1 checkpoints only today) |

Always prints the run's recorded val and `extra_test` top-1/top-5 from `metrics.json`. **Output:** `Workspace/scratch/evaluation/<name>.json` *(planned)*. See [`evaluation`](../TrainingPipeline/evaluation/README.md) for what each script does and its limits.

```bash
./lsm evaluate --run lab_cap5_cnn
```

### `lsm report`

Compare runs: metrics table and training curves.

| Flag | Default | Meaning |
|---|---|---|
| `--runs A B ...` | every folder in `Runs/` with a `metrics.json` | Runs to include |

**Output:** table on stdout (model, layout, epochs, val and extra_test top-1/top-5), plus `Workspace/scratch/report_<timestamp>.md` and a combined val-accuracy curves image *(planned)*.

```bash
./lsm report --runs lab_cap5_cnn lab_cap5_tcn lab_cap5_ablation_fixed lab_cap5_ablation_legacy
```

### `lsm test`

Run the pytest suite (`TrainingPipeline/tests`, see [`tests`](../TrainingPipeline/tests/README.md)).

| Flag | Default | Meaning |
|---|---|---|
| `--quick` | off | Skip the slow and platform-dependent tests (resume subprocess run, Core ML round-trip). Used in CI |

```bash
./lsm test
./lsm test --quick
```

### `lsm pipeline`

End to end in one command: extract → prepare → train → convert → deploy, with numbered stages.

| Flag | Default | Meaning |
|---|---|---|
| `--videos DIR` | one of the two | Start from videos (extraction is macOS only) |
| `--landmarks DIR` | one of the two | Start from existing landmarks (skips extraction) |
| `--models LIST` | `cnn,tcn` | Comma-separated: `cnn`, `tcn`, `3dcnn` |
| `--overlay` | off | Extraction writes overlay videos |
| `--skip-extract` | off | With `--videos`, reuse landmarks already in the default output folder |
| `--no-deploy` | off | Stop after conversion |
| `--name PREFIX` | derived *(planned)* | Cache is `Workspace/prepared/<PREFIX>.npz`; runs are `<PREFIX>_<model>` |
| `--epochs N` † | per-model recipe | Override epochs for every model (use `2` for a smoke test) |

It applies the cap-5 `v2_162` defaults of `prepare`, the lab recipes of `train` (`--pool attn`; TCN `--causal 0`), then `convert` (with parity where supported) and `deploy`. Stops at the first failing stage (exit `2`) and prints how to resume with the individual commands. **Output:** everything above, then a summary with run folders, `.pt`, `.mlpackage`, key metrics.

```bash
./lsm pipeline --videos ~/LSMVideos --name lab_cap5                    # macOS, full
./lsm pipeline --landmarks ~/landmarks --models cnn --epochs 2 --no-deploy --name smoke   # quick check
./lsm --dry-run pipeline --landmarks ~/landmarks                       # print the commands only
```

## Platform support

| Stage | macOS | Linux | Windows |
|---|---|---|---|
| setup, doctor, test, prepare, train, evaluate, report | yes (MPS) | yes (CUDA/CPU) | yes (CUDA/CPU) |
| extract | yes (Xcode) | no | no |
| convert | yes, with parity | yes, no parity (no `predict`) | via WSL2 |
| deploy | yes | copies files | copies files |

Landmarks from a Mac can be shared with Linux/Windows machines; training there starts at `lsm prepare --landmarks`.

## Reproducing the lab cap-5 cycle

[`Runs/lab_cap5/reproduce.sh`](../Runs/lab_cap5/reproduce.sh) lists the exact `./lsm` commands.
