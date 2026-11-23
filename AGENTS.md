# AGENTS.md — LSMPytorch

> For humans: [README.md](./README.md). Skills: [`.cursor/skills/`](.cursor/skills/).

## Where to start

1. Read [README.md](README.md) for clone, platform notes, and `./lsm` quickstart.
2. Run `./lsm doctor` before changing pipeline behavior.
3. Prefer `./lsm <command>` over calling scripts directly; scripts still work under their folders.
4. Skills:
   - [`lsm-development`](.cursor/skills/lsm-development/SKILL.md) — architectures, layouts, Swift↔Python sync
   - [`lsm-run-pipeline`](.cursor/skills/lsm-run-pipeline/SKILL.md) — interview user → run orchestrator
   - [`lsm-document-run`](.cursor/skills/lsm-document-run/SKILL.md) — fill `Runs/<name>/README.md` from metrics

## Layout

| Path | Own |
|---|---|
| `Orchestrator/` | Cross-platform CLI |
| `LandmarkExtraction/` | macOS extractors (Vision); **macOS only** |
| `TrainingPipeline/` | `train.py`, `features.py`, models, pytest |
| `CoreMLConversion/` | convert + deploy-to-app |
| `TestingTools/` | iOS app |
| `Runs/` | New experiment artifacts + README per run |
| `Models/{Pytorch,CoreML}/` | Canonical best weights |
| `Legacy/` | Frozen history — **do not extend** |
| `Utils/` | Misc tools |
| `Workspace/` | Local caches/logs (not committed) |

Import root for Python training code: **`TrainingPipeline/`** (`pytest.ini` sets `pythonpath = .`). Other areas use a local `_paths.py` helper.

## Critical invariants

| Contract | Rule |
|---|---|
| Feature layouts | `v1_135` (legacy) vs `v2_162` (current). Core ML metadata key `feature_layout`. |
| v2 vector | 21×(x,y,c) per hand + 12×(x,y,c) pose; missing joints confidence 0; shoulder-centered / median shoulder width. |
| JSON schema v2 | `{schema_version, fps, width, height, orientation, frames}` with null joints. |
| Class keys | Use `word (category)` when a word is ambiguous (e.g. Rosa). |
| Parity gate | Export must pass max logit diff ≤ 0.05 and top-1 agreement ≥ 99% when `--parity` runs (macOS). |
| Extractor sync | CLI and GUI engine sources must match (`./lsm` / `Utils/check_extractor_sync.py`). |
| New runs | Always under `Runs/<name>/` with `README.md`, `metrics.json`, `classes.json`. Copy `best_model.pt` → `Models/Pytorch/<name>.pt`. |
| Legacy | Do not add features to `Legacy/`. Port forward into the active tree instead. |

## How to work

```bash
./lsm test                  # TrainingPipeline pytest
./lsm test --quick
python Utils/check_extractor_sync.py
./lsm pipeline --dry-run --landmarks /path --models cnn --name trial
```

After model/layout changes:

1. Update `TrainingPipeline/features.py` and matching Swift (`LSMConstants`, `SequencePreprocessorV2`, `LandmarkPipeline`).
2. Re-run pytest + a 2-epoch smoke (`./lsm train ... --epochs 2`).
3. Convert with parity + fixture; confirm iOS Phase2ParityCheck on simulator (`computeUnits = .cpuOnly`).

## Conventions

- Prefer kebab-case CLI flags; keep legacy snake_case aliases working.
- Do not commit `Workspace/`, intermediate `last.pt` / `checkpoint_epoch*.pt`, or secrets.
- Track best `.pt` and `.mlpackage` via Git LFS (see `.gitattributes`).
- Document every meaningful training run in its `Runs/<name>/README.md` (skill: `lsm-document-run`).
