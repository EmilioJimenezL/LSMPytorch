---
name: lsm-document-run
description: Write or update Runs/<name>/README.md from metrics.json, history.json, and split manifests. Use after training finishes or when documenting an existing run.
---

# Document a training run

## Inputs

For `Runs/<name>/` (or `Legacy/Runs/<name>/`):

- `metrics.json` — best val, extra_test, layout, model_config, epochs
- `history.json` — optional curves narrative
- `split_manifest.json` / `classes.json` — class count, cap, seed
- CLI flags or `lsm` invocation used

## README template

```markdown
# <name>

## Goal
One sentence: what this run tested.

## Config
- Model / layout / pool / causal
- Cap, min videos, seed, epochs, batch, lr
- Augmentation notes

## Results
- Val top-1 / top-5
- Extra-test top-1 / top-5 (if present)
- Early stop epoch

## Artifacts
- `best_model.pt` → `Models/Pytorch/<name>.pt` (if promoted)
- Core ML: `Models/CoreML/<name>.mlpackage` (if converted)

## Reproduce
```bash
./lsm train ...
```
```

Fill numbers from JSON; do not invent. If metrics are missing, say so and only document config.
