# evaluation

Post-training analysis scripts. They were written for the first LSMOutput cycle (307 classes, v1 layout) to compare **Fase 1** (fingerprint + DTW, now in [`Legacy/Fase1Fingerprints`](../../Legacy/Fase1Fingerprints/README.md)) with **Fase 2** (neural). For the current cap-5 runs the primary evaluation is built into `train.py`: `Runs/<name>/metrics.json` already holds val and `extra_test` top-1/top-5.

## Files

| File | Purpose | Inputs | Outputs |
|---|---|---|---|
| `evaluate_nn.py` | Evaluate a checkpoint on the full dataset; split out val using the run's `split_manifest.json` | `--checkpoint` (required), `--dataset` (required), `--min-videos 2`, `--output` | JSON: top-1/top-5, per-class, per-category, confusion, ms/video, `val_split` block |
| `compare_models.py` | Head-to-head Fase 1 LOO vs Fase 2 reports; global, per-category, per-class | `--fase1`, `--fase2 A.json B.json ...`, `--output` (all required) | `comparison.json` |
| `error_analysis.py` | Worst classes from a comparison, with suggested actions | `--comparison` (required), `--top 20`, `--output` | findings JSON (stdout summary) |
| `generate_training_report.py` | Consolidated readiness report: metrics table + whether each export exists | `--root .`, `--reports ./reports` | `training_report.json`, `training_report_summary.txt` |
| `run_all.sh` | Runs dataset inventory, Fase 1 LOO, per-checkpoint eval, comparison, error analysis, report | dataset dir (arg, default `../LSMOutput`) | files under `reports/` |

Protocol note: `evaluate_nn.py` runs inference on every video including training videos (so its "full-dataset" number is inflated). Use its `val_split` block or `metrics.json` for held-out accuracy.

## Known limitations

- `evaluate_nn.py` builds the model with default `input_dim` (135) and loads data through the v1 path, so it does **not** support `v2_162` checkpoints yet. For `lab_cap5*` runs use `metrics.json`.
- `generate_training_report.py` contains the hardcoded 2026-06 metrics table (Fase 1 and LSMOutput CNN/TCN/3D CNN) and legacy paths (`runs_lsmoutput_*`, `exports/`).
- `run_all.sh` and the script docstrings still use pre-reorganization paths (`review/`, `scripts/`, `lsm_fingerprints/`, `runs_lsmoutput_*`). `dataset_report.py` now lives in [`Utils/`](../../Utils/README.md), Fase 1 in `Legacy/`.
- `lsm evaluate` and `lsm report` wrap these scripts (see the [Orchestrator](../../Orchestrator/README.md)); update those scripts to read `Runs/` and v2 when they are extended.
