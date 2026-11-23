---
name: lsm-run-pipeline
description: Run the LSM training pipeline via the lsm orchestrator. Use when the user wants to extract, prepare, train, convert, deploy, or run an end-to-end cycle.
---

# Run the LSM pipeline

## Interview the user (ask only what is missing)

1. **Input:** videos directory (macOS extract) **or** landmarks JSON tree?
2. **Models:** `cnn`, `tcn`, both, or `3dcnn`?
3. **Layout:** default `v2_162` (current) vs `v1_135` (legacy).
4. **Cap / min videos:** default cap 5, min 5 for lab-style runs.
5. **Platform:** macOS (full), Linux (train+convert), Windows (train; convert in WSL2).
6. **Deploy to iOS app?** default yes on macOS after convert.
7. **Run name prefix** for `Runs/` and `Models/`.

## Procedure

```bash
./lsm doctor
# Preview:
./lsm pipeline --dry-run --landmarks DIR --models cnn,tcn --name PREFIX
# Or execute:
./lsm pipeline --landmarks DIR --models cnn,tcn --name PREFIX -y
# From videos on macOS:
./lsm pipeline --videos DIR --models cnn,tcn --name PREFIX -y
```

Step-by-step when the user wants control: `extract` → `prepare` → `train` → `convert` → `deploy`.

## While training

- Point the user at `Workspace/logs/` and the live terminal plan/progress.
- After completion, read `Runs/<name>/metrics.json` and summarize val / extra_test top-1.
- Suggest next command from the CLI summary (usually `lsm convert` or `lsm deploy`).
- Offer to document the run with the `lsm-document-run` skill.

## Failures

Map errors to fixes: missing `.venv` → `lsm setup`; non-macOS extract → share landmarks; parity fail → check layout mismatch; no CUDA → `--cpu` or install CUDA torch via `lsm setup --cuda 12.4`.
