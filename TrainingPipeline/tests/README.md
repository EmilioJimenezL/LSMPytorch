# tests

pytest suite for the training package. `TrainingPipeline/pytest.ini` sets `pythonpath = .` and `testpaths = tests`.

```bash
./lsm test                                   # wraps pytest
cd TrainingPipeline && ../.venv/bin/python -m pytest -q
```

Last run: 34 passed, 1 skipped (~22 s). `test_resume.py` runs `train.py` in subprocesses; `test_models.py` includes a Core ML conversion + predict round-trip, which needs macOS or a platform where `coremltools` can predict.

| File | Covers |
|---|---|
| `test_features.py` | v2 null joints → confidence 0, v1 null tolerance, shoulder-width scale normalization, per-joint interpolation does not blend missing frames, v1 vector from v2 JSON keeps shoulders |
| `test_augment.py` | v1/v2 x/y/confidence index masks, confidence channels untouched, missing joints stay zero, shapes preserved |
| `test_dataset.py` | `Rosa` split into `Rosa (Category)` classes, cap-5 split deterministic and exact, `extra_test` disjoint from train/val, v1 and v2 JSON schemas load, `min_videos` counts valid sequences |
| `test_models.py` | CNN/TCN forward shapes for both layouts and both pools, TCN receptive-field formula, legacy TCN defaults, old checkpoint (no `model_config`) loads, tiny Core ML round-trip (diff < 0.05) |
| `test_resume.py` | Training 2 epochs then `--resume` for a 3rd continues the epoch counter and history |

Skips: `test_dataset.py` (two tests) and `test_features.py` (one) read sample JSON from `LandmarkExtraction/LSMExtractorCLI/build/sample_v2_out` (a local, gitignored build artifact) and skip when it is missing. The Core ML test skips if `coremltools` is not installed. The Core ML round-trip test and the sample-data tests are the platform-dependent ones to watch in CI.
