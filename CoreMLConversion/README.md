# CoreMLConversion

Converts a trained checkpoint to a Core ML `.mlpackage`, proves the Core ML model matches PyTorch (the **parity gate**), and copies the result into the iOS test app.

```
Runs/<name>/best_model.pt ──convert_to_coreml.py──► Models/CoreML/<name>.mlpackage + <name>_fixture.json
                                                    ──export_to_app.py──► TestingTools/LSMMobileModelTesting/...
```

Normally run as `./lsm convert --run <name>` then `./lsm deploy --run <name>` ([Orchestrator](../Orchestrator/README.md)).

## Files

| File | Role |
|---|---|
| `convert_to_coreml.py` | Checkpoint → TorchScript trace → Core ML, metadata, parity gate, fixture |
| `export_to_app.py` | Copy `.mlpackage`, classes JSON and fixture into the app bundle folder |
| `_paths.py` | Puts `TrainingPipeline` on `sys.path`; defines `REPO_ROOT` and `IOS_APP` (`TestingTools/LSMMobileModelTesting/LSMMobileModelTesting`) |
| `requirements.txt` | torch, numpy, coremltools ≥ 7, rich (also covered by `TrainingPipeline/requirements.txt`) |

## convert_to_coreml.py

```bash
python CoreMLConversion/convert_to_coreml.py \
  --checkpoint Runs/lab_cap5_cnn/best_model.pt \
  --output Models/CoreML/lab_cap5_cnn \
  --cache Workspace/prepared/lab_v2_cap5.npz --parity \
  --fixture Models/CoreML/lab_cap5_cnn_fixture.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | required | `best_model.pt` from a run |
| `--output` | `./lsm_model` | Output path; `.mlpackage` is appended if missing |
| `--cache` | – | Prepared `.npz`; its validation tensors drive parity and the fixture |
| `--parity` | off | Fail the conversion if Core ML differs from PyTorch. Parity also runs whenever `--cache` is given; it requires `--cache` |
| `--fixture` | – | Write a golden fixture JSON (needs `--cache`) |

The model is rebuilt from the checkpoint's `model_config` (`input_dim`, `layout`, `pool`, `causal`), traced with a `(1, 85, input_dim)` dummy, and converted for **iOS 16+** with `ComputeUnit.ALL`.

**Model interface:** input `landmarks` float32 `(1, 85, feature_dim)` (162 for `v2_162`, 135 for `v1_135`), output `logits` `(1, n_classes)` (raw logits; apply softmax in the app). Batch size is fixed to 1.

**Embedded metadata** (`user_defined_metadata`): `classes` (JSON list), `n_frames`, `feature_dim`, `n_classes`, `val_acc`, `model_type`, `feature_layout` (`v1_135`/`v2_162`; missing means v1), `preprocess_version` (`v2_shoulder_scale_1` or `v1_shoulder`), `model_config`. The app uses `feature_layout` to choose its preprocessor.

### Parity gate

Runs the first up to 256 validation sequences from the cache through both PyTorch and the saved `.mlpackage` and compares logits. **Pass requires max |Δlogit| ≤ 0.05 and top-1 agreement ≥ 99%**; otherwise the script exits with an error and the diff numbers. Measured for the lab cap-5 models (256 sequences): CNN max |Δlogit| 0.017, TCN 0.018, top-1 agreement 100%, PASS.

Core ML `predict` only works on macOS. On Linux or Windows the conversion itself can succeed but the random-input smoke test is skipped and the parity step cannot run, so convert without `--parity`/`--cache` there and run the gate on a Mac before shipping. The `lsm convert` command skips parity automatically with a warning on such platforms. Windows needs WSL2 for `coremltools`.

### Fixture JSON

First validation sample of the cache: `layout`/`feature_layout`, `feature_dim`, `n_frames`, source `path`, `class`, `prepared` (and the same data as `prepared_sequence`, `85 × D`), PyTorch `logits`, and the Python `parity` result. The iOS DEBUG check compares Core ML logits against it (max diff < 0.05 and same top class).

## export_to_app.py

```bash
python CoreMLConversion/export_to_app.py \
  --mlpackage Models/CoreML/lab_cap5_cnn.mlpackage \
  --classes Models/CoreML/lab_cap5_cnn_classes.json \
  --run-stem lab_cap5_cnn --fixture Models/CoreML/lab_cap5_cnn_fixture.json \
  --name "CNN — Lab cap5 (v2)" --family cnn --sign-set labCap5
```

| Flag | Default | Meaning |
|---|---|---|
| `--mlpackage` | required | `.mlpackage` directory to copy |
| `--classes` | required | Non-empty JSON array of class names |
| `--run-stem` | required | Prefix for `{stem}_classes.json` and `{stem}_phase2_fixture.json` |
| `--fixture` | – | Parity fixture to bundle |
| `--name` / `--description` | package name / `N clases` | Used only in the printed snippet |
| `--family` | `cnn` | `cnn`, `tcn`, `cnn3d`, `fingerprint`, `legacy` |
| `--sign-set` | `labCap5` | App `SignSet` case |

Copies (replacing any existing copy of the same package name) into `TestingTools/LSMMobileModelTesting/LSMMobileModelTesting/`, then **prints** a Swift `LSMModel(...)` entry. It does not edit `LandmarkPipeline.swift`: paste the entry into `AVAILABLE_MODELS` by hand. Bundle names must match `bundleName` (package stem) and `classesBundleName` (`{stem}_classes`); the DEBUG `Phase2ParityCheck` looks up `{stem}_phase2_fixture` and derives the package name from it unless the fixture has a `model_bundle` field (see [`TestingTools`](../TestingTools/README.md)).

## Outputs

`convert_to_coreml.py` writes the `.mlpackage` and the fixture. `<name>_classes.json` is a copy of the run's `classes.json` (made by `lsm convert`, or copy it by hand). Final layout in `Models/CoreML/`: `<name>.mlpackage` (Git LFS), `<name>_classes.json`, `<name>_fixture.json`; see [`Models/CoreML`](../Models/CoreML/README.md).
