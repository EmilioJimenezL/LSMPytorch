# Models/CoreML

Core ML exports of the deployable runs. Each model has three files (the `.mlpackage` is Git LFS):

| File | Content |
|---|---|
| `<run>.mlpackage` | Core ML model (iOS 16+) |
| `<run>_classes.json` | Ordered class names (518 entries), copy of the run's `classes.json`; index = logit index |
| `<run>_fixture.json` | Golden fixture: first validation sequence (`prepared`, 85×162), PyTorch `logits`, and the Python parity result. Used by the app's DEBUG `Phase2ParityCheck` |

Current models:

| Run | Model | Layout | Classes | Val / Extra top-1 | Parity max \|Δlogit\| |
|---|---|---|---|---|---|
| `lab_cap5_cnn` | 1D CNN, attn pool | v2_162 | 518 | 36.68% / 26.63% | 0.017 |
| `lab_cap5_tcn` | TCN non-causal, attn pool | v2_162 | 518 | 33.20% / 23.81% | 0.018 |

Parity was measured on 256 validation sequences; top-1 agreement with PyTorch was 100% (gate: ≤ 0.05 and ≥ 99%). See [`CoreMLConversion`](../../CoreMLConversion/README.md).

## Interface

- Input `landmarks`: float32 `(1, 85, 162)` (`v2_162`; a `v1_135` model would be `(1, 85, 135)`). Prepare it exactly like training: shoulder-centered, width-scaled, per-joint interpolated to 85 frames (`SequencePreprocessorV2` on iOS).
- Output `logits`: `(1, n_classes)`; apply softmax for probabilities.
- Metadata (`user_defined_metadata`): `classes` (JSON), `n_frames`, `feature_dim`, `n_classes`, `val_acc`, `model_type`, `feature_layout`, `preprocess_version`, `model_config`. Check `feature_layout` to pick the right preprocessing.

Predicting requires macOS (Core ML runtime). The files are copied to the iOS app with `lsm deploy` / `CoreMLConversion/export_to_app.py`; the app folder keeps its own older stems (`runs_lab_cap5_*`), see [`TestingTools`](../../TestingTools/README.md).
