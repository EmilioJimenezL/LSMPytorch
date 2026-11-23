# TestingTools

Holds **LSMMobileModelTesting**, an iOS SwiftUI app that runs the trained models live on the camera so you can grade predictions on a device. It is the last stage of the pipeline: models arrive via `lsm deploy` ([`CoreMLConversion`](../CoreMLConversion/README.md)).

## What the app does

- Captures camera video (front/back), extracts hand + pose landmarks per frame with Vision (same joints as [`LandmarkExtraction`](../LandmarkExtraction/README.md)), and builds the 85-frame input for the selected model.
- **v1 models** (Fase 1 and old CNN/TCN/3D CNN): rolling buffer, inference every 15 frames once 85 frames are buffered, `SequencePreprocessor`.
- **v2 models** (`labCap5`): `LiveSegmenter` cuts rest-to-rest sign segments (starts on hand present + wrist motion, ends after rest frames or at 170 frames), then `SequencePreprocessorV2` (shoulder center + width scale, per-joint interpolation) builds the 162-dim tensor.
- Shows Top-5 as buttons plus **Otra** (searchable list of all classes with a confirm step); each choice is logged to `Documents/live_test_log.csv`.
- Switching model shows a loading overlay; if the new model fails to load, the previous one stays active. Debug panel (ⓘ): FPS, inference latency, detection state, buffer progress, and loaded **Clases** count (must match the active model).

Requirements: Xcode, iOS 26.2+ deployment target, a physical device for the camera (the simulator is used for parity checks only). Signing uses automatic signing with the project's existing team ID; change it to your own team to run on a device, and do not commit signing or `xcuserdata` changes.

## Layout

```
TestingTools/
└── LSMMobileModelTesting/
    ├── LSMMobileModelTesting.xcodeproj
    └── LSMMobileModelTesting/            # synchronized folder: files here are bundled automatically
```

| File (in `LSMMobileModelTesting/LSMMobileModelTesting/`) | Role |
|---|---|
| `LSMInferenceApp.swift`, `ContentView.swift` | Entry point, camera UI, grading panel, overlays |
| `LandmarkPipeline.swift` | Vision (v1 + v2), buffering, inference, `AVAILABLE_MODELS`, `SignSet`, model loading, layout resolution |
| `CameraManager.swift` | AVCapture session, permissions, camera flip |
| `LSMConstants.swift` | v1 (135) and v2 (162) constants; must mirror `TrainingPipeline/features.py` |
| `SequencePreprocessor.swift` / `SequencePreprocessorV2.swift` | v1 and v2 preprocessing |
| `LiveSegmenter.swift` | Rest-to-rest segmentation for v2 |
| `FingerprintExtractor.swift`, `FingerprintMatcher.swift` | Fase 1 (810-dim fingerprint, cosine + DTW voting); legacy, see [`Legacy/Fase1Fingerprints`](../Legacy/Fase1Fingerprints/README.md) |
| `ModelBrowserView.swift`, `ModelPreparingView.swift` | Model catalog (by sign set and family, with search) and loading screen |
| `GradingLog.swift`, `GradingViews.swift` | CSV session log (`import Combine` required), Otra search and share sheet |
| `Phase1ParityCheck.swift`, `Phase2ParityCheck.swift` | DEBUG-only parity checks (below) |
| `runs_*_classes.json`, `*.mlpackage`, `*_phase2_fixture.json`, `golden_sequence.json`, `classes.json`, `fingerprints.bin` | Bundled model artifacts |

## Sign sets (`SignSet`) and models

The picker groups models by the vocabulary they were trained on. `AVAILABLE_MODELS` (18 entries) registers each model with `bundleName`, `classesBundleName`, `family` (`fingerprint`, `cnn`, `tcn`, `cnn3d`, `legacy`), `signSet`, and optional `preferredLayout`.

| SignSet | Classes | Layout | Models |
|---|---|---|---|
| `labCap5` ("LSM Videos Lab, 5 por palabra") | 518 | v2_162 | CNN val 36.68% / extra 26.63%, TCN val 33.20% / extra 23.81% (from [`Runs/`](../Runs/README.md)) |
| `labMin5` ("LSM Videos Lab ≥5") | 517 | v1_135 | Previous cycle CNN 47.89%, TCN 48.57% val |
| `lsmoutput307` | 307 | v1_135 | Fase 1 fingerprint, CNN 47.16%, TCN 44.78%, 3D CNN 15.52% |
| `min12` / `min7` / `min5` | 28 / 54 / 286 | v1_135 | Nested video-count tiers, CNN/TCN/3D CNN each |
| `legacy187` | 187 | v1_135 | `coreml_models` (pre-LSMOutput) |

**Layout resolution:** `preferredLayout` from the entry, else Core ML metadata `feature_layout`, else `v1_135`.

**Class labels:** index `i` of the logits maps to `classes[i]`. Load order: (1) Core ML metadata `classes` (key `classes` or ending `.classes`), (2) bundled JSON named by `classesBundleName`, (3) `bundleName` minus a trailing `_lsmoutput` plus `_classes.json`, (4) `{bundleName}_classes.json`. Fase 1 reads names from `fingerprints.bin`. If labels are fewer than outputs the UI shows `clase_N`; fix by checking that `classesBundleName` points at the JSON from the same run as the model, and that the debug **Clases** count matches.

**Naming:** the app folder still carries the pre-reorganization stems (`runs_lab_cap5_cnn_lsmoutput`, `runs_lab_cap5_cnn_classes`, `runs_lab_cap5_cnn_phase2_fixture`), while `Models/CoreML` uses short names (`lab_cap5_cnn`). `export_to_app.py` copies a package under its own name, writes `{run-stem}_classes.json` and `{run-stem}_phase2_fixture.json`, and prints the `LSMModel(...)` entry to paste into `AVAILABLE_MODELS`. The registration is manual, and `Phase2ParityCheck.verifyBundledFixtures()` has a hardcoded list of fixture names that must be updated for new models.

## Adding a model

1. `./lsm convert --run <name>` then `./lsm deploy --run <name> [--sign-set labCap5]` (or the two scripts by hand).
2. Paste the printed `LSMModel` entry into `AVAILABLE_MODELS`, set `signSet` and `preferredLayout`.
3. Build in Debug; check the console for the parity lines below and the **Clases** count in the debug panel.
4. Commit the `.mlpackage` (Git LFS), the classes JSON and the Swift change.

## Parity checks (DEBUG builds only; run when `LandmarkPipeline` is created)

| Check | Compares | Pass |
|---|---|---|
| `Phase1ParityCheck` | Swift fingerprint vs `golden_sequence.json` | max diff < 1e-3 (`✓ Parity check passed`) |
| `Phase2ParityCheck` | Per fixture: Swift v2 preprocessing vs Python `prepared` tensor (when `raw_frames` is in the fixture), then Core ML logits vs fixture `logits`, CPU-only | preprocessing diff < 1e-3; logit diff < 0.05 and same top class (`✓ Phase2 Core ML parity`) |

Last cycle: Fase 1 parity passed and Phase 2 Core ML parity passed (cpuOnly, simulator) for both cap-5 fixtures. Release builds skip these checks.

## Live grading and CSV

`Documents/live_test_log.csv` on the device (not committed; export with ↑, delete with the trash button, which asks for confirmation). `session_id` (`YYYY-MM-DD-xxxxxxxx`) is created once and kept in `UserDefaults`.

Columns: `session_id, timestamp, model_name, inference_ms, guess1, conf1_pct … guess5, conf5_pct, selected_sign, selection_source` where `selection_source` is `top_pick` (tapped a Top-5 button) or `otra_search`. The template [`Utils/live_test_log_template.csv`](../Utils/README.md) has a smaller column set (`sign, category, backend, correct_yes_no, …`); derive `correct_yes_no` offline by comparing `selected_sign` to the sign performed.

Device protocol for comparing backends: pick ~10 signs spread over categories, switch models in the same session without restarting, grade each, export the CSV. Check load (Clases count), live top-1, latency (target < 200 ms), and one Release build smoke test.

## Build checks

```bash
cd TestingTools/LSMMobileModelTesting
xcodebuild -project LSMMobileModelTesting.xcodeproj -scheme LSMMobileModelTesting \
  -destination 'platform=iOS Simulator,name=<simulator>' build
```

The Xcode user-state folders (`xcuserdata`) and `build/` are gitignored. `fingerprints.bin` (~90 MB, Fase 1) is not tracked; regenerate it with `Legacy/Fase1Fingerprints` (`generate_sample_bin.py` for a dev stub).
