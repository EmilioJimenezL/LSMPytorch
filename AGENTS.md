# AGENTS.md — TrainingPipeline

> **Audience:** AI-assisted and agentic development. For human onboarding, see [README.md](./README.md).
>
> **Sibling repo:** [AppLSMTests/AGENTS.md](../AppLSMTests/AGENTS.md) — macOS extractor + iOS inference (includes Xcode signing conventions).

This repo is the **Python side** of LSM (Lengua de Señas Mexicana): Fase 1 statistical fingerprint + DTW (`lsm_fingerprints/`) and Fase 2 neural training + Core ML export.

---

## Goals and success metrics

| Phase | Goal | Success metric |
|---|---|---|
| **Fase 1** | Recognize signs without training a neural net | ≥60% top-1 leave-one-out (minimum 40%); top-5 ≥70% (goal 85%) |
| **Fase 2** | Beat Fase 1 with deep learning | TCN/CNN accuracy > Fase 1 baseline on same dataset |
| **Mobile** | Real-time inference on iOS | Fase 1 latency <200ms (goal <100ms) |

**Vocabulary:** 221 words in `themes.json` (11 categories). Current trained model: ~187 classes (classes with ≥2 valid videos after `--min-videos` filter).

---

## Ecosystem architecture

```mermaid
flowchart LR
  subgraph videos [Input]
    LSMVideos["LSMVideos/category/word/*.mp4"]
  end

  subgraph applsm [AppLSMTests]
    Extractor[LSMExtractorGUI]
    Mobile[LSMMobileModelTesting]
  end

  subgraph training [TrainingPipeline]
    Fase1[lsm_fingerprints]
    Fase2[train.py + convert_to_coreml]
  end

  LSMVideos --> Extractor
  Extractor -->|"*_landmarks.json"| Fase1
  Extractor -->|"*_landmarks.json"| Fase2
  Fase1 -->|"fingerprints.bin"| Mobile
  Fase2 -->|"coreml_models.mlpackage"| Mobile
```

**End-to-end flow:**

1. Organize raw videos (optional): `organize_from_json.py` + `themes.json`
2. Extract landmarks: LSMExtractorGUI → `LSMVideosOutput/` (see AppLSMTests)
3. Fase 1: `build_db.py` → `evaluate.py` → copy `fingerprints.bin` to iOS
4. Fase 2: `train.py` → `convert_to_coreml.py` → copy `.mlpackage` + `classes.json` to iOS
5. Test on device: LSMMobileModelTesting

---

## Critical invariants (do not break)

| Contract | Value / rule |
|---|---|
| Landmark vector | 135 dims/frame: `[0:42]` left hand, `[42:84]` right hand, `[84:135]` pose (17× x,y,visibility) |
| Sequence length | 85 frames after linear interpolation |
| JSON frame keys | `frame`, `timestamp_ms`, `leftHand`, `rightHand`, `pose`; Y=0 at top in stored JSON |
| Extractor input | `{root}/{category}/{word}/*.mp4` — folder path is source of truth |
| Extractor output | Mirrors input under separate output root + `dataset_manifest.json` |
| Training dataset | Nested `category/word/*_landmarks.json` or flat `word/*_landmarks.json` — see `dataset.py` |
| Fase 1 binary | `fingerprints.bin` magic `0x4C534D46`, version 1 — see `lsm_fingerprints/export_bin.py` |
| Fase 1 fingerprint | 810-dim, L2-normalized statistical fingerprint |

---

## File index — if task X, read Y

| Task | Files |
|---|---|
| Load / augment landmarks | `dataset.py` |
| Train model | `train.py`, `split_utils.py`, `models/` |
| Stratified split | `split_utils.py` → `{output}/split_manifest.json` |
| Dataset inventory | `scripts/dataset_report.py` |
| Review / compare | `review/evaluate_nn.py`, `review/compare_models.py`, `review/error_analysis.py`, `review/generate_training_report.py`, `review/run_all.sh` |
| Train all models | `train_all_lsmoutput.sh` |
| Modular models (smoke tests) | `models/` (`__init__.py`, `cnn1d.py`, `tcn.py`, `cnn3d.py`) |
| Core ML export | `convert_to_coreml.py` |
| Organize raw videos | `organize_from_json.py`, `themes.json` |
| Vocabulary listing | `list_words.py`, `palabras.txt` |
| Fase 1 build DB | `lsm_fingerprints/build_db.py` |
| Fase 1 evaluate (LOO) | `lsm_fingerprints/evaluate.py` |
| Fase 1 matching | `lsm_fingerprints/matcher.py` |
| Fase 1 preprocessing | `lsm_fingerprints/preprocess.py` |
| Fase 1 fingerprint | `lsm_fingerprints/fingerprint.py` |
| Fase 1 constants | `lsm_fingerprints/constants.py` |
| Fase 1 ↔ Swift parity | `lsm_fingerprints/tests/test_all.py`, `lsm_fingerprints/tests/fixtures/golden_sequence.json` |
| Generate iOS sample bin | `lsm_fingerprints/generate_sample_bin.py` |
| Export golden fixtures | `lsm_fingerprints/export_fixtures.py` |
| Model smoke tests | `test_models_smoke.py` |
| Training configs | `configs/*.yaml` |

### Fase 1 module reference

| File | Role |
|---|---|
| `preprocess.py` | Shoulder anchor normalization, valid-frame filter, 85-frame interpolation |
| `fingerprint.py` | 810-dim L2-normalized statistical fingerprint |
| `build_db.py` | Index landmark JSON → `fingerprints.npz`; optional `.bin` export |
| `export_bin.py` | Write `fingerprints.bin` (magic `0x4C534D46`) for Swift |
| `matcher.py` | Cosine pre-filter + FastDTW + class voting |
| `evaluate.py` | Leave-one-out cross-validation |
| `constants.py` | Shared dims/thresholds — must stay in sync with Swift `LSMConstants.swift` |

---

## Technologies and dependencies

| Stack | Details |
|---|---|
| Python | 3.13+ recommended |
| ML | PyTorch 2.1+, torchvision, scipy, scikit-learn |
| Export | coremltools 7+ |
| Fase 1 | numpy, scipy, fastdtw |
| Logging | tensorboard |
| Config | pyyaml |

See `requirements.txt` and `lsm_fingerprints/requirements.txt`.

**Dev environment:** shared venv at `TrainingPipeline/.venv`. Fase 1 tests:

```bash
cd lsm_fingerprints
PYTHONPATH="../.venv/lib/python3.13/site-packages:." python tests/test_all.py
```

---

## Common workflows

### Build Fase 1 database

```bash
cd lsm_fingerprints
python build_db.py \
  --dataset ../output \
  --output ./fingerprints.npz \
  --export-bin ./fingerprints.bin
```

### Evaluate Fase 1 (leave-one-out)

```bash
cd lsm_fingerprints
python evaluate.py --db ./fingerprints.npz
```

### Add Fase 1 class (no retraining)

```bash
cd lsm_fingerprints
python -c "
from build_db import add_class
from preprocess import prepare_sequence
seq = prepare_sequence(json_path='nueva_seña_landmarks.json')
add_class('./fingerprints.npz', 'NuevaPalabra', [seq], category='Familia',
          export_bin_path='./fingerprints.bin')
"
```

Copy resulting `fingerprints.bin` to LSMMobileModelTesting Xcode target.

### Train Fase 2

```bash
python train.py --dataset ./output --model tcn --epochs 100 --batch 64 --output ./runs
```

### Export Core ML

```bash
python convert_to_coreml.py --checkpoint ./runs/best_model.pt --output ./lsm_model
```

### Organize videos from themes.json

```bash
python organize_from_json.py --json themes.json --input ./videos --output ./dataset
```

### Run tests

```bash
python test_models_smoke.py
cd lsm_fingerprints && python tests/test_all.py
```

---

## Phase 1 testing workflow (LSMOutput)

**Dataset:** `../LSMOutput` (2,131 landmark JSON, 14 categories, 501 words in manifest).

```bash
export DATASET=/path/to/LSMOutput
export PYTHONPATH=".venv/lib/python3.13/site-packages:lsm_fingerprints:."
cd lsm_fingerprints

python tests/test_all.py
python build_db.py --dataset "$DATASET/Colores" --output ./runs_pilot/colores_fingerprints.npz --min-videos 1
python evaluate.py --db ./runs_pilot/colores_fingerprints.npz
python build_db.py --dataset "$DATASET/Familia" --output ./runs_pilot/familia_fingerprints.npz
python evaluate.py --db ./runs_pilot/familia_fingerprints.npz
python build_db.py --dataset "$DATASET" --output ./fingerprints.npz --export-bin ./fingerprints.bin --min-videos 2
python evaluate.py --db ./fingerprints.npz 2>&1 | tee loo_report.log
```

**Local artifacts** (gitignored): `fingerprints.npz`, `fingerprints.bin`, `loo_report.log`, `runs_pilot/`.

---

## Phase 1 baseline (LSMOutput)

Measured **2026-06-17** on full LSMOutput extract.

| Metric | Result | Target |
|---|---|---|
| Classes indexed | 307 | — |
| Videos indexed | 1,934 | — |
| Build skipped | 3 | — |
| **Top-1 LOO** | **6.57%** | ≥40% (goal 60%) |
| **Top-5 LOO** | **16.18%** | ≥70% (goal 85%) |

**Pilots:**

| Subset | Classes | Videos | Top-1 | Top-5 |
|---|---|---|---|---|
| Colores | 2 | 12 | 25.00% | 100.00% |
| Familia | 21 | 126 | 11.11% | 31.75% |

**Per-category Top-1 (full LOO):** Festividades 24.4%, Personajes_Historicos 17.3%, Frutas_Verduras_y_Plantas 11.1%, Personas 10.9%, Colores 8.3%, Profesiones 7.9%, Alimentos_y_Bebidas 6.9%, Familia 4.0%, Animales_e_Insectos 3.8%, Locaciones 2.3%, Abecedario 2.1%, Objetos 2.0%, Actividades_Cotidianas 0.0%.

**Conclusion:** baseline not met; Fase 2 neural training on LSMOutput is the next path to beat 6.57% Top-1.

---

## Phase 2 baseline (LSMOutput)

Training **complete** on **307 classes**, **1934 videos**, Fase 1 preprocessing (shoulder norm). Artifacts gitignored under `runs_lsmoutput_*/`, `exports/`, `reports/`.

| Model | Val Top-1 (stratified) | Val Top-5 | Full-dataset Top-1 | Export | iOS deployed |
|---|---|---|---|---|---|
| **1D CNN** | **47.16%** | 64.78% | 90.28% | `exports/runs_lsmoutput_cnn_lsmoutput.mlpackage` | Yes (local) |
| TCN | 44.78% | **67.76%** | 89.87% | `exports/runs_lsmoutput_tcn_lsmoutput.mlpackage` | Yes (local) |
| 3D CNN | 15.52% | 32.84% | 74.72% | `exports/runs_lsmoutput_3dcnn_lsmoutput.mlpackage` | Yes (local) |
| Legacy v1 | 9.68% | — | — | `coreml_models.mlpackage` (187 cls) | Yes (bundled) |

**Recommended iOS default:** CNN (best val Top-1). Regenerate reports: `./review/run_all.sh ../LSMOutput` → `reports/training_report.json`.

Configs: `configs/*.yaml` with `num_classes: 307`. Linux export skips Core ML predict verification — smoke-test on Mac/Xcode.

---

## Model comparison results

Generate with `review/compare_models.py` after Fase 1 LOO JSON + Fase 2 eval JSONs exist.

| Report | Path |
|---|---|
| Dataset inventory | `reports/dataset_inventory.json` |
| Class overlap (iOS A/B) | `reports/class_overlap.json` |
| Fase 1 LOO | `reports/fase1_loo.json` |
| Fase 2 per model | `reports/fase2_runs_lsmoutput_*.json` |
| Head-to-head | `reports/comparison.json` |
| Error drill-down | `reports/error_analysis.json` |
| Consolidated report | `reports/training_report.json`, `reports/training_report_summary.txt` |
| Live test log template | `review/live_test_log_template.csv` |

---

## iOS validation results

Integration work in AppLSMTests (**2026-06-22**). Device camera tests still pending on Mac.

| Backend | Artifacts copied | Load smoke (C2) | Live A/B | Notes |
|---|---|---|---|---|
| Fase 1 fingerprint | Pass | Pending device | Pending | ~95 MB bin, 307 classes |
| CNN LSMOutput | Pass | Pending device | Pending | Recommended default; val 47.16% |
| TCN LSMOutput | Pass | Pending device | Pending | val 44.78%, best val Top-5 |
| 3D CNN LSMOutput | Pass | Pending device | Pending | val 15.52% |
| Legacy Core ML v1 | Pass (pre-existing) | Pending device | Pending | 187 classes |

| Check | Result | Notes |
|---|---|---|
| Bin bundled locally | Pass | ~95 MB, magic `LSMF`, 307 classes, 1934 videos |
| Bin parse (offline) | Pass | ~0.1 s load on Linux |
| Golden fingerprint parity | Pass | max diff 0.0 vs `golden_sequence.json` (Python); Swift Debug check pending Mac |
| All 5 backends in model picker | Pass | `AVAILABLE_MODELS` updated 2026-06-22 |
| Async bin load + loading UI | Implemented | `LandmarkPipeline.loadModel` |
| Background inference + DTW radius | Implemented | `dtwRadius=10`; match on detached queue |
| Live Top-1 / confidence | **Pending device** | Use `review/live_test_log_template.csv` |
| Inference latency (device) | **Pending device** | Python desktop reference ~930 ms/cycle (Fase 1 FastDTW) |
| Release build smoke | **Pending device** | Parity skipped in Release (`#if DEBUG`) |
| Mac Core ML predict verify | **Pending device** | Linux export skips predict step |

Full checklist: [AppLSMTests/AGENTS.md — Phase 1/2 iOS validation](../AppLSMTests/AGENTS.md#phase-1-ios-validation-checklist)

---

## Handoff to AppLSMTests

After `fingerprints.bin` is built locally, iOS validation is owned by AppLSMTests:

1. Copy production `fingerprints.bin` to LSMMobileModelTesting bundle
2. Debug build — `Phase1ParityCheck` against `golden_sequence.json`
3. Live Fase 1 smoke test (5–10 known signs from 307-class DB)
4. Latency check via debug panel (<200ms target)

Full checklist: [AppLSMTests/AGENTS.md — Phase 1 iOS validation](../AppLSMTests/AGENTS.md#phase-1-ios-validation-checklist)

---

## Next steps / backlog

1. **Device validation (AppLSMTests):** B2–B5 (Fase 1) + C2–C5 (Fase 2) on Mac — use `review/live_test_log_template.csv`
2. **Mac Core ML predict smoke** — first Xcode run per `.mlpackage` (Linux export skips predict)
3. Improve weak categories (Abecedario, Actividades_Cotidianas) via data collection or category-scoped bins
4. Wire YAML config loader into `train.py` (warmup, AMP, patience, joint_dropout)
5. Close vocab gap: 501 words extracted vs 307 classes with ≥2 valid videos

See [Live-testing roadmap](#live-testing-roadmap) for iterative accuracy improvement.

---

## Live-testing roadmap

Structured loops to close the Python val → device accuracy gap. Log results in `live_test_log.csv` (copy from `review/live_test_log_template.csv`).

### Loop A — Measure (weeks 1–2)

- Run device A/B matrix across all 5 backends; fill `live_test_log.csv`
- Compute **live Top-1 per backend** vs Python val (gap analysis)
- Tag failure modes: wrong class, low confidence, no detection, latency timeout

### Loop B — Diagnose (ongoing)

- Merge live failures with `reports/error_analysis.json`
- Prioritize categories: Abecedario, Actividades_Cotidianas, Familia (weakest in Fase 1)
- Identify **systematic errors** (confusion pairs) vs **data gaps** (missing angles/lighting)

### Loop C — Data collection (weeks 2–4)

- Use [LSMExtractorGUI](../AppLSMTests/LSMExtractorGUI/) to record **live-failure signs** with multiple performers/angles
- Target 194 single-video words (`--min_videos 2` exclusion list from inventory)
- Re-run `build_db.py` + `train.py` on expanded LSMOutput

### Loop D — Model iteration (weeks 4–8)

| Experiment | Goal |
|---|---|
| Retrain CNN/TCN on expanded data | Close val/live gap |
| Category-scoped Fase 1 bins | Boost weak categories without full retrain |
| Early stopping + reduced augmentation | Reduce overfit (train ~99% vs val ~47%) |
| TCN vs CNN on live Top-1 | Pick production default |
| Optional: `--min_videos 1` pilot | Coverage vs accuracy tradeoff |

### Loop E — Production hardening

- Mac Core ML predict verification for all exports
- Optional `Phase2ParityCheck.swift` (golden logits vs Python)
- Default model selection in app (CNN unless live tests favor TCN)
- Consider on-demand model download if bundle size blocks TestFlight

### Success metrics (8-week targets)

| Metric | Current (Python) | Live target |
|---|---|---|
| Fase 1 Top-1 | 6.57% LOO | Baseline reference only |
| CNN val Top-1 | 47.16% | Live Top-1 > 35% |
| CNN latency | ~0.2 ms/video (Python) | <200 ms on device |
| Category Festividades | 24% Fase 1 LOO | Live > 40% with best backend |
| Confusion pairs | From error_analysis | −50% on top-10 pairs after data pass |

### Mac device A/B protocol

1. **Load smoke (C2):** Each backend loads 307 classes (187 for legacy), no `loadError`
2. **10-sign A/B:** Mix from `reports/class_overlap.json` — Festividades, Personajes, Colores, Familia, Abecedario
3. **Same session:** Switch picker without restart; record Top-1 + confidence + ms
4. **Latency:** Fase 1 (Inferencia/Cosine/DTW), Fase 2 (Inferencia only)
5. **Release smoke:** One Release build with CNN default path

CSV columns: `session_id, sign, category, backend, top1, confidence, inference_ms, correct_yes_no, notes`

---

## Agent conventions

### Do

- Match existing naming and conventions; keep diffs minimal and focused
- Reuse `dataset.py` vector layout; keep Fase 1 Python/Swift constants in sync
- Update AGENTS.md and README.md when changing contracts, workflows, or backlog
- Check sibling [AppLSMTests/AGENTS.md](../AppLSMTests/AGENTS.md) for cross-repo changes

### Don't

- Commit `.venv/`, `runs/*.pt`, `dataset/`, `fingerprints.npz`, `fingerprints.bin` (gitignored artifacts)
- Reintroduce `themes.json` filename matching in the extractor — path-based discovery is canonical (AppLSMTests)
- Create new `.md` files — only `README.md` and `AGENTS.md` are tracked (see `.gitignore`)
- Modify Xcode signing settings — see AppLSMTests AGENTS.md for iOS/macOS work

---

## Documentation maintenance protocol

After completing a task that changes architecture, contracts, workflows, or backlog status:

1. Update **this repo's** `AGENTS.md`
2. Update **this repo's** `README.md` if human-facing behavior changed
3. Check whether [AppLSMTests/AGENTS.md](../AppLSMTests/AGENTS.md) needs the same invariant or cross-link update
4. Do **not** add other markdown files

| Change type | Update |
|---|---|
| New module / renamed file | File index in both AGENTS.md; structure in repo README |
| Data contract change | Invariants in both AGENTS.md; examples in both READMEs |
| New CLI flag or workflow | Commands here; workflow in README |
| Completed backlog item | Remove or mark done in both AGENTS.md |
| Cross-repo integration | Ecosystem diagram + sibling cross-link in both files |
