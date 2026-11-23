# Legacy

Frozen material from earlier phases of the project. It moved here during the repository reorganization so the active areas (`LandmarkExtraction`, `TrainingPipeline`, `CoreMLConversion`, `TestingTools`, `Runs`, `Models`) only contain what the current v2 pipeline uses. **Do not extend or fix anything here**; new work goes in the active folders. Files keep their original contents, so paths inside scripts and docstrings still point at the old layout and may not run unchanged.

| Folder | What it is | Why it is here |
|---|---|---|
| [`Scripts/`](./Scripts/README.md) | Old shell/Python helpers: per-model re-export files, CUDA setup, Apple-silicon and tier training scripts, the original cap-5 cycle script | Replaced by `lsm` and `TrainingPipeline/models/` |
| [`Fase1Fingerprints/`](./Fase1Fingerprints/README.md) | Fase 1: 810-dim statistical fingerprint + DTW matcher, no neural network | Baseline reached 6.57% top-1, far below the neural models; v1-only |
| [`Runs/`](./Runs/README.md) | Run folders of the 2026-06 LSMOutput tiers and the first lab cycle (v1, uncapped) | Superseded by the cap-5 v2 runs in `../Runs/` |
| [`Models/`](./Models/README.md) | Weights and Core ML exports of those runs, plus the original 187-class model | Superseded; kept for comparison and for the app's older sign sets |

## Timeline of what is archived

1. **Fase 1 (fingerprint + DTW):** LOO top-1 6.57% / top-5 16.18% on 307 classes (1,934 videos). Target was 40% minimum.
2. **LSMOutput neural models (307 classes, v1):** CNN 47.16%, TCN 44.78%, 3D CNN 15.52% val top-1.
3. **Video-count tiers (`m4_min12/7/5`):** nested vocabularies of 28 / 54 / 286 classes, trained on Apple silicon.
4. **Lab cycle ≥5 (`lab_min5`, v1, uncapped, 517 classes):** CNN 47.89%, TCN 48.57%, 3D CNN aborted at ~18%.

The v1 layout (135 dims) these use is still supported by the training code; the legacy app sign sets (`lsmoutput307`, `min12`, `min7`, `min5`, `labMin5`, `legacy187`) still load these models.

Numbers above come from the older READMEs and the lab cap-5 report; `Legacy/Runs` folders do not keep their `metrics.json`.
