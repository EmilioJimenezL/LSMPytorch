# Legacy/Scripts

Archived helpers from before the `lsm` orchestrator. Not maintained. Most `cd` to their own folder and expect `train.py`, `convert_to_coreml.py`, `requirements.txt` and `scripts/` next to them (the pre-reorganization root), so they do not run from here without edits.

| File | What it did | Replacement |
|---|---|---|
| `model.py`, `model_tcn.py`, `model_3dcnn.py` | Compatibility re-exports of `models/cnn1d.py`, `tcn.py`, `cnn3d.py` (`LSM_CNN`, `TemporalConvNet`, ...) | `TrainingPipeline/models` |
| `test_models_smoke.py` | Shape smoke test of all registered models (330 classes, 135 dims) | `TrainingPipeline/tests/test_models.py` |
| `testCoreml.py` | Prints metadata of `coreml_models.mlpackage` (the 187-class legacy model) | `lsm doctor`, Core ML metadata in `Models/CoreML/README.md` |
| `setup_env.sh` | `python3.11` venv + PyTorch CUDA 12.4 wheels (RTX 3060); wrong for Mac | `lsm setup` |
| `train_all_lsmoutput.sh` | Train CNN (150 ep, batch 64), TCN (120, 16), 3D CNN (100, 12) on LSMOutput (`--min_videos 2`, 307 classes), resume if possible, export Core ML, log to `reports/training.log` | `lsm train` / `lsm pipeline` |
| `train_m4.sh` | Same three models on Apple silicon (MPS) with auto venv creation; `smoke` mode = 1-epoch CNN on `Colores` | `lsm train` |
| `train_m4_tiers.sh` | The three models at `--min_videos` 12, 7, 5 (nested vocabularies, 28 / 54 / 286 classes), then export each | `lsm prepare --min-videos` + `lsm train` |
| `train_lab_cap5.sh` | The lab cap-5 v2 cycle: prepare v2 and v1 caches, two 20-epoch CNN ablations, full CNN (150 ep) and TCN (120 ep), Core ML with parity gate, copy to the app; hardcoded `/Users/emilio/...` dataset paths | `Runs/lab_cap5/reproduce.sh` (same steps via `./lsm`) |

To see exactly how the cap-5 runs were produced, `train_lab_cap5.sh` is the original record; the `./lsm` equivalent is documented in [`Runs/lab_cap5/reproduce.sh`](../../Runs/lab_cap5/reproduce.sh).
