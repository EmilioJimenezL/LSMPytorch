# LandmarkExtraction — agent notes

macOS-only Vision landmark extractors. Prefer **LSMExtractorCLI** for batch jobs (`./lsm extract`). Keep CLI and GUI engine files in sync (`Utils/check_extractor_sync.py`).

Schema v2 JSON is the contract with `TrainingPipeline/features.py` and the iOS app. Do not change joint order or confidence semantics without updating all three.
