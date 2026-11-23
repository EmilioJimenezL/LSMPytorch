---
name: lsm-development
description: Develop LSMPytorch architectures, feature layouts, and Swift↔Python parity. Use when adding models, changing layouts, or editing extractor/app code.
---

# LSM development

## Before changing code

1. Read [AGENTS.md](../../../AGENTS.md) invariants (layouts `v1_135` / `v2_162`, parity gate, extractor sync).
2. Prefer extending `TrainingPipeline/` and `TestingTools/`; never grow `Legacy/`.

## Adding an architecture

1. Implement under `TrainingPipeline/models/` and register in `models/__init__.py` (`MODEL_REGISTRY`, `TRAIN_MODEL_MAP`).
2. Accept `input_dim` and optional `pool` (`avg`/`attn`); keep legacy checkpoint load paths working.
3. Add unit tests in `TrainingPipeline/tests/`.
4. Wire `./lsm train --model <name>` (orchestrator model list in `Orchestrator/lsm_cli/tasks.py`).
5. Export via `./lsm convert` and confirm Core ML metadata includes `feature_layout`.

## Changing a feature layout

1. Update `TrainingPipeline/features.py` constants + vectorizers + `prepare_sequence_*`.
2. Mirror in Swift: `LSMConstants.swift`, preprocessors, `LandmarkPipeline` / extractor `Models.swift`.
3. Run `python Utils/check_extractor_sync.py` (CLI↔GUI).
4. `./lsm test` and a 2-epoch smoke train + convert with `--parity` on macOS.
5. Confirm Phase2ParityCheck on simulator uses `.cpuOnly`.

## Required checks before finishing

```bash
./lsm test
python Utils/check_extractor_sync.py
./lsm convert --run <smoke> --fixture   # macOS parity
```
