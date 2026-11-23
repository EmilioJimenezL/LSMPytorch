# Legacy/Models

Weights and Core ML exports from the earlier cycles (v1 layout, 135 dims). Superseded by [`../../Models`](../../Models/README.md); not maintained. Files are stored with Git LFS (`Legacy/Models/**/*.pt`, `**/*.mlpackage/**`).

## CoreML/

| Package | Source | Classes | Val top-1 |
|---|---|---|---|
| `m4_min12_{cnn,tcn,3dcnn}.mlpackage` | Video-count tier ≥12 | 28 | 48.21% / 35.71% / 3.57% |
| `m4_min7_{cnn,tcn,3dcnn}.mlpackage` | Tier ≥7 | 54 | 52.44% / 39.02% / 15.85% |
| `m4_min5_{cnn,tcn,3dcnn}.mlpackage` | Tier ≥5 | 286 | 47.45% / 47.45% / 13.69% |
| `lab_min5_{cnn,tcn}.mlpackage` | Lab dataset ≥5 videos, uncapped (no 3D CNN export) | 517 | 47.89% / 48.57% |
| `coreml_models.mlpackage` | Original model from before the LSMOutput dataset | 187 | 9.68% |

(Order in the accuracy column is CNN / TCN / 3D CNN.) All take input `landmarks (1, 85, 135)` (v1). They predate the v2 layout; a package without `feature_layout` metadata is treated as `v1_135`.

## Pytorch/

`lab_min5_cnn.pt`, `lab_min5_tcn.pt`, `lab_min5_3dcnn.pt`: best checkpoints of the uncapped lab ≥5 cycle (3D CNN is from the aborted run). The tier and LSMOutput checkpoints are not stored here.

The iOS app still bundles these models (sign sets `min12`, `min7`, `min5`, `labMin5`, `legacy187`) under its own file names, e.g. `runs_m4_min12_cnn_lsmoutput.mlpackage`; class lists are the matching `*_classes.json` files in the app bundle and `Legacy/Runs/*/classes.json`.
