# Models

Deployable model artifacts, one set per training run. Training history and metrics stay in [`../Runs/`](../Runs/README.md); superseded models are in [`../Legacy/Models`](../Legacy/Models/README.md).

| Folder | Content | Produced by |
|---|---|---|
| [`Pytorch/`](./Pytorch/README.md) | `<run>.pt`, the best checkpoint of a run | `lsm train` copies `Runs/<run>/best_model.pt` here |
| [`CoreML/`](./CoreML/README.md) | `<run>.mlpackage`, `<run>_classes.json`, `<run>_fixture.json` | `lsm convert` (`CoreMLConversion/convert_to_coreml.py`) |

```
Runs/<run>/best_model.pt ─► Models/Pytorch/<run>.pt ─► Models/CoreML/<run>.mlpackage ─► TestingTools app (lsm deploy)
```

- The file stem is always the run name (`lab_cap5_cnn`, `lab_cap5_tcn`, `lab_cap5_ablation_fixed`, `lab_cap5_ablation_legacy`).
- Weights and `.mlpackage` contents are stored with **Git LFS** (`Models/Pytorch/*.pt`, `**/*.mlpackage/**`); run `git lfs pull` after cloning. Intermediate checkpoints (`last.pt`, `checkpoint_epoch*.pt`) are not kept here.
- Current deployable set (lab cap-5, layout `v2_162`, 518 classes): CNN (val 36.68% / extra 26.63%) and TCN (val 33.20% / extra 23.81%). Core ML exists only for these two; the ablation `.pt` files are for reference.
