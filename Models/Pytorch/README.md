# Models/Pytorch

Best-checkpoint weights, one `<run>.pt` per run (Git LFS). Each is a copy of `Runs/<run>/best_model.pt`.

| File | Model | Layout | Classes | Val top-1 | Extra top-1 | Run |
|---|---|---|---|---|---|---|
| `lab_cap5_cnn.pt` | CNN, attn pool | v2_162 | 518 | 36.68% | 26.63% | [`lab_cap5_cnn`](../../Runs/lab_cap5_cnn/README.md) |
| `lab_cap5_tcn.pt` | TCN non-causal, attn pool | v2_162 | 518 | 33.20% | 23.81% | [`lab_cap5_tcn`](../../Runs/lab_cap5_tcn/README.md) |
| `lab_cap5_ablation_fixed.pt` | CNN, attn pool, 20 epochs | v2_162 | 518 | 17.18% | 13.80% | [`lab_cap5_ablation_fixed`](../../Runs/lab_cap5_ablation_fixed/README.md) |
| `lab_cap5_ablation_legacy.pt` | CNN, avg pool, 20 epochs, static aug | v1_135 | 518 | 24.71% | 15.06% | [`lab_cap5_ablation_legacy`](../../Runs/lab_cap5_ablation_legacy/README.md) |

## Checkpoint format

A pickled dict (load with `weights_only=False`): `epoch`, `model` (state dict), `optimizer`, `scheduler`, `best_val_acc`, `history`, `classes`, `n_frames` (85), `feature_dim`, `model_type` (`cnn`/`tcn`/`3dcnn`), `model_config` (`input_dim`, `layout`, `causal`, `pool`, `dropout`).

Older checkpoints may lack `model_config`; treat them as v1 (135), average pool, causal TCN.

## Load

```python
import sys, torch
sys.path.insert(0, "TrainingPipeline")
from models import TRAIN_MODEL_MAP, create_model

ckpt = torch.load("Models/Pytorch/lab_cap5_cnn.pt", map_location="cpu", weights_only=False)
cfg, kind = ckpt["model_config"], ckpt["model_type"]
kwargs = {"input_dim": cfg["input_dim"], "pool": cfg["pool"]}
if kind == "tcn":
    kwargs["causal"] = cfg["causal"]
model = create_model(TRAIN_MODEL_MAP[kind], len(ckpt["classes"]), **kwargs)
model.load_state_dict(ckpt["model"]); model.eval()
# input: float32 (batch, 85, cfg["input_dim"]); output: logits over ckpt["classes"]
```

Inputs must be prepared with the matching layout (`TrainingPipeline/features.py`, `prepare_sequence_v2` for `v2_162`).
