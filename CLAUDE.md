# LSM Multi-Model Training Project

## Project Overview
Implementing TCN (Temporal Convolutional Network) and 3D CNN models alongside existing 1D CNN baseline for Mexican Sign Language recognition.

## Current Structure
- `model.py` - Existing 1D CNN baseline
- `train.py` - Training script
- `dataset.py` - Data loading
- `convert_to_coreml.py` - Model export

## Target Structure (After Implementation)
```
models/
  ├── __init__.py          # Model factory
  ├── cnn1d.py            # Refactored 1D CNN
  ├── tcn.py              # NEW: TCN implementation
  └── cnn3d.py            # NEW: 3D CNN implementation

train_multimodel.py        # Unified trainer
compare_models.py          # Comparison framework
configs/
  ├── cnn1d_config.yaml
  ├── tcn_config.yaml
  └── cnn3d_config.yaml
```

## Key Implementation Details
- **Input:** (batch, 85, 135) - 85 frames, 135 features (skeleton joints)
- **Output:** (batch, 330) - 330 LSM word classes
- **Framework:** PyTorch + PyTorch Lightning
- **Training:** RTX 3060 Mobile, ~20 hours total

## Instructions for Claude
1. Implement in 5 phases (see AI_AGENT_PROMPT_MultiModel_Implementation.md)
2. Test after each phase before proceeding
3. Use code templates from AI_AGENT_CODE_TEMPLATES.md
4. Reference TCN_vs_3DCNN_Detailed_Analysis.md for architecture details
5. Ask for clarification if any requirement is ambiguous