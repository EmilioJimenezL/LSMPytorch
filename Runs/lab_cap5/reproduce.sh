#!/usr/bin/env bash
# Reproduce the lab cap-5 cycle (REPORT.md, 2026-11-22) with the `lsm` orchestrator.
#
# This script DOCUMENTS the steps and runs them when executed. Read it first:
# training takes hours on Apple MPS (CNN ~87 epochs, TCN ~114 epochs with early stopping).
#
# Requirements:
#   * Landmarks extracted with schema v2 (macOS: `./lsm extract`, or copy an existing extraction).
#   * Landmarks for the v1 ablation baseline (same videos, v1-era extraction is fine: v1_135 can
#     also be built from v2 JSON, so one v2 folder is enough to re-run the ablation).
#   * Core ML parity needs macOS (Core ML `predict`); elsewhere stop after step 4.
#
# Usage:
#   LANDMARKS_V2=/path/to/LSM_LAB_OUTPUT_V2/landmarks ./Runs/lab_cap5/reproduce.sh
#   LANDMARKS_V2=... LANDMARKS_V1=/path/to/LSM_LAB_OUTPUT/landmarks ./Runs/lab_cap5/reproduce.sh
#
# Prepared caches go to Workspace/prepared/, runs to Runs/<name>/, weights to Models/Pytorch/,
# Core ML to Models/CoreML/. Add `--dry-run` to any `./lsm` call to print the underlying commands.
#
# Original script (pre-reorganization, hardcoded paths, calls train.py directly):
#   Legacy/Scripts/train_lab_cap5.sh
#
# NOTE: `lsm` is being implemented (Orchestrator/README.md is the command spec). The flags below
# follow that spec. If a flag is not available yet, use the equivalent direct call shown in the
# comments next to each step (TrainingPipeline/*.py, CoreMLConversion/*.py).
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

: "${LANDMARKS_V2:?set LANDMARKS_V2 to the schema-v2 landmarks folder}"
LANDMARKS_V1="${LANDMARKS_V1:-$LANDMARKS_V2}"

# --- 1. Prepare caches: >=5 valid videos per class, cap 5 per class, seed 42 -------------------
# Direct: python TrainingPipeline/prepare_dataset.py --dataset DIR --output Workspace/prepared/NAME.npz \
#           --layout v2_162 --min-videos 5 --max-videos-per-class 5
# Expected: 518 classes, train 2072 / val 518 / extra_test 1029 (hash 1849851f3acc5842 for v2).
./lsm prepare --landmarks "$LANDMARKS_V2" --layout v2_162 --min-videos 5 --cap 5 --seed 42 --name lab_v2_cap5
./lsm prepare --landmarks "$LANDMARKS_V1" --layout v1_135 --min-videos 5 --cap 5 --seed 42 --name lab_v1_cap5

# --- 2. Ablation: CNN, 20 epochs each -----------------------------------------------------------
# legacy = v1 + static CPU augmentation 10x, avg pool, no warmup, no early stopping  -> 24.71% val
./lsm train --model cnn --cache Workspace/prepared/lab_v1_cap5.npz --name lab_cap5_ablation_legacy \
  --epochs 20 --batch 64 --augment 10 --legacy-augment --warmup 0 --patience 0 --pool avg
# fixed  = v2 + GPU augmentation, attention pool, warmup 5, no early stopping       -> 17.18% val
./lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_ablation_fixed \
  --epochs 20 --batch 64 --augment 10 --warmup 5 --patience 0 --pool attn

# --- 3. Full runs (early stopping, patience 15) -------------------------------------------------
# CNN: 36.68% val / 26.63% extra_test (stopped at epoch 87)
./lsm train --model cnn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_cnn \
  --epochs 150 --batch 64 --augment 10 --warmup 10 --patience 15 --pool attn
# TCN (non-causal): 33.20% val / 23.81% extra_test (stopped at epoch 114)
./lsm train --model tcn --cache Workspace/prepared/lab_v2_cap5.npz --name lab_cap5_tcn \
  --epochs 120 --batch 16 --augment 10 --warmup 10 --patience 15 --pool attn --causal 0
# (joint dropout 0.1 is the train.py default.)

# --- 4. Core ML with parity gate (macOS) --------------------------------------------------------
# Gate: max|dlogit| <= 0.05 and top-1 agreement >= 99%. Observed 0.017 (CNN) / 0.018 (TCN), 100%.
./lsm convert --run lab_cap5_cnn
./lsm convert --run lab_cap5_tcn

# --- 5. Deploy into the iOS app (sign set labCap5) ----------------------------------------------
# Then paste the printed LSMModel(...) entries into AVAILABLE_MODELS (TestingTools README).
./lsm deploy --run lab_cap5_cnn --sign-set labCap5 --display-name "CNN — Lab cap5 (v2)"
./lsm deploy --run lab_cap5_tcn --sign-set labCap5 --display-name "TCN — Lab cap5 (v2)"

echo "Done. Compare Runs/lab_cap5_*/metrics.json with Runs/lab_cap5/REPORT.md"
