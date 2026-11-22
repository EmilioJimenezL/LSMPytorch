#!/usr/bin/env bash
# Fixed CNN + TCN cycle: prepare v2 cap5 cache, ablation, full trains, export → app.
set -euo pipefail
cd "$(dirname "$0")"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1
PY="${PWD}/.venv/bin/python"
DS_V2="/Users/emilio/Desktop/Universidad/LSM_LAB_OUTPUT_V2/landmarks"
DS_V1="/Users/emilio/Desktop/Universidad/LSM_LAB_OUTPUT/landmarks"
CACHE_V2="prepared/lab_v2_cap5.npz"
CACHE_V1="prepared/lab_v1_cap5.npz"
LOG="reports/lab_cap5_training.log"
mkdir -p prepared reports exports

{
  echo "=== 1. Prepare v2 cap5 cache ==="
  "$PY" scripts/prepare_dataset.py \
    --dataset "$DS_V2" \
    --output "$CACHE_V2" \
    --layout v2_162 \
    --min-videos 5 \
    --max-videos-per-class 5

  echo "=== 1b. Prepare v1 cap5 cache (ablation baseline) ==="
  "$PY" scripts/prepare_dataset.py \
    --dataset "$DS_V1" \
    --output "$CACHE_V1" \
    --layout v1_135 \
    --min-videos 5 \
    --max-videos-per-class 5

  echo "=== 2. Ablation: CNN legacy (v1 + legacy-augment), 20 epochs ==="
  "$PY" train.py \
    --model cnn --cache "$CACHE_V1" --layout v1_135 \
    --epochs 20 --batch 64 --augment 10 --legacy-augment \
    --warmup 0 --patience 0 --pool avg \
    --output ./runs_lab_cap5_ablation_legacy \
    --resume ./runs_lab_cap5_ablation_legacy/last.pt

  echo "=== 2b. Ablation: CNN fixed (v2 + GPU aug + attn), 20 epochs ==="
  "$PY" train.py \
    --model cnn --cache "$CACHE_V2" --layout v2_162 \
    --epochs 20 --batch 64 --augment 10 \
    --warmup 5 --patience 0 --pool attn --joint_dropout 0.1 \
    --output ./runs_lab_cap5_ablation_fixed \
    --resume ./runs_lab_cap5_ablation_fixed/last.pt

  echo "=== 3. CNN full (v2 fixed), 150 epochs ==="
  "$PY" train.py \
    --model cnn --cache "$CACHE_V2" --layout v2_162 \
    --epochs 150 --batch 64 --augment 10 \
    --warmup 10 --patience 15 --pool attn --joint_dropout 0.1 \
    --output ./runs_lab_cap5_cnn \
    --resume ./runs_lab_cap5_cnn/last.pt

  echo "=== 4. TCN full (non-causal + attn), 120 epochs ==="
  "$PY" train.py \
    --model tcn --cache "$CACHE_V2" --layout v2_162 \
    --epochs 120 --batch 16 --augment 10 \
    --warmup 10 --patience 15 --pool attn --causal 0 --joint_dropout 0.1 \
    --output ./runs_lab_cap5_tcn \
    --resume ./runs_lab_cap5_tcn/last.pt

  echo "=== 5. Export CNN + TCN with parity gate ==="
  "$PY" convert_to_coreml.py \
    --checkpoint ./runs_lab_cap5_cnn/best_model.pt \
    --output ./exports/runs_lab_cap5_cnn_lsmoutput \
    --cache "$CACHE_V2" --parity \
    --fixture ./exports/runs_lab_cap5_cnn_fixture.json

  "$PY" convert_to_coreml.py \
    --checkpoint ./runs_lab_cap5_tcn/best_model.pt \
    --output ./exports/runs_lab_cap5_tcn_lsmoutput \
    --cache "$CACHE_V2" --parity \
    --fixture ./exports/runs_lab_cap5_tcn_fixture.json

  echo "=== 6. Copy into iOS app ==="
  "$PY" scripts/export_to_app.py \
    --mlpackage ./exports/runs_lab_cap5_cnn_lsmoutput.mlpackage \
    --classes ./runs_lab_cap5_cnn/classes.json \
    --run-stem runs_lab_cap5_cnn \
    --fixture ./exports/runs_lab_cap5_cnn_fixture.json \
    --name "CNN — Lab cap5 (v2)" --family cnn --sign-set labCap5

  "$PY" scripts/export_to_app.py \
    --mlpackage ./exports/runs_lab_cap5_tcn_lsmoutput.mlpackage \
    --classes ./runs_lab_cap5_tcn/classes.json \
    --run-stem runs_lab_cap5_tcn \
    --fixture ./exports/runs_lab_cap5_tcn_fixture.json \
    --name "TCN — Lab cap5 (v2)" --family tcn --sign-set labCap5

  echo LAB_CAP5_DONE
} 2>&1 | tee "$LOG"
