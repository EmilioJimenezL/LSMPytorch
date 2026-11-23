#!/usr/bin/env bash
# Train all Fase 2 models on LSMOutput (307 classes). CPU/GPU via PyTorch auto-detect.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/.venv/lib/python3.13/site-packages:$ROOT"

DATASET="${1:-../LSMOutput}"
LOG="$ROOT/reports/training.log"
mkdir -p "$ROOT/reports" "$ROOT/exports"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

train_model() {
  local model=$1 epochs=$2 batch=$3 out=$4
  log "=== Training $model ($epochs epochs, batch=$batch) → $out ==="
  if [[ -f "$out/best_model.pt" && -f "$out/checkpoint_epoch${epochs}.pt" ]]; then
    log "Skip $model — already has checkpoint_epoch${epochs}.pt"
    return
  fi
  local resume=()
  if [[ -f "$out/best_model.pt" ]]; then
    resume=(--resume "$out/best_model.pt")
    log "Resuming from $out/best_model.pt"
  fi
  python3 train.py --model "$model" --dataset "$DATASET" --epochs "$epochs" \
    --batch "$batch" --output "$out" --min_videos 2 "${resume[@]}" 2>&1 | tee -a "$LOG"
  log "=== Exporting Core ML: $out ==="
  python3 convert_to_coreml.py --checkpoint "$out/best_model.pt" \
    --output "$ROOT/exports/${out#./}_lsmoutput" 2>&1 | tee -a "$LOG"
}

train_model cnn   150 64 ./runs_lsmoutput_cnn
train_model tcn   120 16 ./runs_lsmoutput_tcn
train_model 3dcnn 100 12 ./runs_lsmoutput_3dcnn

log "=== All training complete ==="
