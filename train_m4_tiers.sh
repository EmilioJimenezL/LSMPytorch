#!/usr/bin/env bash
# Nested video-count tiers on Apple Silicon.
# min 7 includes min 12; min 5 includes both. Same epochs and batches as train_m4.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DATASET="${DATASET:-$ROOT/../LSMOutput}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONUNBUFFERED=1

pick_python() {
  local c ver
  for c in python3.13 python3.12 python3.11 python3.10 \
           /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           /opt/homebrew/opt/python@3.13/bin/python3.13 \
           /opt/homebrew/opt/python@3.12/bin/python3.12 \
           /opt/homebrew/opt/python@3.11/bin/python3.11; do
    if [[ -x "$c" ]] || command -v "$c" >/dev/null 2>&1; then
      ver="$("$c" -c 'import sys; print(sys.version_info >= (3, 10))' 2>/dev/null || true)"
      if [[ "$ver" == "True" ]]; then
        echo "$c"
        return
      fi
    fi
  done
}

venv_is_new_enough() {
  [[ -x "$ROOT/.venv/bin/python" ]] || return 1
  "$ROOT/.venv/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
}

if ! venv_is_new_enough; then
  PY_BOOT="$(pick_python)"
  if [[ -z "$PY_BOOT" ]]; then
    echo "Se necesita Python 3.10 o posterior. En este Mac: brew install python@3.12" >&2
    exit 1
  fi
  rm -rf "$ROOT/.venv"
  "$PY_BOOT" -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
fi

PY="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/reports" "$ROOT/exports"

train_model() {
  local model=$1 epochs=$2 batch=$3 out=$4 min_videos=$5
  echo "=== Training $model min_videos=$min_videos ($epochs epochs, batch=$batch) → $out ==="
  if [[ -f "$out/best_model.pt" && -f "$out/checkpoint_epoch${epochs}.pt" ]]; then
    echo "Skip $model — already has checkpoint_epoch${epochs}.pt"
    return
  fi
  if [[ -f "$out/best_model.pt" ]]; then
    echo "Resuming from $out/best_model.pt"
    "$PY" train.py --model "$model" --dataset "$DATASET" --epochs "$epochs" \
      --batch "$batch" --output "$out" --min_videos "$min_videos" \
      --resume "$out/best_model.pt"
  else
    "$PY" train.py --model "$model" --dataset "$DATASET" --epochs "$epochs" \
      --batch "$batch" --output "$out" --min_videos "$min_videos"
  fi
}

export_model() {
  local out=$1
  local dest="$ROOT/exports/${out#./}_lsmoutput.mlpackage"
  if [[ -d "$dest" ]]; then
    echo "Skip export — $dest exists"
    return
  fi
  echo "=== Exporting Core ML: $out ==="
  "$PY" convert_to_coreml.py --checkpoint "$out/best_model.pt" --output "$dest"
}

train_tier() {
  local min_videos=$1
  train_model cnn   150 64 "./runs_m4_min${min_videos}_cnn"   "$min_videos"
  export_model "./runs_m4_min${min_videos}_cnn"
  train_model tcn   120 16 "./runs_m4_min${min_videos}_tcn"   "$min_videos"
  export_model "./runs_m4_min${min_videos}_tcn"
  train_model 3dcnn 100 12 "./runs_m4_min${min_videos}_3dcnn" "$min_videos"
  export_model "./runs_m4_min${min_videos}_3dcnn"
}

train_tier 12
train_tier 7
train_tier 5

echo "=== All tier training complete ==="
