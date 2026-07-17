#!/usr/bin/env bash
# Run full Fase 1 + Fase 2 review pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATASET="${1:-../LSMOutput}"
REPORTS="$ROOT/reports"
mkdir -p "$REPORTS"

echo "=== Dataset inventory ==="
python3 scripts/dataset_report.py --dataset "$DATASET" --output "$REPORTS"

if [[ -f lsm_fingerprints/fingerprints.npz ]]; then
  if [[ -f "$REPORTS/fase1_loo.json" ]]; then
    echo "=== Fase 1 LOO (skipped, using existing fase1_loo.json) ==="
  else
    echo "=== Fase 1 LOO ==="
    python3 lsm_fingerprints/evaluate.py \
      --db lsm_fingerprints/fingerprints.npz \
      --output "$REPORTS/fase1_loo.json"
  fi
fi

FASE2_JSONS=()
for ckpt in runs_lsmoutput_*/best_model.pt; do
  [[ -f "$ckpt" ]] || continue
  name="$(basename "$(dirname "$ckpt")")"
  out="$REPORTS/fase2_${name}.json"
  echo "=== Fase 2 eval: $ckpt ==="
  python3 review/evaluate_nn.py \
    --checkpoint "$ckpt" \
    --dataset "$DATASET" \
    --output "$out"
  FASE2_JSONS+=("$out")
done

if [[ -f "$REPORTS/fase1_loo.json" && ${#FASE2_JSONS[@]} -gt 0 ]]; then
  echo "=== Comparison ==="
  python3 review/compare_models.py \
    --fase1 "$REPORTS/fase1_loo.json" \
    --fase2 "${FASE2_JSONS[@]}" \
    --output "$REPORTS/comparison.json"
  python3 review/error_analysis.py \
    --comparison "$REPORTS/comparison.json" \
    --output "$REPORTS/error_analysis.json" \
    | tee "$REPORTS/summary.txt"
fi

echo "=== Training report ==="
python3 review/generate_training_report.py --root "$ROOT" --reports "$REPORTS"

echo "Done. Reports in $REPORTS"
