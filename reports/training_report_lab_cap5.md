# Training report — LSM Videos Lab cap-5 (v2 layout)

**Date:** 2026-11-22  
**Script:** `train_lab_cap5.sh`  
**Cache:** `prepared/lab_v2_cap5.npz` (hash `1849851f3acc5842`)  
**Curves:** [`training_curves_lab_cap5.png`](./training_curves_lab_cap5.png)

## Setup

| Item | Value |
|---|---|
| Source videos | `LSM_VIDEOS_LAB` → 3,774 extracted to `LSM_LAB_OUTPUT_V2` (schema v2) |
| Layout | `v2_162` (hands 21×(x,y,c) ×2 + pose 12×(x,y,c)); Fase 1 / old models stay on `v1_135` |
| Filter | `--min_videos 5` on **valid** sequences → 518 classes (Rosa disambiguated), 41 excluded |
| Cap | `--max_videos_per_class 5` (seed 42) → train 2,072 / val 518 / **extra_test 1,029** |
| Device | Apple MPS |
| Augment | GPU batch, 10×/epoch, joint dropout 0.1; warmup 10, patience 15 |

## Ablation (CNN, 20 epochs, same cap)

| Setting | Layout | Augment | Val Top-1 | Extra Top-1 |
|---|---|---|---|---|
| Legacy | v1_135 | static CPU 10× (`--legacy-augment`) | **24.71%** | 15.06% |
| Fixed | v2_162 | GPU + attn pool | 17.18% | 13.80% |

At a matched 20-epoch budget the legacy path wins: it memorizes the tiny train set faster (train acc ~99% by epoch 10). The fixed path is intentionally harder (fresh GPU aug + warmup) and needs more epochs. **Full training of the fixed CNN reaches 36.68% val / 26.63% extra**, beating the 20-epoch legacy baseline on both splits.

## Full runs (this cycle)

| Model | Config | Epochs | Val Top-1 | Val Top-5 | Extra Top-1 | Extra Top-5 |
|---|---|---|---|---|---|---|
| **CNN** | v2, attn, causal N/A | 87 (early stop) | **36.68%** | 57.72% | **26.63%** | 52.96% |
| **TCN** | v2, non-causal, attn | 114 (early stop) | 33.20% | 53.09% | 23.81% | 46.45% |

Core ML parity (Python gate): CNN max|Δlogit|=0.017, TCN 0.018, top-1 agree 100% — **PASS**.  
iOS Debug simulator: Fase 1 parity ✓; Phase 2 Core ML parity ✓ (cpuOnly) for both fixtures. Class counts: labMin5 **517**, labCap5 **518**.

## Previous cycle (uncapped ≥5, v1) — for comparison

| Model | Epochs | Val Top-1 | Notes |
|---|---|---|---|
| CNN | 150 | **47.89%** | More videos/class → higher val; overfit risk |
| TCN | 120 | **48.57%** | Same |
| 3D CNN | ~74/100 | ~18.4% | Aborted; not continued |

Cap-5 numbers are lower than uncapped ≥5 because each class has only 4 train videos. That is the trade-off for balance and a held-out `extra_test`.

## Artifacts

| Path | Role |
|---|---|
| `runs_lab_cap5_cnn/`, `runs_lab_cap5_tcn/` | Checkpoints, `history.json`, `metrics.json`, curves |
| `exports/runs_lab_cap5_{cnn,tcn}_lsmoutput.mlpackage` | Core ML |
| App SignSet `labCap5` | New v2 models |
| App SignSet `labMin5` | Previous v1 lab CNN/TCN (517) |

## Getting more from all videos (without overfitting popular words)

This cycle uses only the first idea (`extra_test`). Candidates for the next experiment:

1. **Extras for eval / early stopping** — already done; steadier signal at no train risk.
2. **Rotating balanced sampling** — each epoch draw 5 different videos/class from all available. Same class weight every epoch; all videos used over time. Compare extra_test vs this cycle.
3. **Logit adjustment / balanced softmax** — if training on all videos, subtract log class prior at inference, or inverse-sqrt frequency loss weights.
4. **5-fold CV on the capped set** — every video validated once; optional fold ensemble.
5. **Self-supervised pretraining** — masked-landmark reconstruction on all 3,774 (incl. excluded words), then fine-tune on the balanced labeled set.
6. **Signer-/session-held-out splits** once metadata exists — today’s random split overstates new-signer accuracy.

Harness: same cache/seed, compare extra_test top-1 and per-class variance, plus live A/B via in-app grading CSV.
