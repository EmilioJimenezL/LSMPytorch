# lab_cap5 (cycle)

No weights here; this folder summarizes the whole lab cap-5 experiment cycle (2026-11-22). The full write-up is [`REPORT.md`](./REPORT.md); `reproduce.sh` lists the commands. `training_curves.png` plots validation top-1 for the previous uncapped ≥5 cycle (left) and the cap-5 runs including the ablations (right); `REPORT.md` links it under an older name, `training_curves_lab_cap5.png`.

## The cycle

Goal: retrain the CNN and TCN on the **v2_162** landmark layout with a **balanced** dataset (cap of 5 videos per class) and a held-out `extra_test`, and see whether the fixed pipeline (GPU augmentation, attention pool, non-causal TCN, warmup, early stopping) beats the previous v1 setup.

1. **Prepare** the v2 and v1 caches (`min_videos 5`, `max_videos_per_class 5`, seed 42): 3,774 videos extracted, 3,619 valid, 518 classes (`Rosa` disambiguated), 41 words excluded. Split: train 2,072 / val 518 / extra_test 1,029. All runs share this exact split.
2. **Ablation** (CNN, 20 epochs each): [`lab_cap5_ablation_legacy`](../lab_cap5_ablation_legacy/README.md) (v1 + static augmentation) vs [`lab_cap5_ablation_fixed`](../lab_cap5_ablation_fixed/README.md) (v2 + GPU augmentation + attention pool). Legacy wins at 20 epochs (24.71% vs 17.18% val) because it memorizes the small train set faster; the fixed path is deliberately harder.
3. **Full runs** with early stopping (patience 15): [`lab_cap5_cnn`](../lab_cap5_cnn/README.md) 36.68% val / 26.63% extra, [`lab_cap5_tcn`](../lab_cap5_tcn/README.md) 33.20% val / 23.81% extra. The fixed CNN beats the 20-epoch legacy baseline on both splits.
4. **Convert + parity gate** to Core ML (`Models/CoreML/lab_cap5_{cnn,tcn}.mlpackage`): max |Δlogit| 0.017 (CNN) / 0.018 (TCN), 100% top-1 agreement over 256 validation sequences. PASS.
5. **Deploy** to the iOS app as sign set `labCap5` (518 classes); `labMin5` (517 classes, v1) keeps the previous cycle's models. Simulator parity passed for both fixtures.

## Takeaways

- Cap-5 numbers are lower than the previous uncapped ≥5 cycle (CNN 47.89%, TCN 48.57% val, v1) because each class has 4 training videos. That is the cost of a balanced set and a clean `extra_test`; the two val sets are not comparable.
- Train accuracy reaches ~98–99% while val stays at 33–37%: overfitting on 4 videos/class is the main limit. `extra_test` (never seen, never used for stopping) is the better estimate of generalization: 26.6% CNN, 23.8% TCN.
- Random splits can put the same signer in train and val, so these numbers likely overstate accuracy for new signers.
- Next ideas are listed in `REPORT.md` (rotating balanced sampling, logit adjustment, 5-fold CV, self-supervised pretraining, signer-held-out splits).

## Reproduce

`./reproduce.sh` (documents the `./lsm` commands; see the header for dataset paths and options).
