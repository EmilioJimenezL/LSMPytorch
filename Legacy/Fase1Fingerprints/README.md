# Legacy/Fase1Fingerprints

**Fase 1: sign recognition without training.** Each video becomes an 810-dim statistical fingerprint; a query is matched with a cosine pre-filter, FastDTW on the shortlisted sequences, and class voting. Archived: it only supports the v1 layout and scored far below the neural models.

| Metric (LSMOutput, 307 classes, 1,934 videos, leave-one-out) | Result | Target |
|---|---|---|
| Top-1 | 6.57% | ≥ 40% (goal 60%) |
| Top-5 | 16.18% | ≥ 70% (goal 85%) |

For comparison, the neural CNN reached 47.16% val top-1 on the same vocabulary (different protocol: held-out split vs LOO).

## Layout

```
Fase1Fingerprints/
├── _paths.py             # puts lsm_fingerprints/ and TrainingPipeline on sys.path; defines IOS_APP
└── lsm_fingerprints/
    ├── constants.py      # dims and thresholds (135/810, 85 frames, TOP_K_FILTER 20, TOP_K_DTW 5, DTW radius 10); mirrored in Swift LSMConstants
    ├── preprocess.py     # shoulder normalization, valid-frame filter, 85-frame interpolation (v1)
    ├── fingerprint.py    # 810-dim L2-normalized fingerprint (135 × 6 statistics)
    ├── build_db.py       # landmark JSON → fingerprints.npz (+ optional fingerprints.bin); add_class() adds a word without retraining
    ├── export_bin.py     # writes fingerprints.bin (magic 0x4C534D46 "LSMF", version 1, 64-byte header) for the iOS app
    ├── matcher.py        # cosine filter + FastDTW + voting
    ├── evaluate.py       # leave-one-out evaluation, optional JSON report
    ├── export_fixtures.py, generate_sample_bin.py
    ├── tests/            # test_all.py, test_dataset_parity.py, fixtures/golden_sequence.json
    └── requirements.txt  # numpy, scipy, fastdtw
```

## Commands (from this folder)

```bash
pip install -r lsm_fingerprints/requirements.txt
python lsm_fingerprints/build_db.py --dataset <landmarks> --output lsm_fingerprints/fingerprints.npz \
  --export-bin lsm_fingerprints/fingerprints.bin --min-videos 2        # defaults: output ./fingerprints.npz, min-videos 2
python lsm_fingerprints/evaluate.py --db lsm_fingerprints/fingerprints.npz [--output report.json]
python lsm_fingerprints/generate_sample_bin.py                          # dev stub written into the iOS app folder
python lsm_fingerprints/tests/test_all.py
```

## Inputs and outputs

- **In:** nested or flat `*_landmarks.json` (v1 or v2 JSON; always converted to the 135-dim v1 vector).
- **Out:** `fingerprints.npz` (database), `fingerprints.bin` (iOS), evaluation JSON. `fingerprints.npz`, `fingerprints.bin`, and `runs_pilot/` are gitignored and regenerated locally.

## Notes

- The iOS app (`TestingTools`) still contains the Fase 1 matcher (`FingerprintExtractor.swift`, `FingerprintMatcher.swift`) and checks it against `golden_sequence.json` in DEBUG builds. Keep `constants.py` and `LSMConstants.swift` in sync if you ever touch either.
- `tests/test_dataset_parity.py` compares against a sibling `LSMOutput` folder and skips when it is absent.
