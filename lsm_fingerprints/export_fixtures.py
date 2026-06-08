#!/usr/bin/env python3
"""Export golden fixtures for Swift parity tests."""

import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
_TP = _PKG.parent
sys.path.insert(0, str(_PKG))
sys.path.insert(0, str(_TP / ".venv" / "lib" / "python3.13" / "site-packages"))

from constants import FEATURE_DIM, FINGERPRINT_DIM, N_FRAMES, POSE_OFFSET
from fingerprint import FingerprintExtractor
from preprocess import filter_valid_frames, interpolate_sequence, normalize_sequence


def _make_valid_frame(hand_offset: float = 0.05) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    ls = POSE_OFFSET + 11 * 3
    rs = POSE_OFFSET + 12 * 3
    vec[ls], vec[ls + 1], vec[ls + 2] = 0.45, 0.4, 0.9
    vec[rs], vec[rs + 1], vec[rs + 2] = 0.55, 0.4, 0.9
    vec[0], vec[1] = 0.5 + hand_offset, 0.45
    return vec


def main():
    frames = [_make_valid_frame(0.05 + 0.001 * t) for t in range(25)]
    raw = np.stack(frames)
    filtered = filter_valid_frames(raw)
    normed = normalize_sequence(filtered)
    seq = interpolate_sequence(normed, N_FRAMES)
    fp = FingerprintExtractor().extract(seq)

    out_dir = Path(__file__).resolve().parent / "tests" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture = {
        "n_frames": N_FRAMES,
        "feature_dim": FEATURE_DIM,
        "fingerprint_dim": FINGERPRINT_DIM,
        "sequence": seq.tolist(),
        "fingerprint": fp.tolist(),
    }
    path = out_dir / "golden_sequence.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
