#!/usr/bin/env python3
"""Generate a minimal fingerprints.bin for iOS bundle testing."""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from constants import FEATURE_DIM, FINGERPRINT_DIM, N_FRAMES, POSE_OFFSET
from export_bin import export_bin
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


def _make_sequence(n: int = 25, offset: float = 0.0) -> np.ndarray:
    frames = [_make_valid_frame(0.05 + offset + 0.001 * t) for t in range(n)]
    raw = np.stack(frames)
    filtered = filter_valid_frames(raw)
    normed = normalize_sequence(filtered)
    return interpolate_sequence(normed, N_FRAMES)


def main():
    _repo = Path(__file__).resolve().parent.parent.parent
    out = _repo / "AppLSMTests" / "LSMMobileModelTesting" / "LSMMobileModelTesting" / "fingerprints.bin"

    extractor = FingerprintExtractor()
    classes = ["Agua", "Perro"]
    seqs = [_make_sequence(offset=0.0), _make_sequence(offset=0.0), _make_sequence(offset=0.3)]
    fps = np.stack([extractor.extract(s) for s in seqs])
    labels = np.array([0, 0, 1], dtype=np.uint32)

    export_bin(fps, labels, np.stack(seqs), classes, out)
    print(f"Generated {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
