"""
Golden tests for LSM Phase 1 fingerprint pipeline.
Run: cd TrainingPipeline/lsm_fingerprints && python -m pytest tests/ -v
Or:  cd TrainingPipeline/lsm_fingerprints && python tests/test_all.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import FEATURE_DIM, FINGERPRINT_DIM, N_FRAMES, POSE_OFFSET
from preprocess import (
    compute_anchor,
    is_valid_frame,
    normalize_frame,
    prepare_sequence,
)
from fingerprint import FingerprintExtractor
from export_bin import export_bin, load_bin
from build_db import add_class, build_database, load_database
from matcher import FingerprintMatcher


def _make_valid_frame(
    shoulder_x: float = 0.5,
    shoulder_y: float = 0.4,
    hand_offset: float = 0.05,
) -> np.ndarray:
    """Synthetic frame with visible shoulders and left hand."""
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    ls = POSE_OFFSET + 11 * 3
    rs = POSE_OFFSET + 12 * 3
    vec[ls] = shoulder_x - 0.05
    vec[ls + 1] = shoulder_y
    vec[ls + 2] = 0.9
    vec[rs] = shoulder_x + 0.05
    vec[rs + 1] = shoulder_y
    vec[rs + 2] = 0.9
    vec[0] = shoulder_x + hand_offset
    vec[1] = shoulder_y + hand_offset
    return vec


def _make_sequence(n: int = 30, class_offset: float = 0.0) -> np.ndarray:
    frames = []
    for t in range(n):
        f = _make_valid_frame(hand_offset=0.05 + class_offset + 0.001 * t)
        frames.append(f)
    return np.stack(frames, axis=0)


class TestPreprocess(unittest.TestCase):
    def test_anchor_computed(self):
        frame = _make_valid_frame()
        anchor = compute_anchor(frame)
        self.assertIsNotNone(anchor)
        self.assertAlmostEqual(anchor[0], 0.5, places=2)

    def test_normalize_subtracts_anchor(self):
        frame = _make_valid_frame()
        normed = normalize_frame(frame)
        self.assertIsNotNone(normed)
        ls = POSE_OFFSET + 11 * 3
        self.assertAlmostEqual(normed[ls], -0.05, places=3)
        self.assertAlmostEqual(normed[rs := POSE_OFFSET + 12 * 3], 0.05, places=3)

    def test_valid_frame_heuristic(self):
        self.assertTrue(is_valid_frame(_make_valid_frame()))

    def test_prepare_sequence_shape(self):
        seq = _make_sequence(30)
        from preprocess import filter_valid_frames, normalize_sequence, interpolate_sequence

        filtered = filter_valid_frames(seq)
        self.assertGreaterEqual(len(filtered), 10)
        normed = normalize_sequence(filtered)
        out = interpolate_sequence(normed, N_FRAMES)
        self.assertEqual(out.shape, (N_FRAMES, FEATURE_DIM))


class TestFingerprint(unittest.TestCase):
    def test_fingerprint_dim_and_normalized(self):
        from preprocess import filter_valid_frames, normalize_sequence, interpolate_sequence

        seq = _make_sequence(30)
        filtered = filter_valid_frames(seq)
        normed = normalize_sequence(filtered)
        fixed = interpolate_sequence(normed, N_FRAMES)

        fp = FingerprintExtractor().extract(fixed)
        self.assertEqual(fp.shape, (FINGERPRINT_DIM,))
        norm = np.linalg.norm(fp)
        self.assertAlmostEqual(norm, 1.0, places=5)


class TestBinRoundtrip(unittest.TestCase):
    def test_export_load(self):
        n = 3
        fps = np.random.randn(n, FINGERPRINT_DIM).astype(np.float32)
        fps /= np.linalg.norm(fps, axis=1, keepdims=True)
        labels = np.array([0, 0, 1], dtype=np.uint32)
        seqs = np.random.randn(n, N_FRAMES, FEATURE_DIM).astype(np.float32)
        classes = ["Agua", "Perro"]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.bin"
            export_bin(fps, labels, seqs, classes, path)
            loaded = load_bin(path)

        np.testing.assert_allclose(loaded["fingerprints"], fps)
        np.testing.assert_array_equal(loaded["labels"], labels)
        np.testing.assert_allclose(loaded["sequences"], seqs)
        self.assertEqual(loaded["classes"], classes)


class TestMatcher(unittest.TestCase):
    def test_match_identical_sequence(self):
        from preprocess import filter_valid_frames, normalize_sequence, interpolate_sequence

        seq_a = interpolate_sequence(normalize_sequence(filter_valid_frames(_make_sequence(30, 0.0))), N_FRAMES)
        seq_b = interpolate_sequence(normalize_sequence(filter_valid_frames(_make_sequence(30, 0.5))), N_FRAMES)

        extractor = FingerprintExtractor()
        fps = np.stack([extractor.extract(seq_a), extractor.extract(seq_a), extractor.extract(seq_b)])
        labels = np.array([0, 0, 1], dtype=np.int64)
        seqs = np.stack([seq_a, seq_a, seq_b])
        classes = ["Agua", "Perro"]

        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "db.npz"
            np.savez_compressed(
                npz,
                fingerprints=fps,
                labels=labels,
                sequences=seqs,
                classes=np.array(classes, dtype=object),
                categories=np.array(["", "", ""], dtype=object),
                metadata='{}',
            )
            matcher = FingerprintMatcher(str(npz))
            results = matcher.match(seq_a)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["class"], "Agua")


class TestBuildDbIntegration(unittest.TestCase):
    def test_build_from_synthetic_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            word_dir = tmp / "Familia" / "Agua"
            word_dir.mkdir(parents=True)

            for i in range(2):
                frames = []
                for t in range(25):
                    vec = _make_valid_frame(hand_offset=0.05 + 0.001 * t)
                    lh = [{"x": float(vec[j]), "y": float(vec[j + 1])} for j in range(0, 42, 2)]
                    pose = []
                    for pid in range(17):
                        base = POSE_OFFSET + pid * 3
                        pose.append({
                            "id": pid,
                            "x": float(vec[base]),
                            "y": float(vec[base + 1]),
                            "visibility": float(vec[base + 2]),
                        })
                    frames.append({
                        "frame": t,
                        "timestamp_ms": t * 33.0,
                        "leftHand": lh,
                        "rightHand": None,
                        "pose": pose,
                    })
                json_path = word_dir / f"Agua{i + 1}_landmarks.json"
                json_path.write_text(json.dumps(frames), encoding="utf-8")

            out_npz = tmp / "fingerprints.npz"
            out_bin = tmp / "fingerprints.bin"
            meta = build_database(str(tmp), str(out_npz), min_videos=2, export_bin_path=str(out_bin))

            self.assertGreaterEqual(meta["n_videos"], 2)
            self.assertTrue(out_bin.exists())

            db = load_database(out_npz)
            self.assertEqual(db["fingerprints"].shape[1], FINGERPRINT_DIM)

            matcher = FingerprintMatcher(str(out_npz))
            results = matcher.match(db["sequences"][0])
            self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()
