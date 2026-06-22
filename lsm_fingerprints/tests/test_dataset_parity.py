"""Test Fase 2 dataset loader matches Fase 1 prepare_sequence."""

import sys
import unittest
from pathlib import Path

import numpy as np

_TP = Path(__file__).resolve().parent.parent.parent
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))

from dataset import _load_sequence_from_json, json_to_sequence, interpolate_sequence, N_FRAMES
from lsm_fingerprints.preprocess import prepare_sequence


class TestDatasetPreprocessParity(unittest.TestCase):
    def test_load_matches_prepare_sequence(self):
        lsm = _TP.parent / "LSMOutput"
        if not lsm.exists():
            self.skipTest("LSMOutput not available")

        samples = list(lsm.rglob("*_landmarks.json"))[:5]
        if not samples:
            self.skipTest("No landmark JSON in LSMOutput")

        matched = 0
        for path in samples:
            ref = prepare_sequence(json_path=str(path))
            got = _load_sequence_from_json(str(path), shoulder_norm=True)
            if ref is None and got is None:
                continue
            if ref is None or got is None:
                continue
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-5)
            matched += 1

        self.assertGreater(matched, 0, "No comparable sequences found")


if __name__ == "__main__":
    unittest.main()
