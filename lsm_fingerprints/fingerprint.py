"""
LSM Phase 1 — Statistical fingerprint extractor (135 × 6 = 810).
"""

from __future__ import annotations

import numpy as np

from constants import FEATURE_DIM, FINGERPRINT_DIM, N_FRAMES
from preprocess import prepare_sequence


class FingerprintExtractor:
    """
    Extrae huella estadística de una secuencia de landmarks.
    Entrada:  sequence (T, 135) — debe estar preprocesada e interpolada a N_FRAMES
    Salida:   fingerprint (810,) — normalizada L2
    """

    def extract(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
            raise ValueError(f"Expected (T, {FEATURE_DIM}), got {sequence.shape}")

        stats = []
        for ch in range(FEATURE_DIM):
            canal = sequence[:, ch].astype(np.float64)
            stats.extend([
                float(np.mean(canal)),
                float(np.std(canal)),
                float(np.min(canal)),
                float(np.max(canal)),
                float(np.percentile(canal, 25)),
                float(np.percentile(canal, 75)),
            ])

        fp = np.array(stats, dtype=np.float32)
        norm = np.linalg.norm(fp)
        if norm > 0:
            fp = fp / norm
        return fp

    def extract_from_json(self, json_path: str) -> np.ndarray | None:
        seq = prepare_sequence(json_path=json_path, target_len=N_FRAMES)
        if seq is None:
            return None
        return self.extract(seq)
