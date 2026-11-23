"""
LSM Phase 1 — Two-stage matcher: cosine filter + FastDTW + class voting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from build_db import load_database
from constants import (
    CONFIDENCE_THR,
    DTW_RADIUS,
    FEATURE_DIM,
    N_FRAMES,
    TOP_K_DTW,
    TOP_K_FILTER,
)
from fingerprint import FingerprintExtractor
from preprocess import prepare_sequence


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors assumed L2-normalized."""
    dot = float(np.dot(a, b))
    return 1.0 - dot


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Multivariate DTW using Euclidean per-frame distance."""
    distance, _ = fastdtw(seq_a, seq_b, dist=euclidean, radius=DTW_RADIUS)
    return float(distance)


class FingerprintMatcher:
    """
    Carga fingerprints.npz y realiza búsqueda en dos etapas.
    """

    def __init__(self, db_path: str):
        db = load_database(db_path)
        self.fingerprints: np.ndarray = db["fingerprints"]
        self.labels: np.ndarray = db["labels"]
        self.sequences: np.ndarray = db["sequences"]
        self.classes: list[str] = db["classes"]
        self.extractor = FingerprintExtractor()
        self.db_path = Path(db_path)

    def match(
        self,
        query_sequence: np.ndarray,
        top_k_filter: int = TOP_K_FILTER,
        top_k_dtw: int = TOP_K_DTW,
    ) -> list[dict]:
        if query_sequence.shape != (N_FRAMES, FEATURE_DIM):
            raise ValueError(f"Query must be ({N_FRAMES}, {FEATURE_DIM})")

        query_fp = self.extractor.extract(query_sequence)

        # Stage 1: cosine filter
        dists = np.array([
            _cosine_distance(query_fp, self.fingerprints[i])
            for i in range(len(self.fingerprints))
        ])
        filter_k = min(top_k_filter, len(dists))
        candidate_idx = np.argsort(dists)[:filter_k]

        # Stage 2: DTW on candidates
        dtw_results = []
        for idx in candidate_idx:
            dtw_dist = _dtw_distance(query_sequence, self.sequences[idx])
            dtw_results.append((idx, dtw_dist))

        dtw_results.sort(key=lambda x: x[1])
        top_dtw = dtw_results[:top_k_dtw]

        # Stage 3: class voting
        class_scores: dict[int, float] = {}
        for idx, dtw_dist in top_dtw:
            label = int(self.labels[idx])
            score = 1.0 / (dtw_dist + 1e-6)
            class_scores[label] = class_scores.get(label, 0.0) + score

        total = sum(class_scores.values()) or 1.0
        ranked = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for rank, (label, score) in enumerate(ranked[:top_k_dtw]):
            norm_score = score / total
            if norm_score < CONFIDENCE_THR and rank > 0:
                continue
            results.append({
                "class": self.classes[label],
                "score": float(norm_score),
                "rank": rank + 1,
                "label_idx": label,
            })

        return results

    def match_json(self, json_path: str, **kwargs) -> list[dict]:
        seq = prepare_sequence(json_path=json_path)
        if seq is None:
            return []
        return self.match(seq, **kwargs)

    def add_class(
        self,
        class_name: str,
        sequences: list[np.ndarray],
        category: str = "",
        export_bin_path: str | None = None,
    ) -> None:
        from build_db import add_class as _add_class

        _add_class(
            str(self.db_path),
            class_name,
            sequences,
            category=category,
            export_bin_path=export_bin_path,
        )
        # Reload
        db = load_database(self.db_path)
        self.fingerprints = db["fingerprints"]
        self.labels = db["labels"]
        self.sequences = db["sequences"]
        self.classes = db["classes"]
