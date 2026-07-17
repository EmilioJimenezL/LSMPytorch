"""Stratified train/val split utilities for LSM datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def stratified_split_indices(
    y: np.ndarray,
    meta: list[dict],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-class stratified split.
    - 2 videos  → 1 train / 1 val
    - 3+ videos → ~val_fraction val (minimum 1)
    """
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(y):
        by_class[int(label)].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []

    for _cls, indices in sorted(by_class.items()):
        indices = list(indices)
        rng.shuffle(indices)
        n = len(indices)
        if n == 2:
            val_idx.append(indices[0])
            train_idx.append(indices[1])
        else:
            n_val = max(1, int(round(n * val_fraction)))
            n_val = min(n_val, n - 1)
            val_idx.extend(indices[:n_val])
            train_idx.extend(indices[n_val:])

    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def export_split_manifest(
    meta: list[dict],
    y: np.ndarray,
    classes: list[str],
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    output_path: str | Path,
) -> None:
    """Write split_manifest.json for reproducible review."""
    val_set = set(int(i) for i in idx_val)
    entries = []
    for i, m in enumerate(meta):
        label = int(y[i])
        entries.append({
            "index": i,
            "split": "val" if i in val_set else "train",
            "class": m["clase"],
            "class_idx": label,
            "category": m.get("categoria", ""),
            "path": m["archivo"],
        })

    payload = {
        "n_total": len(meta),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_classes": len(classes),
        "classes": classes,
        "entries": entries,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
