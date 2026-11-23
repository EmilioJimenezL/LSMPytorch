"""
Export / import fingerprints.bin for Swift (LSMF binary format).
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from constants import (
    BIN_HEADER_SIZE,
    BIN_MAGIC,
    BIN_VERSION,
    FEATURE_DIM,
    FINGERPRINT_DIM,
    N_FRAMES,
)


def export_bin(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    classes: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    n_videos = fingerprints.shape[0]
    n_classes = len(classes)

    header = struct.pack(
        "<IIIIII",
        BIN_MAGIC,
        BIN_VERSION,
        n_videos,
        n_classes,
        FINGERPRINT_DIM,
        N_FRAMES,
    )
    header += struct.pack("<I", FEATURE_DIM)
    header += b"\x00" * (BIN_HEADER_SIZE - len(header))

    fp_bytes = fingerprints.astype(np.float32).tobytes(order="C")
    label_bytes = labels.astype(np.uint32).tobytes(order="C")
    seq_bytes = sequences.astype(np.float32).tobytes(order="C")

    class_bytes = b""
    for name in classes:
        encoded = name.encode("utf-8")
        class_bytes += struct.pack("<H", len(encoded))
        class_bytes += encoded

    output_path.write_bytes(header + fp_bytes + label_bytes + seq_bytes + class_bytes)


def load_bin(path: str | Path) -> dict:
    """Load fingerprints.bin (for round-trip tests and Python verification)."""
    data = Path(path).read_bytes()
    if len(data) < BIN_HEADER_SIZE:
        raise ValueError("File too small for header")

    magic, version, n_videos, n_classes, fp_dim, n_frames, feat_dim = struct.unpack(
        "<IIIIIII", data[:28]
    )
    if magic != BIN_MAGIC:
        raise ValueError(f"Invalid magic: {hex(magic)}")
    if version != BIN_VERSION:
        raise ValueError(f"Unsupported version: {version}")

    offset = BIN_HEADER_SIZE
    fp_size = n_videos * fp_dim * 4
    label_size = n_videos * 4
    seq_size = n_videos * n_frames * feat_dim * 4

    fingerprints = np.frombuffer(
        data[offset: offset + fp_size], dtype=np.float32
    ).reshape(n_videos, fp_dim)
    offset += fp_size

    labels = np.frombuffer(
        data[offset: offset + label_size], dtype=np.uint32
    )
    offset += label_size

    sequences = np.frombuffer(
        data[offset: offset + seq_size], dtype=np.float32
    ).reshape(n_videos, n_frames, feat_dim)
    offset += seq_size

    classes = []
    for _ in range(n_classes):
        if offset + 2 > len(data):
            break
        (length,) = struct.unpack("<H", data[offset: offset + 2])
        offset += 2
        name = data[offset: offset + length].decode("utf-8")
        offset += length
        classes.append(name)

    return {
        "fingerprints": fingerprints,
        "labels": labels,
        "sequences": sequences,
        "classes": classes,
        "metadata": {
            "n_videos": n_videos,
            "n_classes": n_classes,
            "n_frames": n_frames,
            "feature_dim": feat_dim,
            "fingerprint_dim": fp_dim,
        },
    }
