"""
LSM Phase 1 — Preprocessing: shoulder anchor, valid frames, interpolation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from constants import (
    FEATURE_DIM,
    MIN_VALID_FRAMES,
    N_FRAMES,
    N_LEFT_HAND,
    N_POSE,
    N_RIGHT_HAND,
    POSE_OFFSET,
    LEFT_SHOULDER_IDX,
    RIGHT_SHOULDER_IDX,
    VISIBILITY_BIN_THRESHOLD,
    VISIBILITY_THRESHOLD,
)

# Reuse TrainingPipeline helpers (parent package)
_TP = Path(__file__).resolve().parent.parent
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))

from dataset import frame_to_vector, interpolate_sequence  # noqa: E402


def _shoulder_base(idx: int) -> int:
    return POSE_OFFSET + idx * 3


def compute_anchor(vec: np.ndarray) -> tuple[float, float] | None:
    """Midpoint of visible shoulders, or single shoulder if only one visible."""
    ls = _shoulder_base(LEFT_SHOULDER_IDX)
    rs = _shoulder_base(RIGHT_SHOULDER_IDX)

    left_vis = vec[ls + 2] > VISIBILITY_THRESHOLD and (vec[ls] != 0 or vec[ls + 1] != 0)
    right_vis = vec[rs + 2] > VISIBILITY_THRESHOLD and (vec[rs] != 0 or vec[rs + 1] != 0)

    if left_vis and right_vis:
        return (float((vec[ls] + vec[rs]) / 2), float((vec[ls + 1] + vec[rs + 1]) / 2))
    if left_vis:
        return (float(vec[ls]), float(vec[ls + 1]))
    if right_vis:
        return (float(vec[rs]), float(vec[rs + 1]))
    return None


def _count_nonzero_pose_xy(vec: np.ndarray) -> int:
    count = 0
    for i in range(17):
        base = POSE_OFFSET + i * 3
        if vec[base] != 0 or vec[base + 1] != 0:
            count += 1
    return count


def _hand_has_points(vec: np.ndarray, offset: int, dim: int) -> bool:
    segment = vec[offset: offset + dim]
    return bool(np.any(segment != 0))


def is_valid_frame(vec: np.ndarray) -> bool:
    anchor = compute_anchor(vec)
    if anchor is None:
        return False

    has_left = _hand_has_points(vec, 0, N_LEFT_HAND)
    has_right = _hand_has_points(vec, N_LEFT_HAND, N_RIGHT_HAND)
    has_pose = _count_nonzero_pose_xy(vec) >= 3

    return has_left or has_right or has_pose


def normalize_frame(vec: np.ndarray) -> np.ndarray | None:
    """
    Shoulder-relative normalization + visibility binarization.
    Returns None if anchor cannot be computed.
    """
    out = vec.copy().astype(np.float32)
    anchor = compute_anchor(out)
    if anchor is None:
        return None

    ax, ay = anchor

    # Hands x,y
    for i in range(N_LEFT_HAND // 2):
        base = i * 2
        if out[base] != 0 or out[base + 1] != 0:
            out[base] -= ax
            out[base + 1] -= ay

    for i in range(N_RIGHT_HAND // 2):
        base = N_LEFT_HAND + i * 2
        if out[base] != 0 or out[base + 1] != 0:
            out[base] -= ax
            out[base + 1] -= ay

    # Pose x,y + binarized visibility
    for i in range(17):
        base = POSE_OFFSET + i * 3
        if out[base] != 0 or out[base + 1] != 0:
            out[base] -= ax
            out[base + 1] -= ay
        vis = out[base + 2]
        out[base + 2] = 1.0 if vis >= VISIBILITY_BIN_THRESHOLD else 0.0

    return out


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Apply per-frame normalization; invalid frames become zeros."""
    out = np.zeros_like(seq, dtype=np.float32)
    for t in range(len(seq)):
        normed = normalize_frame(seq[t])
        if normed is not None:
            out[t] = normed
    return out


def filter_valid_frames(seq: np.ndarray) -> np.ndarray:
    """Keep only frames passing the valid-frame heuristic."""
    valid = [seq[t] for t in range(len(seq)) if is_valid_frame(seq[t])]
    if not valid:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack(valid, axis=0)


def load_raw_sequence(json_path: str) -> np.ndarray:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data if isinstance(data, list) else data.get("frames", [])
    if not frames:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.stack([frame_to_vector(f) for f in frames], axis=0)


def prepare_sequence(
    json_path: str | None = None,
    raw_seq: np.ndarray | None = None,
    target_len: int = N_FRAMES,
) -> np.ndarray | None:
    """
    Full pipeline: load → filter valid → normalize → interpolate.
    Returns (target_len, 135) or None if insufficient valid frames.
    """
    if raw_seq is None:
        if json_path is None:
            raise ValueError("Provide json_path or raw_seq")
        raw_seq = load_raw_sequence(json_path)

    if len(raw_seq) == 0:
        return None

    filtered = filter_valid_frames(raw_seq)
    if len(filtered) < MIN_VALID_FRAMES:
        return None

    normalized = normalize_sequence(filtered)
    return interpolate_sequence(normalized, target_len)
