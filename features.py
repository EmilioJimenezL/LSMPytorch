"""
LSM feature layouts and sequence preparation.

v1 (135): left hand 42 + right hand 42 + pose 51 (17×x,y,vis) — Fase 1 / legacy.
v2 (162): left hand 63 + right hand 63 + pose 36 (12×x,y,conf) — new models.
          Shoulder-centered and scaled by median shoulder width.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# ── Layout constants ──────────────────────────────────────────────────────────

N_FRAMES = 85
MIN_VALID_FRAMES = 10

# v1
N_LEFT_HAND_V1 = 21 * 2   # 42
N_RIGHT_HAND_V1 = 21 * 2  # 42
N_POSE_V1 = 17 * 3        # 51
FEATURE_DIM_V1 = N_LEFT_HAND_V1 + N_RIGHT_HAND_V1 + N_POSE_V1  # 135
POSE_OFFSET_V1 = N_LEFT_HAND_V1 + N_RIGHT_HAND_V1  # 84
LEFT_SHOULDER_IDX_V1 = 11
RIGHT_SHOULDER_IDX_V1 = 12

# v2
N_HAND_JOINTS = 21
N_POSE_JOINTS_V2 = 12
COORDS_PER_JOINT_V2 = 3  # x, y, confidence
N_LEFT_HAND_V2 = N_HAND_JOINTS * COORDS_PER_JOINT_V2   # 63
N_RIGHT_HAND_V2 = N_HAND_JOINTS * COORDS_PER_JOINT_V2  # 63
N_POSE_V2 = N_POSE_JOINTS_V2 * COORDS_PER_JOINT_V2     # 36
FEATURE_DIM_V2 = N_LEFT_HAND_V2 + N_RIGHT_HAND_V2 + N_POSE_V2  # 162
POSE_OFFSET_V2 = N_LEFT_HAND_V2 + N_RIGHT_HAND_V2  # 126
# v2 pose ids: 0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear, 5 neck,
#              6 L shoulder, 7 R shoulder, 8 L elbow, 9 R elbow, 10 L wrist, 11 R wrist
LEFT_SHOULDER_IDX_V2 = 6
RIGHT_SHOULDER_IDX_V2 = 7

VISIBILITY_THRESHOLD = 0.1
VISIBILITY_BIN_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.1

LAYOUT_V1 = "v1_135"
LAYOUT_V2 = "v2_162"
PREPROCESS_VERSION_V2 = "v2_shoulder_scale_1"


# ── JSON loading ──────────────────────────────────────────────────────────────

def load_landmark_json(path: str | Path) -> tuple[list[dict], dict]:
    """
    Load landmark JSON. Returns (frames, meta).
    meta includes schema_version (1 or 2) and optional fps/width/height/orientation.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, {"schema_version": 1}
    if isinstance(data, dict) and "frames" in data:
        meta = {
            "schema_version": int(data.get("schema_version", 2)),
            "fps": data.get("fps"),
            "width": data.get("width"),
            "height": data.get("height"),
            "orientation": data.get("orientation"),
        }
        return data["frames"], meta
    raise ValueError(f"Unrecognized landmark JSON: {path}")


def _point_xy(pt) -> tuple[float, float] | None:
    """Extract (x, y) from a point dict or None if missing/null."""
    if pt is None:
        return None
    if not isinstance(pt, dict):
        return None
    if "x" not in pt or "y" not in pt:
        return None
    return float(pt["x"]), float(pt["y"])


def _point_conf(pt, default: float = 1.0) -> float:
    if pt is None or not isinstance(pt, dict):
        return 0.0
    if "c" in pt:
        return float(pt["c"])
    if "visibility" in pt:
        return float(pt["visibility"])
    return default


# ── Vectorizers ───────────────────────────────────────────────────────────────

def frame_to_vector_v1(frame: dict) -> np.ndarray:
    """
    Convert one frame to 135-d v1 vector.
    Missing / null joints → 0.0 (legacy contract for Fase 1).
    Works with schema v1 and v2 JSON.
    """
    vec = np.zeros(FEATURE_DIM_V1, dtype=np.float32)

    lh = frame.get("leftHand") or frame.get("left_hand")
    if lh:
        for i, pt in enumerate(lh[:21]):
            xy = _point_xy(pt)
            if xy is None:
                continue
            vec[i * 2] = xy[0]
            vec[i * 2 + 1] = xy[1]

    rh = frame.get("rightHand") or frame.get("right_hand")
    if rh:
        offset = N_LEFT_HAND_V1
        for i, pt in enumerate(rh[:21]):
            xy = _point_xy(pt)
            if xy is None:
                continue
            vec[offset + i * 2] = xy[0]
            vec[offset + i * 2 + 1] = xy[1]

    pose = frame.get("pose")
    if pose:
        offset = POSE_OFFSET_V1
        # Schema v2 has 12 joints with names; map into legacy 17-slot layout.
        name_to_legacy = {
            "nose": 0,
            "left_eye": 2,
            "right_eye": 5,
            "left_ear": 7,
            "right_ear": 8,
            "neck": 9,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
        }
        for pt in pose:
            if not isinstance(pt, dict):
                continue
            xy = _point_xy(pt)
            if xy is None:
                continue
            vis = _point_conf(pt, default=1.0)
            if "name" in pt and pt["name"] in name_to_legacy:
                idx = name_to_legacy[pt["name"]]
                # Duplicate eye/neck slots to match legacy MediaPipe-shaped layout.
                slots = [idx]
                if pt["name"] == "left_eye":
                    slots = [1, 2, 3]
                elif pt["name"] == "right_eye":
                    slots = [4, 5, 6]
                elif pt["name"] == "neck":
                    slots = [9, 10]
                for idx in slots:
                    base = offset + idx * 3
                    vec[base] = xy[0]
                    vec[base + 1] = xy[1]
                    vec[base + 2] = vis
            elif "id" in pt:
                idx = int(pt["id"])
                if 0 <= idx < 17:
                    base = offset + idx * 3
                    vec[base] = xy[0]
                    vec[base + 1] = xy[1]
                    vec[base + 2] = vis

    return vec


def frame_to_vector_v2(frame: dict) -> np.ndarray:
    """
    Convert one frame to 162-d raw v2 vector (before shoulder scale).
    Missing joints: x=y=0, confidence=0.
    """
    vec = np.zeros(FEATURE_DIM_V2, dtype=np.float32)

    def fill_hand(points, offset: int) -> None:
        if not points:
            return
        for i, pt in enumerate(points[:N_HAND_JOINTS]):
            xy = _point_xy(pt)
            if xy is None:
                continue
            conf = _point_conf(pt, default=1.0)
            if conf < CONFIDENCE_THRESHOLD:
                continue
            base = offset + i * 3
            vec[base] = xy[0]
            vec[base + 1] = xy[1]
            vec[base + 2] = conf

    fill_hand(frame.get("leftHand") or frame.get("left_hand"), 0)
    fill_hand(frame.get("rightHand") or frame.get("right_hand"), N_LEFT_HAND_V2)

    pose = frame.get("pose") or []
    name_to_idx = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "neck": 5, "left_shoulder": 6, "right_shoulder": 7,
        "left_elbow": 8, "right_elbow": 9, "left_wrist": 10, "right_wrist": 11,
    }
    for pt in pose:
        if not isinstance(pt, dict):
            continue
        xy = _point_xy(pt)
        if xy is None:
            continue
        conf = _point_conf(pt, default=1.0)
        if conf < CONFIDENCE_THRESHOLD:
            continue
        if "name" in pt and pt["name"] in name_to_idx:
            idx = name_to_idx[pt["name"]]
        elif "id" in pt:
            idx = int(pt["id"])
            # If schema v1 17-slot id, map shoulders etc.
            legacy_to_v2 = {
                0: 0, 2: 1, 5: 2, 7: 3, 8: 4, 9: 5,
                11: 6, 12: 7, 13: 8, 14: 9, 15: 10, 16: 11,
            }
            if idx not in legacy_to_v2:
                continue
            idx = legacy_to_v2[idx]
        else:
            continue
        if not (0 <= idx < N_POSE_JOINTS_V2):
            continue
        base = POSE_OFFSET_V2 + idx * 3
        vec[base] = xy[0]
        vec[base + 1] = xy[1]
        vec[base + 2] = conf

    return vec


# ── v1 prepare (shoulder + valid + interpolate) — mirrors Fase 1 ──────────────

def _shoulder_base_v1(idx: int) -> int:
    return POSE_OFFSET_V1 + idx * 3


def compute_anchor_v1(vec: np.ndarray) -> tuple[float, float] | None:
    ls = _shoulder_base_v1(LEFT_SHOULDER_IDX_V1)
    rs = _shoulder_base_v1(RIGHT_SHOULDER_IDX_V1)
    left_vis = vec[ls + 2] > VISIBILITY_THRESHOLD and (vec[ls] != 0 or vec[ls + 1] != 0)
    right_vis = vec[rs + 2] > VISIBILITY_THRESHOLD and (vec[rs] != 0 or vec[rs + 1] != 0)
    if left_vis and right_vis:
        return float((vec[ls] + vec[rs]) / 2), float((vec[ls + 1] + vec[rs + 1]) / 2)
    if left_vis:
        return float(vec[ls]), float(vec[ls + 1])
    if right_vis:
        return float(vec[rs]), float(vec[rs + 1])
    return None


def is_valid_frame_v1(vec: np.ndarray) -> bool:
    if compute_anchor_v1(vec) is None:
        return False
    has_left = bool(np.any(vec[0:N_LEFT_HAND_V1] != 0))
    has_right = bool(np.any(vec[N_LEFT_HAND_V1:POSE_OFFSET_V1] != 0))
    pose_count = 0
    for i in range(17):
        base = POSE_OFFSET_V1 + i * 3
        if vec[base] != 0 or vec[base + 1] != 0:
            pose_count += 1
    return has_left or has_right or pose_count >= 3


def normalize_frame_v1(vec: np.ndarray) -> np.ndarray | None:
    out = vec.copy().astype(np.float32)
    anchor = compute_anchor_v1(out)
    if anchor is None:
        return None
    ax, ay = anchor
    for i in range(N_LEFT_HAND_V1 // 2):
        base = i * 2
        if out[base] != 0 or out[base + 1] != 0:
            out[base] -= ax
            out[base + 1] -= ay
    for i in range(N_RIGHT_HAND_V1 // 2):
        base = N_LEFT_HAND_V1 + i * 2
        if out[base] != 0 or out[base + 1] != 0:
            out[base] -= ax
            out[base + 1] -= ay
    for i in range(17):
        base = POSE_OFFSET_V1 + i * 3
        if out[base] != 0 or out[base + 1] != 0:
            out[base] -= ax
            out[base + 1] -= ay
        out[base + 2] = 1.0 if out[base + 2] >= VISIBILITY_BIN_THRESHOLD else 0.0
    return out


def interpolate_sequence(seq: np.ndarray, target_len: int = N_FRAMES) -> np.ndarray:
    T = len(seq)
    if T == target_len:
        return seq.astype(np.float32)
    if T == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim == 2 else FEATURE_DIM_V1), dtype=np.float32)
    if T == 1:
        return np.repeat(seq.astype(np.float32), target_len, axis=0)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    interp = interp1d(x_old, seq, axis=0, kind="linear")
    return interp(x_new).astype(np.float32)


def prepare_sequence_v1(
    json_path: str | None = None,
    raw_seq: np.ndarray | None = None,
    target_len: int = N_FRAMES,
) -> np.ndarray | None:
    if raw_seq is None:
        if json_path is None:
            raise ValueError("Provide json_path or raw_seq")
        frames, _ = load_landmark_json(json_path)
        if not frames:
            return None
        raw_seq = np.stack([frame_to_vector_v1(f) for f in frames], axis=0)

    if len(raw_seq) == 0:
        return None

    valid = [raw_seq[t] for t in range(len(raw_seq)) if is_valid_frame_v1(raw_seq[t])]
    if len(valid) < MIN_VALID_FRAMES:
        return None

    normalized = []
    for v in valid:
        n = normalize_frame_v1(v)
        if n is not None:
            normalized.append(n)
        else:
            normalized.append(np.zeros(FEATURE_DIM_V1, dtype=np.float32))
    return interpolate_sequence(np.stack(normalized, axis=0), target_len)


# ── v2 prepare ────────────────────────────────────────────────────────────────

def _joint_present_v2(vec: np.ndarray, base: int) -> bool:
    return vec[base + 2] >= CONFIDENCE_THRESHOLD and (vec[base] != 0 or vec[base + 1] != 0 or True)


def compute_anchor_v2(vec: np.ndarray) -> tuple[float, float] | None:
    ls = POSE_OFFSET_V2 + LEFT_SHOULDER_IDX_V2 * 3
    rs = POSE_OFFSET_V2 + RIGHT_SHOULDER_IDX_V2 * 3
    left_ok = vec[ls + 2] >= CONFIDENCE_THRESHOLD
    right_ok = vec[rs + 2] >= CONFIDENCE_THRESHOLD
    if left_ok and right_ok:
        return float((vec[ls] + vec[rs]) / 2), float((vec[ls + 1] + vec[rs + 1]) / 2)
    if left_ok:
        return float(vec[ls]), float(vec[ls + 1])
    if right_ok:
        return float(vec[rs]), float(vec[rs + 1])
    return None


def compute_shoulder_width_v2(vec: np.ndarray) -> float | None:
    ls = POSE_OFFSET_V2 + LEFT_SHOULDER_IDX_V2 * 3
    rs = POSE_OFFSET_V2 + RIGHT_SHOULDER_IDX_V2 * 3
    if vec[ls + 2] < CONFIDENCE_THRESHOLD or vec[rs + 2] < CONFIDENCE_THRESHOLD:
        return None
    dx = float(vec[ls] - vec[rs])
    dy = float(vec[ls + 1] - vec[rs + 1])
    w = float(np.hypot(dx, dy))
    return w if w > 1e-4 else None


def is_valid_frame_v2(vec: np.ndarray) -> bool:
    return compute_anchor_v2(vec) is not None


def _joint_bases_v2() -> list[int]:
    bases = []
    for i in range(N_HAND_JOINTS):
        bases.append(i * 3)
        bases.append(N_LEFT_HAND_V2 + i * 3)
    for i in range(N_POSE_JOINTS_V2):
        bases.append(POSE_OFFSET_V2 + i * 3)
    return bases


def interpolate_joints_v2(seq: np.ndarray, target_len: int = N_FRAMES) -> np.ndarray:
    """
    Per-joint interpolation: only blend frames where the joint is present.
    Edge gaps stay missing (conf=0). Confidence is interpolated among present frames.
    """
    T, D = seq.shape
    assert D == FEATURE_DIM_V2
    out = np.zeros((target_len, D), dtype=np.float32)
    t_new = np.linspace(0, 1, target_len)

    for base in _joint_bases_v2():
        present = [t for t in range(T) if seq[t, base + 2] >= CONFIDENCE_THRESHOLD]
        if not present:
            continue
        if len(present) == 1:
            t0 = present[0]
            # Map single observation to nearest resampled frames around that time.
            t_old = np.linspace(0, 1, T)
            nearest = int(np.argmin(np.abs(t_new - t_old[t0])))
            out[nearest, base:base + 3] = seq[t0, base:base + 3]
            continue

        t_old = np.linspace(0, 1, T)[present]
        for ch in range(3):
            vals = seq[present, base + ch]
            interp = interp1d(
                t_old, vals, kind="linear",
                bounds_error=False, fill_value=np.nan,
            )
            sampled = interp(t_new)
            for i, v in enumerate(sampled):
                if np.isnan(v):
                    continue
                out[i, base + ch] = float(v)
        # Zero confidence outside the span of present frames.
        t_min, t_max = float(t_old[0]), float(t_old[-1])
        for i, tn in enumerate(t_new):
            if tn < t_min - 1e-9 or tn > t_max + 1e-9:
                out[i, base:base + 3] = 0.0

    return out


def prepare_sequence_v2(
    json_path: str | None = None,
    raw_seq: np.ndarray | None = None,
    target_len: int = N_FRAMES,
) -> np.ndarray | None:
    """
    Full v2 pipeline: load → filter valid → shoulder-center → scale by median
    shoulder width → per-joint interpolate to target_len.
    """
    if raw_seq is None:
        if json_path is None:
            raise ValueError("Provide json_path or raw_seq")
        frames, _ = load_landmark_json(json_path)
        if not frames:
            return None
        raw_seq = np.stack([frame_to_vector_v2(f) for f in frames], axis=0)

    if len(raw_seq) == 0:
        return None

    valid_idx = [t for t in range(len(raw_seq)) if is_valid_frame_v2(raw_seq[t])]
    if len(valid_idx) < MIN_VALID_FRAMES:
        return None
    seq = raw_seq[valid_idx].astype(np.float32).copy()

    widths = []
    for t in range(len(seq)):
        w = compute_shoulder_width_v2(seq[t])
        if w is not None:
            widths.append(w)
    scale = float(np.median(widths)) if widths else 1.0
    if scale < 1e-4:
        scale = 1.0

    for t in range(len(seq)):
        anchor = compute_anchor_v2(seq[t])
        if anchor is None:
            seq[t] = 0.0
            continue
        ax, ay = anchor
        for base in _joint_bases_v2():
            if seq[t, base + 2] < CONFIDENCE_THRESHOLD:
                seq[t, base:base + 3] = 0.0
                continue
            seq[t, base] = (seq[t, base] - ax) / scale
            seq[t, base + 1] = (seq[t, base + 1] - ay) / scale
            # Keep confidence; binarize lightly for stability.
            seq[t, base + 2] = 1.0 if seq[t, base + 2] >= VISIBILITY_BIN_THRESHOLD else seq[t, base + 2]

    return interpolate_joints_v2(seq, target_len)


def prepare_sequence(
    json_path: str | None = None,
    raw_seq: np.ndarray | None = None,
    target_len: int = N_FRAMES,
    layout: str = LAYOUT_V2,
) -> np.ndarray | None:
    if layout == LAYOUT_V1:
        return prepare_sequence_v1(json_path=json_path, raw_seq=raw_seq, target_len=target_len)
    if layout == LAYOUT_V2:
        return prepare_sequence_v2(json_path=json_path, raw_seq=raw_seq, target_len=target_len)
    raise ValueError(f"Unknown layout: {layout}")


def feature_dim_for_layout(layout: str) -> int:
    if layout == LAYOUT_V1:
        return FEATURE_DIM_V1
    if layout == LAYOUT_V2:
        return FEATURE_DIM_V2
    raise ValueError(layout)
