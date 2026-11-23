"""Feature vectorizer / prepare_sequence_v2 tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from features import (
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    N_FRAMES,
    POSE_OFFSET_V2,
    frame_to_vector_v1,
    frame_to_vector_v2,
    prepare_sequence_v2,
)


def _pose_pt(name, idx, x, y, c=1.0):
    return {"id": idx, "name": name, "x": x, "y": y, "visibility": c, "c": c}


def _hand(n=21, present=True):
    if not present:
        return [None] * n
    return [{"x": 0.1 + i * 0.01, "y": 0.2, "c": 0.9} for i in range(n)]


def _frame_with_shoulders(missing_hand_joint=False):
    lh = _hand()
    if missing_hand_joint:
        lh[5] = None
    return {
        "leftHand": lh,
        "rightHand": _hand(),
        "pose": [
            _pose_pt("nose", 0, 0.5, 0.2),
            _pose_pt("left_eye", 1, 0.48, 0.18),
            _pose_pt("right_eye", 2, 0.52, 0.18),
            _pose_pt("left_ear", 3, 0.45, 0.2),
            _pose_pt("right_ear", 4, 0.55, 0.2),
            _pose_pt("neck", 5, 0.5, 0.3),
            _pose_pt("left_shoulder", 6, 0.4, 0.4),
            _pose_pt("right_shoulder", 7, 0.6, 0.4),
            _pose_pt("left_elbow", 8, 0.35, 0.55),
            _pose_pt("right_elbow", 9, 0.65, 0.55),
            _pose_pt("left_wrist", 10, 0.3, 0.7),
            _pose_pt("right_wrist", 11, 0.7, 0.7),
        ],
    }


def test_v2_null_joint_confidence_zero():
    fr = _frame_with_shoulders(missing_hand_joint=True)
    v = frame_to_vector_v2(fr)
    assert v.shape == (FEATURE_DIM_V2,)
    assert v[5 * 3] == 0 and v[5 * 3 + 1] == 0 and v[5 * 3 + 2] == 0


def test_v1_null_tolerant():
    fr = _frame_with_shoulders(missing_hand_joint=True)
    v = frame_to_vector_v1(fr)
    assert v.shape == (FEATURE_DIM_V1,)
    assert v[5 * 2] == 0 and v[5 * 2 + 1] == 0


def test_scale_normalization():
    raw = np.stack([frame_to_vector_v2(_frame_with_shoulders()) for _ in range(20)], axis=0)
    seq = prepare_sequence_v2(raw_seq=raw)
    assert seq is not None
    assert seq.shape == (N_FRAMES, FEATURE_DIM_V2)
    ls = POSE_OFFSET_V2 + 6 * 3
    rs = POSE_OFFSET_V2 + 7 * 3
    mid = (seq[:, ls] + seq[:, rs]) / 2
    mask = seq[:, ls + 2] > 0
    assert float(np.mean(np.abs(mid[mask]))) < 0.05


def test_interpolate_does_not_blend_missing():
    frames = []
    for t in range(30):
        fr = _frame_with_shoulders()
        if t < 10 or t > 20:
            fr["leftHand"][3] = None
        frames.append(fr)
    raw = np.stack([frame_to_vector_v2(f) for f in frames], axis=0)
    seq = prepare_sequence_v2(raw_seq=raw)
    assert seq is not None
    base = 3 * 3
    conf = seq[:, base + 2]
    assert conf[0] < 0.15 or conf[-1] < 0.15


SAMPLE_JSON = next(
    Path(
        "/Users/emilio/Desktop/Universidad/LSMPytorch/AppLSMTests/"
        "LSMExtractorCLI/build/sample_v2_out"
    ).rglob("*_landmarks.json"),
    None,
)


@pytest.mark.skipif(SAMPLE_JSON is None, reason="no sample json")
def test_v1_from_v2_json_has_shoulders():
    doc = json.loads(SAMPLE_JSON.read_text())
    frames = doc["frames"]
    v1 = frame_to_vector_v1(frames[0])
    v2 = frame_to_vector_v2(frames[0])
    assert v1.shape == (FEATURE_DIM_V1,)
    assert v2.shape == (FEATURE_DIM_V2,)
    # If shoulders present in v2, legacy slots 11/12 should be filled in v1
    ls = POSE_OFFSET_V2 + 6 * 3
    if v2[ls + 2] > 0:
        assert v1[84 + 11 * 3] != 0 or v1[84 + 11 * 3 + 1] != 0
