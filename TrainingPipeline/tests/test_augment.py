"""Augmentation layout / mask tests."""

from __future__ import annotations

import torch

from augment import augment_batch, indices_for_layout, xy_indices_v1, xy_indices_v2
from features import FEATURE_DIM_V1, FEATURE_DIM_V2, LAYOUT_V1, LAYOUT_V2, POSE_OFFSET_V1


def test_v1_pose_xy_indices():
    x_idx, y_idx, c_idx = xy_indices_v1()
    # First pose joint at offset 84
    assert x_idx[42] == POSE_OFFSET_V1
    assert y_idx[42] == POSE_OFFSET_V1 + 1
    assert c_idx[0] == POSE_OFFSET_V1 + 2
    assert len(c_idx) == 17
    assert max(x_idx) < FEATURE_DIM_V1


def test_v2_indices_cover_all_joints():
    x_idx, y_idx, c_idx = xy_indices_v2()
    assert len(x_idx) == FEATURE_DIM_V2 // 3
    assert set(x_idx) | set(y_idx) | set(c_idx) == set(range(FEATURE_DIM_V2))


def test_confidence_untouched_v2():
    B, T, D = 2, 85, FEATURE_DIM_V2
    x = torch.randn(B, T, D)
    # Set confidence channels to known pattern
    _, _, c_idx = indices_for_layout(LAYOUT_V2)
    x[:, :, c_idx] = 0.77
    # Zero some joints (missing)
    x[:, :, 0:3] = 0
    out = augment_batch(
        x, layout=LAYOUT_V2, joint_dropout=0.0,
        p_scale=1.0, p_translate=1.0, p_warp=0.0, p_noise=1.0,
    )
    # Confidence on present joints unchanged (noise/scale only on xy)
    present_bases = [i for i in range(0, D, 3) if (x[0, 0, i] != 0 or x[0, 0, i + 1] != 0)]
    for b in present_bases:
        assert torch.allclose(out[:, :, b + 2], torch.full((B, T), 0.77)), b


def test_missing_joints_stay_zero_v2():
    x = torch.zeros(1, 85, FEATURE_DIM_V2)
    # Only one joint present
    x[0, :, 9:12] = torch.tensor([0.2, -0.1, 1.0])
    out = augment_batch(
        x, layout=LAYOUT_V2, joint_dropout=0.0,
        p_scale=1.0, p_translate=1.0, p_warp=0.0, p_noise=1.0,
    )
    # Missing joint 0 stays zero
    assert torch.all(out[0, :, 0:3] == 0)
    # Present joint moved but conf intact
    assert torch.all(out[0, :, 11] == 1.0)
    assert not torch.allclose(out[0, :, 9], x[0, :, 9])


def test_shapes_preserved():
    for layout, dim in [(LAYOUT_V1, FEATURE_DIM_V1), (LAYOUT_V2, FEATURE_DIM_V2)]:
        x = torch.randn(4, 85, dim)
        out = augment_batch(x, layout=layout, joint_dropout=0.1)
        assert out.shape == x.shape
