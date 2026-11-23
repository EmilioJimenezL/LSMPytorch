"""
GPU batch augmentation for LSM sequences.

Layout-aware masks: pose x/y indices are correct; visibility/confidence
channels are never scaled, translated, or noised.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from features import (
    FEATURE_DIM_V1,
    FEATURE_DIM_V2,
    LAYOUT_V1,
    LAYOUT_V2,
    N_LEFT_HAND_V1,
    N_LEFT_HAND_V2,
    N_POSE_JOINTS_V2,
    N_RIGHT_HAND_V1,
    POSE_OFFSET_V1,
    POSE_OFFSET_V2,
)


def xy_indices_v1() -> tuple[list[int], list[int], list[int]]:
    """Return (x_idx, y_idx, conf_or_vis_idx) for v1 layout."""
    x_idx, y_idx, c_idx = [], [], []
    # Hands: interleaved x,y
    for i in range((N_LEFT_HAND_V1 + N_RIGHT_HAND_V1) // 2):
        x_idx.append(i * 2)
        y_idx.append(i * 2 + 1)
    # Pose: x,y,visibility triples
    for i in range(17):
        base = POSE_OFFSET_V1 + i * 3
        x_idx.append(base)
        y_idx.append(base + 1)
        c_idx.append(base + 2)
    return x_idx, y_idx, c_idx


def xy_indices_v2() -> tuple[list[int], list[int], list[int]]:
    """Return (x_idx, y_idx, conf_idx) for v2 layout."""
    x_idx, y_idx, c_idx = [], [], []
    n_joints = (N_LEFT_HAND_V2 // 3) + (N_LEFT_HAND_V2 // 3) + N_POSE_JOINTS_V2
    # Hands + pose are all x,y,c triples
    for i in range(n_joints):
        base = i * 3
        x_idx.append(base)
        y_idx.append(base + 1)
        c_idx.append(base + 2)
    return x_idx, y_idx, c_idx


def indices_for_layout(layout: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if layout == LAYOUT_V2:
        x, y, c = xy_indices_v2()
    else:
        x, y, c = xy_indices_v1()
    return (
        torch.tensor(x, dtype=torch.long),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(c, dtype=torch.long),
    )


def joint_bases_for_layout(layout: str) -> torch.Tensor:
    """Base index of each joint (start of x,y,c or x,y for hands in v1)."""
    if layout == LAYOUT_V2:
        n = FEATURE_DIM_V2 // 3
        return torch.arange(n, dtype=torch.long) * 3
    # v1: hands are 2-d, pose is 3-d — dropout zeroes the joint coords
    bases = []
    for i in range((N_LEFT_HAND_V1 + N_RIGHT_HAND_V1) // 2):
        bases.append(i * 2)
    for i in range(17):
        bases.append(POSE_OFFSET_V1 + i * 3)
    return torch.tensor(bases, dtype=torch.long)


@torch.no_grad()
def augment_batch(
    x: torch.Tensor,
    layout: str = LAYOUT_V2,
    scale_range: tuple[float, float] = (0.9, 1.1),
    translation: float = 0.05,
    time_warp: tuple[float, float] = (0.8, 1.2),
    noise_std: float = 0.01,
    joint_dropout: float = 0.1,
    p_scale: float = 0.8,
    p_translate: float = 0.8,
    p_warp: float = 0.7,
    p_noise: float = 0.6,
) -> torch.Tensor:
    """
    Augment a batch of sequences (B, T, D) on the same device as x.
    Fresh randomness every call.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (B,T,D), got {tuple(x.shape)}")
    B, T, D = x.shape
    out = x.clone()
    device = x.device
    x_idx, y_idx, c_idx = indices_for_layout(layout)
    x_idx = x_idx.to(device)
    y_idx = y_idx.to(device)
    c_idx = c_idx.to(device)

    # Presence mask: joint present if any of its xy nonzero OR conf>0
    # For translation/scale we only move present joints.
    for b in range(B):
        seq = out[b]  # (T, D)

        # ── Scale (per-axis around origin = shoulder-centered 0) ──────────
        if torch.rand(1, device=device).item() < p_scale:
            sx = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(1, device=device)
            sy = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(1, device=device)
            present_x = seq[:, x_idx] != 0
            present_y = seq[:, y_idx] != 0
            seq[:, x_idx] = torch.where(present_x, seq[:, x_idx] * sx, seq[:, x_idx])
            seq[:, y_idx] = torch.where(present_y, seq[:, y_idx] * sy, seq[:, y_idx])

        # ── Translation ──────────────────────────────────────────────────
        if torch.rand(1, device=device).item() < p_translate:
            tx = (torch.rand(1, device=device) * 2 - 1) * translation
            ty = (torch.rand(1, device=device) * 2 - 1) * translation
            present_x = seq[:, x_idx] != 0
            present_y = seq[:, y_idx] != 0
            seq[:, x_idx] = torch.where(present_x, seq[:, x_idx] + tx, seq[:, x_idx])
            seq[:, y_idx] = torch.where(present_y, seq[:, y_idx] + ty, seq[:, y_idx])

        # ── Time warp ────────────────────────────────────────────────────
        if torch.rand(1, device=device).item() < p_warp:
            speed = time_warp[0] + (time_warp[1] - time_warp[0]) * torch.rand(1, device=device).item()
            new_len = max(10, int(round(T * speed)))
            # (1, D, T) → interpolate on time → back to T
            tw = seq.transpose(0, 1).unsqueeze(0)  # (1, D, T)
            tw = F.interpolate(tw, size=new_len, mode="linear", align_corners=True)
            tw = F.interpolate(tw, size=T, mode="linear", align_corners=True)
            seq = tw.squeeze(0).transpose(0, 1).contiguous()

        # ── Gaussian noise on xy only ─────────────────────────────────────
        if torch.rand(1, device=device).item() < p_noise:
            noise_x = torch.randn(T, len(x_idx), device=device) * noise_std
            noise_y = torch.randn(T, len(y_idx), device=device) * noise_std
            present_x = seq[:, x_idx] != 0
            present_y = seq[:, y_idx] != 0
            seq[:, x_idx] = torch.where(present_x, seq[:, x_idx] + noise_x, seq[:, x_idx])
            seq[:, y_idx] = torch.where(present_y, seq[:, y_idx] + noise_y, seq[:, y_idx])

        # ── Joint dropout ────────────────────────────────────────────────
        if joint_dropout > 0:
            bases = joint_bases_for_layout(layout).to(device)
            drop = torch.rand(len(bases), device=device) < joint_dropout
            for base, do_drop in zip(bases.tolist(), drop.tolist()):
                if not do_drop:
                    continue
                if layout == LAYOUT_V2:
                    seq[:, base:base + 3] = 0
                else:
                    # v1 hand joint: 2 coords; pose: 3
                    if base < POSE_OFFSET_V1:
                        seq[:, base:base + 2] = 0
                    else:
                        seq[:, base:base + 3] = 0

        out[b] = seq

    return out


def expand_with_augmentation(
    x: torch.Tensor,
    y: torch.Tensor,
    factor: int,
    layout: str,
    joint_dropout: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return originals + ``factor`` augmented copies (concatenated)."""
    if factor <= 0:
        return x, y
    xs = [x]
    ys = [y]
    for _ in range(factor):
        xs.append(augment_batch(x, layout=layout, joint_dropout=joint_dropout))
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
