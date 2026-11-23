"""Dataset / split tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import _class_key, load_raw_dataset
from features import LAYOUT_V1, LAYOUT_V2, FEATURE_DIM_V2, N_FRAMES
from split_utils import cap_and_split_indices


def test_rosa_becomes_two_classes():
    ambiguous = {"Rosa"}
    assert _class_key("Rosa", "Colores", ambiguous) == "Rosa (Colores)"
    assert _class_key("Rosa", "Frutas_Verduras_y_Plantas", ambiguous) == (
        "Rosa (Frutas_Verduras_y_Plantas)"
    )
    assert _class_key("Azul", "Colores", ambiguous) == "Azul"


def test_cap_exactly_five_deterministic():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.full(12, 0), np.full(8, 1), np.full(5, 2)])
    meta = [{"clase": f"c{int(yi)}", "categoria": "C", "archivo": f"{i}.json"} for i, yi in enumerate(y)]
    a = cap_and_split_indices(y, meta, max_videos_per_class=5, seed=42)
    b = cap_and_split_indices(y, meta, max_videos_per_class=5, seed=42)
    for x, yv in zip(a, b):
        assert np.array_equal(x, yv)
    train, val, extra = a
    used = np.concatenate([train, val])
    # Cap 5 → for class 0: 5 kept, 7 extra; class 1: 5 kept, 3 extra; class 2: 5 kept, 0 extra
    assert len(extra) == 7 + 3
    for cls in (0, 1, 2):
        n_kept = int(((y[used] == cls).sum()))
        assert n_kept == 5


def test_extra_test_disjoint():
    y = np.array([0] * 20 + [1] * 15)
    meta = [{"clase": "a", "categoria": "C", "archivo": f"{i}.json"} for i in range(len(y))]
    train, val, extra = cap_and_split_indices(y, meta, max_videos_per_class=5, seed=42)
    assert set(train).isdisjoint(set(val))
    assert set(train).isdisjoint(set(extra))
    assert set(val).isdisjoint(set(extra))
    assert len(set(train) | set(val) | set(extra)) == len(y)


SAMPLE_V2 = Path(
    "/Users/emilio/Desktop/Universidad/LSMPytorch/AppLSMTests/"
    "LSMExtractorCLI/build/sample_v2_out"
)


@pytest.mark.skipif(not SAMPLE_V2.exists(), reason="sample_v2_out missing")
def test_both_schemas_load(tmp_path):
    # Symlink Colores subset into a mini dataset tree
    src = SAMPLE_V2 / "Colores"
    if not src.exists():
        pytest.skip("no Colores in sample")
    dest = tmp_path / "landmarks" / "Colores"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(src)
    X, y, classes, meta = load_raw_dataset(
        str(tmp_path / "landmarks"),
        min_videos=1,
        layout=LAYOUT_V2,
    )
    assert X.shape[1:] == (N_FRAMES, FEATURE_DIM_V2)
    assert len(classes) == len(set(y))
    # Also v1 path on same JSON
    X1, y1, classes1, _ = load_raw_dataset(
        str(tmp_path / "landmarks"),
        min_videos=1,
        layout=LAYOUT_V1,
    )
    assert X1.shape[-1] == 135
    assert len(classes1) == len(classes)


@pytest.mark.skipif(not SAMPLE_V2.exists(), reason="sample_v2_out missing")
def test_min_videos_counts_valid(tmp_path):
    src = SAMPLE_V2 / "Colores"
    if not src.exists():
        pytest.skip("no Colores")
    dest = tmp_path / "landmarks" / "Colores"
    dest.parent.mkdir(parents=True)
    dest.symlink_to(src)
    # High min_videos should exclude most/all classes in the tiny sample
    try:
        X, y, classes, meta = load_raw_dataset(
            str(tmp_path / "landmarks"),
            min_videos=100,
            layout=LAYOUT_V2,
        )
    except RuntimeError:
        return
    assert len(classes) == 0
    assert len(X) == 0
