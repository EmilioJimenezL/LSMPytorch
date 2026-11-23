"""Resume from last.pt continues epoch counter."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from features import FEATURE_DIM_V2, LAYOUT_V2, N_FRAMES


PIPELINE = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE.parent


def _make_tiny_cache(path: Path, n_classes: int = 4, per_class: int = 5):
    n = n_classes * per_class
    X = np.random.randn(n, N_FRAMES, FEATURE_DIM_V2).astype(np.float32)
    y = np.repeat(np.arange(n_classes), per_class).astype(np.int64)
    classes = np.array([f"c{i}" for i in range(n_classes)], dtype=object)
    idx = np.arange(n)
    # simple split: last of each class → val
    idx_val = np.array([i * per_class + per_class - 1 for i in range(n_classes)], dtype=np.int64)
    idx_train = np.array([i for i in idx if i not in set(idx_val.tolist())], dtype=np.int64)
    idx_extra = np.array([], dtype=np.int64)
    settings = {
        "layout": LAYOUT_V2,
        "feature_dim": FEATURE_DIM_V2,
        "min_videos": 2,
        "max_videos_per_class": per_class,
    }
    meta = [
        {"clase": f"c{int(yi)}", "categoria": "T", "archivo": f"{i}.json"}
        for i, yi in enumerate(y)
    ]
    # train.load_cache expects meta reconstructed — check load_cache
    np.savez_compressed(
        path,
        X=X,
        y=y,
        classes=classes,
        paths=np.array([m["archivo"] for m in meta], dtype=object),
        categories=np.array([m["categoria"] for m in meta], dtype=object),
        class_names=np.array([m["clase"] for m in meta], dtype=object),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_extra=idx_extra,
        settings=json.dumps(settings),
    )


def test_resume_epoch_continues(tmp_path):
    cache = tmp_path / "tiny.npz"
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    _make_tiny_cache(cache)

    py = REPO_ROOT / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path(sys.executable)

    cmd_base = [
        str(py), str(PIPELINE / "train.py"),
        "--model", "cnn",
        "--cache", str(cache),
        "--layout", LAYOUT_V2,
        "--pool", "attn",
        "--batch", "8",
        "--augment", "0",
        "--warmup", "0",
        "--patience", "0",
        "--workers", "0",
    ]
    r1 = subprocess.run(
        cmd_base + ["--epochs", "2", "--output", str(out1)],
        cwd=str(PIPELINE), capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr
    last = out1 / "last.pt"
    assert last.exists()
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 1  # 0-indexed after 2 epochs

    # Copy last into out2 and resume for 1 more epoch
    out2.mkdir()
    import shutil
    shutil.copy2(last, out2 / "last.pt")
    r2 = subprocess.run(
        cmd_base + ["--epochs", "3", "--output", str(out2), "--resume", str(out2 / "last.pt")],
        cwd=str(PIPELINE), capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "Resumiendo desde epoch 2" in r2.stdout
    ckpt2 = torch.load(out2 / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt2["epoch"] == 2
    assert len(ckpt2["history"]["val_acc"]) == 3
