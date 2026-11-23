"""Model forward shapes, pools, legacy load, Core ML round-trip."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from features import FEATURE_DIM_V1, FEATURE_DIM_V2, N_FRAMES
from models import create_model
from models.tcn import TemporalConvNet


@pytest.mark.parametrize("layout_dim,pool", [
    (FEATURE_DIM_V1, "avg"),
    (FEATURE_DIM_V1, "attn"),
    (FEATURE_DIM_V2, "avg"),
    (FEATURE_DIM_V2, "attn"),
])
def test_cnn_forward_shapes(layout_dim, pool):
    m = create_model("1d_cnn", num_classes=10, input_dim=layout_dim, pool=pool)
    m.eval()
    x = torch.randn(2, N_FRAMES, layout_dim)
    y = m(x)
    assert y.shape == (2, 10)


@pytest.mark.parametrize("causal,pool,dim", [
    (True, "avg", FEATURE_DIM_V1),
    (False, "attn", FEATURE_DIM_V2),
])
def test_tcn_forward_shapes(causal, pool, dim):
    m = create_model("tcn", num_classes=7, input_dim=dim, causal=causal, pool=pool)
    m.eval()
    x = torch.randn(2, N_FRAMES, dim)
    y = m(x)
    assert y.shape == (2, 7)


def test_receptive_field_counts_two_convs():
    m = TemporalConvNet(num_classes=3, num_layers=4, kernel_size=3, causal=False)
    # 1 + sum_i 2*(k-1)*2^i for i in 0..3 = 1 + 2*2*(1+2+4+8)=1+60=61
    assert m.receptive_field() == 1 + sum(2 * 2 * (2 ** i) for i in range(4))


def test_legacy_defaults():
    m = create_model("tcn", num_classes=5)
    m.eval()
    assert getattr(m, "causal", True) is True
    x = torch.randn(2, N_FRAMES, FEATURE_DIM_V1)
    assert m(x).shape == (2, 5)


def test_old_checkpoint_load_compat(tmp_path):
    m = create_model("1d_cnn", num_classes=4, input_dim=FEATURE_DIM_V1, pool="avg")
    path = tmp_path / "old.pt"
    torch.save({"model": m.state_dict(), "classes": ["a", "b", "c", "d"]}, path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config") or {}
    m2 = create_model(
        "1d_cnn",
        num_classes=len(ckpt["classes"]),
        input_dim=int(cfg.get("input_dim", 135)),
        pool=cfg.get("pool", "avg"),
    )
    m2.load_state_dict(ckpt["model"])
    m2.eval()
    x = torch.randn(2, N_FRAMES, FEATURE_DIM_V1)
    assert m2(x).shape == (2, 4)


def test_coreml_roundtrip_tiny():
    coremltools = pytest.importorskip("coremltools")
    m = create_model("1d_cnn", num_classes=3, input_dim=FEATURE_DIM_V2, pool="attn")
    m.eval()
    dummy = torch.randn(1, N_FRAMES, FEATURE_DIM_V2)
    with torch.no_grad():
        pt = m(dummy).numpy()
    traced = torch.jit.trace(m, dummy)
    ml = coremltools.convert(
        traced,
        inputs=[coremltools.TensorType(name="landmarks", shape=dummy.shape, dtype=np.float32)],
        outputs=[coremltools.TensorType(name="logits")],
        minimum_deployment_target=coremltools.target.iOS16,
    )
    out = ml.predict({"landmarks": dummy.numpy().astype(np.float32)})
    cm = np.asarray(list(out.values())[0]).reshape(-1)
    assert cm.shape == pt.reshape(-1).shape
    assert np.max(np.abs(cm - pt.reshape(-1))) < 0.05
