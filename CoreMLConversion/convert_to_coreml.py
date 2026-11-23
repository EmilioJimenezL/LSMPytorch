"""
LSM — Conversión PyTorch → Core ML
====================================
Uso (desde la raíz del repo; normalmente vía `./lsm convert`):
    python CoreMLConversion/convert_to_coreml.py \
        --checkpoint Runs/lab_cap5_cnn/best_model.pt --output Models/CoreML/lab_cap5_cnn
    python CoreMLConversion/convert_to_coreml.py \
        --checkpoint Runs/lab_cap5_cnn/best_model.pt --output Models/CoreML/lab_cap5_cnn \
        --cache Workspace/prepared/lab_v2_cap5.npz --parity --fixture Models/CoreML/lab_cap5_cnn_fixture.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _paths  # noqa: F401  — puts TrainingPipeline on sys.path
import coremltools as ct
import numpy as np
import torch

from features import LAYOUT_V1, N_FRAMES, PREPROCESS_VERSION_V2
from models import TRAIN_MODEL_MAP, create_model


def _model_from_ckpt(ckpt: dict):
    classes = ckpt["classes"]
    n_classes = len(classes)
    model_type = ckpt.get("model_type", "cnn")
    cfg = ckpt.get("model_config") or {}
    feature_dim = int(cfg.get("input_dim", ckpt.get("feature_dim", 135)))
    layout = cfg.get("layout", LAYOUT_V1)
    pool = cfg.get("pool", "avg")
    causal = cfg.get("causal", True)
    registry = TRAIN_MODEL_MAP.get(model_type, model_type)

    kwargs = {"input_dim": feature_dim, "pool": pool}
    if model_type == "tcn":
        kwargs["causal"] = causal

    model = create_model(registry, num_classes=n_classes, **kwargs)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes, feature_dim, layout, model_type, cfg


def _parity_check(pt_model, mlmodel, X_val: np.ndarray, classes: list[str],
                  max_logit_diff: float = 0.05, min_top1_agree: float = 0.99) -> dict:
    """Compare PyTorch vs Core ML on validation tensors."""
    n = min(len(X_val), 256)
    X = X_val[:n].astype(np.float32)
    with torch.no_grad():
        pt_logits = pt_model(torch.from_numpy(X)).numpy()

    cm_logits = []
    for i in range(n):
        out = mlmodel.predict({"landmarks": X[i:i+1]})
        logits = list(out.values())[0]
        cm_logits.append(np.asarray(logits).reshape(-1))
    cm_logits = np.stack(cm_logits, axis=0)
    if cm_logits.ndim == 1:
        cm_logits = cm_logits.reshape(1, -1)

    # Align shapes
    if cm_logits.shape[0] != n:
        cm_logits = cm_logits.reshape(n, -1)

    diff = np.max(np.abs(pt_logits - cm_logits))
    agree = float((pt_logits.argmax(1) == cm_logits.argmax(1)).mean())
    report = {
        "n": n,
        "max_abs_logit_diff": float(diff),
        "top1_agreement": agree,
        "pass": bool(diff <= max_logit_diff and agree >= min_top1_agree),
    }
    print(f"  Parity: max|Δlogit|={diff:.6f}  top1_agree={agree:.2%}  "
          f"{'PASS' if report['pass'] else 'FAIL'}")
    if not report["pass"]:
        raise SystemExit(
            f"Core ML parity gate failed (diff={diff:.6f}, agree={agree:.2%})"
        )
    return report


def convert(
    checkpoint_path: str,
    output_path: str,
    cache_path: str | None = None,
    require_parity: bool = False,
    fixture_out: str | None = None,
    skip_parity: bool = False,
):
    print(f"\nCargando checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, classes, feature_dim, layout, model_type, cfg = _model_from_ckpt(ckpt)
    n_classes = len(classes)

    print(f"  Clases      : {n_classes}")
    print(f"  Feature dim : {feature_dim}")
    print(f"  Layout      : {layout}")
    print(f"  Val Acc     : {ckpt.get('best_val_acc', 0):.4f}")
    print(f"  Arquitectura: {model_type}  config={cfg}")

    dummy = torch.randn(1, N_FRAMES, feature_dim)
    traced = torch.jit.trace(model, dummy)
    print("\n  TorchScript trace OK")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="landmarks",
            shape=(1, N_FRAMES, feature_dim),
            dtype=np.float32,
        )],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    mlmodel.short_description = "LSM — Reconocimiento de Lengua de Señas Mexicana"
    mlmodel.input_description["landmarks"] = (
        f"Secuencia de {N_FRAMES} frames × {feature_dim} ({layout})"
    )
    mlmodel.output_description["logits"] = f"Logits para {n_classes} clases"

    mlmodel.user_defined_metadata["classes"] = json.dumps(classes, ensure_ascii=False)
    mlmodel.user_defined_metadata["n_frames"] = str(N_FRAMES)
    mlmodel.user_defined_metadata["feature_dim"] = str(feature_dim)
    mlmodel.user_defined_metadata["n_classes"] = str(n_classes)
    mlmodel.user_defined_metadata["val_acc"] = str(round(ckpt.get("best_val_acc", 0), 4))
    mlmodel.user_defined_metadata["model_type"] = model_type
    mlmodel.user_defined_metadata["feature_layout"] = layout
    mlmodel.user_defined_metadata["preprocess_version"] = (
        PREPROCESS_VERSION_V2 if layout.startswith("v2") else "v1_shoulder"
    )
    mlmodel.user_defined_metadata["model_config"] = json.dumps(cfg)

    if not output_path.endswith(".mlpackage"):
        output_path = output_path + ".mlpackage"
    mlmodel.save(output_path)
    print(f"\n✅ Modelo guardado: {output_path}")

    # Random smoke (predict needs the macOS Core ML runtime)
    print("\nVerificando modelo...")
    loaded = ct.models.MLModel(output_path)
    if sys.platform != "darwin":
        print("  Skip random predict: Core ML predict only works on macOS")
    else:
        try:
            dummy_in = {"landmarks": np.random.randn(1, N_FRAMES, feature_dim).astype(np.float32)}
            out = loaded.predict(dummy_in)
            logits = list(out.values())[0]
            probs = np.exp(logits) / np.exp(logits).sum()
            flat = probs.reshape(-1)
            top5 = np.argsort(flat)[::-1][:5]
            print("  Top-5 (input aleatorio):")
            for i, idx in enumerate(top5):
                print(f"    {i+1}. {classes[idx]:<40} {flat[idx]*100:5.2f}%")
        except Exception as e:
            print(f"  Skip random predict: {e}")
            loaded = ct.models.MLModel(output_path)

    parity = None
    if not skip_parity and (require_parity or cache_path):
        if not cache_path:
            raise SystemExit("--parity requires --cache with validation tensors")
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        idx_val = data["idx_val"].astype(np.int64)
        X_val = X[idx_val]
        parity = _parity_check(model, loaded, X_val, classes)

    if fixture_out:
        # Golden fixture: first val sample
        data = np.load(cache_path, allow_pickle=True) if cache_path else None
        if data is not None:
            idx_val = data["idx_val"].astype(np.int64)
            i0 = int(idx_val[0])
            seq = data["X"][i0].astype(np.float32)
            with torch.no_grad():
                logits = model(torch.from_numpy(seq[None])).numpy()[0]
            fixture = {
                "layout": layout,
                "feature_layout": layout,
                "feature_dim": feature_dim,
                "n_frames": N_FRAMES,
                "path": str(data["paths"][i0]),
                "class": str(data["class_names"][i0]),
                "prepared": seq.tolist(),
                "prepared_sequence": seq.tolist(),
                "logits": logits.tolist(),
                "parity": parity,
            }
            Path(fixture_out).write_text(
                json.dumps(fixture, ensure_ascii=False), encoding="utf-8",
            )
            print(f"  Fixture: {fixture_out}")

    print(f"\n  Listo para Xcode → {output_path}")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults (skipping None/False) and keep the line breaks of the epilog."""

    def _get_help_string(self, action):
        text = action.help or ""
        if "%(default)" in text or action.default is None or action.default is False \
                or action.default is argparse.SUPPRESS:
            return text
        return text + " (default: %(default)s)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a trained PyTorch checkpoint (best_model.pt) to a Core ML .mlpackage, "
                    "optionally checking Core ML == PyTorch and writing a golden fixture.",
        formatter_class=_HelpFormatter,
        epilog=(
            "Examples:\n"
            "  python convert_to_coreml.py --checkpoint ../Runs/lab_cap5_cnn/best_model.pt "
            "--output ../Models/CoreML/lab_cap5_cnn\n"
            "  python convert_to_coreml.py --checkpoint ../Runs/lab_cap5_cnn/best_model.pt "
            "--output ../Models/CoreML/lab_cap5_cnn \\\n"
            "      --cache ../Workspace/prepared/lab_v2_cap5.npz --parity "
            "--fixture ../Models/CoreML/lab_cap5_cnn_fixture.json\n\n"
            "Notes:\n"
            "  * With --cache the parity check runs by default (needs macOS: coremltools predict); "
            "use --no-parity to skip it.\n"
            "  * Prefer the orchestrator: ./lsm convert --run NAME"
        ),
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt from train.py")
    parser.add_argument("--output", default="./lsm_model",
                        help="Output path; '.mlpackage' is appended if missing")
    parser.add_argument("--cache", default=None, help="Prepared .npz (prepare_dataset.py) for parity/fixture")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--parity", action="store_true",
                       help="Require the parity check (needs --cache); fails if Core ML != PyTorch")
    group.add_argument("--no-parity", action="store_true",
                       help="Skip the parity check even when --cache is given (e.g. on Linux/Windows)")
    parser.add_argument("--fixture", default=None, help="Write golden fixture JSON here (needs --cache)")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(
        args.checkpoint, args.output,
        cache_path=args.cache,
        require_parity=args.parity,
        fixture_out=args.fixture,
        skip_parity=args.no_parity,
    )
