"""
LSM — Conversión PyTorch → Core ML
====================================
Convierte best_model.pt a un .mlpackage listo para iOS/macOS.

Uso:
    pip install coremltools
    python convert_to_coreml.py --checkpoint ./runs/best_model.pt --output ./lsm_model
"""

import argparse
import json
import numpy as np
import torch
import coremltools as ct

from dataset import N_FRAMES, FEATURE_DIM
from model import LSM_CNN


def convert(checkpoint_path: str, output_path: str):
    # ── Cargar checkpoint ─────────────────────────────────────────────────────
    print(f"\nCargando checkpoint: {checkpoint_path}")
    ckpt    = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    n_classes = len(classes)

    print(f"  Clases     : {n_classes}")
    print(f"  N frames   : {ckpt.get('n_frames', N_FRAMES)}")
    print(f"  Feature dim: {ckpt.get('feature_dim', FEATURE_DIM)}")
    print(f"  Val Acc    : {ckpt.get('best_val_acc', 0):.4f}")

    # ── Reconstruir modelo ────────────────────────────────────────────────────
    model_type = ckpt.get("model_type", "cnn")
    print(f"  Arquitectura: {model_type}")

    if model_type == "cnn":
        model = LSM_CNN(n_classes=n_classes)
    elif model_type == "tcn":
        from model_tcn import LSM_TCN
        model = LSM_TCN(n_classes=n_classes)
        print(f"  Campo recep.: {model.receptive_field()} frames")
    elif model_type == "3dcnn":
        from model_3dcnn import LSM_3DCNN
        model = LSM_3DCNN(n_classes=n_classes)
        print(f"  Campo recep.: {model.receptive_field()}")
    else:
        raise ValueError(f"model_type desconocido: '{model_type}'")

    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Trazar con TorchScript ────────────────────────────────────────────────
    dummy_input = torch.randn(1, N_FRAMES, FEATURE_DIM)
    traced = torch.jit.trace(model, dummy_input)
    print("\n  TorchScript trace OK")

    # ── Convertir a Core ML ───────────────────────────────────────────────────
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="landmarks",
            shape=(1, N_FRAMES, FEATURE_DIM),
            dtype=np.float32
        )],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
        compute_units=ct.ComputeUnit.ALL,
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    mlmodel.short_description = "LSM — Reconocimiento de Lengua de Señas Mexicana"
    mlmodel.input_description["landmarks"] = \
        f"Secuencia de {N_FRAMES} frames × {FEATURE_DIM} valores " \
        f"[left_hand(42), right_hand(42), pose(51)]"
    mlmodel.output_description["logits"] = \
        f"Logits sin normalizar para {n_classes} clases"

    # Guardar lista de clases como metadata del modelo
    mlmodel.user_defined_metadata["classes"]     = json.dumps(classes, ensure_ascii=False)
    mlmodel.user_defined_metadata["n_frames"]    = str(N_FRAMES)
    mlmodel.user_defined_metadata["feature_dim"] = str(FEATURE_DIM)
    mlmodel.user_defined_metadata["n_classes"]   = str(n_classes)
    mlmodel.user_defined_metadata["val_acc"]     = str(round(ckpt.get("best_val_acc", 0), 4))

    # ── Guardar ───────────────────────────────────────────────────────────────
    output_path = output_path if output_path.endswith(".mlpackage") \
                               else output_path + ".mlpackage"
    mlmodel.save(output_path)
    print(f"\n✅ Modelo guardado: {output_path}")

    # ── Verificación rápida ───────────────────────────────────────────────────
    print("\nVerificando modelo...")
    loaded = ct.models.MLModel(output_path)
    dummy  = {"landmarks": np.random.randn(1, N_FRAMES, FEATURE_DIM).astype(np.float32)}
    out    = loaded.predict(dummy)
    logits = list(out.values())[0]
    probs  = np.exp(logits) / np.exp(logits).sum()
    top5_idx = np.argsort(probs[0])[::-1][:5]

    print("  Top-5 predicciones (input aleatorio):")
    for i, idx in enumerate(top5_idx):
        print(f"    {i+1}. {classes[idx]:<40} {probs[0][idx]*100:5.2f}%")

    print(f"\n  Listo para arrastrar al proyecto Xcode → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte best_model.pt a Core ML")
    parser.add_argument("--checkpoint", required=True, help="Ruta al best_model.pt")
    parser.add_argument("--output",     default="./lsm_model",
                        help="Ruta de salida (sin .mlpackage)")
    args = parser.parse_args()
    convert(args.checkpoint, args.output)
