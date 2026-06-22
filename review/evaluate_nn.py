"""
Fase 2 neural model evaluation — stratified val + full-dataset inference.

Usage:
    python review/evaluate_nn.py --checkpoint ./runs_lsmoutput_tcn/best_model.pt --dataset ../LSMOutput
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import load_raw_dataset  # noqa: E402
from models import create_model, TRAIN_MODEL_MAP  # noqa: E402


@torch.no_grad()
def predict_all(model, X: np.ndarray, device, batch_size: int = 64) -> np.ndarray:
    model.eval()
    logits_list = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        logits_list.append(model(batch).cpu().numpy())
    return np.concatenate(logits_list, axis=0)


def _accuracy(logits: np.ndarray, y: np.ndarray, k: int = 1) -> tuple[int, int]:
    topk = np.argsort(logits, axis=1)[:, ::-1][:, :k]
    hits = sum(int(y[i] in topk[i]) for i in range(len(y)))
    return hits, len(y)


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_dir: str,
    min_videos: int = 2,
    device: str | None = None,
) -> dict:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    classes = ckpt["classes"]
    model_type = ckpt.get("model_type", "cnn")
    registry = TRAIN_MODEL_MAP.get(model_type, model_type)

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model = create_model(registry, num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    print(f"Loading dataset {dataset_dir}...")
    X, y, loaded_classes, meta = load_raw_dataset(
        dataset_dir, min_videos=min_videos, shoulder_norm=True,
    )
    assert loaded_classes == classes, "Class list mismatch vs checkpoint"

    t0 = time.time()
    logits = predict_all(model, X, device)
    infer_ms = (time.time() - t0) / len(X) * 1000

    top1_h, n = _accuracy(logits, y, k=1)
    top5_h, _ = _accuracy(logits, y, k=5)

    per_class: dict[str, list[bool]] = defaultdict(list)
    per_category: dict[str, list[bool]] = defaultdict(list)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    preds = logits.argmax(axis=1)
    for i in range(n):
        true_cls = classes[y[i]]
        pred_cls = classes[preds[i]]
        hit = preds[i] == y[i]
        per_class[true_cls].append(hit)
        cat = meta[i].get("categoria") or "Sin categoría"
        per_category[cat].append(hit)
        confusion[true_cls][pred_cls] += 1

    class_acc = {c: sum(v) / len(v) for c, v in per_class.items() if v}
    category_acc = {c: sum(v) / len(v) for c, v in per_category.items() if v}

    report = {
        "model_type": model_type,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "n_videos": n,
        "n_classes": len(classes),
        "eval_protocol": "full_dataset_inference",
        "note": "Not leave-one-out; model trained with stratified split. Use val_acc from checkpoint for holdout.",
        "checkpoint_val_acc": ckpt.get("best_val_acc"),
        "top1_accuracy": top1_h / n,
        "top5_accuracy": top5_h / n,
        "top1_correct": top1_h,
        "top5_correct": top5_h,
        "inference_ms_per_video": infer_ms,
        "per_class_accuracy": class_acc,
        "per_category_accuracy": category_acc,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    # Val-only metrics if split manifest exists
    manifest_path = Path(checkpoint_path).parent / "split_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        val_indices = [e["index"] for e in manifest["entries"] if e["split"] == "val"]
        if val_indices:
            val_idx = np.array(val_indices, dtype=np.int64)
            v_logits = logits[val_idx]
            v_y = y[val_idx]
            v1, vn = _accuracy(v_logits, v_y, k=1)
            v5, _ = _accuracy(v_logits, v_y, k=5)
            report["val_split"] = {
                "n_videos": vn,
                "top1_accuracy": v1 / vn,
                "top5_accuracy": v5 / vn,
                "top1_correct": v1,
                "top5_correct": v5,
            }

    print(f"\n  Full-dataset Top-1: {report['top1_accuracy']:.2%} ({top1_h}/{n})")
    print(f"  Full-dataset Top-5: {report['top5_accuracy']:.2%} ({top5_h}/{n})")
    print(f"  Inference: {infer_ms:.1f} ms/video")
    if "val_split" in report:
        vs = report["val_split"]
        print(f"  Val split Top-1: {vs['top1_accuracy']:.2%} ({vs['top1_correct']}/{vs['n_videos']})")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fase 2 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--min-videos", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = evaluate_checkpoint(args.checkpoint, args.dataset, args.min_videos)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  JSON: {out}")


if __name__ == "__main__":
    main()
