#!/usr/bin/env python3
"""
Parse landmark JSON once and write a shared prepared cache + split.

Usage:
  python scripts/prepare_dataset.py \
    --dataset ../LSM_LAB_OUTPUT_V2/landmarks \
    --output prepared/lab_v2_cap5.npz \
    --layout v2_162 --min-videos 5 --max-videos-per-class 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import load_raw_dataset  # noqa: E402
from features import LAYOUT_V1, LAYOUT_V2, PREPROCESS_VERSION_V2, feature_dim_for_layout  # noqa: E402
from split_utils import cap_and_split_indices, export_split_manifest  # noqa: E402


def settings_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LSM dataset cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, help="Path to .npz cache")
    parser.add_argument("--layout", default=LAYOUT_V2, choices=[LAYOUT_V1, LAYOUT_V2])
    parser.add_argument("--min-videos", type=int, default=5)
    parser.add_argument("--max-videos-per-class", type=int, default=5)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shoulder-norm", action="store_true",
                        help="Only meaningful for v1 legacy path")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out.with_suffix(".split.json")

    print(f"Loading {args.dataset} (layout={args.layout})...")
    X, y, classes, meta = load_raw_dataset(
        args.dataset,
        min_videos=args.min_videos,
        shoulder_norm=not args.no_shoulder_norm,
        layout=args.layout,
        max_videos_per_class=args.max_videos_per_class,
    )

    idx_train, idx_val, idx_extra = cap_and_split_indices(
        y, meta,
        max_videos_per_class=args.max_videos_per_class,
        val_fraction=args.val_split,
        seed=args.seed,
    )

    settings = {
        "layout": args.layout,
        "preprocess_version": PREPROCESS_VERSION_V2 if args.layout == LAYOUT_V2 else "v1_shoulder",
        "feature_dim": feature_dim_for_layout(args.layout),
        "min_videos": args.min_videos,
        "max_videos_per_class": args.max_videos_per_class,
        "val_split": args.val_split,
        "seed": args.seed,
        "shoulder_norm": not args.no_shoulder_norm,
        "n_classes": len(classes),
        "n_total": len(y),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_extra_test": len(idx_extra),
    }
    settings["hash"] = settings_hash(settings)

    export_split_manifest(
        meta, y, classes, idx_train, idx_val, manifest_path, idx_extra=idx_extra,
    )

    # Store paths as object array
    paths = np.array([m["archivo"] for m in meta], dtype=object)
    cats = np.array([m["categoria"] for m in meta], dtype=object)
    class_names = np.array([m["clase"] for m in meta], dtype=object)

    np.savez_compressed(
        out,
        X=X,
        y=y,
        classes=np.array(classes, dtype=object),
        paths=paths,
        categories=cats,
        class_names=class_names,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_extra=idx_extra,
        settings=json.dumps(settings, ensure_ascii=False),
    )

    print(f"\n  Cache        : {out}")
    print(f"  Split        : {manifest_path}")
    print(f"  Classes      : {len(classes)}")
    print(f"  Train/Val/Extra: {len(idx_train)}/{len(idx_val)}/{len(idx_extra)}")
    print(f"  Settings hash: {settings['hash']}")
    print(f"  X shape      : {X.shape}")


if __name__ == "__main__":
    main()
