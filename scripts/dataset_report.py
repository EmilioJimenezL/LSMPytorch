#!/usr/bin/env python3
"""
Scan LSMOutput (or any landmark dataset) and export inventory + class overlap reports.

Usage:
    python scripts/dataset_report.py --dataset ../LSMOutput --output ./reports
    python scripts/dataset_report.py --dataset ../LSMOutput --fase2-classes ./runs/classes.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset import load_raw_dataset  # noqa: E402


def scan_dataset(dataset_dir: str, min_videos: int = 2) -> dict:
    X, y, classes, meta = load_raw_dataset(dataset_dir, min_videos=min_videos, shoulder_norm=True)

    by_class: dict[str, list[dict]] = defaultdict(list)
    by_category: dict[str, int] = defaultdict(int)
    for i, m in enumerate(meta):
        by_class[m["clase"]].append({
            "path": m["archivo"],
            "category": m.get("categoria", ""),
            "index": i,
        })
        cat = m.get("categoria") or "Sin categoría"
        by_category[cat] += 1

    class_details = []
    for cls in classes:
        videos = by_class[cls]
        cats = sorted({v["category"] for v in videos if v["category"]})
        class_details.append({
            "class": cls,
            "n_videos": len(videos),
            "categories": cats,
            "paths": [v["path"] for v in videos],
        })

    return {
        "generated": date.today().isoformat(),
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "min_videos": min_videos,
        "n_classes": len(classes),
        "n_videos": len(meta),
        "n_categories": len(by_category),
        "categories": dict(sorted(by_category.items())),
        "classes": class_details,
        "class_names": classes,
    }


def class_overlap(inventory_classes: list[str], fase2_classes_path: str | None) -> dict:
    inv = set(inventory_classes)
    if not fase2_classes_path:
        return {
            "fase1_classes": len(inv),
            "fase2_classes": 0,
            "intersection": sorted(inv),
            "n_intersection": len(inv),
            "only_fase1": sorted(inv),
            "only_fase2": [],
        }

    with open(fase2_classes_path, encoding="utf-8") as f:
        f2 = set(json.load(f))

    both = sorted(inv & f2)
    return {
        "fase1_classes": len(inv),
        "fase2_classes": len(f2),
        "intersection": both,
        "n_intersection": len(both),
        "only_fase1": sorted(inv - f2),
        "only_fase2": sorted(f2 - inv),
        "fase2_classes_path": str(Path(fase2_classes_path).resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description="Dataset inventory and class overlap")
    parser.add_argument("--dataset", required=True, help="Landmark dataset root")
    parser.add_argument("--output", default="./reports", help="Output directory")
    parser.add_argument("--min-videos", type=int, default=2)
    parser.add_argument("--fase2-classes", default=None, help="Optional Fase 2 classes.json for overlap")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.dataset}...")
    inventory = scan_dataset(args.dataset, min_videos=args.min_videos)
    inv_path = out_dir / "dataset_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, ensure_ascii=False, indent=2)
    print(f"  {inventory['n_classes']} classes, {inventory['n_videos']} videos → {inv_path}")

    overlap = class_overlap(inventory["class_names"], args.fase2_classes)
    overlap_path = out_dir / "class_overlap.json"
    with open(overlap_path, "w", encoding="utf-8") as f:
        json.dump(overlap, f, ensure_ascii=False, indent=2)
    print(f"  Intersection: {overlap['n_intersection']} classes → {overlap_path}")


if __name__ == "__main__":
    main()
