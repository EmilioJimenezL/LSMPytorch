"""
Head-to-head comparison of Fase 1 (fingerprint LOO) vs Fase 2 (neural) reports.

Usage:
    python review/compare_models.py \
        --fase1 reports/fase1_loo.json \
        --fase2 reports/fase2_tcn.json reports/fase2_cnn.json \
        --output reports/comparison.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(fase1: dict, fase2_reports: list[tuple[str, dict]]) -> dict:
    global_section = {
        "fase1": {
            "top1": fase1.get("top1_accuracy"),
            "top5": fase1.get("top5_accuracy"),
            "n_videos": fase1.get("n_videos"),
            "protocol": "leave_one_out",
        },
    }
    for name, r in fase2_reports:
        global_section[name] = {
            "top1": r.get("top1_accuracy"),
            "top5": r.get("top5_accuracy"),
            "val_top1": r.get("val_split", {}).get("top1_accuracy"),
            "checkpoint_val_acc": r.get("checkpoint_val_acc"),
            "n_videos": r.get("n_videos"),
            "protocol": r.get("eval_protocol"),
            "inference_ms": r.get("inference_ms_per_video"),
        }

    f1_cat = fase1.get("per_category_accuracy", {})
    per_category = {}
    for cat, acc1 in f1_cat.items():
        row = {"fase1": acc1}
        for name, r in fase2_reports:
            row[name] = r.get("per_category_accuracy", {}).get(cat)
        best = max((v for v in row.values() if v is not None), default=0)
        row["best"] = best
        row["fase2_beats_fase1"] = any(
            row.get(n) is not None and row[n] > acc1 for n, _ in fase2_reports
        )
        per_category[cat] = row

    f1_cls = fase1.get("per_class_accuracy", {})
    per_class = []
    all_classes = set(f1_cls.keys())
    for _, r in fase2_reports:
        all_classes.update(r.get("per_class_accuracy", {}).keys())

    for cls in sorted(all_classes):
        a1 = f1_cls.get(cls)
        row = {"class": cls, "fase1": a1}
        deltas = []
        for name, r in fase2_reports:
            a2 = r.get("per_class_accuracy", {}).get(cls)
            row[name] = a2
            if a1 is not None and a2 is not None:
                deltas.append(a2 - a1)
        row["max_delta"] = max(deltas) if deltas else None
        per_class.append(row)

    per_class.sort(key=lambda x: (x["max_delta"] is None, -(x["max_delta"] or -999)))

    return {
        "generated": date.today().isoformat(),
        "global": global_section,
        "per_category": per_category,
        "per_class": per_class,
        "summary": {
            "fase1_top1": fase1.get("top1_accuracy"),
            "categories_fase2_beats_fase1": sum(
                1 for v in per_category.values() if v.get("fase2_beats_fase1")
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Fase 1 vs Fase 2 reports")
    parser.add_argument("--fase1", required=True, help="Fase 1 LOO JSON")
    parser.add_argument("--fase2", nargs="+", required=True, help="Fase 2 eval JSON(s)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    f1 = _load(args.fase1)
    f2_list = []
    for path in args.fase2:
        r = _load(path)
        name = f"fase2_{r.get('model_type', Path(path).stem)}"
        f2_list.append((name, r))

    report = compare(f1, f2_list)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Comparison written to {out}")
    print(f"  Fase 1 Top-1: {report['summary']['fase1_top1']:.2%}")
    for name, r in f2_list:
        print(f"  {name} Top-1: {r['top1_accuracy']:.2%}")


if __name__ == "__main__":
    main()
