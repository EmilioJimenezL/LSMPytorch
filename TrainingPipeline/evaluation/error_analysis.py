"""
Drill into worst-performing classes from a comparison report.

Usage:
    python review/error_analysis.py --comparison reports/comparison.json --top 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SUGGESTIONS = {
    "Abecedario": "High confusion expected — consider category-scoped DB or separate letter model",
    "default_low_data": "Collect more videos (currently near min-videos threshold)",
    "both_fail": "Review preprocessing parity and landmark quality for this sign",
    "fase2_wins": "Neural model captures pattern Fase 1 misses — prioritize Fase 2 for this class",
    "fase1_wins": "DTW template matching works — check if NN needs more augmentation for this class",
}


def analyze(comparison: dict, top_n: int = 20) -> dict:
    per_class = comparison.get("per_class", [])
    f1_top1 = comparison.get("summary", {}).get("fase1_top1", 0)

    worst = []
    for row in per_class:
        f1 = row.get("fase1")
        if f1 is None:
            continue
        f2_vals = {k: v for k, v in row.items() if k.startswith("fase2_") and v is not None}
        best_f2 = max(f2_vals.values()) if f2_vals else 0.0
        worst.append({
            "class": row["class"],
            "fase1_acc": f1,
            "best_fase2_acc": best_f2,
            "delta": best_f2 - f1,
            "both_below_20pct": f1 < 0.2 and best_f2 < 0.2,
        })

    worst.sort(key=lambda x: min(x["fase1_acc"], x["best_fase2_acc"]))

    findings = []
    for item in worst[:top_n]:
        cls = item["class"]
        actions = []
        if item["both_below_20pct"]:
            actions.append(SUGGESTIONS["both_fail"])
        if item["delta"] > 0.1:
            actions.append(SUGGESTIONS["fase2_wins"])
        elif item["delta"] < -0.1:
            actions.append(SUGGESTIONS["fase1_wins"])
        if "Abecedario" in cls or cls in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            actions.append(SUGGESTIONS["Abecedario"])
        if not actions:
            actions.append(SUGGESTIONS["default_low_data"])

        findings.append({**item, "suggested_actions": actions})

    return {
        "fase1_baseline_top1": f1_top1,
        "n_classes_analyzed": len(worst),
        "worst_classes": findings,
    }


def main():
    parser = argparse.ArgumentParser(description="Error analysis from comparison report")
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.comparison, encoding="utf-8") as f:
        comparison = json.load(f)

    report = analyze(comparison, args.top)
    text_lines = [
        f"Error analysis — Fase 1 baseline Top-1: {report['fase1_baseline_top1']:.2%}",
        "",
    ]
    for i, item in enumerate(report["worst_classes"], 1):
        text_lines.append(
            f"{i}. {item['class']}: Fase1={item['fase1_acc']:.1%} "
            f"best_Fase2={item['best_fase2_acc']:.1%} delta={item['delta']:+.1%}"
        )
        for a in item["suggested_actions"]:
            text_lines.append(f"   → {a}")

    text = "\n".join(text_lines)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        summary = out.with_suffix(".txt")
        summary.write_text(text, encoding="utf-8")
        print(f"\nJSON: {out}\nSummary: {summary}")


if __name__ == "__main__":
    main()
