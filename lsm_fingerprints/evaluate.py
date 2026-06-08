"""
LSM Phase 1 — Leave-one-out evaluation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from build_db import load_database
from constants import CONFIDENCE_THR, TOP_K_DTW, TOP_K_FILTER
from fingerprint import FingerprintExtractor
from matcher import _cosine_distance, _dtw_distance


def _match_exclude_self(
    query_seq: np.ndarray,
    query_fp: np.ndarray,
    fingerprints: np.ndarray,
    labels: np.ndarray,
    sequences: np.ndarray,
    classes: list[str],
    exclude_idx: int,
    top_k_filter: int = TOP_K_FILTER,
    top_k_dtw: int = TOP_K_DTW,
) -> list[dict]:
    mask = np.ones(len(labels), dtype=bool)
    mask[exclude_idx] = False

    fp_subset = fingerprints[mask]
    label_subset = labels[mask]
    seq_subset = sequences[mask]

    dists = np.array([_cosine_distance(query_fp, fp_subset[i]) for i in range(len(fp_subset))])
    filter_k = min(top_k_filter, len(dists))
    candidate_local = np.argsort(dists)[:filter_k]

    dtw_results = []
    for local_i in candidate_local:
        dtw_dist = _dtw_distance(query_seq, seq_subset[local_i])
        dtw_results.append((int(label_subset[local_i]), dtw_dist))

    dtw_results.sort(key=lambda x: x[1])
    top_dtw = dtw_results[:top_k_dtw]

    class_scores: dict[int, float] = {}
    for label, dtw_dist in top_dtw:
        score = 1.0 / (dtw_dist + 1e-6)
        class_scores[label] = class_scores.get(label, 0.0) + score

    total = sum(class_scores.values()) or 1.0
    ranked = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (label, score) in enumerate(ranked[:top_k_dtw]):
        norm_score = score / total
        if norm_score < CONFIDENCE_THR and rank > 0:
            continue
        results.append({
            "class": classes[label],
            "score": float(norm_score),
            "rank": rank + 1,
            "label_idx": label,
        })
    return results


def evaluate_leave_one_out(db_path: str) -> dict:
    db = load_database(db_path)
    fingerprints = db["fingerprints"]
    labels = db["labels"]
    sequences = db["sequences"]
    classes = db["classes"]
    categories = db.get("categories") or [""] * len(labels)

    extractor = FingerprintExtractor()
    n = len(labels)

    top1_correct = 0
    top5_correct = 0
    per_class: dict[str, list[bool]] = defaultdict(list)
    per_category: dict[str, list[bool]] = defaultdict(list)

    print(f"\n  Leave-one-out sobre {n} videos, {len(classes)} clases...")

    for i in range(n):
        true_label = int(labels[i])
        true_class = classes[true_label]
        cat = categories[i] if i < len(categories) else ""

        query_fp = fingerprints[i]
        query_seq = sequences[i]

        results = _match_exclude_self(
            query_seq, query_fp,
            fingerprints, labels, sequences, classes,
            exclude_idx=i,
        )

        pred_labels = [r["label_idx"] for r in results]
        hit1 = len(pred_labels) > 0 and pred_labels[0] == true_label
        hit5 = true_label in pred_labels[:5]

        top1_correct += int(hit1)
        top5_correct += int(hit5)
        per_class[true_class].append(hit1)
        per_category[cat or "Sin categoría"].append(hit1)

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n} — running top-1: {top1_correct / (i + 1):.2%}")

    top1_acc = top1_correct / n
    top5_acc = top5_correct / n

    class_acc = {c: sum(v) / len(v) for c, v in per_class.items() if v}
    category_acc = {c: sum(v) / len(v) for c, v in per_category.items() if v}

    report = {
        "n_videos": n,
        "n_classes": len(classes),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "per_class_accuracy": class_acc,
        "per_category_accuracy": category_acc,
    }

    print(f"\n  Resultados LOO:")
    print(f"    Top-1 : {top1_acc:.2%} ({top1_correct}/{n})")
    print(f"    Top-5 : {top5_acc:.2%} ({top5_correct}/{n})")
    print(f"\n  Por categoría:")
    for cat, acc in sorted(category_acc.items(), key=lambda x: -x[1]):
        print(f"    {cat:<35} {acc:.2%}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evalúa fingerprints.npz (LOO)")
    parser.add_argument("--db", required=True, help="Ruta a fingerprints.npz")
    args = parser.parse_args()
    evaluate_leave_one_out(args.db)


if __name__ == "__main__":
    main()
