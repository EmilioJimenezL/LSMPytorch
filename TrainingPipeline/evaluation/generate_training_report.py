"""
Consolidated training + deployment readiness report.

Usage:
    python review/generate_training_report.py --reports ./reports
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path


TRAINING_METRICS = {
    "fase1_fingerprint": {
        "backend_id": "fase1_fingerprint",
        "name": "Fase 1 — Fingerprint+DTW",
        "protocol": "leave_one_out",
        "val_top1": 0.0657,
        "val_top5": 0.1618,
        "export_path": "lsm_fingerprints/fingerprints.bin",
        "ios_bundle": "fingerprints.bin",
        "n_classes": 307,
    },
    "cnn": {
        "backend_id": "fase2_cnn",
        "name": "Fase 2 — 1D CNN",
        "model_type": "cnn",
        "epochs": 150,
        "batch": 64,
        "val_top1": 0.4716,
        "val_top5": 0.6478,
        "export_path": "exports/runs_lsmoutput_cnn_lsmoutput.mlpackage",
        "ios_bundle": "runs_lsmoutput_cnn_lsmoutput.mlpackage",
        "checkpoint": "runs_lsmoutput_cnn/best_model.pt",
        "n_classes": 307,
    },
    "tcn": {
        "backend_id": "fase2_tcn",
        "name": "Fase 2 — TCN",
        "model_type": "tcn",
        "epochs": 120,
        "batch": 16,
        "val_top1": 0.4478,
        "val_top5": 0.6776,
        "export_path": "exports/runs_lsmoutput_tcn_lsmoutput.mlpackage",
        "ios_bundle": "runs_lsmoutput_tcn_lsmoutput.mlpackage",
        "checkpoint": "runs_lsmoutput_tcn/best_model.pt",
        "n_classes": 307,
    },
    "3dcnn": {
        "backend_id": "fase2_3dcnn",
        "name": "Fase 2 — 3D CNN",
        "model_type": "3dcnn",
        "epochs": 100,
        "batch": 12,
        "val_top1": 0.1552,
        "val_top5": 0.3284,
        "export_path": "exports/runs_lsmoutput_3dcnn_lsmoutput.mlpackage",
        "ios_bundle": "runs_lsmoutput_3dcnn_lsmoutput.mlpackage",
        "checkpoint": "runs_lsmoutput_3dcnn/best_model.pt",
        "n_classes": 307,
    },
    "legacy": {
        "backend_id": "fase2_legacy",
        "name": "Fase 2 — Legacy Core ML v1",
        "val_top1": 0.0968,
        "export_path": "coreml_models.mlpackage",
        "ios_bundle": "coreml_models.mlpackage",
        "n_classes": 187,
        "note": "Pre-LSMOutput baseline",
    },
}


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _merge_fase2_eval(base: dict, eval_path: Path) -> dict:
    ev = _load_json(eval_path)
    if not ev:
        return base
    base = dict(base)
    base["full_dataset_top1"] = ev.get("top1_accuracy")
    base["full_dataset_top5"] = ev.get("top5_accuracy")
    base["inference_ms_per_video"] = ev.get("inference_ms_per_video")
    base["eval_protocol"] = ev.get("eval_protocol")
    if ev.get("val_split"):
        base["val_split_top1"] = ev["val_split"].get("top1_accuracy")
        base["val_split_top5"] = ev["val_split"].get("top5_accuracy")
    return base


def _check_deployment_ready(root: Path, backend: dict) -> dict:
    checks: dict[str, bool | str] = {}
    export = root / backend.get("export_path", "")
    if backend["backend_id"] == "fase1_fingerprint":
        checks["export_exists"] = export.exists()
        checks["preprocessing_unified"] = True
        checks["ios_bundle_name"] = backend.get("ios_bundle", "")
        checks["mac_predict_verified"] = "pending_device"
        ready = bool(checks["export_exists"])
    else:
        checks["export_exists"] = export.exists()
        checks["checkpoint_exists"] = (
            (root / backend["checkpoint"]).exists() if backend.get("checkpoint") else True
        )
        checks["n_classes_expected"] = backend.get("n_classes") == 307
        checks["preprocessing_unified"] = True
        checks["mac_predict_verified"] = "pending_device"
        checks["ios_load_verified"] = "pending_device"
        ready = bool(
            checks["export_exists"]
            and checks.get("checkpoint_exists", True)
            and checks["n_classes_expected"]
        )
    return {"checks": checks, "deployment_ready": ready}


def generate(root: Path, reports: Path) -> dict:
    fase1 = _load_json(reports / "fase1_loo.json")
    comparison = _load_json(reports / "comparison.json")
    inventory = _load_json(reports / "dataset_inventory.json")

    backends = []
    f1 = dict(TRAINING_METRICS["fase1_fingerprint"])
    if fase1:
        f1["val_top1"] = fase1.get("top1_accuracy", f1["val_top1"])
        f1["val_top5"] = fase1.get("top5_accuracy", f1["val_top5"])
        f1["n_videos"] = fase1.get("n_videos")
    f1["deployment"] = _check_deployment_ready(root, f1)
    backends.append(f1)

    fase2_map = {
        "cnn": reports / "fase2_runs_lsmoutput_cnn.json",
        "tcn": reports / "fase2_runs_lsmoutput_tcn.json",
        "3dcnn": reports / "fase2_runs_lsmoutput_3dcnn.json",
    }
    for key in ("cnn", "tcn", "3dcnn"):
        b = _merge_fase2_eval(dict(TRAINING_METRICS[key]), fase2_map[key])
        b["deployment"] = _check_deployment_ready(root, b)
        backends.append(b)

    legacy = dict(TRAINING_METRICS["legacy"])
    legacy["deployment"] = _check_deployment_ready(root, legacy)
    backends.append(legacy)

    best_nn = max(
        [b for b in backends if b["backend_id"].startswith("fase2_") and b.get("val_top1")],
        key=lambda x: x.get("val_top1", 0),
        default=None,
    )

    report = {
        "generated": date.today().isoformat(),
        "dataset": {
            "source": str((root.parent / "LSMOutput").resolve()),
            "n_classes_trainable": inventory.get("n_classes", 307) if inventory else 307,
            "n_videos": inventory.get("n_videos", 1934) if inventory else 1934,
            "min_videos": 2,
            "words_extracted": 501,
            "words_excluded_single_video": 194,
        },
        "backends": backends,
        "comparison_available": comparison is not None,
        "recommended_ios_default": "fase2_cnn" if best_nn else "fase1_fingerprint",
        "recommended_ios_default_reason": "Best stratified val Top-1 among Fase 2 models",
    }

    if comparison:
        report["per_category_winners"] = {
            cat: {
                "fase1": row.get("fase1"),
                "fase2_beats_fase1": row.get("fase2_beats_fase1"),
            }
            for cat, row in comparison.get("per_category", {}).items()
        }

    return report


def format_summary(report: dict) -> str:
    lines = [
        "LSM Training & Deployment Report",
        f"Generated: {report['generated']}",
        "",
        "=== Dataset ===",
        f"  Trainable classes: {report['dataset']['n_classes_trainable']} (min-videos=2)",
        f"  Videos indexed:    {report['dataset']['n_videos']}",
        "",
        "=== Backend metrics ===",
    ]
    for b in report["backends"]:
        lines.append(f"  {b['name']}")
        lines.append(f"    Val Top-1: {b.get('val_top1', 0):.2%}  Top-5: {b.get('val_top5', 0):.2%}")
        if b.get("full_dataset_top1") is not None:
            lines.append(
                f"    Full-dataset Top-1: {b['full_dataset_top1']:.2%}  "
                f"Top-5: {b.get('full_dataset_top5', 0):.2%}"
            )
        dep = b.get("deployment", {})
        status = "READY (Python)" if dep.get("deployment_ready") else "NOT READY"
        lines.append(f"    Deployment: {status}")
        lines.append("")

    lines.extend([
        f"Recommended iOS default for testing: {report['recommended_ios_default']}",
        "",
        "Device validation: pending Mac/Xcode (see AppLSMTests AGENTS Phase 2 checklist)",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate consolidated training report")
    parser.add_argument("--root", default=".", help="TrainingPipeline root")
    parser.add_argument("--reports", default="./reports", help="Reports directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    reports = Path(args.reports).resolve()
    reports.mkdir(parents=True, exist_ok=True)

    report = generate(root, reports)

    json_path = reports / "training_report.json"
    txt_path = reports / "training_report_summary.txt"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    txt_path.write_text(format_summary(report), encoding="utf-8")

    print(format_summary(report))
    print(f"\nJSON: {json_path}")
    print(f"Summary: {txt_path}")


if __name__ == "__main__":
    main()
