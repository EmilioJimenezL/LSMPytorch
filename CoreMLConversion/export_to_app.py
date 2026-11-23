#!/usr/bin/env python3
"""
Copy a Core ML export + classes JSON (+ optional fixture) into the iOS app bundle.

Usage:
  python scripts/export_to_app.py \
    --mlpackage exports/runs_lab_cap5_cnn_lsmoutput.mlpackage \
    --classes runs_lab_cap5_cnn/classes.json \
    --run-stem runs_lab_cap5_cnn \
    --fixture exports/runs_lab_cap5_cnn_fixture.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from _paths import IOS_APP, REPO_ROOT

IOS = IOS_APP


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy a Core ML export + classes JSON into the iOS testing app bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Example:\n  python export_to_app.py --mlpackage ../Models/CoreML/lab_cap5_cnn.mlpackage "
               "--classes ../Models/CoreML/lab_cap5_cnn_classes.json --run-stem lab_cap5_cnn",
    )
    parser.add_argument("--mlpackage", required=True, help="Path to .mlpackage directory")
    parser.add_argument("--classes", required=True, help="classes.json from the training run")
    parser.add_argument("--run-stem", required=True,
                        help="Stem for bundle files, e.g. lab_cap5_cnn → lab_cap5_cnn_classes.json")
    parser.add_argument("--fixture", default=None, help="Optional Phase-2 parity fixture JSON")
    parser.add_argument("--name", default=None, help="Display name for AVAILABLE_MODELS snippet")
    parser.add_argument("--description", default="")
    parser.add_argument("--family", default="cnn", choices=["cnn", "tcn", "cnn3d", "fingerprint", "legacy"])
    parser.add_argument("--sign-set", default="labCap5")
    args = parser.parse_args()

    src = Path(args.mlpackage)
    if not src.exists():
        print(f"Missing mlpackage: {src}", file=sys.stderr)
        return 1
    classes = json.loads(Path(args.classes).read_text(encoding="utf-8"))
    if not isinstance(classes, list) or not classes:
        print("classes.json must be a non-empty JSON array", file=sys.stderr)
        return 1

    dest_pkg = IOS / src.name
    if dest_pkg.exists():
        shutil.rmtree(dest_pkg)
    shutil.copytree(src, dest_pkg)

    classes_name = f"{args.run_stem}_classes.json"
    classes_dest = IOS / classes_name
    classes_dest.write_text(
        json.dumps(classes, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.fixture:
        fix_src = Path(args.fixture)
        fix_dest = IOS / f"{args.run_stem}_phase2_fixture.json"
        shutil.copy2(fix_src, fix_dest)
        print(f"Fixture → {fix_dest}")

    bundle_name = src.stem  # without .mlpackage
    classes_bundle = classes_name.replace(".json", "")
    print(f"Copied {src.name} → {dest_pkg}")
    print(f"Copied classes ({len(classes)}) → {classes_dest}")
    print("\nRegister in LandmarkPipeline.swift AVAILABLE_MODELS:\n")
    print(f"""    LSMModel(
        name:        "{args.name or bundle_name}",
        bundleName:  "{bundle_name}",
        classesBundleName: "{classes_bundle}",
        description: "{args.description or f'{len(classes)} clases'}",
        family: .{args.family},
        signSet: .{args.sign_set}
    ),""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
