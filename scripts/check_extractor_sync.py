#!/usr/bin/env python3
"""Fail if LSMExtractorCLI and LSMExtractorGUI engine sources have drifted.

Compares Models.swift, DatasetScanner.swift, and LandmarkExtractor.swift
(ignoring import lines, which differ: CoreGraphics vs CoreImage/AppKit).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "AppLSMTests" / "LSMExtractorCLI" / "LSMExtractorCLI"
GUI = ROOT / "AppLSMTests" / "LSMExtractorGUI" / "LSMExtractorGUI"

ENGINE_FILES = ("Models.swift", "DatasetScanner.swift", "LandmarkExtractor.swift")


def strip_imports(text: str) -> str:
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("import "):
            continue
        lines.append(line)
    return "".join(lines)


def main() -> int:
    failed = False
    for name in ENGINE_FILES:
        cli_path = CLI / name
        gui_path = GUI / name
        if not cli_path.exists() or not gui_path.exists():
            print(f"FAIL  missing file: {name}")
            failed = True
            continue
        cli = strip_imports(cli_path.read_text(encoding="utf-8"))
        gui = strip_imports(gui_path.read_text(encoding="utf-8"))
        if cli == gui:
            print(f"OK    {name}")
        else:
            print(f"FAIL  {name} differs between CLI and GUI (beyond imports)")
            failed = True
    if failed:
        print("\nExtractor sources are out of sync. Copy CLI → GUI (or vice versa).")
        return 1
    print("\nExtractor engine sources are in sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
