"""Error types, exit codes and known-failure message mapping."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

EXIT_OK = 0
EXIT_USER = 1  # bad arguments, missing files, wrong platform, ...
EXIT_STEP = 2  # a wrapped step (training, conversion, build, ...) failed
EXIT_INTERRUPT = 130  # Ctrl-C (shell convention)


class CliError(Exception):
    """An error that should be shown to the user without a traceback.

    Attributes:
        message:   one-line description of what went wrong
        hint:      actionable next step ("run `lsm setup`", ...)
        exit_code: 1 for user errors, 2 for step failures
        details:   optional extra text (e.g. the tail of a failing command)
        log_path:  log file containing full output, if any
    """

    exit_code = EXIT_USER

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        exit_code: Optional[int] = None,
        details: Optional[str] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details
        self.log_path = log_path
        if exit_code is not None:
            self.exit_code = exit_code


class UserError(CliError):
    """Invalid arguments / environment problem (exit code 1)."""

    exit_code = EXIT_USER


class StepError(CliError):
    """A wrapped command ran but failed (exit code 2)."""

    exit_code = EXIT_STEP


# (regex, message template, hint). Group references (\1) are expanded in message.
_KNOWN: Sequence[Tuple[str, str, str]] = (
    (
        r"externally-managed-environment",
        "This Python is 'externally managed' (PEP 668); pip refuses to install into it.",
        "Use the project virtualenv: run `lsm setup` (creates .venv) and retry.",
    ),
    (
        r"No module named '?(torch|torchvision|numpy|scipy|coremltools|sklearn|yaml|pandas|matplotlib|pytest)\b",
        "Python package '\\1' is not installed in the interpreter that was used.",
        "Run `lsm setup` to create .venv and install dependencies, then `lsm doctor` to verify.",
    ),
    (
        r"Could not find a version that satisfies the requirement torch|No matching distribution found for torch",
        "pip could not find a matching PyTorch wheel for this Python/platform.",
        "PyTorch wheels usually lag new Python releases: use Python 3.10-3.12. "
        "On Linux/Windows pick a wheel index with `lsm setup --cuda 12.4` or `lsm setup --cpu`.",
    ),
    (
        r"CUDA out of memory|CUDA error: out of memory",
        "The GPU ran out of memory.",
        "Lower the batch size (e.g. `--batch 16`), reduce `--augment`, or use a smaller model.",
    ),
    (
        r"MPS backend out of memory|MPS.{0,40}out of memory",
        "The Apple GPU (MPS) ran out of memory.",
        "Lower the batch size (e.g. `--batch 16`) or reduce `--augment`.",
    ),
    (
        r"Core ML parity gate failed",
        "The Core ML model disagrees with the PyTorch model (parity gate failed).",
        "Do not ship this export. Re-train/re-convert, or pass `--no-parity` only to inspect it.",
    ),
    (
        r"BlobWriter not loaded|libcoremlpython|libmilstoragepython|Failed to load _MLModelProxy",
        "coremltools native libraries are not available for this Python/OS combination.",
        "Use Python 3.10-3.12 in .venv (`lsm setup`), then check `lsm doctor`. "
        "Core ML prediction/parity only works on macOS.",
    ),
    (
        r"Error\(s\) in loading state_dict|size mismatch for",
        "The checkpoint does not match the model architecture.",
        "Make sure the checkpoint, --model, --pool and --causal match how it was trained.",
    ),
    (
        r"invalid load key, 'v'|version https://git-lfs",
        "The checkpoint is a Git LFS pointer file, not the real weights.",
        "Run `git lfs install && git lfs pull` and retry (see `lsm doctor`).",
    ),
    (
        r"No valid sequences loaded",
        "No usable landmark sequences were found in the dataset.",
        "Check that the folder has {category}/{word}/*_landmarks.json files and that "
        "--min-videos is not higher than the videos per word (try `--min-videos 2`).",
    ),
    (
        r"--dataset or --cache is required",
        "Training needs data.",
        "Pass `--landmarks DIR` (raw landmark JSONs) or `--cache FILE.npz` (from `lsm prepare`).",
    ),
    (
        r"Missing mlpackage",
        "The Core ML package to deploy does not exist.",
        "Run `lsm convert --run NAME` first.",
    ),
    (
        r"xcode-select: error|requires Xcode, but active developer directory|"
        r"agreed to the Xcode license|xcodebuild: error: .*(SDK|scheme|project)",
        "Xcode is not set up for command-line builds.",
        "Install Xcode, then: `sudo xcode-select -s /Applications/Xcode.app` and "
        "`sudo xcodebuild -license accept`. Or build LSMExtractorCLI once from Xcode (Release).",
    ),
    (
        r"No space left on device",
        "The disk is full.",
        "Free some space (old Runs/ checkpoints, Workspace/prepared/) and retry.",
    ),
    (
        r"PermissionError|Permission denied",
        "Permission denied while reading or writing a file.",
        "Check ownership/permissions of the paths shown in the log.",
    ),
)

_COMPILED = [(re.compile(p, re.IGNORECASE | re.DOTALL), m, h) for p, m, h in _KNOWN]


def explain_failure(output: str) -> Optional[Tuple[str, str]]:
    """Map raw command output to a (message, hint) pair if it is a known failure."""
    for rx, message, hint in _COMPILED:
        match = rx.search(output)
        if match:
            try:
                message = match.expand(message)
            except (re.error, IndexError):
                pass
            return message, hint
    return None
