"""Repository root discovery and the standard directory layout.

Always access directories as ``paths.WORKSPACE`` (module attribute) rather than
``from .paths import WORKSPACE`` so that ``--workspace`` overrides take effect.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

# Orchestrator/lsm_cli/paths.py -> parents[0]=lsm_cli, [1]=Orchestrator, [2]=repo root
ORCHESTRATOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ORCHESTRATOR_DIR.parent

TRAINING = REPO_ROOT / "TrainingPipeline"
CONVERSION = REPO_ROOT / "CoreMLConversion"
EXTRACTOR_DIR = REPO_ROOT / "LandmarkExtraction" / "LSMExtractorCLI"
EXTRACTOR_PROJECT = EXTRACTOR_DIR / "LSMExtractorCLI.xcodeproj"
EXTRACTOR_BUILD = EXTRACTOR_DIR / "build"
IOS_APP_DIR = REPO_ROOT / "TestingTools" / "LSMMobileModelTesting"

RUNS = REPO_ROOT / "Runs"
MODELS_PYTORCH = REPO_ROOT / "Models" / "Pytorch"
MODELS_COREML = REPO_ROOT / "Models" / "CoreML"
REQUIREMENTS = TRAINING / "requirements.txt"
VENV = REPO_ROOT / ".venv"

# Script entry points that the orchestrator wraps.
PREPARE_SCRIPT = TRAINING / "prepare_dataset.py"
TRAIN_SCRIPT = TRAINING / "train.py"
EVALUATE_SCRIPT = TRAINING / "evaluation" / "evaluate_nn.py"
CONVERT_SCRIPT = CONVERSION / "convert_to_coreml.py"
EXPORT_SCRIPT = CONVERSION / "export_to_app.py"

# --- Workspace (overridable with --workspace) -------------------------------
WORKSPACE = REPO_ROOT / "Workspace"
LOGS_DIR = WORKSPACE / "logs"
PREPARED_DIR = WORKSPACE / "prepared"
LANDMARKS_DIR = WORKSPACE / "landmarks"
EXPORTS_DIR = WORKSPACE / "exports"


def set_workspace(path: Optional[os.PathLike]) -> None:
    """Re-point every Workspace/* directory (used by the global --workspace flag)."""
    global WORKSPACE, LOGS_DIR, PREPARED_DIR, LANDMARKS_DIR, EXPORTS_DIR
    if path is None:
        return
    root = Path(os.path.expanduser(str(path)))
    if not root.is_absolute():
        root = Path.cwd() / root
    WORKSPACE = root.resolve()
    LOGS_DIR = WORKSPACE / "logs"
    PREPARED_DIR = WORKSPACE / "prepared"
    LANDMARKS_DIR = WORKSPACE / "landmarks"
    EXPORTS_DIR = WORKSPACE / "exports"


def is_windows() -> bool:
    return os.name == "nt"


def is_macos() -> bool:
    return sys.platform == "darwin"


def venv_python(venv: Path = VENV) -> Path:
    """Path of the interpreter inside the virtualenv (may not exist yet)."""
    if is_windows():
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def venv_exists() -> bool:
    return venv_python().exists()


def rel(path: os.PathLike) -> str:
    """Display a path relative to the repo root when it lives inside it."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO_ROOT)) or "."
    except (ValueError, OSError):
        return str(p)


def resolve_user_path(raw: os.PathLike) -> Path:
    """Expand ~, strip quotes (drag-and-drop into a terminal) and make absolute."""
    text = str(raw).strip().strip("'\"")
    p = Path(os.path.expanduser(text))
    if not p.is_absolute():
        p = Path.cwd() / p
    return Path(os.path.normpath(str(p)))


def check_repo_layout() -> List[str]:
    """Return the standard repo directories that are missing (empty = healthy)."""
    needed = [TRAINING, CONVERSION, TRAIN_SCRIPT, PREPARE_SCRIPT, CONVERT_SCRIPT]
    return [str(p) for p in needed if not p.exists()]
