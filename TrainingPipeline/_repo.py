"""Locate the repository root from anywhere under LSMPytorch."""

from __future__ import annotations

from pathlib import Path

_TRAINING = Path(__file__).resolve().parent
REPO_ROOT = _TRAINING.parent
TRAINING_PIPELINE = _TRAINING


def ensure_training_on_path() -> Path:
    """Insert TrainingPipeline on sys.path and return it."""
    import sys

    path = str(TRAINING_PIPELINE)
    if path not in sys.path:
        sys.path.insert(0, path)
    return TRAINING_PIPELINE
