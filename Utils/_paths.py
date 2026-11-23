"""Bootstrap imports from TrainingPipeline."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_PIPELINE = REPO_ROOT / "TrainingPipeline"
LANDMARK_EXTRACTION = REPO_ROOT / "LandmarkExtraction"
CLI_ENGINE = LANDMARK_EXTRACTION / "LSMExtractorCLI" / "LSMExtractorCLI"
GUI_ENGINE = LANDMARK_EXTRACTION / "LSMExtractorGUI" / "LSMExtractorGUI"

_path = str(TRAINING_PIPELINE)
if _path not in sys.path:
    sys.path.insert(0, _path)
