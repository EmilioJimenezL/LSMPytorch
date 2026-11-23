"""Bootstrap Fase 1 package + TrainingPipeline imports."""

from __future__ import annotations

import sys
from pathlib import Path

_FASE1 = Path(__file__).resolve().parent
PKG = _FASE1 / "lsm_fingerprints"
REPO_ROOT = _FASE1.parent.parent
TRAINING_PIPELINE = REPO_ROOT / "TrainingPipeline"
IOS_APP = (
    REPO_ROOT
    / "TestingTools"
    / "LSMMobileModelTesting"
    / "LSMMobileModelTesting"
)

for p in (str(PKG), str(TRAINING_PIPELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)
