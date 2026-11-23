"""Entry point: ``python -m lsm_cli`` or ``python Orchestrator/lsm_cli``."""

import os
import sys

if __package__ in (None, ""):
    # Executed as a directory/script (python Orchestrator/lsm_cli): make the package importable.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lsm_cli.main import main
else:
    from .main import main

if __name__ == "__main__":
    sys.exit(main())
