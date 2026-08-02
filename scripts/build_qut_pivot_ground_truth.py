#!/usr/bin/env python3
"""Build report-backed runtime-pivot candidates for QUT-DV25."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pivot_audit.qut import main  # noqa: E402


if __name__ == "__main__":
    main()
