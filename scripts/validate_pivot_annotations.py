#!/usr/bin/env python3
"""Validate manual runtime-pivot annotations."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pivot_audit.validation import *  # noqa: F401,F403,E402
from pivot_audit.validation import main  # noqa: E402


if __name__ == "__main__":
    main()
