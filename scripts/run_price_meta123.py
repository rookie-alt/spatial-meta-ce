#!/usr/bin/env python3
"""Isolated, explicitly authorized execution entry point for experiment 123."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.price_meta123_runtime import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
