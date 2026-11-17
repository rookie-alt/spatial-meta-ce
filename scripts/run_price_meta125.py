#!/usr/bin/env python3
"""Explicit official26 training entry point; no target evaluation commands."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.price_meta125_runtime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
