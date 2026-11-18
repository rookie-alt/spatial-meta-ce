#!/usr/bin/env python3
"""Authorized four-target evaluation of sealed PRICE checkpoints."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.price_meta126_runtime import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
