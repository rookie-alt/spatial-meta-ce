#!/usr/bin/env python3
"""Independently summarise 112 source predictions with NumPy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.plans import EXPECTED_QD  # noqa: E402
from spatial_meta_ce.source_summary import (  # noqa: E402
    summarize_predictions,
    write_summary,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summary = summarize_predictions(args.predictions, expected_qd=EXPECTED_QD)
    write_summary(args.out, summary)
    print(json.dumps(summary["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
