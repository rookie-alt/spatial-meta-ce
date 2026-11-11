#!/usr/bin/env python3
"""Evaluate one frozen 112 checkpoint on the sealed source plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial_meta_ce.cli import data_settings, read_json  # noqa: E402
from spatial_meta_ce.price_protocol import A_ARMS  # noqa: E402
from spatial_meta_ce.source_eval import ARMS, evaluate_source  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arm", required=True, choices=sorted((*ARMS, *A_ARMS)))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--data-config", default="configs/data.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    data_config = data_settings(read_json(args.data_config), args.data_config)
    protocol = evaluate_source(
        args.checkpoint,
        args.arm,
        args.seed,
        args.plan,
        data_config,
        args.out,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
