#!/usr/bin/env python3
"""Paired architecture comparison on SOURCE VALIDATION only.

Why this exists: the 2026-09-27 selection picked ``sum`` from four
architectures on three seeds, but the gap between architectures was smaller
than the seed-to-seed SD of the arms themselves. That comparison also had two
confounds this script is built to avoid:

1. Each arm used its own early-stopped ``best.pt``, chosen on the same
   validation databases it was later reported on.
2. Selection used whole-table macro *log*-Q95, while the project's headline
   metric is the 4+ table bucket Q95 in Q space.

So the comparison here is **paired by seed**: for a given seed, both
architectures share the training stream and the evaluation plan, so their
difference cancels the seed variance instead of being drowned by it. No
significance test is computed -- n is small and the SD estimate itself rests
on few observations, so a p-value would overstate what this data supports.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path

ALL_ARCHS = ("set", "sum", "attn", "attn_pair")
# Primary metric; the secondary comparison reuses the whole-table log-Q95.
FOUR_PLUS_KEY = "four_plus_q95"
REFERENCE_KEY = "macro_log_q95"
MIN_PAIRS_FOR_VERDICT = 2
# Differences below this are floating-point noise, not a direction.
_TINY = 1e-12


def paired_verdict(deltas: list[float]) -> dict:
    """Judge one paired difference series; never silently rank.

    ``deltas`` are per-seed (left - right) on a lower-is-better metric. A sign
    only counts as a finding when the mean difference is at least as large as
    its own spread; otherwise the honest answer is "cannot tell", which is what
    ``inconclusive`` says.
    """
    clean = [float(d) for d in deltas]
    if any(not math.isfinite(d) for d in clean):
        raise ValueError("Paired differences must be finite")
    if len(clean) < MIN_PAIRS_FOR_VERDICT:
        return {
            "verdict": "inconclusive",
            "reason": f"fewer than {MIN_PAIRS_FOR_VERDICT} paired seeds",
            "n_pairs": len(clean),
            "mean_delta": None,
            "sd_delta": None,
            "wins_left": None,
        }
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean)
    wins_left = sum(d < 0 for d in clean)
    if abs(mean) < sd or abs(mean) < _TINY:
        verdict = "inconclusive"
    elif mean < 0:
        verdict = "lower_is_better_left"
    else:
        verdict = "lower_is_better_right"
    return {
        "verdict": verdict,
        "n_pairs": len(clean),
        "mean_delta": mean,
        "sd_delta": sd,
        "wins_left": wins_left,
        "wins_right": len(clean) - wins_left,
    }


def select_cells(cells: list[dict]) -> dict[str, dict[int, dict]]:
    """arch -> seed -> cell, refusing an incomplete or duplicated matrix."""
    grid: dict[str, dict[int, dict]] = {}
    for cell in cells:
        arch, seed = cell.get("arch"), cell.get("seed")
        if arch not in ALL_ARCHS:
            raise ValueError(f"Unknown architecture {arch!r}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"Invalid seed {seed!r}")
        macro = cell.get(FOUR_PLUS_KEY, {}).get("macro")
        if macro is None:
            raise ValueError(
                f"Cell {arch}/seed{seed} has no 4+ macro; the comparison needs a "
                "database set that actually contains 4+ table queries"
            )
        if not math.isfinite(macro) or macro < 1:
            raise ValueError(f"Cell {arch}/seed{seed} has an invalid 4+ macro")
        by_seed = grid.setdefault(arch, {})
        if seed in by_seed:
            raise ValueError(f"Duplicate cell {arch}/seed{seed}")
        by_seed[seed] = cell
    missing = [a for a in ALL_ARCHS if a not in grid]
    if missing:
        raise ValueError(f"Missing architectures: {missing}")
    seed_sets = {a: set(v) for a, v in grid.items()}
    first = seed_sets[ALL_ARCHS[0]]
    for arch, seeds in seed_sets.items():
        if seeds != first:
            raise ValueError(f"Seed sets differ: {arch} has {sorted(seeds)}")
    return grid


def metric_value(cell: dict, key: str):
    """Read one scalar metric out of a cell, refusing a missing value.

    The 4+ metric is nested (``four_plus_q95`` -> ``macro``); every other key
    is a flat field on the cell.
    """
    value = cell[FOUR_PLUS_KEY]["macro"] if key == FOUR_PLUS_KEY else cell.get(key)
    if value is None:
        raise ValueError(f"Cell {cell.get('arch')}/seed{cell.get('seed')} lacks {key}")
    return value


def compare(grid: dict[str, dict[int, dict]], key: str) -> list[dict]:
    """All architecture pairs, paired by seed, reading ``key`` from each cell."""
    rows = []
    for left, right in combinations(ALL_ARCHS, 2):
        seeds = sorted(grid[left])
        deltas = [
            metric_value(grid[left][seed], key) - metric_value(grid[right][seed], key)
            for seed in seeds
        ]
        rows.append(
            {
                "left": left,
                "right": right,
                "metric": key,
                **paired_verdict(deltas),
                "per_seed_delta": deltas,
            }
        )
    return rows


def load_cells(paths: list[Path]) -> list[dict]:
    cells = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for name, result in data["results"].items():
            arch, _, seed = name.rpartition("-s")
            cells.append(
                {
                    "arch": arch,
                    "seed": int(seed),
                    FOUR_PLUS_KEY: result[FOUR_PLUS_KEY]["zero_shot"],
                    REFERENCE_KEY: result[REFERENCE_KEY],
                    "checkpoint": result.get("path"),
                    "parameters": result.get("parameters"),
                    "source": str(path),
                }
            )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        type=Path,
        help="arch_select output JSON files",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()

    cells = load_cells(args.results)
    grid = select_cells(cells)
    rule = "|mean paired delta| < paired delta SD -> inconclusive; no ranking emitted"
    summary = {
        "schema": 1,
        "purpose": "paired architecture comparison on source validation only",
        "target_databases_read": [],
        "metric": "four_plus_q95 (macro over databases reporting the bucket)",
        "rule": rule,
        "archs": list(ALL_ARCHS),
        "seeds": sorted(grid[ALL_ARCHS[0]]),
        "cells": cells,
        "comparisons": compare(grid, FOUR_PLUS_KEY),
        "reference_comparisons": compare(grid, REFERENCE_KEY),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    args.protocol.write_text(
        json.dumps(
            {
                "schema": 1,
                "rule": rule,
                "min_pairs_for_verdict": MIN_PAIRS_FOR_VERDICT,
                "architecture_set": list(ALL_ARCHS),
                "aggregation": "per-seed paired delta; mean and sample SD (ddof=1)",
                "no_significance_test": (
                    "n is small and the SD rests on few observations; a p-value would "
                    "overstate what this data supports"
                ),
                "inputs": [str(p) for p in args.results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for row in summary["comparisons"]:
        print(
            f"{row['left']:>9} vs {row['right']:<9} {row['verdict']:<22} "
            f"mean={row['mean_delta']} sd={row['sd_delta']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
