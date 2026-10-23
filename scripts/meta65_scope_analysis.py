"""Descriptive contrasts on certified scope outputs; never a candidate selector."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

BUCKETS = ("low4", "low13", "nonlow4", "nonlow13")
MIN_COUNT = 30


def _mean(values: list[float]) -> float:
    return math.fsum(v / len(values) for v in values)


def summarize_runs(runs: list[dict]) -> dict[str, Any]:
    """Apply all frozen descriptive AND conditions without dropping empty runs."""
    if len(runs) != 5 or {r["run"] for r in runs} != set(range(5)):
        raise ValueError("Five complete distinct runs required")
    for row in runs:
        if not math.isfinite(row["mean_delta_logq"]):
            raise ValueError("Nonfinite paired log-Q change")
        if set(row["buckets"]) != set(BUCKETS):
            raise ValueError("Four complete contrast buckets required")
        for b in row["buckets"].values():
            if type(b["count"]) is not int or b["count"] < 0:
                raise ValueError("Invalid bucket count")
            for field in ("none_q95", "head_q95"):
                value = b[field]
                if b["count"] == 0:
                    if value is not None:
                        raise ValueError("Empty bucket must remain null")
                elif (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 1
                ):
                    raise ValueError("Invalid standard Q95")
    sufficient = all(r["buckets"]["low4"]["count"] >= MIN_COUNT for r in runs)
    if not sufficient:
        stable, ratio, differences = None, None, []
    else:
        ratio = _mean([r["buckets"]["low4"]["head_q95"] for r in runs]) / _mean(
            [r["buckets"]["low4"]["none_q95"] for r in runs]
        )
        differences = [
            r["buckets"]["low4"]["head_q95"] - r["buckets"]["low4"]["none_q95"]
            for r in runs
        ]
        stable = (
            ratio > 1.10
            and sum(x > 0 for x in differences) >= 4
            and all(_mean(differences[:i] + differences[i + 1 :]) > 0 for i in range(5))
            and sum(r["mean_delta_logq"] > 0 for r in runs) >= 4
        )
    adequate = all(r["buckets"][b]["count"] >= MIN_COUNT for r in runs for b in BUCKETS)
    interactions, stronger = [], []
    if adequate:
        for row in runs:
            ratios = {
                b: math.log10(
                    row["buckets"][b]["head_q95"] / row["buckets"][b]["none_q95"]
                )
                for b in BUCKETS
            }
            interactions.append(
                ratios["low4"]
                - ratios["low13"]
                - ratios["nonlow4"]
                + ratios["nonlow13"]
            )
            stronger.append(
                ratios["low4"] > ratios["low13"] and ratios["low4"] > ratios["nonlow4"]
            )
    positive = (
        (
            sum(stronger) >= 4
            and sum(x > 0 for x in interactions) >= 4
            and all(
                _mean(interactions[:i] + interactions[i + 1 :]) > 0 for i in range(5)
            )
        )
        if adequate
        else None
    )
    return {
        "low4_ratio_vs_none": ratio,
        "stable_aggravation_descriptive": stable,
        "positive_q95_runs": sum(x > 0 for x in differences) if sufficient else None,
        "positive_mean_delta_logq_runs": sum(r["mean_delta_logq"] > 0 for r in runs),
        "joint_stronger_runs": sum(stronger) if adequate else None,
        "positive_interaction_runs": sum(x > 0 for x in interactions)
        if adequate
        else None,
        "interaction_mean": _mean(interactions) if adequate else None,
        "interaction_runs": interactions if adequate else [None] * 5,
        "positive_interaction_descriptive": positive,
    }


def analyze(output: Path, *, formal: bool = True) -> dict[str, Any]:
    """Read authenticated arrays after independent matrix verification."""
    if not formal:
        return {"formal_result": False, "summary": None}
    import numpy as np

    from scripts.meta65_scope_verify import _cell_arrays, read_json

    cells = read_json(output / "cells.json")["cells"]
    lookup = {(c["arm"], c["operator"], c["database"], c["run"]): c for c in cells}
    expected = {
        (a, op, db, run)
        for a in ("R_head3", "T_all3")
        for op in ("none", "head3")
        for db in ("chembl", "legalacts", "telstra")
        for run in range(5)
    }
    if len(cells) != 60 or set(lookup) != expected:
        raise ValueError("Complete 60-cell matrix required for descriptive analysis")
    summaries = []
    for arm in ("R_head3", "T_all3"):
        for db in ("chembl", "legalacts", "telstra"):
            runs = []
            for run in range(5):
                identity, none = _cell_arrays(
                    output, lookup[arm, "none", db, run], formal=True
                )
                _, head = _cell_arrays(
                    output, lookup[arm, "head3", db, run], formal=True
                )
                true, tables = identity["true"], identity["n_tables"]
                low, complex_query = true < 100, tables >= 4
                masks = dict(
                    zip(
                        BUCKETS,
                        (
                            low & complex_query,
                            low & ~complex_query,
                            ~low & complex_query,
                            ~low & ~complex_query,
                        ),
                    )
                )
                qlogs = {
                    op: np.abs(np.maximum(pred, 0) - np.log(np.maximum(true, 1)))
                    for op, pred in (("none", none), ("head", head))
                }
                with np.errstate(over="raise", invalid="raise"):
                    qs = {op: np.exp(q) for op, q in qlogs.items()}
                buckets = {
                    b: {
                        "count": int(mask.sum()),
                        **{
                            f"{op}_q95": float(
                                np.quantile(q[mask], 0.95, method="linear")
                            )
                            if mask.any()
                            else None
                            for op, q in qs.items()
                        },
                    }
                    for b, mask in masks.items()
                }
                if not masks["low4"].any():
                    raise ValueError("Main descriptive bucket empty")
                runs.append(
                    {
                        "run": run,
                        "buckets": buckets,
                        "mean_delta_logq": float(
                            (qlogs["head"] - qlogs["none"])[masks["low4"]].mean()
                        ),
                    }
                )
            summaries.append(
                {
                    "arm": arm,
                    "database": db,
                    "runs": runs,
                    "contrasts": summarize_runs(runs),
                }
            )
    return {
        "schema": 1,
        "formal_result": True,
        "scope": "descriptive_only_not_candidate",
        "metric": "five-run Q95 means; natural log-Q changes; log10 ratio interaction",
        "summary": summaries,
    }
