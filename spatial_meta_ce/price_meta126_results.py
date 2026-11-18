"""Target-side aggregation for the frozen PRICE meta-learning target protocol.

The module only consumes completed evaluation cells and refuses anything that
does not match the frozen matrix: wrong identity, duplicate cell, missing
cell, non-finite metric, failed cell or drifted query denominator.  Invalid
cells are rejected, never filtered.

Aggregation order is fixed by the protocol:

* few-shot headline: per seed and database the arithmetic mean over the five
  support draws of Q95 is taken first, then the logarithm, then the
  arithmetic mean over the three few-shot databases gives ``L``; the paired
  ratio is ``exp(mean_seed(L_M - L_E))``.  Cells are never pooled before the
  logarithm.
* zero-shot headline: the ratio per database is the geometric-mean ratio over
  the four target databases.
"""

import math
from collections.abc import Mapping

DATABASES = ("imdb", "stats", "ergastf1", "genome")
FEWSHOT_DATABASES = ("stats", "ergastf1", "genome")
SEEDS = (42, 43, 44)
K_VALUES = (0, 5, 20)
RUNS = range(5)
ARMS = ("P", "E", "M", "PG")
METRICS = ("q50", "q90", "q95", "q99")
ALPHA = 1e-6
EXPECTED_DENOMINATORS = {
    "imdb": 521,
    "stats": 2603,
    "ergastf1": 1258,
    "genome": 1699,
}
THRESHOLDS = {
    "relative_gain": 0.95,
    "useful_adaptation": 0.95,
    "k0_protect": 1.05,
    "per_database": 1.10,
    "improved_streams_min": 2,
    "reference": (
        "preregistered thresholds of experiment 123; the E_alt arm and its "
        "learning-rate selection are cancelled, no new target-side threshold "
        "was introduced"
    ),
}


def _streams_for(arm):
    return (0,) if arm in ("P", "PG") else SEEDS


def _stream_key(arm, seed):
    """Stream actually held by an arm for a paired seed slot."""
    return 0 if arm in ("P", "PG") else seed


def _key(unit):
    return (
        unit["arm"],
        unit["stream"],
        unit["database"],
        unit["k"],
        unit["run"],
    )


def _check_unit(unit, alpha):
    try:
        arm = unit["arm"]
        stream = unit["stream"]
        database = unit["database"]
        k = unit["k"]
        run = unit["run"]
        n = unit["n"]
        lr = unit.get("lr")
    except (KeyError, TypeError) as exc:
        raise ValueError("Evaluation cell is missing a required field") from exc
    if arm not in ARMS:
        raise ValueError(f"Invalid arm in evaluation cell: {arm!r}")
    if stream not in _streams_for(arm):
        raise ValueError(f"Invalid stream for arm {arm}: {stream!r}")
    if database not in DATABASES:
        raise ValueError(f"Invalid database in evaluation cell: {database!r}")
    if k not in K_VALUES:
        raise ValueError(f"Invalid K in evaluation cell: {k!r}")
    if not isinstance(run, int) or isinstance(run, bool):
        raise ValueError("Invalid run index in evaluation cell")
    if k == 0:
        if run != 0:
            raise ValueError("K0 cells must use run 0")
        if arm == "PG" and k != 0:
            raise ValueError("PG has no few-shot cells")
    elif run not in RUNS:
        raise ValueError(f"Invalid run index in evaluation cell: {run!r}")
    if arm == "PG" and k != 0:
        raise ValueError("PG must not report few-shot cells")
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"Invalid query denominator in evaluation cell: {n!r}")
    if lr is None:
        if arm in ("E", "M") and k != 0:
            raise ValueError(
                f"Few-shot continuation cell must bind the frozen alpha {alpha!r}"
            )
    else:
        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise ValueError("Evaluation cell must carry a numeric learning rate")
        expected_lr = 0.0 if k == 0 else alpha
        if float(lr) != float(expected_lr):
            raise ValueError(
                f"Learning rate {lr!r} does not match the frozen alpha {expected_lr!r}"
            )
    status = unit.get("status")
    if status != "completed":
        raise ValueError(
            f"Incomplete evaluation cell {unit.get('arm')}/"
            f"{unit.get('database')}/k={unit.get('k')}: {status!r} "
            f"{unit.get('error', '')}"
        )
    metrics = unit.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Evaluation cell has no metric mapping")
    for name in METRICS:
        if name not in metrics:
            raise ValueError(f"Evaluation cell is missing metric {name}")
        value = metrics[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Metric {name} is not numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"Metric {name} is not finite")
        if float(value) < 1.0 - 1e-9:
            raise ValueError(f"Metric {name} is below the Q-error lower bound")


def expected_cells():
    """The frozen 242-cell target matrix.

    238 cells come from the seven models (one shared P, E/M on three paired
    streams) over four zero-shot cells plus three few-shot databases with two
    K values and five support draws each; PG adds four zero-shot cells.
    """
    cells = set()
    for arm in ("P", "E", "M"):
        for stream in _streams_for(arm):
            for database in DATABASES:
                cells.add((arm, stream, database, 0, 0))
                if database in FEWSHOT_DATABASES:
                    for k in (5, 20):
                        for run in RUNS:
                            cells.add((arm, stream, database, k, run))
    for database in DATABASES:
        cells.add(("PG", 0, database, 0, 0))
    return cells


def _index(units, alpha):
    values = units.values() if isinstance(units, Mapping) else units
    cells = {}
    for unit in values:
        if not isinstance(unit, Mapping):
            raise ValueError("Evaluation cell must be a mapping")
        _check_unit(unit, alpha)
        key = _key(unit)
        if key in cells:
            raise ValueError(f"Duplicate evaluation cell: {key}")
        cells[key] = unit
    return cells


def _draw_mean(cells, arm, stream, database, k, metric):
    runs = (0,) if k == 0 else RUNS
    stream = _stream_key(arm, stream)
    values = [
        float(cells[(arm, stream, database, k, run)]["metrics"][metric]) for run in runs
    ]
    return sum(values) / len(values)


def _stream_log_score(cells, arm, stream, k, metric="q95"):
    """L = mean over the few-shot databases of log(mean support-draw Q)."""
    return sum(
        math.log(_draw_mean(cells, arm, stream, database, k, metric))
        for database in FEWSHOT_DATABASES
    ) / len(FEWSHOT_DATABASES)


def _paired_ratio(cells, numerator, denominator):
    """exp(mean_seed(L_numerator - L_denominator)) over the paired streams."""
    per_seed = {}
    for seed in SEEDS:
        left = _stream_log_score(
            cells, numerator[0], _stream_key(numerator[0], seed), numerator[1]
        )
        right = _stream_log_score(
            cells, denominator[0], _stream_key(denominator[0], seed), denominator[1]
        )
        per_seed[str(seed)] = {
            "L_numerator": left,
            "L_denominator": right,
            "delta": left - right,
        }
    deltas = [per_seed[str(seed)]["delta"] for seed in SEEDS]
    ratio = math.exp(sum(deltas) / len(deltas))
    return {
        "ratio": ratio,
        "percent_change": 100.0 * (ratio - 1.0),
        "per_stream_deltas": deltas,
        "per_seed": per_seed,
        "improved_streams": sum(1 for delta in deltas if delta < 0),
        "streams": list(SEEDS),
        "denominator_streams": list(_streams_for(denominator[0])),
        "method": (
            "per seed L = mean_db log(mean_5_support_draws_q95) over the three "
            "few-shot databases; ratio = exp(mean_seed(L_numerator - L_denominator))"
        ),
    }


def _k0_geometric_mean(cells, numerator_arm, denominator_arm):
    per_database = {}
    for database in DATABASES:
        numerator = sum(
            _draw_mean(cells, numerator_arm, seed, database, 0, "q95")
            for seed in _streams_for(numerator_arm)
        ) / len(_streams_for(numerator_arm))
        denominator = sum(
            _draw_mean(cells, denominator_arm, seed, database, 0, "q95")
            for seed in _streams_for(denominator_arm)
        ) / len(_streams_for(denominator_arm))
        per_database[database] = numerator / denominator
    ratio = math.exp(
        sum(math.log(v) for v in per_database.values()) / len(per_database)
    )
    return {
        "ratio": ratio,
        "percent_change": 100.0 * (ratio - 1.0),
        "per_database": per_database,
        "method": (
            "per-database ratio of the zero-shot Q95 means, then the geometric "
            "mean over the four target databases"
        ),
    }


def _per_database_ratio(cells, database, metric, numerator, denominator):
    numerator_mean = sum(
        _draw_mean(cells, numerator[0], seed, database, numerator[1], metric)
        for seed in _streams_for(numerator[0])
    ) / len(_streams_for(numerator[0]))
    denominator_mean = sum(
        _draw_mean(cells, denominator[0], seed, database, denominator[1], metric)
        for seed in _streams_for(denominator[0])
    ) / len(_streams_for(denominator[0]))
    return numerator_mean / denominator_mean


def _per_database_rollup(cells, database):
    rollup = {}
    for arm in ("P", "E", "M", "PG"):
        streams = {}
        for seed in _streams_for(arm):
            per_k = {}
            for k in K_VALUES:
                if k != 0 and (arm == "PG" or database not in FEWSHOT_DATABASES):
                    continue
                runs = (0,) if k == 0 else RUNS
                q95_runs = [
                    float(cells[(arm, seed, database, k, run)]["metrics"]["q95"])
                    for run in runs
                ]
                q50_runs = [
                    float(cells[(arm, seed, database, k, run)]["metrics"]["q50"])
                    for run in runs
                ]
                per_k[str(k)] = {
                    "q95": sum(q95_runs) / len(q95_runs),
                    "q50": sum(q50_runs) / len(q50_runs),
                    "q95_runs": q95_runs,
                    "q50_runs": q50_runs,
                }
            streams[str(seed)] = per_k
        entry = {"streams": streams}
        for metric in ("q95", "q50"):
            values = {}
            for k in K_VALUES:
                if not any(str(k) in s for s in streams.values()):
                    continue
                per_seed_values = {
                    seed: s[str(k)][metric] for seed, s in streams.items()
                }
                values[str(k)] = sum(per_seed_values.values()) / len(per_seed_values)
            entry[metric] = values
        if len(_streams_for(arm)) > 1:
            across = {}
            for k in K_VALUES:
                per_seed_values = {
                    seed: s[str(k)]["q95"] for seed, s in streams.items() if str(k) in s
                }
                if not per_seed_values:
                    continue
                numbers = list(per_seed_values.values())
                across[str(k)] = {
                    "metric": "q95",
                    "mean": sum(numbers) / len(numbers),
                    "min": min(numbers),
                    "max": max(numbers),
                    "range": [min(numbers), max(numbers)],
                    "values": per_seed_values,
                }
            entry["across_seeds_q95"] = across
        rollup[arm] = entry
    return rollup


def aggregate(units, denominators=None, *, alpha=ALPHA):
    """Aggregate the frozen target matrix into preregistered headline ratios."""
    if denominators is None:
        denominators = dict(EXPECTED_DENOMINATORS)
    if not isinstance(denominators, Mapping):
        raise ValueError("Query denominators must be a mapping")
    if dict(denominators) != EXPECTED_DENOMINATORS:
        raise ValueError(
            "Query denominators drifted from the frozen target protocol: "
            f"{dict(denominators)} != {EXPECTED_DENOMINATORS}"
        )
    cells = _index(units, alpha)
    expected = expected_cells()
    present = set(cells)
    if present != expected:
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        raise ValueError(
            "Evaluation matrix does not match the frozen target protocol: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    for database, denominator in denominators.items():
        observed = {
            unit["n"] for unit in cells.values() if unit["database"] == database
        }
        if observed != {denominator}:
            raise ValueError(
                f"Query denominator drift for {database}: {sorted(observed)} "
                f"!= [{denominator}]"
            )

    primary = _paired_ratio(cells, ("M", 20), ("E", 20))
    ratios = {
        "M20/E20": primary,
        "M20/P20": _paired_ratio(cells, ("M", 20), ("P", 20)),
        "M20/M0": _paired_ratio(cells, ("M", 20), ("M", 0)),
        "M20/P0": _paired_ratio(cells, ("M", 20), ("P", 0)),
        "M20/E0": _paired_ratio(cells, ("M", 20), ("E", 0)),
    }
    k0_ratios = {
        "M0/P0": _k0_geometric_mean(cells, "M", "P"),
        "E0/P0": _k0_geometric_mean(cells, "E", "P"),
        "M0/E0": _k0_geometric_mean(cells, "M", "E"),
    }
    pg_reference = {
        "P0/PG0": _k0_geometric_mean(cells, "P", "PG"),
        "E0/PG0": _k0_geometric_mean(cells, "E", "PG"),
        "M0/PG0": _k0_geometric_mean(cells, "M", "PG"),
        "role": (
            "reference only; the PG baseline is not part of the preregistered "
            "gain, K0-protection or per-database gates"
        ),
    }

    protection = {}
    comparisons = {
        "M20/E20": (("M", 20), ("E", 20)),
        "M20/P20": (("M", 20), ("P", 20)),
        "M20/M0": (("M", 20), ("M", 0)),
        "M20/P0": (("M", 20), ("P", 0)),
        "M20/E0": (("M", 20), ("E", 0)),
        "M0/P0": (("M", 0), ("P", 0)),
        "M0/E0": (("M", 0), ("E", 0)),
    }
    for database in DATABASES:
        for metric in ("q95", "q50"):
            for name, (numerator, denominator) in comparisons.items():
                if numerator[1] != 0 and database not in FEWSHOT_DATABASES:
                    continue
                protection[f"{database}/{metric}/{name}"] = _per_database_ratio(
                    cells, database, metric, numerator, denominator
                )

    k0_cells = []
    for seed in SEEDS:
        for database in FEWSHOT_DATABASES:
            baseline = _draw_mean(cells, "M", seed, database, 0, "q95")
            adapted = _draw_mean(cells, "M", seed, database, 20, "q95")
            k0_cells.append(
                {
                    "stream": seed,
                    "database": database,
                    "m0_q95": baseline,
                    "m20_q95": adapted,
                    "improved": adapted < baseline,
                }
            )
    improved = sum(1 for cell in k0_cells if cell["improved"])
    block_cells = []
    for seed in SEEDS:
        for database in FEWSHOT_DATABASES:
            baseline = _draw_mean(cells, "M", seed, database, 0, "q95")
            for run in RUNS:
                value = float(cells[("M", seed, database, 20, run)]["metrics"]["q95"])
                block_cells.append(
                    {
                        "stream": seed,
                        "database": database,
                        "run": run,
                        "m20_q95": value,
                        "m0_q95": baseline,
                        "worsened": value > baseline,
                    }
                )
    worsened_blocks = sum(1 for cell in block_cells if cell["worsened"])
    k0_protection = {
        "by_seed_database": {
            "total": len(k0_cells),
            "improved": improved,
            "degraded": len(k0_cells) - improved,
            "fraction_improved": improved / len(k0_cells),
            "cells": k0_cells,
            "comparison": (
                "M20 mean-over-blocks Q95 against its own M0 Q95, per stream "
                "and database (3 streams x 3 databases)"
            ),
        },
        "by_block": {
            "total": len(block_cells),
            "worsened": worsened_blocks,
            "not_worsened": len(block_cells) - worsened_blocks,
            "fraction_worsened": worsened_blocks / len(block_cells),
            "cells": block_cells,
            "comparison": (
                "each K20 support block Q95 against its own stream/database "
                "M0 Q95 (3 streams x 3 databases x 5 blocks)"
            ),
        },
    }

    relative_gain = (
        ratios["M20/E20"]["ratio"] <= THRESHOLDS["relative_gain"]
        and ratios["M20/P20"]["ratio"] <= THRESHOLDS["relative_gain"]
        and primary["improved_streams"] >= THRESHOLDS["improved_streams_min"]
    )
    useful_adaptation = (
        ratios["M20/M0"]["ratio"] <= THRESHOLDS["useful_adaptation"]
        and ratios["M20/P0"]["ratio"] <= THRESHOLDS["useful_adaptation"]
        and ratios["M20/E0"]["ratio"] <= THRESHOLDS["useful_adaptation"]
        and ratios["M20/M0"]["improved_streams"] >= THRESHOLDS["improved_streams_min"]
    )
    k0_protect = (
        k0_ratios["M0/P0"]["ratio"] <= THRESHOLDS["k0_protect"]
        and k0_ratios["M0/E0"]["ratio"] <= THRESHOLDS["k0_protect"]
    )
    per_database_ok = all(
        value <= THRESHOLDS["per_database"] for value in protection.values()
    )
    gates = {
        "relative_gain": bool(relative_gain),
        "useful_adaptation": bool(useful_adaptation),
        "k0_protect": bool(k0_protect),
        "per_database": bool(per_database_ok),
    }

    if primary["ratio"] >= 1.0:
        decision = "no_target_gain"
    elif all(gates.values()):
        decision = "target_fewshot_gain"
    else:
        decision = "partial_target_signal"

    notes = [
        (
            "Target few-shot coverage is 3 of 4 target databases "
            f"({', '.join(FEWSHOT_DATABASES)}); imdb is zero-shot only because "
            "its independent K>0 support set is missing, so no K>0 imdb cell "
            "exists; the imdb test rows were never used as support."
        ),
        (
            "Every target number is a fixed-evaluation output: there was "
            "no target-side hyper-parameter selection and no target-side "
            "strategy, step or checkpoint selection."
        ),
        (
            "The three paired continuation streams (42/43/44) share one "
            "official PRICE pretraining initialization, so this target result "
            "is not replicated across independent pretraining runs."
        ),
        (
            "P is a single shared pretraining model; its five support draws "
            "are repeated evaluations of that one model, not independent "
            "pretrained models."
        ),
        (
            "The zero-shot headline ratios are geometric means over the four "
            "target databases; the K20 headline ratios use the paired "
            "mean-then-log form over the three few-shot databases only."
        ),
    ]

    return {
        "decision": decision,
        "gates": gates,
        "thresholds": dict(THRESHOLDS),
        "primary_metric": {
            **primary,
            "databases": list(FEWSHOT_DATABASES),
            "k": 20,
            "scope": "target-side few-shot, three of four target databases",
        },
        "ratios": ratios,
        "k0_geometric_mean": k0_ratios,
        "pg_reference": pg_reference,
        "per_database": {
            database: _per_database_rollup(cells, database) for database in DATABASES
        },
        "per_database_protection": protection,
        "k0_protection": k0_protection,
        "coverage": {
            "cells_expected": len(expected),
            "cells_present": len(cells),
            "fewshot_coverage": f"{len(FEWSHOT_DATABASES)}/{len(DATABASES)}",
            "k0_coverage": f"{len(DATABASES)}/{len(DATABASES)}",
            "imdb_fewshot": "not_evaluated",
            "imdb_fewshot_reason": "unavailable_missing_independent_support",
            "fewshot_databases": list(FEWSHOT_DATABASES),
            "k0_databases": list(DATABASES),
        },
        "denominators": dict(denominators),
        "selection": {
            "alpha": alpha,
            "target_side_selection": "none",
            "e_arm_lr_selection": "cancelled",
            "e_alt_arm": "not_run",
        },
        "notes": notes,
        "cells": len(cells),
    }
