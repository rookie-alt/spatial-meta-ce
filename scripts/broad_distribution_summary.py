"""Cross-database distribution summary for the 26-source K=0 sweep (doc 101 §5.5).

``scripts/price_population_eval.py`` reports one database at a time.  The
plan also requires a *distribution* answer over all 26 source databases:
how many we beat PG on, how many we beat PRICE on, the ratio distribution,
and a three-tier classification of every database.  That aggregation lives
here rather than inside the evaluated script, so the PRICE script stays a
pure per-database evaluator and its default output stays byte-comparable
with task C.

Read-only: this script consumes one already-written summary and computes
counts.  It never re-runs a model, and it refuses a summary that read a
target database (``target_databases_read`` must be empty) or that is
missing databases from the expected source list.

Every database is classified; nothing is dropped, reordered by result, or
reported as a subset.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METHODS = ("ours_k0", "price_k0", "pg")
TIERS = (
    "beats_pg_and_price",
    "beats_pg_loses_price",
    "loses_pg",
)
MIN_RATIO = 1e-12


def _load_price_population_module():
    """Reuse task C's atomic JSON writer instead of re-implementing it."""
    path = Path(__file__).resolve().parent / "price_population_eval.py"
    spec = importlib.util.spec_from_file_location("price_population_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ratio(numerator, denominator):
    """numerator / denominator, or None when the denominator is unusable."""
    if numerator is None or denominator is None:
        return None
    if denominator <= MIN_RATIO:
        return None
    return float(numerator) / float(denominator)


def flatten(row):
    """Accept either a full ``database_report`` or a ``headline`` record.

    ``price_population_eval.py`` writes the full per-database report under
    ``summary["databases"]`` (methods at the top level, each holding
    ``q50/q95/log_q95``) and a compact ``headline`` view alongside it
    (``q95.{method}``).  Both carry the same numbers; normalising here keeps
    the caller's choice of input file from changing the classification.
    """
    if "q95" in row and "ours_k0" in row["q95"]:
        return {
            "q95": dict(row["q95"]),
            "bucket_4plus_q95": dict(row.get("bucket_4plus_q95", {})),
        }
    four_plus = row.get("bucket_4plus") or {}
    return {
        "q95": {method: (row[method] or {}).get("q95") for method in METHODS},
        "bucket_4plus_q95": {
            method: (four_plus.get(method) or {}).get("q95") for method in METHODS
        },
    }


def classify(row):
    """Three-tier label for one database on the headline Q95.

    ``beats_pg_and_price``   our Q95 below both baselines
    ``beats_pg_loses_price`` our Q95 below PG but not below PRICE
    ``loses_pg``             our Q95 at or above PG (ties count as losses)
    """
    row = flatten(row)
    ours = row["q95"]["ours_k0"]
    price = row["q95"]["price_k0"]
    pg = row["q95"]["pg"]
    if ours is None or price is None or pg is None:
        raise ValueError("Headline Q95 missing for a database")
    if ours < pg and ours < price:
        return "beats_pg_and_price"
    if ours < pg:
        return "beats_pg_loses_price"
    return "loses_pg"


def describe(values):
    """median / min / max / count of a ratio list, ignoring undefined ratios."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {
        "n": len(clean),
        "median": float(statistics.median(clean)),
        "min": float(min(clean)),
        "max": float(max(clean)),
    }


def build_distribution(databases, expected):
    """All rows in the expected order plus the plan's §5.5 statistics."""
    missing = [db for db in expected if db not in databases]
    unexpected = [db for db in databases if db not in expected]
    if missing or unexpected:
        raise ValueError(
            f"database set mismatch: missing={missing} unexpected={unexpected}"
        )
    rows = {db: flatten(databases[db]) for db in expected}
    tiers = {db: classify(row) for db, row in rows.items()}
    price_over_ours = [
        ratio(row["q95"]["price_k0"], row["q95"]["ours_k0"]) for row in rows.values()
    ]
    pg_over_ours = [
        ratio(row["q95"]["pg"], row["q95"]["ours_k0"]) for row in rows.values()
    ]
    price_over_ours_4 = [
        ratio(
            row["bucket_4plus_q95"]["price_k0"],
            row["bucket_4plus_q95"]["ours_k0"],
        )
        for row in rows.values()
    ]
    pg_over_ours_4 = [
        ratio(row["bucket_4plus_q95"]["pg"], row["bucket_4plus_q95"]["ours_k0"])
        for row in rows.values()
    ]
    empty_4plus = [
        db for db, row in rows.items() if row["bucket_4plus_q95"]["ours_k0"] is None
    ]
    # The plan asks for two *independent* pairwise counts, then a three-tier
    # classification.  Counting "beats PRICE" as "beats both" (the tier) would
    # silently drop a database that beats PRICE but loses to PG -- tpc_h is
    # exactly that case -- so the pairwise counts are computed separately.
    beats_pg = sum(
        1 for row in rows.values() if row["q95"]["ours_k0"] < row["q95"]["pg"]
    )
    beats_price = sum(
        1 for row in rows.values() if row["q95"]["ours_k0"] < row["q95"]["price_k0"]
    )
    total = len(rows)
    return {
        "n_databases": total,
        "ours_better_than_pg": beats_pg,
        "ours_better_than_price": beats_price,
        "pairwise_note": (
            "ours_better_than_pg and ours_better_than_price are independent "
            "pairwise counts (our Q95 strictly below that baseline); they do "
            "not sum to n_databases and are not the same as tier_counts"
        ),
        "tier_counts": {
            tier: sum(1 for t in tiers.values() if t == tier) for tier in TIERS
        },
        "tier_by_database": tiers,
        "q95_ratio_price_over_ours": describe(price_over_ours),
        "q95_ratio_pg_over_ours": describe(pg_over_ours),
        "bucket_4plus_ratio_price_over_ours": describe(price_over_ours_4),
        "bucket_4plus_ratio_pg_over_ours": describe(pg_over_ours_4),
        "bucket_4plus_empty": {
            "databases": empty_4plus,
            "n": len(empty_4plus),
            "reason": (
                "every query in these databases has fewer than four tables, "
                "so the 4+ bucket is empty and its Q95 is undefined (null); "
                "they are excluded from the 4+ ratio distribution only"
            ),
        },
        "ratio_convention": (
            "ratio > 1 means we have the lower (better) Q error; ratio < 1 "
            "means the baseline is better than us"
        ),
        "tier_rule": (
            "beats_pg_and_price: our Q95 strictly below both; "
            "beats_pg_loses_price: strictly below PG but not below PRICE; "
            "loses_pg: greater than or equal to PG (ties count as losses)"
        ),
    }


def summarize(databases, expected, protocol):
    if protocol.get("target_databases_read") != []:
        raise ValueError(
            "refusing to summarise a run that read a target database: "
            f"{protocol.get('target_databases_read')!r}"
        )
    distribution = build_distribution(databases, expected)
    return {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-price-population-broad",
        "target_databases_read": [],
        "scope": "in_sample_fit_diagnostic",
        "scope_note": (
            "PRICE was trained on all 26 of these databases and we trained on "
            "23; a win here measures fit, not resistance to spatial drift. "
            "Drift evidence lives only on the four target databases."
        ),
        "evidence_level": "single_seed_frozen_checkpoint_full_population",
        "needs_evidence": True,
        "needs_evidence_reason": (
            "the ours_k0 side is one seed (42); seed coverage is not part of this sweep"
        ),
        "metric": "unique-query population Q95 (single pass, NOT episode mean)",
        "price": {
            "maxima": protocol["price"]["maxima"],
            "checkpoint_sha256": protocol["price"]["checkpoint_sha256"],
            "maxima_rule": protocol["price"].get("maxima_rule"),
        },
        "inputs": protocol.get("inputs", {}),
        "distribution": distribution,
        "databases": databases,
    }


def main(argv):
    parser = argparse.ArgumentParser(
        description="Aggregate the 26-source K=0 sweep (read-only)"
    )
    parser.add_argument("--summary", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-protocol", required=True)
    args = parser.parse_args(argv)

    price_population = _load_price_population_module()
    raw = json.loads(Path(args.summary).read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    combined = summarize(raw["databases"], list(args.expected), protocol)
    price_population.write_json(args.output, combined)

    out_protocol = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-price-population-broad",
        "target_databases_read": [],
        "metric": combined["metric"],
        "databases_expected": list(args.expected),
        "databases_reported": list(combined["databases"]),
        "price": protocol.get("price", {}),
        "inputs": {
            **protocol.get("inputs", {}),
            "raw_summary": str(Path(args.summary).resolve()),
            "raw_protocol": str(Path(args.protocol).resolve()),
        },
        "aggregation": {
            "source": "scripts/broad_distribution_summary.py",
            "rule": combined["distribution"]["tier_rule"],
            "ratio_convention": combined["distribution"]["ratio_convention"],
        },
        "comparability": {
            "comparable_to": (
                "the rows inside this report only: one maxima, one query "
                "population definition, one checkpoint"
            ),
            "not_comparable_to": [
                "the task-C three-database table (maxima [11,12,7,10] vs the "
                "26-source [12,12,7,10]): see the layout sensitivity check in "
                "the report before quoting the two together",
                "the four target-database rows, which measure drift rather "
                "than in-sample fit",
            ],
        },
    }
    price_population.write_json(args.output_protocol, out_protocol)
    print(json.dumps(combined["distribution"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
