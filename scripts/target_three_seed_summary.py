"""Aggregate the four target databases across three training seeds (doc 106).

Read-only.  No training, no inference, no model selection, no adaptation.

Why this exists
---------------
Docs 102 and 103 measured the four target databases from a **single**
checkpoint (the seed-42 baseline).  Two later findings make that single
measurement hard to defend:

* the project's own screening runs measured a **6.28 pp seed-to-seed
  standard deviation** on the source-side 4+ Q95 macro (doc 94 §6.1b),
  which is larger than every effect size any screened intervention produced;
* the direction audit showed that in-database signal strength and
  cross-database transfer both vary by seed (doc 98).

So the headline "four targets, four losses to PRICE" rested on n=1.  This
script aggregates the same measurement over the three seeds that already
exist -- seed 42, 43 and 44 of the *identical* ERM-sum baseline (only the
``seed`` field differs; every other training hyper-parameter is byte
identical, verified against the three ``config.json`` files on the server).

Everything here is read from the per-seed ``summary.json`` files the
read-only ``target_population_eval.py`` wrote.  Nothing is recomputed from
predictions and nothing is typed by hand.

Two guards are deliberate:

* PRICE and PostgreSQL rows are re-checked for **bitwise identity across
  seeds**.  PRICE has one provided checkpoint and PG is a fixed estimate, so
  any difference would mean the three runs are not comparable and the
  aggregate must not be published.
* The per-seed Q95 is reported as **mean +/- sample SD (ddof=1)**, never as
  a best-of-seeds value, and the per-seed win/loss count against PRICE is
  reported alongside, so a reader can see whether the direction holds for
  every seed rather than only on average.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATABASES = ("stats", "genome", "ergastf1", "imdb")
SEEDS = (42, 43, 44)
ESTIMATORS = ("ours_k0", "price_k0", "pg")


def _mean_sd(values):
    """Mean and sample SD (ddof=1) over seeds; SD is 0 for a single seed."""
    clean = [float(value) for value in values]
    count = len(clean)
    mean = sum(clean) / count
    if count < 2:
        return {"mean": mean, "std": 0.0, "min": mean, "max": mean, "n": count}
    variance = sum((value - mean) ** 2 for value in clean) / (count - 1)
    return {
        "mean": mean,
        "std": variance**0.5,
        "min": min(clean),
        "max": max(clean),
        "n": count,
    }


def _assert_identical_across_seeds(per_seed, label, databases=DATABASES):
    """Fixed estimators must not move between seeds; refuse to aggregate if they do."""
    for database in databases:
        values = [per_seed[seed][database][label]["q95"] for seed in SEEDS]
        if len(set(values)) != 1:
            raise ValueError(
                f"{label} q95 differs across seeds for {database!r}: {values}. "
                "The three runs are not comparable; refusing to aggregate."
            )


def collect(per_seed, databases=DATABASES):
    """Build the per-database, per-estimator aggregate block."""
    aggregate = {}
    for database in databases:
        entry = {"n": per_seed[SEEDS[0]][database]["n"]}
        for estimator in ESTIMATORS:
            q95 = [per_seed[seed][database][estimator]["q95"] for seed in SEEDS]
            q50 = [per_seed[seed][database][estimator]["q50"] for seed in SEEDS]
            entry[estimator] = {
                "q95": _mean_sd(q95),
                "q50": _mean_sd(q50),
                "per_seed_q95": {str(seed): value for seed, value in zip(SEEDS, q95)},
                "per_seed_q50": {str(seed): value for seed, value in zip(SEEDS, q50)},
            }
        ours = entry["ours_k0"]["per_seed_q95"]
        price = entry["price_k0"]["per_seed_q95"]
        pg = entry["pg"]["per_seed_q95"]
        entry["per_seed_q95_ratio_price_over_ours"] = {
            str(seed): price[str(seed)] / ours[str(seed)] for seed in SEEDS
        }
        entry["per_seed_q95_ratio_pg_over_ours"] = {
            str(seed): pg[str(seed)] / ours[str(seed)] for seed in SEEDS
        }
        entry["seeds_beating_price_on_q95"] = sum(
            1 for seed in SEEDS if ours[str(seed)] < price[str(seed)]
        )
        entry["seeds_beating_pg_on_q95"] = sum(
            1 for seed in SEEDS if ours[str(seed)] < pg[str(seed)]
        )
        entry["median_q95_ratio_price_over_ours"] = statistics.median(
            entry["per_seed_q95_ratio_price_over_ours"].values()
        )
        entry["median_q95_ratio_pg_over_ours"] = statistics.median(
            entry["per_seed_q95_ratio_pg_over_ours"].values()
        )
        aggregate[database] = entry
    return aggregate


def four_plus_aggregate(per_seed, databases=DATABASES):
    """Same aggregation for the 4+ table bucket, which is where the tail lives."""
    aggregate = {}
    for database in databases:
        entry = {"n": per_seed[SEEDS[0]][database]["bucket_4plus"]["n"]}
        for estimator in ESTIMATORS:
            q95 = [
                per_seed[seed][database]["bucket_4plus"][estimator]["q95"]
                for seed in SEEDS
            ]
            entry[estimator] = {
                "q95": _mean_sd(q95),
                "per_seed_q95": {str(seed): value for seed, value in zip(SEEDS, q95)},
            }
        aggregate[database] = entry
    return aggregate


def headline(aggregate, databases=DATABASES):
    """The one-paragraph claim this aggregate supports, computed from it."""
    price_medians = [
        aggregate[database]["median_q95_ratio_price_over_ours"]
        for database in databases
    ]
    pg_medians = [
        aggregate[database]["median_q95_ratio_pg_over_ours"] for database in databases
    ]
    all_seeds_lose = all(
        aggregate[database]["seeds_beating_price_on_q95"] == 0 for database in databases
    )
    return {
        "databases": list(databases),
        "seeds": list(SEEDS),
        "median_q95_ratio_price_over_ours": {
            "value": statistics.median(price_medians),
            "min": min(price_medians),
            "max": max(price_medians),
        },
        "median_q95_ratio_pg_over_ours": {
            "value": statistics.median(pg_medians),
            "min": min(pg_medians),
            "max": max(pg_medians),
        },
        "every_seed_loses_to_price_on_q95": all_seeds_lose,
        "total_seed_database_pairs": len(SEEDS) * len(databases),
        "seed_database_pairs_beating_price": sum(
            aggregate[database]["seeds_beating_price_on_q95"] for database in databases
        ),
        "seed_database_pairs_beating_pg": sum(
            aggregate[database]["seeds_beating_pg_on_q95"] for database in databases
        ),
    }


def main(argv):
    parser = argparse.ArgumentParser(
        description="Aggregate target databases across three training seeds"
    )
    parser.add_argument("--seed-summary", action="append", required=True)
    parser.add_argument("--seed-protocol", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--databases",
        nargs="+",
        default=list(DATABASES),
        help="target databases to aggregate (default: the four of doc 108)",
    )
    parser.add_argument(
        "--audit",
        default="p2-2-target-3seed",
        help="audit id written into the protocol and summary",
    )
    args = parser.parse_args(argv)
    databases = tuple(args.databases)

    if len(args.seed_summary) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} --seed-summary arguments")
    if len(args.seed_protocol) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} --seed-protocol arguments")

    per_seed = {}
    protocols = {}
    for seed, summary_path, protocol_path in zip(
        SEEDS, args.seed_summary, args.seed_protocol
    ):
        summary = json.loads(Path(summary_path).read_text())
        protocol = json.loads(Path(protocol_path).read_text())
        if summary["target_databases_read"] != list(databases):
            raise ValueError(
                f"seed {seed} read targets {summary['target_databases_read']}, "
                f"expected {list(databases)}"
            )
        if summary["checkpoint"]["seed"] != seed:
            raise ValueError(
                f"{summary_path} carries checkpoint seed "
                f"{summary['checkpoint']['seed']}, expected {seed}"
            )
        per_seed[seed] = summary["databases"]
        protocols[seed] = protocol

    # PRICE and PG must not move between seeds, or the runs are not comparable.
    _assert_identical_across_seeds(per_seed, "price_k0", databases)
    _assert_identical_across_seeds(per_seed, "pg", databases)

    aggregate = collect(per_seed, databases)
    four_plus = four_plus_aggregate(per_seed, databases)

    protocol = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": args.audit,
        "target_databases_read": list(databases),
        "allowed_targets": list(databases),
        "metric": protocols[SEEDS[0]]["metric"],
        "metric_definitions": protocols[SEEDS[0]]["metric_definitions"],
        "estimators": protocols[SEEDS[0]]["estimators"],
        "seeds": list(SEEDS),
        "checkpoints": {
            str(seed): {
                "path": protocols[seed]["checkpoint"]["path"],
                "sha256": protocols[seed]["checkpoint"]["sha256"],
                "seed": protocols[seed]["checkpoint"]["seed"],
                "method": protocols[seed]["checkpoint"]["method"],
                "train_databases": protocols[seed]["checkpoint"]["train_databases"],
                "validation_databases": protocols[seed]["checkpoint"][
                    "validation_databases"
                ],
                "model_config": protocols[seed]["checkpoint"]["model_config"],
                "transform": protocols[seed]["checkpoint"]["transform"],
            }
            for seed in SEEDS
        },
        "price": protocols[SEEDS[0]]["price"],
        "inputs": protocols[SEEDS[0]]["inputs"],
        "comparability": {
            "training_config": (
                "the three checkpoints differ only in the seed field; every other "
                "training hyper-parameter (method, steps, meta_batch, query_size, "
                "outer_lr, validate_every, validation_runs, validation_seed, "
                "patience, min_delta, loss, model arch) is byte identical, verified "
                "against the three config.json files on the server"
            ),
            "price_and_pg_fixed_across_seeds": True,
            "fixed_estimator_check": (
                "PRICE (one provided checkpoint) and PostgreSQL (a fixed estimate) "
                "produce bitwise identical Q95 on every evaluated database in all "
                "three runs; the aggregator refuses to publish otherwise"
            ),
            "aggregation": (
                "mean and sample SD (ddof=1) over the three seeds, never a "
                "best-of-seeds value; the per-seed win/loss count against PRICE "
                "and PG is reported alongside"
            ),
        },
        "limitations": [
            "three seeds is the project's seed coverage; it bounds the seed "
            "spread, it does not make any single database's Q95 a stable "
            "point estimate",
            *(
                [
                    "genome's seed spread is large relative to the other three "
                    "databases; the per-seed values are reported rather than hidden "
                    "behind a mean"
                ]
                if "genome" in databases
                else []
            ),
            "PRICE contributes one provided checkpoint, so its column has no "
            "seed spread and is not a training-matched comparison",
            "the IMDB row is a K=0 supplement, not a fourth complete few-shot "
            "target acceptance (no finetune split exists for IMDB)",
        ],
    }

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": args.audit,
        "target_databases_read": list(databases),
        "metric": protocol["metric"],
        "evidence_level": "three_seed_frozen_checkpoints_full_population",
        "needs_evidence": False,
        "needs_evidence_reason": (
            "the three seeds share a byte-identical training configuration and "
            "differ only in the seed; the remaining limits are the small seed "
            "count and the single provided PRICE checkpoint"
        ),
        "seeds": list(SEEDS),
        "checkpoints": protocol["checkpoints"],
        "price": protocol["price"],
        "inputs": protocol["inputs"],
        "databases": aggregate,
        "bucket_4plus": four_plus,
        "headline": headline(aggregate, databases),
    }

    Path(args.protocol).parent.mkdir(parents=True, exist_ok=True)
    Path(args.protocol).write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )
    Path(args.output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary["headline"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
