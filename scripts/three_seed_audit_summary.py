"""Aggregate the P2-2 direction audit across seeds (doc 97, task A).

Read-only.  No training, no inference, no model selection, no target database.

Why this exists
---------------
Doc 96's direction-signal audit (AUC 0.92-0.98 in-database, 0.38-0.60
leave-one-database-out) rests on a single checkpoint: the seed-42 baseline.
Doc 97 §4.3 froze the acceptance thresholds *before* the extra seeds were
run, and this script applies them verbatim:

    in-database AUC, minimum over all seeds and databases   > 0.85
    LODO AUC, maximum over all seeds and databases          < 0.65
    telstra LODO AUC below 0.5 in at least 2 of 3 seeds

Averages are reported as mean +/- sample SD (ddof=1) over seeds, never as a
best-of-seeds value.  If any threshold fails, doc 96's conclusion is
downgraded to ``inconclusive`` and may not enter the paper body.

Every number here is read from the per-seed ``audit.json`` files that the
audit script wrote; nothing is recomputed from predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATABASES = ("chembl", "legalacts", "telstra")
MODELS = ("hgb", "logistic")

# Frozen in doc 97 §4.3 before the seed-43/44 runs.  Do not edit after seeing
# the results.
IN_DATABASE_AUC_MIN_THRESHOLD = 0.85
LODO_AUC_MAX_THRESHOLD = 0.65
TELSTRA_LODO_BELOW_HALF = 0.5
TELSTRA_LODO_REQUIRED_SEEDS = 2


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


def read_audit(path):
    """Load one per-seed audit.json and pull the metrics the summary needs.

    Only metrics the audit actually produced are kept.  The near-duplicate
    sensitivity pass, for instance, exists for the gradient-boosting model
    only, so a missing model is recorded as "not measured" rather than being
    silently filled in.
    """
    data = json.loads(Path(path).read_text())
    if data.get("target_databases_read") != []:
        raise ValueError(f"Audit {path} did not record an empty target read list")
    per_seed = {"path": str(Path(path).resolve())}
    for database in DATABASES:
        block = data["databases"][database]
        for model in MODELS:
            cell = block["in_database"].get(model)
            if cell is not None:
                per_seed.setdefault("in_database", {}).setdefault(model, {})[
                    database
                ] = cell["auc_pooled_oof"]
            cell = block["near_duplicate_grouped_cv"].get(model)
            if cell is not None:
                per_seed.setdefault("near_duplicate", {}).setdefault(model, {})[
                    database
                ] = cell["auc_pooled_oof"]
        for model in MODELS:
            cell = data["leave_one_database_out"][database].get(model)
            if cell is None:
                continue
            per_seed.setdefault("lodo", {}).setdefault(model, {})[database] = cell[
                "auc"
            ]
            per_seed.setdefault("lodo_accuracy", {}).setdefault(model, {})[database] = (
                cell["accuracy"]
            )
            per_seed.setdefault("lodo_majority", {}).setdefault(model, {})[database] = (
                cell["majority_accuracy"]
            )
    per_seed["inputs"] = data.get("inputs", {})
    per_seed["verdict"] = data.get("verdict", {})
    return per_seed


def summarize(seeds):
    """Mean +/- SD over seeds for every metric, per database and model.

    A cell aggregates only the seeds that measured it; ``n`` in the result
    says how many contributed, so a partially measured metric cannot look
    like a full three-seed result.
    """
    summary = {}
    groups = (
        "in_database",
        "near_duplicate",
        "lodo",
        "lodo_accuracy",
        "lodo_majority",
    )
    for group in groups:
        summary[group] = {}
        for model in MODELS:
            summary[group][model] = {}
            for database in DATABASES:
                values = [
                    seeds[seed][group][model][database]
                    for seed in sorted(seeds)
                    if group in seeds[seed]
                    and model in seeds[seed][group]
                    and database in seeds[seed][group][model]
                ]
                summary[group][model][database] = _mean_sd(values) if values else None
    return summary


def judge(seeds):
    """Apply the frozen doc 97 §4.3 thresholds."""
    missing = [
        (seed, database)
        for seed in seeds
        for database in DATABASES
        if database not in seeds[seed].get("in_database", {}).get("hgb", {})
        or database not in seeds[seed].get("lodo", {}).get("hgb", {})
    ]
    if missing:
        raise ValueError(f"Audit is missing the metrics the thresholds need: {missing}")
    in_database = [
        seeds[seed]["in_database"]["hgb"][database]
        for seed in seeds
        for database in DATABASES
    ]
    lodo = [
        seeds[seed]["lodo"]["hgb"][database] for seed in seeds for database in DATABASES
    ]
    telstra = {seed: seeds[seed]["lodo"]["hgb"]["telstra"] for seed in seeds}
    below = sorted(
        seed for seed, value in telstra.items() if value < TELSTRA_LODO_BELOW_HALF
    )
    checks = {
        "in_database_auc_min_over_seeds_and_databases": {
            "value": min(in_database),
            "threshold": IN_DATABASE_AUC_MIN_THRESHOLD,
            "comparison": ">",
            "pass": min(in_database) > IN_DATABASE_AUC_MIN_THRESHOLD,
        },
        "lodo_auc_max_over_seeds_and_databases": {
            "value": max(lodo),
            "threshold": LODO_AUC_MAX_THRESHOLD,
            "comparison": "<",
            "pass": max(lodo) < LODO_AUC_MAX_THRESHOLD,
        },
        "telstra_lodo_below_half": {
            "value": len(below),
            "seeds": below,
            "per_seed": telstra,
            "threshold": TELSTRA_LODO_REQUIRED_SEEDS,
            "comparison": ">=",
            "pass": len(below) >= TELSTRA_LODO_REQUIRED_SEEDS,
        },
    }
    checks["all_pass"] = all(check["pass"] for check in checks.values())
    checks["conclusion"] = "confirmed" if checks["all_pass"] else "inconclusive"
    checks["note"] = (
        "Thresholds are the ones frozen in doc 97 section 4.3 before the "
        "seed-43/44 runs. A failure downgrades doc 96 to inconclusive; the "
        "affected numbers may then only appear in an appendix."
    )
    return checks


def write_json(path, value):
    """Write JSON atomically and refuse values JSON cannot represent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Three-seed summary of the P2-2 direction audit (read-only)"
    )
    parser.add_argument(
        "--audit",
        action="append",
        required=True,
        metavar="SEED=PATH",
        help="per-seed audit.json, e.g. 43=results/.../audit.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args(argv)

    seeds = {}
    for entry in args.audit:
        if "=" not in entry:
            raise ValueError(f"Expected SEED=PATH, got {entry!r}")
        seed, path = entry.split("=", 1)
        seeds[seed] = read_audit(path)
    if len(seeds) < 2:
        raise ValueError("A seed summary needs at least two seeds")

    summary = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-direction-audit-three-seed",
        "target_databases_read": [],
        "seeds": sorted(seeds),
        "per_seed": seeds,
        "summary": summarize(seeds),
        "verdict": judge(seeds),
    }
    protocol = {
        "schema": 1,
        "purpose": "research-read-only-audit",
        "audit": "p2-2-direction-audit-three-seed",
        "target_databases_read": [],
        "aggregation": "mean and sample SD (ddof=1) over seeds",
        "model": "hgb is the primary model; logistic is reported alongside",
        "thresholds_source": "doc 97 section 4.3, frozen before the runs",
        "thresholds": {
            "in_database_auc_min": IN_DATABASE_AUC_MIN_THRESHOLD,
            "lodo_auc_max": LODO_AUC_MAX_THRESHOLD,
            "telstra_lodo_below_half": TELSTRA_LODO_BELOW_HALF,
            "telstra_lodo_required_seeds": TELSTRA_LODO_REQUIRED_SEEDS,
        },
        "inputs": {seed: seeds[seed]["inputs"] for seed in sorted(seeds)},
    }
    write_json(args.protocol, protocol)
    write_json(args.output, summary)
    print(json.dumps(summary["verdict"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
