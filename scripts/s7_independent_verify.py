"""Independent NumPy/math result verification; no production statistics imports."""

import hashlib
import json
import math
from itertools import pairwise
from pathlib import Path

import numpy as np

EDGES = (0, 1, 10, 100, 1000, 10000, 100000, np.inf)


def percentile(values, percent):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("Empty percentile denominator")
    rank = (len(ordered) - 1) * percent / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def numbers(pred, true):
    if len(pred) != len(true) or not len(true):
        raise ValueError("Incomplete verification denominator")
    label = np.array([math.log(max(float(y), 1)) for y in true])
    effective = np.array([max(float(p), 0) for p in pred])
    q = np.array([math.exp(abs(p - y)) for p, y in zip(effective, label)])
    raw = np.asarray(pred) - label
    signed = effective - label
    return q, raw, signed


def metric_values(pred, true):
    q, raw, signed = numbers(pred, true)
    n = len(true)
    return {
        "n": n,
        "q50": percentile(q, 50),
        "q95": percentile(q, 95),
        "raw_log_mae": math.fsum(abs(float(x)) for x in raw) / n,
        "raw_log_mse": math.fsum(float(x) ** 2 for x in raw) / n,
        "raw_signed_log_error": math.fsum(float(x) for x in raw) / n,
        "under_fraction": float(np.count_nonzero(signed < 0) / n),
        "over_fraction": float(np.count_nonzero(signed > 0) / n),
        "exact_fraction": float(np.count_nonzero(signed == 0) / n),
    }


def close(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Independent statistic field mismatch")
        for k, v in expected.items():
            close(actual[k], v)
    elif isinstance(expected, (int, float)):
        if (
            isinstance(actual, bool)
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError(f"Independent statistic mismatch: {actual} != {expected}")
    elif actual != expected:
        raise ValueError("Independent string/status mismatch")


def pair_check(record, data, mask):
    n = int(mask.sum())
    if n < 200:
        close(record, {"n": n, "status": "below_min_cell_n"})
        return
    t, a, b = (data[k][mask] for k in ("true", "A", "B"))
    pg = np.log(np.maximum(data["pg"][mask], 1))
    qa, _, _ = numbers(a, t)
    qb, _, _ = numbers(b, t)
    qp, _, _ = numbers(pg, t)
    close(
        record,
        {
            "n": n,
            "status": "descriptive",
            "A": metric_values(a, t),
            "B": metric_values(b, t),
            "PG": metric_values(pg, t),
            "q95_relative_change_b_vs_a": percentile(qb, 95) / percentile(qa, 95) - 1,
            "improved_fraction_b_vs_a": float(np.mean(qb < qa)),
            "worse_fraction_b_vs_a": float(np.mean(qb > qa)),
            "unchanged_fraction_b_vs_a": float(np.mean(qb == qa)),
            "improved_vs_pg": {
                "A": float(np.mean(qa < qp)),
                "B": float(np.mean(qb < qp)),
            },
            "mean_raw_prediction_shift_b_minus_a": math.fsum(b - a) / n,
            "mean_correction_vs_pg": {
                "A": math.fsum(a - pg) / n,
                "B": math.fsum(b - pg) / n,
            },
        },
    )


def arrays(rows):
    return {
        key: np.array([r[key] for r in rows], dtype=np.float64)
        for key in ("true", "pg", "n_tables", "A", "B")
    }


def cards(data):
    return {
        f"[{a},{b})": (data["true"] >= a) & (data["true"] < b)
        for a, b in pairwise(EDGES)
    }


def directions(pred, true):
    _, _, signed = numbers(pred, true)
    return {"under": signed < 0, "over": signed > 0, "exact": signed == 0}


def composition_check(record, data, mask):
    if record["n"] != int(mask.sum()):
        raise ValueError("Tail composition denominator mismatch")
    close(
        record["cardinality_counts"],
        {k: int((mask & m).sum()) for k, m in cards(data).items()},
    )
    for key, pred in [
        ("a", data["A"]),
        ("b", data["B"]),
        ("pg", np.log(np.maximum(data["pg"], 1))),
    ]:
        close(
            record[f"{key}_error_direction_counts"],
            {
                k: int((mask & m).sum())
                for k, m in directions(pred, data["true"]).items()
            },
        )
    pair_check(record["stats"], data, mask)


def verify(output, data_root, s6_run, caches, states):
    output, data_root, s6_run = map(Path, (output, data_root, s6_run))
    plan = json.loads((output / "sample-plan.json").read_text())
    sample = [
        json.loads(line)
        for line in (output / "sample-manifest.jsonl").read_text().splitlines()
    ]
    predicted = [
        json.loads(line)
        for line in (output / "source-predictions.jsonl").read_text().splitlines()
    ]
    if len(sample) != len(predicted):
        raise ValueError("Sample/prediction row count mismatch")
    seen_keys = set()
    groups = {}
    for row, pred in zip(sample, predicted):
        db, cohort = row["database"], row["cohort"]
        key = (db, cohort, row["query_hash"])
        if key in seen_keys or db not in states or cohort not in ("seen", "unseen"):
            raise ValueError("Duplicate or invalid source prediction identity")
        seen_keys.add(key)
        c, i = caches[db], row["unique_position"]
        count = int(states[db]["support"][i] + states[db]["query"][i])
        if (
            row["query_hash"] != c["query_hash"][i]
            or row["raw_index"] != int(c["indices"][i])
            or row["presentations"] != count
            or (count > 0) != (cohort == "seen")
        ):
            raise ValueError("Sample identity/membership mismatch")
        for k in ("true", "pg", "n_tables"):
            if row[k] != c[k][i] or pred[k] != row[k]:
                raise ValueError("Sample prediction true/PG/table mismatch")
        if any(pred[k] != row[k] for k in ("database", "cohort", "query_hash")):
            raise ValueError("Prediction sample alignment mismatch")
        groups.setdefault((db, cohort), []).append(pred)
    for entry in plan["cohorts"]:
        db, cohort = entry["database"], entry["cohort"]
        c = caches[db]
        counts = states[db]["support"] + states[db]["query"]
        candidates = [
            i for i in range(len(counts)) if (counts[i] > 0) == (cohort == "seen")
        ]
        keys = [
            (
                hashlib.sha256(
                    f"s7-fixed-v1|{db}|{cohort}|{c['query_hash'][i]}".encode()
                ).hexdigest(),
                i,
            )
            for i in candidates
        ]
        chosen = [i for _, i in sorted(keys)[:2048]]
        actual = [r["query_hash"] for r in groups.get((db, cohort), [])]
        if (
            actual != [c["query_hash"][i] for i in chosen]
            or entry["pool_unique"] != len(candidates)
            or entry["sample_unique"] != len(chosen)
        ):
            raise ValueError("Independent fixed sample selection mismatch")
    held = {}
    for arm, dirname in [("A", "A_mean"), ("B", "B_mean_tail10")]:
        with (s6_run / dirname / "endpoint_rows.jsonl").open() as stream:
            for line in stream:
                r = json.loads(line)
                key = (r["database"], r["query_hash"])
                if key not in held:
                    held[key] = {
                        "true": r["true"],
                        "pg": math.exp(r["pg_reference_log"]),
                        "n_tables": r["n_tables"],
                        "query_hash": r["query_hash"],
                    }
                held[key][arm] = r["log_pred"]
    for db in ("chembl", "legalacts", "telstra"):
        c = caches[db]
        rows = []
        for i, h in enumerate(c["query_hash"]):
            row = held[(db, h)]
            # Use original FP64 PG rather than exponentiating a logged zero.
            row["pg"] = float(c["pg"][i])
            rows.append(row)
        groups[(db, "source_validation_full")] = rows
    gen = json.loads((output / "generalization.json").read_text())
    verified_cells = 0
    for row in gen["rows"]:
        rows = groups.get((row["database"], row["cohort"]), [])
        if not rows:
            close(
                row["statistics"],
                [{"bucket": "all", "n": 0, "status": "below_min_cell_n"}],
            )
            continue
        data = arrays(rows)
        masks = {
            "all": np.ones(len(rows), dtype=bool),
            "four_plus": data["n_tables"] >= 4,
        }
        table = {
            "1-3": data["n_tables"] < 4,
            "4": data["n_tables"] == 4,
            "5": data["n_tables"] == 5,
            "6+": data["n_tables"] >= 6,
        }
        masks.update(
            {
                f"{c}|tables={t}": cm & tm
                for c, cm in cards(data).items()
                for t, tm in table.items()
            }
        )
        if {r["bucket"] for r in row["statistics"]} != set(masks):
            raise ValueError("Missing/extra fixed stratum")
        for cell in row["statistics"]:
            pair_check(
                {k: v for k, v in cell.items() if k != "bucket"},
                data,
                masks[cell["bucket"]],
            )
            verified_cells += 1
    legal = json.loads((output / "legalacts.json").read_text())
    data = arrays(groups[("legalacts", "source_validation_full")])
    base = data["n_tables"] >= 4
    pair_check(legal["all"], data, np.ones(len(base), dtype=bool))
    pair_check(legal["four_plus"], data, base)
    qa, _, _ = numbers(data["A"], data["true"])
    qb, _, _ = numbers(data["B"], data["true"])
    at, bt = (
        base & (qa >= percentile(qa[base], 95)),
        base & (qb >= percentile(qb[base], 95)),
    )
    masks = {
        "persistent": at & bt,
        "entered": ~at & bt,
        "left": at & ~bt,
        "neither": base & ~at & ~bt,
    }
    for group in legal["tail_groups"]:
        composition_check(
            {k: v for k, v in group.items() if k != "group"},
            data,
            masks[group["group"]],
        )
    pair_check(legal["old_tail_stats"], data, at)
    composition_check(
        legal["q95_neighborhood"],
        data,
        base & (qb >= percentile(qb[base], 94)) & (qb <= percentile(qb[base], 96)),
    )
    axes = {
        "true": cards(data),
        "pg_direction": directions(np.log(np.maximum(data["pg"], 1)), data["true"]),
        "a_direction": directions(data["A"], data["true"]),
    }
    for cell in legal["strata"]:
        axis, bucket = cell["axis"], cell["bucket"]
        if axis == "true_x_pg_direction":
            card, pg = bucket.split("|")
            mask = axes["true"][card] & axes["pg_direction"][pg]
        else:
            mask = axes[axis][bucket]
        pair_check(
            {k: v for k, v in cell.items() if k not in ("axis", "bucket")},
            data,
            base & mask,
        )
        verified_cells += 1
    return {
        "passed": True,
        "purpose": "independent-math-quantile-and-identity-verification",
        "sample_queries": len(sample),
        "cohorts": len(plan["cohorts"]),
        "verified_statistic_cells": verified_cells,
        "tail_partition_unique": int(base.sum()),
        "s6_decision": "no_candidate",
        "causal_explanation_verified": False,
    }
