"""Offline P2-B encoding audit. Does not certify true SQL types or real selectivity."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sqlglot import exp, parse_one

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_meta_ce.manifests import file_hash

DATABASES = ("chembl", "legalacts", "telstra")
TOLERANCE = 1e-6


def numeric_bounds(expressions: list[str], column: str, minimum: float, maximum: float) -> tuple[float, float]:
    lower, upper, equals = minimum, maximum, []
    for expression in expressions:
        node = parse_one(expression, read="postgres")
        operator = type(node).__name__
        left, right = node.this, node.expression
        if str(left) != column:
            if str(right) != column:
                raise ValueError("Filter column mismatch")
            operator = {"LT": "GT", "LTE": "GTE", "GT": "LT", "GTE": "LTE"}.get(operator, operator)
            right = left
        if operator not in ("EQ", "LT", "GTE") or not isinstance(right, exp.Literal) or right.is_string:
            raise ValueError("Unsupported numeric predicate")
        value = float(right.this)
        if operator == "EQ":
            equals.append(value)
        elif operator == "LT":
            upper = min(upper, value)
        else:
            lower = max(lower, value)
    if equals:
        lower, upper = max(lower, max(equals)), min(upper, min(equals))
    return lower, upper


def numeric_histogram_selectivity(hist: np.ndarray, edges: np.ndarray,
                                  low: float, high: float, total: float) -> float:
    if total <= 0 or len(edges) != len(hist) + 1 or not np.all(np.diff(edges) > 0):
        raise ValueError("Unusable histogram or total")
    if low >= high:
        return 0.
    cumulative = np.r_[0., np.cumsum(hist, dtype=np.float64)]
    return float((np.interp(high, edges, cumulative) - np.interp(low, edges, cumulative)) / total)


def frozen_summary(values: list[float]) -> np.ndarray:
    if not values:
        return np.ones(3, dtype=np.float64)
    top = sorted(values, reverse=True)[:4]
    return np.array([np.prod(values), min(values),
                     np.prod([x ** (1 / (2**i)) for i, x in enumerate(top)])], dtype=np.float64)


def load_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def calculate_token(column: str, filters: list[str], histogram: dict, summary: dict,
                    types: dict, size: dict) -> np.ndarray:
    alias, name = column.split(".", 1)
    count = size[alias]["size"]
    if count <= 0:
        raise ValueError("Nonpositive table size")
    predicates = [parse_one(x, read="postgres") for x in filters]
    discrete = name in types["col_type"][alias]["dsct"] and len(predicates) == 1 and isinstance(predicates[0], exp.EQ)
    if discrete:
        record = summary[alias][name]
        keys, values = list(record["keys"]), list(record["values"])
        if len(keys) < 40:
            keys.extend([-1e3] * (40 - len(keys)))
            values.extend([0] * (40 - len(values)))
        else:
            pairs = sorted(zip(values, keys), reverse=True)
            keys = [k for _, k in pairs[:39]] + ["OtHeRs"]
            values = [v for v, _ in pairs[:39]] + [sum(v for v, _ in pairs[39:]) / len(pairs[39:])]
        literal = predicates[0].expression
        if not isinstance(literal, exp.Literal):
            raise ValueError("Unsupported discrete equality")
        key = literal.this if literal.is_string else float(literal.this)
        if not literal.is_string and key.is_integer():
            key = int(key)
        if not literal.is_string and str(literal.this) in keys:
            key = str(literal.this)
        try:
            index = keys.index(key)
        except ValueError:
            index = 39
        return np.r_[np.asarray(values, dtype=np.float64) / count,
                     [index / 40, (index + 1) / 40, values[index] / count]]
    numeric = histogram[alias][name]
    bins = np.asarray(numeric["hist"], dtype=np.float64)
    edges = np.asarray(numeric["bin_edges"], dtype=np.float64)
    low, high = numeric_bounds(filters, column, float(edges[0]), float(edges[-1]))
    if len(predicates) == 1 and isinstance(predicates[0], exp.EQ):
        high = low + 1e-5
    denom = edges[-1] - edges[0]
    if denom <= 0:
        raise ValueError("Degenerate histogram bounds")
    return np.r_[bins / numeric["len"],
                 [np.clip((low - edges[0]) / denom, 0, 1),
                  np.clip((high - edges[0]) / denom, 0, 1),
                  numeric_histogram_selectivity(bins, edges, low, high, count)]]


def audit_database(project: Path, source: Path, run: Path, db: str) -> dict:
    stats = source / "datas/statistics/pretrain" / db
    frozen = json.loads((project / "artifacts/cache/pretrain" / db / "manifest.json").read_text())
    gate = json.loads((run / "preflight.json").read_text())
    digest = gate["databases"][db]["input_sha256"]
    for filename in ("histogram40.pkl", "summary40.pkl", "abbrev_col_type.pkl", "size.pkl"):
        if file_hash(stats / filename) != digest[f"statistics/{filename}"]:
            raise ValueError(f"G0 statistics changed: {db}/{filename}")
    cache = project / "artifacts/cache/pretrain" / db
    for filename in ("group2.npy", "group3.npy", "offsets.npy"):
        if file_hash(cache / filename) != digest[f"cache/{filename}"]:
            raise ValueError(f"G0 token array changed: {db}/{filename}")
    histogram, summary = (load_pickle(stats / f"{key}40.pkl") for key in ("histogram", "summary"))
    types, sizes = load_pickle(stats / "abbrev_col_type.pkl"), load_pickle(stats / "size.pkl")
    offsets = np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)
    filter_tokens = np.load(cache / "group3.npy", mmap_mode="r", allow_pickle=False)
    table_tokens = np.load(cache / "group2.npy", mmap_mode="r", allow_pickle=False)
    by_line = {item["source_line"]: index for index, item in enumerate(frozen["rows"])}
    counts = Counter()
    seen = set()
    discrepancies, examples, max_token, max_agg = Counter(), [], 0., 0.
    p = run / f"{db}-parse_rows.jsonl"
    recorded = json.loads((run / f"{db}-parse_summary.json").read_text())
    if file_hash(p) != recorded["parse_rows_sha256"]:
        raise ValueError(f"P2-A row evidence changed: {db}")
    with p.open() as stream:
        for raw in stream:
            item = json.loads(raw)
            frozen_row = frozen["rows"][by_line[item["source_line"]]]
            if any(item[key] != frozen_row[key] for key in ("source_line", "exact_hash", "query_hash")):
                raise ValueError("Source row identity changed")
            # Counts include every unique SQL, but preserve each raw row in the A artifact.
            if item["query_hash"] in seen:
                continue
            seen.add(item["query_hash"])
            counts["unique_queries"] += 1
            if item["status"] != "parsed":
                counts["blocked_queries"] += 1
                continue
            index = by_line[item["source_line"]]
            predicates = [parse_one(x, read="postgres") for x in item["filter_expressions"]]
            per_column = {c: [] for c in item["filter_columns"]}
            for fragment, predicate in zip(item["filter_expressions"], predicates):
                for col in predicate.find_all(exp.Column):
                    if str(col) in per_column:
                        per_column[str(col)].append(fragment)
            per_table = {alias: [] for alias in item["tables"]}
            start = int(offsets[index, 3])
            for number, column in enumerate(item["filter_columns"]):
                counts["filter_column_slots"] += 1
                try:
                    token = calculate_token(column, per_column[column], histogram, summary, types, sizes)
                    observed = filter_tokens[start + number]
                    if token.shape != observed.shape or not np.allclose(token, observed, atol=TOLERANCE, rtol=TOLERANCE):
                        discrepancies["filter_token"] += 1
                        if len(examples) < 12:
                            examples.append({"line": item["source_line"], "column": column, "kind": "filter_token"})
                    else:
                        counts["filter_token_verified"] += 1
                    max_token = max(max_token, float(np.max(np.abs(token - observed))))
                    per_table[column.split(".")[0]].append(float(token[-1]))
                except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError) as error:
                    discrepancies["unsupported_filter"] += 1
                    if len(examples) < 12:
                        examples.append({"line": item["source_line"], "column": column, "reason": str(error)[:160]})
            for number, alias in enumerate(item["tables"]):
                counts["table_slots"] += 1
                if any(column.split(".")[0] == alias for column in item["filter_columns"]) and not per_table[alias]:
                    discrepancies["unverifiable_table"] += 1
                    continue
                expected = frozen_summary(per_table[alias])
                actual = table_tokens[int(offsets[index, 2]) + number, 1:4]
                if not np.allclose(expected, actual, atol=TOLERANCE, rtol=TOLERANCE):
                    discrepancies["table_aggregates"] += 1
                    if len(examples) < 12:
                        examples.append({"line": item["source_line"], "table": alias, "kind": "aggregate"})
                else:
                    counts["table_aggregates_verified"] += 1
                max_agg = max(max_agg, float(np.max(np.abs(expected - actual))))
    result = {"database": db, "status": "completed" if not discrepancies and not counts["blocked_queries"] else "partial",
              "counts": dict(counts), "discrepancies": dict(discrepancies), "examples": examples,
              "max_token_abs_difference": max_token, "max_aggregate_abs_difference": max_agg}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    project, source, run = args.project.resolve(), args.source.resolve(), args.run.resolve()
    summary = json.loads((run / "summary.json").read_text())
    if summary["status"] != "completed" or json.loads((run / "preflight.json").read_text())["status"] != "completed":
        raise ValueError("P2 G0/A have not passed; refusing B workload encoding audit")
    results = []
    for db in DATABASES:
        results.append(audit_database(project, source, run, db))
    g1 = json.loads((run / "g1_inventory.json").read_text())
    outcome = {"status": "blocked" if g1["status"] != "completed" else "partial",
               "reference_self_check": "blocked_postgresql_unavailable",
               "actual_workload_types": "blocked_snapshot_unverified",
               "encoding": "completed" if all(r["status"] == "completed" for r in results) else "partial",
               "independence": "shared_sqlglot_parser; independently implemented histogram interpolation; no PostgreSQL oracle",
               "limitations": "Numeric histograms and dsct EQ checked against frozen tokens; no SQL true selectivity/NULL/collation/type certification.",
               "databases": results}
    (run / "semantic_summary.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
