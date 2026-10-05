#!/usr/bin/env python3
"""Endpoint for the frozen P1 mapping ablation: source validation only.

Metric: unique-query zero-shot q-error, bucketed by number of joined tables. The
primary endpoint pools queries with 4+ tables; the three source validation
databases carry equal weight in the macro mean. Arms are compared pairwise on
exactly the same query_hash set, so a difference cannot come from sampling.

Sidecar risk flags and the shuffle's unchanged fraction are reported verbatim:
they bound how strong the shuffled control can be and must not be smoothed over.
Target databases are never opened.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scripts.arch_select import checkpoint_pool
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.metrics import q_values

TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
PRIMARY_MIN_TABLES = 4
BATCH = 128
PAIRS = (
    ("true", "shuffled", "primary; mapping information only"),
    ("true", "sum", "secondary; capacity is confounded"),
    ("shuffled", "sum", "secondary; capacity is confounded"),
)
RISK_KEYS = (
    "rows",
    "unique_queries",
    "duplicate_rows",
    "changed_rows",
    "changed_fraction",
    "unchanged_rows",
    "unchangeable_rows",
    "random_unchanged_rows",
    "sql_flags",
)


def endpoint_rows(model, store, device: str) -> list:
    """Zero-shot log prediction and truth for every unique query of one store."""
    model.eval()
    indices = store.unique_indices()
    rows = []
    for start in range(0, len(indices), BATCH):
        part = indices[start : start + BATCH]
        batch = store.batch(part, device)
        with torch.no_grad():
            prediction = model(batch).cpu().numpy()
        n_tables = batch["masks"][2].sum(1).cpu().numpy()
        true = batch["true"].cpu().numpy()
        for offset, index in enumerate(part):
            rows.append(
                {
                    "query_hash": store.rows[int(index)]["query_hash"],
                    "n_tables": int(n_tables[offset]),
                    "log_pred": float(prediction[offset]),
                    "true": float(true[offset]),
                }
            )
    return rows


def bucket_q95(rows: list, min_tables: int = 0) -> dict:
    selected = [r for r in rows if r["n_tables"] >= min_tables]
    if not selected:
        raise ValueError(f"No queries with at least {min_tables} tables")
    q = q_values([r["log_pred"] for r in selected], [r["true"] for r in selected])[0]
    return {
        "queries": len(q),
        "q50": float(np.percentile(q, 50)),
        "q95": float(np.percentile(q, 95)),
    }


def plan_digest(rows: list) -> str:
    return digest(sorted((r["query_hash"], r["n_tables"]) for r in rows))


def _log_q(row: dict) -> float:
    return float(q_values([row["log_pred"]], [row["true"]])[2][0])


def paired_comparison(left: list, right: list, left_name: str, right_name: str) -> dict:
    """Compare two arms on identical query hashes, restricted to the primary bucket."""
    left_by = {r["query_hash"]: r for r in left}
    right_by = {r["query_hash"]: r for r in right}
    if len(left_by) != len(left) or set(left_by) != set(right_by):
        raise ValueError("Paired rows must cover exactly the same unique queries")
    keys = [h for h, row in left_by.items() if row["n_tables"] >= PRIMARY_MIN_TABLES]
    if not keys:
        raise ValueError("Primary bucket is empty")
    left_log_q = np.array([_log_q(left_by[h]) for h in keys])
    right_log_q = np.array([_log_q(right_by[h]) for h in keys])
    delta = left_log_q - right_log_q
    left_q95 = float(np.percentile(np.exp(left_log_q), 95))
    right_q95 = float(np.percentile(np.exp(right_log_q), 95))
    return {
        "left": left_name,
        "right": right_name,
        "queries": len(keys),
        "left_q95": left_q95,
        "right_q95": right_q95,
        "q95_ratio": left_q95 / right_q95,
        "left_better_fraction": float((delta < 0).mean()),
        "mean_log_q_delta": float(delta.mean()),
        "median_log_q_delta": float(np.median(delta)),
    }


def sidecar_risks(mapping_root: str, databases: list) -> list:
    """Read the shuffle's own summary: risk flags and unchanged fraction, verbatim."""
    root = Path(mapping_root)
    out = []
    for db in databases:
        if db in TARGETS:
            raise ValueError("Sidecar risk report cannot include target databases")
        summary = read_json(root / db / "summary.json")
        out.append({"database": db, **{k: summary[k] for k in RISK_KEYS}})
    return out


def run_endpoint(model, pool, databases: list, device: str) -> dict:
    rows_by_db = {
        db: endpoint_rows(model, pool.get("pretrain", db), device) for db in databases
    }
    rows = [row for db in databases for row in rows_by_db[db]]
    per_db = {
        db: {
            "all_tables": bucket_q95(rows_by_db[db]),
            "four_plus_tables": bucket_q95(rows_by_db[db], PRIMARY_MIN_TABLES),
        }
        for db in databases
    }
    macro = float(np.mean([per_db[db]["four_plus_tables"]["q95"] for db in databases]))
    return {
        "per_database": per_db,
        "macro_four_plus_q95": macro,
        "plan_digest": plan_digest(rows),
        "unique_queries": len(rows),
        "rows_by_db": rows_by_db,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P1 mapping-ablation endpoint (source only)"
    )
    parser.add_argument(
        "--runs", required=True, help="ablation_runs.json from the training matrix"
    )
    parser.add_argument(
        "--config", required=True, help="any frozen arm config; supplies data_config"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = data_settings(read_json(args.config), args.config)
    databases = data["splits"]["validation"]
    sources = data["splits"]["train"] + databases
    if set(databases) & TARGETS:
        raise ValueError("Validation split contains a target database")
    matrix = read_json(args.runs)
    seeds = matrix["pairing"]["seeds"]
    results, digests, mapping_root = {}, set(), None
    for record in matrix["runs"]:
        name = f"{record['arm']}-seed{record['seed']}"
        model, state = load_checkpoint(Path(record["output"]) / "last.pt", args.device)
        result = run_endpoint(
            model, checkpoint_pool(state, data), databases, args.device
        )
        digests.add(result["plan_digest"])
        results[name] = result
        print(name, f"{result['macro_four_plus_q95']:.4f}", flush=True)
        if mapping_root is None:
            mapping_root = (
                state["metadata"]["training_signature"].get("mapping") or {}
            ).get("root")
    if len(digests) != 1:
        raise ValueError("Arms did not see the same unique-query plan")
    comparisons = []
    for seed in seeds:
        for left, right, role in PAIRS:
            for db in databases:
                a, b = (
                    results[f"{left}-seed{seed}"]["rows_by_db"][db],
                    results[f"{right}-seed{seed}"]["rows_by_db"][db],
                )
                comparisons.append(
                    {
                        "seed": seed,
                        "database": db,
                        "role": role,
                        **paired_comparison(a, b, left, right),
                    }
                )
            pooled = [
                r
                for db in databases
                for r in results[f"{left}-seed{seed}"]["rows_by_db"][db]
            ]
            other = [
                r
                for db in databases
                for r in results[f"{right}-seed{seed}"]["rows_by_db"][db]
            ]
            comparisons.append(
                {
                    "seed": seed,
                    "database": "pooled",
                    "role": role,
                    **paired_comparison(pooled, other, left, right),
                }
            )
    report = {
        "purpose": "p1 mapping ablation endpoint; source-only; not target-tuned",
        "metric": "unique-query zero-shot q-error Q95, 4+ tables, macro over source validation",
        "plan": {
            "databases": databases,
            "unique_queries": results[
                f"{matrix['runs'][0]['arm']}-seed{matrix['runs'][0]['seed']}"
            ]["unique_queries"],
            "plan_digest": digests.pop(),
        },
        "runs": {
            name: {k: v for k, v in result.items() if k != "rows_by_db"}
            for name, result in results.items()
        },
        "comparisons": comparisons,
        "sidecar_risks": sidecar_risks(mapping_root, sources) if mapping_root else [],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
