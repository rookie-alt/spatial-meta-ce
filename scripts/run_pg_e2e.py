#!/usr/bin/env python3
"""Real PostgreSQL end-to-end timing with injected cardinalities (plan L3, doc 110).

What this measures
------------------
Wall-clock execution time of the *unchanged* SQL under four cardinality
regimes, on a real PostgreSQL 13.23 instance holding the real data:

* ``pg_native``  PostgreSQL's own estimates, no hint.
* ``true_card``  every join prefix hinted with its true cardinality -- the
  ceiling of what a perfect estimator could buy.
* ``ours``       join prefixes hinted with our model's zero-shot prediction.
* ``price``      join prefixes hinted with the provided PRICE weights.

Injection uses ``pg_hint_plan``'s ``Rows(t1 t2 ... #n)``, which sets the row
count of the join of exactly those relations.  No query text is rewritten: the
hint is a leading comment, so the measured statement is the original SQL.

Population and support rule
---------------------------
``workloads_all.sql`` renders each query as a flat list of join prefixes
(``/* (a, b) */ SELECT COUNT(*) FROM a, b WHERE ...``).  A query is *supported*
only when every prefix's alias set is a subset of the top-level alias set -- i.e.
one query block, the only case where a single leading hint comment can address
every prefix.  Unsupported queries are counted and never sampled.

Restrictions that travel with the numbers
-----------------------------------------
Single-table prefixes are base relations: PostgreSQL's own scan estimates are
used and no hint is emitted, matching PRICE's convention of substituting the true
cardinality for single-table subqueries.  A multi-table prefix absent from the
derived cache has no estimate, so its query drops out of the sample; the count is
reported.  Execution time is therefore measured on the subset of queries whose
every multi-table prefix was featurizable.

Read-only with respect to data: the statements are ``EXPLAIN (ANALYZE)`` on
COUNT(*) queries, which write nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.baselines import PriceAdapter, price_log_predictions  # noqa: E402
from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.data import StorePool  # noqa: E402
from spatial_meta_ce.manifests import canonical_sql  # noqa: E402
from spatial_meta_ce.meta import forward  # noqa: E402

TAG = re.compile(r"@(\d+)-(\d+)@$")
ALIASES = re.compile(r"/\*\s*\(([^)]*)\)\s*\*/")
ARMS = ("pg_native", "true_card", "ours", "price")
PSQL = "/opt/pgtest/bin/psql"
BATCH = 128


def canonical_hash(sql: str) -> str:
    """The cache's own query identity (docs/protocol.md §划分与防泄漏)."""
    return hashlib.sha256(canonical_sql(sql).encode()).hexdigest()


def parse_workload_all(path: Path) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("||")
            if len(fields) != 3:
                raise ValueError(f"{path.name}:{line_number}: expected 3 fields")
            match = TAG.search(fields[2].strip())
            if match is None:
                raise ValueError(f"{path.name}:{line_number}: bad tag")
            found = ALIASES.search(fields[0])
            grouped.setdefault(int(match.group(1)), []).append(
                {
                    "subquery": int(match.group(2)),
                    "sql": fields[0].strip(),
                    "true": float(fields[1]),
                    "line": line_number,
                    "aliases": tuple(
                        part.strip()
                        for part in (found.group(1) if found else "").split(",")
                        if part.strip()
                    ),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["subquery"])
    return grouped


def top_level_aliases(sql: str) -> tuple[str, ...]:
    """Aliases of the relations in the outermost SELECT only.

    Prefix comments name the aliases PRICE rendered for a join prefix.  Reading
    them against the *outer* block's relations is what makes the one-block test
    meaningful: a prefix mentioning a relation from a nested SELECT cannot be
    reached by a leading hint comment.
    """
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="postgres")
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        return ()
    sources = []
    source = select.args.get("from")
    if source is not None:
        sources.append(source.this)
    for join in select.args.get("joins") or []:
        sources.append(join.this)
    names = []
    for node in sources:
        alias = getattr(node, "alias", "") or ""
        names.append(alias or (node.name if hasattr(node, "name") else ""))
    return tuple(name for name in names if name)


def supported(rows: list[dict], top: tuple[str, ...]) -> bool:
    """One query block: every prefix's aliases sit inside the outer block's."""
    if len(top) < 2 or len(set(top)) != len(top):
        return False
    return all(set(row["aliases"]) <= set(top) for row in rows)


def predictions(store, model, params, price, device):
    """Zero-shot log-cardinality for every cache row, ours and PRICE."""
    indices = list(range(len(store)))
    ours, price_log = [], []
    for start in range(0, len(indices), BATCH):
        batch = store.batch(indices[start : start + BATCH], device)
        with torch.no_grad():
            ours.append(forward(model, params, batch).cpu().numpy())
            price_log.append(price(batch).cpu().numpy())
    return np.concatenate(ours), price_log_predictions(np.concatenate(price_log))


def psql(args, sql, timeout):
    # Parallel workers are pinned off so the plan root stays an Aggregate over a
    # single join chain: a Gather node would make the cardinality probe read a
    # meaningless Plan Rows.  The setting is recorded in the report.
    options = (
        f"-c statement_timeout={args.statement_timeout_ms} "
        "-c max_parallel_workers_per_gather=0"
    )
    environment = dict(os.environ, PGOPTIONS=options)
    result = subprocess.run(
        [
            PSQL,
            "-h",
            args.socket,
            "-p",
            str(args.port),
            "-d",
            args.database,
            "-U",
            args.user,
            "-X",
            "-A",
            "-t",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            sql,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()
        raise RuntimeError(tail[-1] if tail else "psql failed")
    return result.stdout


WRAPPERS = ("Aggregate", "Partial Aggregate", "Gather", "Gather Merge")


def join_rows(node: dict) -> float:
    """Estimated output rows of the topmost join.

    The root of a ``count(*)`` plan is an Aggregate whose ``Plan Rows`` is always
    1, so the root estimate carries no cardinality information.  Parallel plans
    additionally wrap the join in Gather / Partial Aggregate, whose ``Plan Rows``
    is also meaningless here, so the walk skips every wrapper and returns the
    first Join node it meets (falling back to the first non-wrapper).
    """
    seen = node
    while True:
        node_type = str(seen.get("Node Type", ""))
        if "Join" in node_type or node_type == "Nested Loop":
            return float(seen.get("Plan Rows", 0.0))
        children = seen.get("Plans") or []
        if len(children) != 1:
            break
        if seen.get("Node Type") not in WRAPPERS:
            break
        seen = children[0]
    return float(seen.get("Plan Rows", 0.0))


def explain(args, sql, analyze: bool, timeout: float) -> dict:
    prefix = (
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)"
        if analyze
        else "EXPLAIN (FORMAT JSON)"
    )
    payload = json.loads(psql(args, f"{prefix} {sql}", timeout))
    plan = payload[0]
    root = plan["Plan"]
    return {
        "execution_ms": float(plan.get("Execution Time", 0.0)),
        "planning_ms": float(plan.get("Planning Time", 0.0)),
        "total_cost": float(root.get("Total Cost", 0.0)),
        "join_rows": join_rows(root),
        "node_types": [
            node.get("Node Type")
            for node in (root, *(root.get("Plans") or []))
            if node.get("Node Type") != "Aggregate"
        ],
    }


def timed(args, sql, timeout) -> dict:
    """``--runs`` ANALYZE passes, first discarded as the cold run."""
    explain(args, sql, True, timeout)
    passes = [explain(args, sql, True, timeout) for _ in range(args.runs - 1)]
    return {
        "execution_ms": float(np.median([run["execution_ms"] for run in passes])),
        "runs_ms": [run["execution_ms"] for run in passes],
    }


def hint(rows: list[dict], estimates: dict, arm: str) -> str:
    """Leading pg_hint_plan comment covering every multi-table prefix."""
    parts = []
    for row in rows:
        if len(row["aliases"]) < 2:
            continue
        value = estimates[arm].get(row["subquery"])
        if value is None:
            continue
        parts.append(
            "Rows({} #{})".format(" ".join(row["aliases"]), max(1, round(value)))
        )
    return "/*+ " + " ".join(parts) + " */ " if parts else ""


def estimates_for(rows, lookup, ours_log, price_log):
    """Cardinality per prefix per arm; a missing multi-table prefix is recorded."""
    table = {arm: {} for arm in ARMS}
    missing = []
    for row in rows:
        table["true_card"][row["subquery"]] = row["true"]
        if len(row["aliases"]) < 2:
            continue
        index = lookup.get(canonical_hash(row["sql"]))
        if index is None:
            missing.append(row["line"])
            continue
        table["ours"][row["subquery"]] = float(np.exp(ours_log[index]))
        table["price"][row["subquery"]] = float(np.exp(price_log[index]))
    return table, missing


def q_error(estimate: float, truth: float) -> float:
    high, low = max(estimate, truth), min(estimate, truth)
    return float(high / max(low, 1.0))


def measure_arm(args, sql) -> dict:
    """One arm's timed execution and plan, or an error record.

    A statement that exceeds the per-statement timeout is a measurement outcome,
    not a reason to abort: it is recorded so the excluded query stays visible and
    the surviving sample can still be reported.
    """
    try:
        measured = timed(args, sql, args.timeout)
        plan = explain(args, sql, False, args.timeout)
    except Exception as exc:  # noqa: BLE001 - the message is the measurement result
        lines = str(exc).strip().splitlines()
        return {"error": lines[-1][:200] if lines else "unknown failure"}
    return {
        "execution_ms": measured["execution_ms"],
        "runs_ms": measured["runs_ms"],
        "total_cost": plan["total_cost"],
        "join_rows": plan["join_rows"],
        "node_types": plan["node_types"],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="PostgreSQL E2E timing (read-only)")
    parser.add_argument("--database", required=True)
    parser.add_argument("--allow-database", nargs="+", required=True)
    parser.add_argument("--workload-all", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--price-checkpoint", required=True)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=list(ARMS),
        help="arms to measure; a subset is allowed for cost probes and is "
        "recorded in the report",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--socket", default="/var/run/pgsock")
    parser.add_argument("--port", type=int, default=5456)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.database not in args.allow_database:
        raise SystemExit(f"{args.database} is not in --allow-database")
    unknown = set(args.arms) - set(ARMS)
    if unknown:
        raise SystemExit("Unknown arms: " + ", ".join(sorted(unknown)))
    if "pg_native" not in args.arms and len(args.arms) > 1:
        raise SystemExit("Ratios need pg_native in --arms")
    if args.runs < 2:
        raise SystemExit(
            "--runs must be >= 2 (the first pass is the discarded cold run)"
        )
    if Path(args.output).exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")

    torch.set_num_threads(1)
    model, state = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = state["metadata"]
    if args.database in set(metadata.get("train_databases") or ()) | set(
        metadata.get("validation_databases") or ()
    ):
        raise SystemExit("Checkpoint was trained or selected on this database")
    params = dict(model.named_parameters())
    store = StorePool(args.cache_root).get("test", args.database)
    lookup: dict[str, int] = {}
    for index, row in enumerate(store.rows):
        lookup.setdefault(row["query_hash"], index)
    maxima = [int(value) for value in store.manifest["inspection"]["max_counts"]]
    price = PriceAdapter(args.price_checkpoint, maxima, args.device)
    ours_log, price_log = predictions(store, model, params, price, args.device)

    groups = parse_workload_all(Path(args.workload_all))
    usable, unsupported, uninjectable = [], [], []
    for query, rows in sorted(groups.items()):
        top = top_level_aliases(rows[0]["sql"])
        if not supported(rows, top):
            unsupported.append(query)
            continue
        table, missing = estimates_for(rows, lookup, ours_log, price_log)
        if missing:
            uninjectable.append({"query": query, "lines": missing})
            continue
        usable.append((query, rows, table, top))

    sampled = random.Random(args.seed).sample(usable, min(args.sample, len(usable)))
    results = []
    partial = Path(args.output).with_suffix(".partial.jsonl")
    for query, rows, table, top in sampled:
        sql = rows[0]["sql"].rstrip(";")
        widest = max(rows, key=lambda row: len(row["aliases"]))
        entry = {
            "query": query,
            "aliases": list(top),
            "true_card": widest["true"],
            "hinted_prefixes": [
                " ".join(row["aliases"]) for row in rows if len(row["aliases"]) >= 2
            ],
            "arms": {"pg_native": measure_arm(args, sql)},
        }
        for arm in [a for a in args.arms if a != "pg_native"]:
            entry["arms"][arm] = measure_arm(args, hint(rows, table, arm) + sql)
            injected = table[arm].get(widest["subquery"])
            entry["arms"][arm]["injected"] = injected
            entry["arms"][arm]["hint"] = hint(rows, table, arm).strip()
            # An unmatched Rows() hint is silently ignored by pg_hint_plan, which
            # would leave the arm identical to pg_native while still looking like a
            # measurement.  The plan's own join estimate must therefore equal what
            # was injected.
            observed = entry["arms"][arm].get("join_rows")
            entry["arms"][arm]["injection_verified"] = (
                None
                if observed is None or injected is None
                else abs(observed - injected) <= 0.01 * max(injected, 1.0)
            )
        for arm in args.arms:
            if "join_rows" in entry["arms"][arm]:
                entry["arms"][arm]["q_error"] = q_error(
                    entry["arms"][arm]["join_rows"], entry["true_card"]
                )
        results.append(entry)
        with partial.open("a") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"query": query}
                | {
                    arm: entry["arms"].get(arm, {}).get("execution_ms")
                    for arm in args.arms
                }
            ),
            flush=True,
        )

    def ok(entry, arm):
        record = entry["arms"].get(arm, {})
        return record.get("execution_ms", 0) > 0

    # One common subset for every arm: a per-arm subset would let the slowest
    # queries drop out of one arm only, so "median ratio" would silently compare
    # different query sets.
    complete = [entry for entry in results if all(ok(entry, arm) for arm in args.arms)]

    def ratio(arm):
        values = [
            entry["arms"][arm]["execution_ms"]
            / entry["arms"]["pg_native"]["execution_ms"]
            for entry in complete
        ]
        if not values:
            return None
        return {
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p90": float(np.percentile(values, 90)),
            "better_than_pg": int(sum(value < 0.99 for value in values)),
            "worse_than_pg": int(sum(value > 1.01 for value in values)),
            "n": len(values),
        }

    def median_q(arm):
        values = [
            entry["arms"][arm]["q_error"]
            for entry in results
            if "q_error" in entry["arms"].get(arm, {})
        ]
        if not values:
            return None
        return {"median": float(np.median(values)), "n": len(values)}

    report = {
        "schema": 1,
        "purpose": "postgres-e2e-with-injected-cardinalities",
        "database": args.database,
        "includes_feature_extraction": False,
        "includes_plan_generation": True,
        "includes_database_execution": True,
        "claim_boundary": (
            "Execution of the unchanged SQL on PostgreSQL 13.23 with pg_hint_plan "
            "Rows() injection. The mechanism differs from PilotScope's push_card on "
            "PostgreSQL 13.1, so absolute times are not comparable with PRICE's "
            "published E2E figures; ratios within this report are."
        ),
        "population": {
            "queries_total": len(groups),
            "one_block": len(usable) + len(uninjectable),
            "unsupported_multi_block": len(unsupported),
            "uninjectable_missing_estimate": len(uninjectable),
            "sampled": len(results),
            "sample_seed": args.seed,
            "unsupported_queries": unsupported,
            "uninjectable_queries": uninjectable,
        },
        "runs": args.runs,
        "discarded_cold_runs": 1,
        "statement_timeout_ms": args.statement_timeout_ms,
        "arm_failures": [
            {"query": entry["query"], "arm": arm, "error": entry["arms"][arm]["error"]}
            for entry in results
            for arm in args.arms
            if "error" in entry["arms"].get(arm, {})
        ],
        "single_table_policy": "no hint; PostgreSQL's own scan estimate (PRICE convention)",
        "server": {
            "version": psql(args, "select version()", args.timeout).strip(),
            # pg_hint_plan is a shared_preload library here, not an extension, so
            # the effective setting is what has to be recorded.
            "hint_session_preload": psql(
                args,
                "select current_setting('session_preload_libraries')",
                args.timeout,
            ).strip(),
            "hint_enabled": psql(
                args, "select current_setting('pg_hint_plan.enable_hint')", args.timeout
            ).strip(),
            "parallel_workers_per_gather": psql(
                args,
                "select current_setting('max_parallel_workers_per_gather')",
                args.timeout,
            ).strip(),
        },
        "arms_measured": list(args.arms),
        "ratios_to_pg_native": {
            arm: ratio(arm) for arm in args.arms if arm != "pg_native"
        },
        "ratio_subset": (
            "the same queries for every arm: all four arms produced an execution "
            "time; the count is ratios_to_pg_native[*].n"
        ),
        "injection_checks": {
            arm: {
                "verified": sum(
                    1
                    for entry in results
                    if entry["arms"].get(arm, {}).get("injection_verified") is True
                ),
                "unverified": sum(
                    1
                    for entry in results
                    if entry["arms"].get(arm, {}).get("injection_verified") is False
                ),
                "not_checked": sum(
                    1
                    for entry in results
                    if entry["arms"].get(arm, {}).get("injection_verified") is None
                ),
            }
            for arm in args.arms
            if arm != "pg_native"
        },
        "median_q_error": {arm: median_q(arm) for arm in args.arms},
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["ratios_to_pg_native"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
