#!/usr/bin/env python3
"""Per-query cardinality-estimation latency (plan L2, doc 110).

What this measures
------------------
For every subquery of a target database's test workload: the cost of turning
SQL into features plus the cost of one model forward.  Summed per top-level
query, that is the whole estimation cost an optimizer would pay before it can
plan -- unlike the L1 script, which times only the forward pass of an
already-encoded sample.

Why it matters
--------------
PRICE's published ``INFERENCE TIME (MS)`` is a forward-pass figure.  If feature
extraction is the dominant term, a forward-pass comparison flatters every
method equally and says nothing about estimation cost.  This script measures the
split instead of assuming it.

Single-table subqueries
-----------------------
PRICE's own ``benchmark/e2e.py`` cannot featurize a single-table subquery and
substitutes its true cardinality.  This script reports that population
separately (``single_table_true_card``) instead of counting it as a model
success or as a silent exclusion.  Subqueries the extractor cannot handle for
other reasons are listed individually with their reason.

Numerical verification
----------------------
Freshly extracted features are compared element-wise against the derived cache
for a seeded sample of subqueries.  A mismatch aborts the run: timing a code
path whose features no longer match the cached ones would measure something
other than what the reported accuracy was computed on.

Read-only.  No training, no adaptation, no checkpoint selection, no execution.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_inference_latency import sha256_file, summarize, timed_ms  # noqa: E402
from spatial_meta_ce.baselines import PriceAdapter  # noqa: E402
from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.data import GROUP_DIMS, StorePool, batch_samples  # noqa: E402
from spatial_meta_ce.features import extractor, lower_except_quotes  # noqa: E402
from spatial_meta_ce.meta import forward  # noqa: E402

TAG = re.compile(r"@(\d+)-(\d+)@$")
RTOL = 1e-5
ATOL = 1e-7
STAGES = (
    "extraction_ms",
    "ours_forward_ms",
    "ours_total_ms",
    "ours_batched_forward_ms",
    "ours_batched_total_ms",
    "price_forward_ms",
    "price_total_ms",
    "price_batched_forward_ms",
    "price_batched_total_ms",
)


def parse_workload_all(path: Path) -> list[dict]:
    """Parse ``workloads_all.sql``: ``sql||true_cardinality||@query-subquery@``.

    The file carries no PostgreSQL estimate, so a subquery is only estimable if
    the same SQL appears in ``workloads.sql`` and therefore in the cache.  That
    restriction is counted, never hidden.
    """
    rows = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("||")
            if len(fields) != 3:
                raise ValueError(f"{path.name}:{line_number} has {len(fields)} fields")
            match = TAG.search(fields[2].strip())
            if match is None:
                raise ValueError(f"{path.name}:{line_number} tag is not @q-s@")
            rows.append(
                {
                    "sql": fields[0].strip(),
                    "true": float(fields[1]),
                    "query": int(match.group(1)),
                    "subquery": int(match.group(2)),
                    "line": line_number,
                }
            )
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def group_queries(rows: list[dict]) -> list[list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["query"], []).append(row)
    return [
        sorted(grouped[key], key=lambda r: r["subquery"]) for key in sorted(grouped)
    ]


def table_count(sql: str) -> int | None:
    """Distinct relation names in the query, or None if it will not parse."""
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError:
        return None
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return None
    return len({table.sql(dialect="postgres") for table in tree.find_all(exp.Table)})


def exact_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def group_tensor(part, dim: int) -> torch.Tensor:
    if part is None:
        return torch.zeros(0, dim)
    return torch.as_tensor(np.asarray(part), dtype=torch.float32).reshape(-1, dim)


def sample_of(parts, true, pg, identifier) -> dict:
    """A ``batch_samples`` input built from freshly extracted feature groups."""
    return {
        "groups": [group_tensor(part, dim) for part, dim in zip(parts, GROUP_DIMS)],
        "true": true,
        "pg": pg,
        "id": identifier,
        "hash": identifier,
    }


def compare(cached_groups, parts, identifier) -> None:
    """Element-wise equality with the derived cache, per feature group."""
    for group_index, (cached, part) in enumerate(zip(cached_groups, parts)):
        expected = group_tensor(part, GROUP_DIMS[group_index])
        if expected.shape != cached.shape:
            raise ValueError(f"{identifier}: group {group_index} shape changed")
        if not np.allclose(expected.numpy(), cached.numpy(), rtol=RTOL, atol=ATOL):
            raise ValueError(f"{identifier}: group {group_index} values changed")


def measure_database(args, pool, database, model, price_path) -> dict:
    store = pool.get(args.split, database)
    by_hash: dict[str, int] = {}
    for index, row in enumerate(store.rows):
        by_hash.setdefault(row["exact_hash"], index)
    workload = (
        Path(args.data_root)
        / "datas/workloads"
        / args.split
        / database
        / "workloads_all.sql"
    )
    queries = group_queries(parse_workload_all(workload))
    if args.limit_queries:
        queries = queries[: args.limit_queries]

    engine = extractor(args.data_root, database, split=args.split)
    maxima = [int(value) for value in store.manifest["inspection"]["max_counts"]]
    price = PriceAdapter(str(price_path), maxima, args.device) if price_path else None
    params = dict(model.named_parameters())

    verified: list[int] = []
    failures: list[dict] = []
    counts: dict[str, int] = {}
    per_query: list[dict] = []
    warmed = 0
    with contextlib.redirect_stdout(io.StringIO()):
        # Untimed warm-up: the extractor's caches and torch's thread pool are lazily
        # built, so the very first subquery costs several times the steady state
        # (measured 139.4 ms vs a 22.2 ms median).  Timing it would put a pure
        # start-up artefact into the p90/p99 tail.
        for group in queries:
            for row in group:
                if warmed >= args.warmup:
                    break
                index = by_hash.get(exact_hash(row["sql"]))
                if index is None:
                    continue
                parts = engine.create_sql_features(lower_except_quotes(row["sql"]))
                if parts is None:
                    continue
                cached = store[int(index)]
                batch = batch_samples(
                    [sample_of(parts[0], cached["true"], cached["pg"], "warm-up")],
                    args.device,
                )
                with torch.no_grad():
                    forward(model, params, batch)
                    if price is not None:
                        price(batch)
                warmed += 1
            if warmed >= args.warmup:
                break

        for group in queries:
            extraction, ours_single, ours_batch, price_single, price_batch = (
                [],
                [],
                [],
                [],
                [],
            )
            samples = []
            for row in group:
                index = by_hash.get(exact_hash(row["sql"]))
                if index is None:
                    counts["not_in_cache"] = counts.get("not_in_cache", 0) + 1
                    if table_count(row["sql"]) == 1:
                        counts["single_table_true_card"] = (
                            counts.get("single_table_true_card", 0) + 1
                        )
                    else:
                        failures.append(
                            {
                                "database": database,
                                "line": row["line"],
                                "tag": f"{row['query']}-{row['subquery']}",
                                "n_tables": table_count(row["sql"]),
                                "reason": "no PostgreSQL estimate in workloads_all.sql, "
                                "so the model input is unavailable",
                            }
                        )
                    continue
                cached = store[int(index)]
                started = time.perf_counter()
                parts = engine.create_sql_features(lower_except_quotes(row["sql"]))
                extraction.append((time.perf_counter() - started) * 1000.0)
                if parts is None:
                    failures.append(
                        {
                            "database": database,
                            "line": row["line"],
                            "tag": f"{row['query']}-{row['subquery']}",
                            "n_tables": table_count(row["sql"]),
                            "reason": "extractor returned None",
                        }
                    )
                    continue
                parts = parts[0]
                if len(verified) < args.verification_sample:
                    compare(cached["groups"], parts, f"{database}:{row['line']}")
                    verified.append(row["line"])
                sample = sample_of(
                    parts, cached["true"], cached["pg"], f"{database}:{row['line']}"
                )
                samples.append(sample)
                batch = batch_samples([sample], args.device)
                ours_single.append(
                    timed_ms(lambda b=batch: forward(model, params, b), args.device)
                )
                if price is not None:
                    price_single.append(timed_ms(lambda b=batch: price(b), args.device))

            if samples:
                packed = batch_samples(samples, args.device)
                ours_batch.append(
                    timed_ms(lambda b=packed: forward(model, params, b), args.device)
                )
                if price is not None:
                    price_batch.append(timed_ms(lambda b=packed: price(b), args.device))

            if not extraction:
                continue
            entry = {
                "query": group[0]["query"],
                "subqueries": len(group),
                "subqueries_timed": len(extraction),
                "extraction_ms": float(sum(extraction)),
                "ours_forward_ms": float(sum(ours_single)),
                "ours_total_ms": float(sum(extraction) + sum(ours_single)),
            }
            if samples:
                entry["ours_batched_forward_ms"] = float(sum(ours_batch))
                entry["ours_batched_total_ms"] = float(
                    sum(extraction) + sum(ours_batch)
                )
            if price is not None:
                entry["price_forward_ms"] = float(sum(price_single))
                entry["price_total_ms"] = float(sum(extraction) + sum(price_single))
                if samples:
                    entry["price_batched_forward_ms"] = float(sum(price_batch))
                    entry["price_batched_total_ms"] = float(
                        sum(extraction) + sum(price_batch)
                    )
            per_query.append(entry)

    if not per_query:
        raise ValueError(f"{database}: no subquery could be featurized")
    report = {
        "workload": str(workload),
        "queries_total": len(queries),
        "queries_measured": len(per_query),
        "subqueries_total": sum(len(group) for group in queries),
        "population": counts,
        "failures": failures,
        "verification": {
            "checked_lines": verified,
            "checked": len(verified),
            "requested": args.verification_sample,
            "rtol": RTOL,
            "atol": ATOL,
            "result": (
                f"{len(verified)} subqueries compared element-wise, all matched"
                + (
                    ""
                    if len(verified) >= args.verification_sample
                    else f"; fewer than the requested {args.verification_sample} "
                    "featurizable subqueries were available"
                )
            ),
        },
        "warm_up_subqueries": warmed,
    }
    for stage in STAGES:
        values = [entry[stage] for entry in per_query if stage in entry]
        if values:
            report[stage] = summarize(values)
    total = report["ours_total_ms"]["mean"]
    report["extraction_share_of_ours_total"] = (
        report["extraction_ms"]["mean"] / total if total else None
    )
    report["per_query_rows"] = per_query
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Per-query cardinality-estimation latency (read-only)"
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument("--allowed-targets", nargs="+", required=True)
    parser.add_argument("--price-checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verification-sample", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    requested = list(dict.fromkeys(args.databases))
    unapproved = set(requested) - set(args.allowed_targets)
    if unapproved:
        raise SystemExit(
            "Refusing to read databases outside --allowed-targets: "
            + ", ".join(sorted(unapproved))
        )
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.verification_sample < 1:
        raise SystemExit(
            "--verification-sample must be >= 1 (a run with no check is not reported)"
        )
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")

    torch.set_num_threads(args.threads)
    model, state = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = state["metadata"]
    exposed = set(metadata.get("train_databases") or ()) | set(
        metadata.get("validation_databases") or ()
    )
    if exposed & set(requested):
        raise SystemExit("Checkpoint was trained or selected on a requested database")
    if metadata.get("adapted_database"):
        raise SystemExit("An adapted checkpoint is not an independent starting point")

    pool = StorePool(args.cache_root)
    price_path = Path(args.price_checkpoint) if args.price_checkpoint else None
    if price_path is not None and not price_path.exists():
        raise SystemExit(f"PRICE checkpoint not found: {price_path}")

    result = {
        "schema": 1,
        "purpose": "estimation-latency-sql-to-cardinality",
        "unit": "milliseconds per top-level query, summed over its subqueries",
        "includes_feature_extraction": True,
        "includes_label_acquisition": False,
        "includes_plan_generation": False,
        "includes_database_execution": False,
        "claim_boundary": (
            "SQL parsing + feature extraction + model forward. Excludes labels, "
            "plan generation and execution; not a database speedup measurement."
        ),
        "databases": {},
    }
    protocol = {
        "schema": 1,
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "checkpoint_metadata": {
            "method": metadata.get("method"),
            "train_databases": metadata.get("train_databases"),
            "validation_databases": metadata.get("validation_databases"),
            "model_config": state["model_config"],
        },
        "databases_read": requested,
        "allowed_targets": list(args.allowed_targets),
        "split": args.split,
        "threads": args.threads,
        "device": args.device,
        "verification_sample": args.verification_sample,
        "warmup_subqueries": args.warmup,
        "limit_queries": args.limit_queries,
        "extraction_shared_between_arms": (
            "one extraction pass; its cost is attributed to both arms because "
            "both consume the same PRICE feature groups"
        ),
        "single_table_policy": (
            "PRICE's benchmark/e2e.py substitutes true cardinality for "
            "single-table subqueries; those are counted, never timed as model work"
        ),
        "price": None,
    }
    if price_path is not None:
        protocol["price"] = {
            "checkpoint": str(price_path.resolve()),
            "checkpoint_sha256": sha256_file(price_path),
            "maxima": "per-database inspection.max_counts (PRICE's own padding)",
        }

    for database in requested:
        report = measure_database(args, pool, database, model, price_path)
        result["databases"][database] = report
        print(
            json.dumps(
                {
                    "database": database,
                    "queries_measured": report["queries_measured"],
                    "extraction_p50": report["extraction_ms"]["p50"],
                    "ours_total_p50": report["ours_total_ms"]["p50"],
                    "verified": report["verification"]["checked"],
                }
            ),
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    out.with_suffix(".protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
