"""Structural summary for doc 118 steps 1-2: exact key summaries and shared bins.

Two questions, answered on the frozen three-source-database workload only:

    build/verify  can `C_cond` be re-assembled from a stored summary of the
                  join columns, bit for bit identical to the CSV-level value?
    bineval       how far does `C_uncond` degrade when every join edge is
                  bucketed on shared equal-width boundaries?
    queries       emit one structural log-cardinality estimate per workload row

The exact summary is a per-table projection of the join columns; the filter
mask used during verification still comes from the raw table, because filters
act on columns deliberately excluded from the summary.  Binned evaluation
rebuilds `bins`-wide equal-width histograms and re-buckets both sides of every
join edge onto the union range.

    python scripts/struct_summary.py build   --plan DIR/plan.json --dataset-dir D --out O
    python scripts/struct_summary.py verify  --plan DIR/plan.json --dataset-dir D --summary O
    python scripts/struct_summary.py bineval --plan DIR/plan.json --dataset-dir D --out O
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tree_identity as ti  # noqa: E402
from g1_oracle import (  # noqa: E402
    TARGET_DATABASES,
    _mask,
    load_tables,
    parse_ddl,
    percentile,
    q_error,
    write_json,
)
from spatial_meta_ce.manifests import file_hash, read_workload  # noqa: E402
from spatial_meta_ce.struct_summary import (  # noqa: E402
    DEFAULT_BINS,
    bucket_join_sum,
    c_struct,
    equal_width_hist,
    variable_classes,
)

SCHEMA = 1


def filtered_count(frame: pd.DataFrame, predicates) -> int:
    """Rows surviving `predicates`, without materialising a filtered copy.

    `apply_filters` returns `frame[mask]`, which copies every column; callers
    here only need the row count, and on a million-row table that copy is the
    whole cost.
    """
    if not predicates:
        return len(frame)
    mask = None
    for predicate in predicates:
        current = np.asarray(_mask(frame[predicate.column], predicate), dtype=bool)
        mask = current if mask is None else (mask & current)
    return int(mask.sum())


def binned_c_uncond(
    spec: ti.TreeSpec, frames: dict[str, pd.DataFrame], bins: int = DEFAULT_BINS
) -> float:
    """The `C_uncond` closed form with every join edge bucketed on shared bins."""
    classes, table_vars = variable_classes(spec.predicates, spec.tables)
    hists = {}
    for alias in spec.tables:
        for var in table_vars[alias]:
            column = next(c for a, c in classes[var] if a == alias)
            if (alias, column) not in hists:
                hists[(alias, column)] = equal_width_hist(frames[alias][column], bins)
    selectivities, total_rows = {}, {}
    for alias in spec.tables:
        total = len(frames[alias])
        total_rows[alias] = total
        kept = filtered_count(frames[alias], spec.for_alias(alias))
        selectivities[alias] = kept / total if total else 0.0
    return c_struct(
        spec.tables, spec.predicates, hists, selectivities, total_rows, bins
    )


def collect_join_columns(plan: dict) -> dict[str, dict[str, list[str]]]:
    """Database -> table -> sorted join columns used anywhere in the plan."""
    columns: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for db, block in plan["databases"].items():
        for item in block["queries"]:
            spec = ti._spec_from_json(item["spec"])
            classes, table_vars = variable_classes(spec.predicates, spec.tables)
            for alias in spec.tables:
                table = spec.names[spec.tables.index(alias)]
                for var in table_vars[alias]:
                    columns[db][table].update(c for a, c in classes[var] if a == alias)
    return {
        db: {t: sorted(c) for t, c in tables.items()} for db, tables in columns.items()
    }


STR_NULL = "\x00"


def _null_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in ("U", "S"):
        return np.asarray(values == STR_NULL)
    if values.dtype == object:
        return np.asarray(pd.isna(values))
    return np.isnan(values)


def _encoding(series: pd.Series) -> tuple[np.ndarray, str]:
    if pd.api.types.is_integer_dtype(series.dtype):
        return series.to_numpy(dtype=np.float64, na_value=np.nan), "int"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.to_numpy(dtype=np.float64, na_value=np.nan), "float"
    # Object arrays force `allow_pickle` on read; a NUL sentinel keeps the
    # summary loadable without pickle and is not a legal SQL string value.
    values = [STR_NULL if pd.isna(v) else str(v) for v in series.to_numpy()]
    width = max((len(v) for v in values), default=1)
    return np.asarray(values, dtype=f"<U{width}"), "str"


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build(
    plan: dict, dataset_dir: Path, out: Path, bins: int = DEFAULT_BINS
) -> dict[str, object]:
    """Store an exact projection of every join column plus its row counts."""
    out.mkdir(parents=True, exist_ok=True)
    columns = collect_join_columns(plan)
    arrays: dict[str, np.ndarray] = {}
    sizes: dict[str, np.ndarray] = {}
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "bins": bins,
        "databases": {},
        "dataset_dir": str(dataset_dir),
    }
    for db, tables in columns.items():
        if db in TARGET_DATABASES:
            raise ValueError(f"Refusing to summarise target database {db}")
        dataset = dataset_dir / db
        kinds, files = parse_ddl(dataset / f"postgres_create_{db}.sql")
        frames = load_tables(dataset, kinds, files)
        block: dict[str, object] = {"tables": {}}
        for table, table_columns in tables.items():
            frame = frames[table]
            entries = []
            for column in table_columns:
                values, kind = _encoding(frame[column])
                key = _digest(db, table, column)
                arrays[key] = values
                entries.append(
                    {
                        "column": column,
                        "key": key,
                        "kind": kind,
                        "nonnull": int(len(values) - _null_mask(values).sum()),
                    }
                )
            block["tables"][table] = {"rows": int(len(frame)), "columns": entries}
            sizes[_digest(db, table)] = np.asarray([len(frame)], dtype=np.int64)
        manifest["databases"][db] = block
        print(f"{db}: summarised {len(tables)} tables", flush=True)

    np.savez_compressed(out / "key_hist.npz", **arrays)
    np.savez_compressed(out / "size.npz", **sizes)
    manifest["source_sha256"] = {
        db: file_hash(dataset_dir / db / f"postgres_create_{db}.sql")
        for db in manifest["databases"]
    }
    manifest["content_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    write_json(out / "manifest.json", manifest)
    return manifest


def _load_summary(summary: Path) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((summary / "manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "content_hash"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if actual != manifest["content_hash"]:
        raise ValueError("Summary manifest content hash mismatch")
    with np.load(summary / "key_hist.npz", allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return manifest, arrays


def _summary_factors(
    spec: ti.TreeSpec,
    manifest: dict,
    arrays: dict[str, np.ndarray],
    db: str,
    frames: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Filtered per-table factor keys, with join values read from the summary.

    The filter mask comes from the raw table because filters act on columns
    deliberately excluded from the summary.  Everything key-distribution related
    is read back from the stored projection, which is what makes this a
    lossless-summary check rather than a second copy of the CSV path.
    """
    classes, table_vars = variable_classes(spec.predicates, spec.tables)
    block = manifest["databases"][db]["tables"]
    factors: dict[str, dict] = {}
    for alias in spec.tables:
        table = spec.names[spec.tables.index(alias)]
        frame = frames[alias]
        keep = np.ones(len(frame), dtype=bool)
        for predicate in spec.for_alias(alias):
            keep &= np.asarray(_mask(frame[predicate.column], predicate), dtype=bool)
        stored = {entry["column"]: entry for entry in block[table]["columns"]}
        columns = [
            next(c for a, c in classes[var] if a == alias) for var in table_vars[alias]
        ]
        if not columns:
            factors[alias] = {(): float(keep.sum())}
            continue
        values = [arrays[stored[column]["key"]][keep] for column in columns]
        mask = np.ones(len(values[0]), dtype=bool)
        for array in values:
            mask &= ~_null_mask(array)
        if not mask.any():
            factors[alias] = {}
            continue
        keys = list(zip(*(array[mask].tolist() for array in values)))
        counts = pd.Series(keys).value_counts(sort=False)
        factors[alias] = {tuple(key): float(value) for key, value in counts.items()}
    return factors


def verify(plan: dict, dataset_dir: Path, summary: Path) -> dict[str, object]:
    """Recompute C_cond from the stored summary and require bit equality."""
    manifest, arrays = _load_summary(summary)
    mismatches, checked, unfiltered = [], 0, 0
    for db, block in plan["databases"].items():
        if db not in manifest["databases"]:
            raise ValueError(f"{db} missing from summary")
        dataset = dataset_dir / db
        kinds, files = parse_ddl(dataset / f"postgres_create_{db}.sql")
        tables = load_tables(dataset, kinds, files)
        for item in block["queries"]:
            spec = ti._spec_from_json(item["spec"])
            frames = {
                alias: tables[spec.names[i]] for i, alias in enumerate(spec.tables)
            }
            reference = ti.score_query_tree(spec, tables, float(item["true"]))["c_cond"]
            classes, table_vars = variable_classes(spec.predicates, spec.tables)
            column_of = {}
            for index, (alias_a, col_a, alias_b, col_b) in enumerate(spec.predicates):
                column_of[(alias_a, index)] = col_a
                column_of[(alias_b, index)] = col_b
            adjacency: dict[str, list[tuple[str, int]]] = {a: [] for a in spec.tables}
            for index, (alias_a, _, alias_b, _) in enumerate(spec.predicates):
                var = next(
                    root
                    for root, members in classes.items()
                    if (alias_a, column_of[(alias_a, index)]) in members
                )
                adjacency[alias_a].append((alias_b, var))
                adjacency[alias_b].append((alias_a, var))
            factors = _summary_factors(spec, manifest, arrays, db, frames)
            rebuilt = ti._assemble(spec, adjacency, table_vars, factors)
            if rebuilt != reference:
                mismatches.append(
                    {
                        "database": db,
                        "query_hash": item["query_hash"],
                        "summary": rebuilt,
                        "reference": reference,
                    }
                )
            checked += 1
            if not spec.filters:
                unfiltered += 1
    result = {
        "queries_checked": checked,
        "unfiltered_queries": unfiltered,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "bit_identical": not mismatches,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def bineval(
    plan: dict, dataset_dir: Path, bins: int = DEFAULT_BINS
) -> tuple[dict[str, object], pd.DataFrame]:
    """Compare exact C_uncond against its shared-bin estimate on the same queries."""
    rows = []
    for db, block in plan["databases"].items():
        if db in TARGET_DATABASES:
            raise ValueError(f"Refusing to read target database {db}")
        dataset = dataset_dir / db
        kinds, files = parse_ddl(dataset / f"postgres_create_{db}.sql")
        tables = load_tables(dataset, kinds, files)
        for item in block["queries"]:
            spec = ti._spec_from_json(item["spec"])
            truth = max(float(item["true"]), 1.0)
            frames = {
                alias: tables[spec.names[i]] for i, alias in enumerate(spec.tables)
            }
            exact = ti.score_query_tree(spec, tables, float(item["true"]))
            binned = binned_c_uncond(spec, frames, bins)
            rows.append(
                {
                    "database": db,
                    "query_hash": item["query_hash"],
                    "n_tables": len(spec.tables),
                    "q_exact": q_error(exact["c_uncond"], truth),
                    "q_binned": q_error(binned, truth),
                    "exact": exact["c_uncond"],
                    "binned": binned,
                }
            )
        print(f"{db}: binned {len(block['queries'])}", flush=True)
    frame = pd.DataFrame(rows)
    summary = {
        "n": int(len(frame)),
        "bins": bins,
        "exact": {
            "median": percentile(frame["q_exact"].tolist(), 50),
            "q95": percentile(frame["q_exact"].tolist(), 95),
            "mean": float(frame["q_exact"].mean()),
        },
        "binned": {
            "median": percentile(frame["q_binned"].tolist(), 50),
            "q95": percentile(frame["q_binned"].tolist(), 95),
            "mean": float(frame["q_binned"].mean()),
        },
    }
    by_size = {}
    for size, group in frame.groupby("n_tables"):
        by_size[int(size)] = {
            "n": int(len(group)),
            "exact_q95": percentile(group["q_exact"].tolist(), 95),
            "binned_q95": percentile(group["q_binned"].tolist(), 95),
        }
    result = {
        "summary": summary,
        "by_table_count": by_size,
        "degradation_ratio": summary["binned"]["q95"] / summary["exact"]["q95"],
        "within_two_x_exact": summary["binned"]["q95"] <= 2 * summary["exact"]["q95"],
    }
    print(json.dumps(result, indent=2), flush=True)
    return result, frame


def _cached_estimate(spec, tables, bins, hist_cache, class_cache) -> float:
    """C_struct for one query, reusing histograms and edge sums across queries."""
    classes, table_vars = variable_classes(spec.predicates, spec.tables)
    table_of = {alias: spec.names[i] for i, alias in enumerate(spec.tables)}
    hists = {}
    for alias in spec.tables:
        for var in table_vars[alias]:
            column = next(c for a, c in classes[var] if a == alias)
            key = (table_of[alias], column)
            if key not in hist_cache:
                hist_cache[key] = equal_width_hist(tables[key[0]][column], bins)
            hists[(alias, column)] = hist_cache[key]

    selectivities, total_rows = {}, {}
    for alias in spec.tables:
        frame = tables[table_of[alias]]
        total = len(frame)
        total_rows[alias] = total
        kept = filtered_count(frame, spec.for_alias(alias))
        selectivities[alias] = kept / total if total else 0.0

    scale = 1.0
    for alias in spec.tables:
        total = total_rows[alias]
        if total == 0 or selectivities[alias] <= 0:
            return 0.0
        distinct = len(table_vars[alias])
        scale *= selectivities[alias] / (total ** max(distinct - 1, 0))
    product = 1.0
    for members in classes.values():
        ckey = tuple(sorted((table_of[a], c) for a, c in members))
        if ckey not in class_cache:
            class_cache[ckey] = bucket_join_sum(
                [hists[(a, c)] for a, c in members], bins
            )
        product *= class_cache[ckey]
    return scale * product


def queries(
    cache_dir: Path,
    workload_path: Path,
    dataset_dir: Path,
    bins: int = DEFAULT_BINS,
    fallback: str = "fail",
    limit: int | None = None,
    allow_target: bool = False,
) -> dict[str, object]:
    """Write `struct_log.npy` for every row of an existing feature cache.

    The cache manifest fixes the row order; each row's query is looked up by
    `query_hash` in the workload and estimated from the raw tables.  Rows whose
    SQL falls outside the join-tree scope get `fallback`: `fail` (default),
    `pg` (the PostgreSQL estimate, logged) or `zero` (no structural signal).
    """
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    rows = manifest["rows"]
    db, split = cache_dir.name, cache_dir.parent.name
    if db in TARGET_DATABASES and not allow_target:
        raise ValueError(
            f"Refusing to build a struct log for target database {db}; "
            "pass --allow-target for evaluation-time inference input only"
        )
    workload = read_workload(workload_path, db, split)
    by_hash = {}
    for row in workload:
        by_hash.setdefault(row["query_hash"], row)
    dataset = dataset_dir / db
    kinds, files = parse_ddl(dataset / f"postgres_create_{db}.sql")
    tables = load_tables(dataset, kinds, files)

    values, failures = [], []
    hist_cache: dict[tuple[str, str], object] = {}
    class_cache: dict[tuple, float] = {}
    for position, entry in enumerate(rows):
        if limit is not None and position >= limit:
            break
        source = by_hash.get(entry["query_hash"])
        reason = None
        estimate = None
        if source is None:
            reason = "missing_workload_row"
        else:
            try:
                spec = ti.analyze_tree(source["sql"], 2)
            except ti.Unsupported as exc:
                reason = str(exc)
            else:
                estimate = _cached_estimate(spec, tables, bins, hist_cache, class_cache)
        if reason is None:
            values.append(float(np.log(max(estimate, 1.0))))
            if (position + 1) % 5000 == 0:
                print(f"{db}: {position + 1}/{len(rows)}", flush=True)
            continue
        if fallback == "fail":
            raise ValueError(f"{db}: query {entry['id']} outside scope: {reason}")
        failures.append(
            {"id": entry["id"], "query_hash": entry["query_hash"], "reason": reason}
        )
        if fallback == "pg":
            if source is None:
                raise ValueError(
                    f"{db}: query {entry['id']} has no workload row for pg"
                )
            values.append(float(np.log(max(float(source["pg"]), 1.0))))
        elif fallback == "zero":
            values.append(0.0)
        else:
            raise ValueError(f"Unknown fallback {fallback!r}")

    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("struct_log has non-finite values")
    meta = {
        "schema": 1,
        "bins": bins,
        "database": db,
        "split": split,
        "rows": len(rows),
        "processed": len(values),
        "limited": limit is not None,
        "analyzable": len(values) - len(failures),
        "analyzable_fraction": (len(values) - len(failures)) / len(values)
        if values
        else 0.0,
        "fallback": fallback,
        "target_database": db in TARGET_DATABASES,
        "cache_content_hash": manifest["content_hash"],
        "workload_sha256": file_hash(workload_path),
        "failures": failures[:50],
        "failure_count": len(failures),
    }
    if limit is not None:
        # A partial run must not leave a short sidecar behind: a later
        # FeatureStore would load it and fail with a misleading length error.
        print(
            json.dumps(
                {
                    "limited": True,
                    "processed": len(values),
                    "rows": len(rows),
                    "note": "no sidecar written for a partial run",
                }
            ),
            flush=True,
        )
        return {"limited": True, "processed": len(values), "rows": len(rows)}
    np.save(cache_dir / "struct_log.npy", array)
    write_json(cache_dir / "struct_manifest.json", meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "failures"}, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "verify", "bineval", "queries"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--fallback", default="fail", choices=("fail", "pg", "zero"))
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-target", action="store_true")
    args = parser.parse_args(argv)
    if args.bins < 1:
        parser.error("--bins must be positive")

    if args.stage == "queries":
        if args.cache_dir is None or args.workload is None:
            parser.error("queries requires --cache-dir and --workload")
        queries(
            args.cache_dir,
            args.workload,
            args.dataset_dir,
            args.bins,
            args.fallback,
            args.limit,
            args.allow_target,
        )
        return 0

    if args.plan is None:
        parser.error(f"{args.stage} requires --plan")
    plan = json.loads(args.plan.read_text())
    expected = (args.plan.parent / "plan.sha256").read_text().strip()
    actual = hashlib.sha256(
        json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if expected != actual:
        raise SystemExit(f"Plan SHA mismatch: {actual} != {expected}")

    if args.stage == "build":
        if args.out is None:
            parser.error("build requires --out")
        build(plan, args.dataset_dir, args.out, args.bins)
        return 0
    if args.stage == "verify":
        if args.summary is None:
            parser.error("verify requires --summary")
        result = verify(plan, args.dataset_dir, args.summary)
        return 0 if result["bit_identical"] else 1
    if args.out is None:
        parser.error("bineval requires --out")
    result, frame = bineval(plan, args.dataset_dir, args.bins)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "bineval_queries.csv", index=False)
    write_json(args.out / "bineval.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
