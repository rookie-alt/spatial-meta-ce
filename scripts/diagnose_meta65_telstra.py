#!/usr/bin/env python3
"""Descriptive Meta65 telstra diagnosis over frozen source predictions.

This module intentionally uses only NumPy and the standard library.  It never
imports the training runner, loads a checkpoint, connects to a database, or
touches a target database.
"""
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

SOURCE_DATABASES = ("chembl", "legalacts", "telstra")
PRIMARY_DATABASE = "telstra"
ARMS = ("A_all5_head3", "B_head3_head3", "C_last3_last3")
SHOTS = (5, 10, 20, 50, 100)
RUNS = 5
TABLE_BUCKET_LABELS = ("2", "3", "4", "5+")
TRUE_BUCKET_LABELS = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5",
                      "1e5-1e6", ">=1e6")
TRUE_BUCKET_EDGES = (-math.inf, 2.0, 3.0, 4.0, 5.0, 6.0, math.inf)
MIN_BUCKET_N = 30
NPZ_FIELDS = frozenset(("log_pred", "true", "n_tables", "query_hash", "query_id"))
CACHE_FILES = ("cards.npy", "manifest.json", "offsets.npy")
PROTOCOL_RELATIVE = "实验文档/52-Meta65-telstra退化源侧诊断协议.md"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    json.dumps(result, allow_nan=False)
    return result


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def no_symlinks(path):
    path = Path(path).expanduser().absolute()
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"Symlink forbidden: {path}")
    return path.resolve()


def resolved_file(path):
    """Resolve a regular file without rejecting OS-level temp aliases."""
    path = Path(path).expanduser()
    return path.resolve(strict=True)


def _ancestor(path, other):
    try:
        Path(path).relative_to(Path(other))
        return True
    except ValueError:
        return False


def prepare_outputs(args, root=None):
    """Validate CLI safety before any input data are read."""
    if getattr(args, "device", "cpu") != "cpu":
        raise ValueError("Only --device cpu is permitted; GPU execution is forbidden")
    root = no_symlinks(root or Path(__file__).resolve().parents[1])
    inputs = [no_symlinks(args.old_run), no_symlinks(args.completion_run)]
    cache_arg = getattr(args, "cache", None)
    if cache_arg:
        inputs.append(no_symlinks(cache_arg))
    out = Path(args.out).expanduser()
    if out.exists():
        raise ValueError(f"Refusing to overwrite output: {out}")
    out = no_symlinks(out)
    if any(_ancestor(out, item) or _ancestor(item, out) for item in inputs):
        raise ValueError("Output target must not be inside an input directory")
    if _ancestor(out, root / "spatial_meta_ce") or _ancestor(out, root / "PRICE-master"):
        raise ValueError("Output must not be written into source or PRICE paths")
    cache_dirs = [no_symlinks(cache_arg)] if cache_arg else []
    for cache_dir in cache_dirs:
        if _ancestor(out, cache_dir) or _ancestor(cache_dir, out):
            raise ValueError("Output target must not be written into cache paths")
    for item in inputs:
        text = str(item)
        if text.endswith("/PRICE-master") or "/PRICE-master/" in text:
            raise ValueError("PRICE paths are read-only; output is forbidden there")
    csv_dir = None
    if getattr(args, "csv_dir", None):
        csv_dir = Path(args.csv_dir).expanduser()
        if csv_dir.exists():
            raise ValueError(f"Refusing to overwrite CSV directory: {csv_dir}")
        csv_dir = no_symlinks(csv_dir)
        if any(_ancestor(csv_dir, item) or _ancestor(item, csv_dir) for item in inputs):
            raise ValueError("CSV directory must not be inside an input directory")
    return out, csv_dir


def q_error(log_pred, true):
    """Standard Q-error with the project convention: clip to 1, clamp true."""
    log_pred = np.asarray(log_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if not len(true) or log_pred.shape != true.shape:
        raise ValueError("Invalid prediction/label dimensions")
    if (not np.isfinite(log_pred).all() or not np.isfinite(true).all()
            or (true < 0).any()):
        raise ValueError("Invalid or nonfinite prediction/label")
    safe_log_pred = np.maximum(log_pred, 0.0)
    safe_true = np.maximum(true, 1.0)
    log_q = np.abs(safe_log_pred - np.log(safe_true))
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.exp(log_q)
    if not np.isfinite(q).all():
        raise FloatingPointError("Standard Q-error overflow")
    return q, log_q


def quantile_metrics(log_pred, true_or_log_q=None):
    """Return standard and log-space metrics.

    The two-argument form accepts raw log predictions and true cardinalities.
    The one-argument form accepts already-computed log_q values.
    """
    if true_or_log_q is None:
        log_q = np.asarray(log_pred, dtype=np.float64).reshape(-1)
        q = np.exp(log_q)
    else:
        q, log_q = q_error(log_pred, true_or_log_q)
    return {
        "count": len(q),
        "q50": float(np.percentile(q, 50)),
        "q95": float(np.percentile(q, 95)),
        "log_q50": float(np.percentile(log_q, 50)),
        "log_q95": float(np.exp(np.percentile(log_q, 95))),
    }


def _bucket_summary(values, labels, boundaries, min_n=MIN_BUCKET_N):
    values = np.asarray(values)
    result = {}
    for index, label in enumerate(labels):
        lower, upper = boundaries[index], boundaries[index + 1]
        if index == 0:
            mask = values < upper
        elif index == len(labels) - 1:
            mask = values >= lower
        else:
            mask = (values >= lower) & (values < upper)
        count = int(mask.sum())
        if count < min_n:
            row = {"count": count, "status": "below_min_n"}
        else:
            row = {"count": count, "share": float(count / len(values)) if len(values) else 0.0,
                   "status": "ok"}
        result[label] = row
    return result


def _add_metrics(rows, mask, log_pred, true, min_n):
    if not mask.any():
        return {"count": 0, "status": "below_min_n"}
    values = quantile_metrics(np.asarray(log_pred)[mask], np.asarray(true)[mask])
    values.update(count=int(mask.sum()), status="ok" if int(mask.sum()) >= min_n
                  else "below_min_n")
    if values["count"] < min_n:
        return {"count": values["count"], "status": "below_min_n"}
    return values


def table_bucket_rows(tables, log_pred=None, true=None, min_n=MIN_BUCKET_N):
    tables = np.asarray(tables, dtype=np.int64).reshape(-1)
    # Protocol fixes D1 to {2,3,4,5+}; the source also has one-table queries, so
    # they are assigned to the first bucket rather than silently dropped.
    bucketed = np.maximum(tables, 2).astype(np.float64)
    boundaries = (2.0, 3.0, 4.0, 5.0, math.inf)
    rows = _bucket_summary(bucketed, TABLE_BUCKET_LABELS,
                           boundaries, min_n=min_n)
    if log_pred is not None and true is not None:
        log_pred = np.asarray(log_pred, dtype=np.float64)
        true = np.asarray(true, dtype=np.float64)
        masks = (tables == 2, tables == 3, tables == 4, tables >= 5)
        for label, mask in zip(TABLE_BUCKET_LABELS, masks):
            rows[label].update(_add_metrics({}, mask, log_pred, true, min_n))
    return rows


def true_bucket_rows(true, log_pred=None, min_n=MIN_BUCKET_N):
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if (not np.isfinite(true).all() or (true < 0).any()):
        raise ValueError("Invalid true cardinality")
    logs = np.log10(np.maximum(true, 1.0))
    rows = _bucket_summary(logs, TRUE_BUCKET_LABELS, TRUE_BUCKET_EDGES, min_n=min_n)
    if log_pred is not None:
        log_pred = np.asarray(log_pred, dtype=np.float64)
        masks = [logs < 2.0]
        masks.extend(((logs >= edge) & (logs < TRUE_BUCKET_EDGES[index + 1]))
                     for index, edge in enumerate((2.0, 3.0, 4.0, 5.0), start=1))
        masks.append(logs >= 6.0)
        for label, mask in zip(TRUE_BUCKET_LABELS, masks):
            if rows[label]["status"] == "ok":
                rows[label].update(_add_metrics({}, mask, log_pred, true, min_n))
    return rows


def direction_labels(a_log_pred, true):
    a_log_pred = np.asarray(a_log_pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    delta = a_log_pred - np.log(np.maximum(true, 1.0))
    over = delta >= 0
    return np.where(over, "over", "under").astype(str)


def pg_direction_labels(pg, true):
    pg = np.asarray(pg, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    delta = np.log(np.maximum(pg, 1.0)) - np.log(np.maximum(true, 1.0))
    return np.where(delta > 0, "over", "under").astype(str)


def verify_query_sets(query_sets, label):
    """Require the same set of query identities across the compared cells.

    The frozen plan keeps a common query set for all K within one run and for all
    arms within one cell, but the on-disk row order is not guaranteed to be
    identical, so membership (a set) is the correct invariant here.
    """
    if not query_sets:
        raise ValueError(f"Empty {label} query sets")
    reference = {item[0] if isinstance(item, tuple) else item for item in query_sets[0]}
    if len(reference) != len(query_sets[0]):
        raise ValueError(f"Duplicate query identity in {label} query set")
    for index, current in enumerate(query_sets[1:], start=1):
        keys = {item[0] if isinstance(item, tuple) else item for item in current}
        if len(keys) != len(current):
            raise ValueError(f"Duplicate query identity in {label} query set {index}")
        if keys != reference:
            raise ValueError(f"{label} query set changed between repetitions: {index}")
    return list(reference)


def delta_summary(left, right, hashes, ids, top_n=20):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or len(left) != len(hashes) or len(hashes) != len(ids):
        raise ValueError("Paired delta inputs disagree")
    delta = right - left
    order = np.argsort(-np.abs(delta), kind="stable")[:top_n]
    largest = [{"query_hash": str(hashes[index]), "query_id": str(ids[index]),
                "delta": float(delta[index])} for index in order]
    return {
        "count": len(delta),
        "mean": float(np.mean(delta)),
        "p10": float(np.percentile(delta, 10)),
        "p50": float(np.percentile(delta, 50)),
        "p90": float(np.percentile(delta, 90)),
        "largest": largest,
    }


def leave_one_out_stability(per_k):
    """Report sign/order stability after removing each run in turn."""
    values = {key: arms for key, arms in per_k.items() if "A" in arms and "C" in arms}
    if not values:
        return {"sign_flips": [], "k_order_changed": False, "leave_one_out": []}
    records = []
    sign_flips = []
    for k, arms in values.items():
        a = np.asarray(arms["A"], dtype=np.float64)
        c = np.asarray(arms["C"], dtype=np.float64)
        full_sign = math.copysign(1.0, float(np.mean(c - a)))
        for index in range(len(a)):
            keep = np.arange(len(a)) != index
            delta = float(np.mean(c[keep] - a[keep]))
            sign = math.copysign(1.0, delta)
            if sign != full_sign:
                sign_flips.append({"k": k, "removed_run": index})
            records.append({"k": k, "removed_run": index, "delta": delta,
                            "continued_positive": delta > 0})
    order_changed = False
    if len(values) > 1:
        full_order = sorted(values, key=lambda key: float(np.mean(
            np.asarray(values[key]["C"]) - np.asarray(values[key]["A"]))))
        run_count = len(np.asarray(next(iter(values.values()))["A"]))
        for index in range(run_count):
            order = sorted(values, key=lambda key: float(np.mean(
                (np.asarray(values[key]["C"]) - np.asarray(values[key]["A"]))[
                    np.arange(run_count) != index])))
            if order != full_order:
                order_changed = True
    return {"sign_flips": sign_flips, "k_order_changed": order_changed,
            "leave_one_out": records}


def top_log_contribution(log_q, fraction=0.05):
    log_q = np.asarray(log_q, dtype=np.float64).reshape(-1)
    if not len(log_q) or not np.isfinite(log_q).all() or (log_q < 0).any():
        raise ValueError("Invalid log_q values")
    count = max(1, math.ceil(len(log_q) * fraction))
    selected = np.sort(log_q)[-count:]
    total = float(np.sum(log_q))
    contribution = float(np.sum(selected) / total) if total > 0 else 0.0
    return {
        "definition": "sum(log_q of top 5% q-error queries) / sum(all log_q)",
        "selected_count": int(count),
        "tail_count_share": float(count / len(log_q)),
        "contribution_ratio": contribution,
    }


def four_tuple(a_value, c_value):
    a_value = float(a_value)
    c_value = float(c_value)
    if not math.isfinite(a_value) or not math.isfinite(c_value) or a_value <= 0 or c_value <= 0:
        raise ValueError("Four-tuple values must be positive and finite")
    ratio = c_value / a_value
    return {"A": a_value, "C_minus_A": c_value - a_value, "C_over_A": ratio,
            "log10_C_over_A": math.log10(ratio)}


def _manifest_rows(manifest):
    rows = manifest.get("rows") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest.rows must be a nonempty list")
    return rows


def anchor_pg(cache_dir, query_id, query_hash, true, n_tables):
    """Strictly bind source cache PG values to NPZ query identities.

    Returns (status, details, pg).  Any malformed or mismatching input returns
    status='failed' and pg=None; callers must not use PG for that database.
    """
    cache_dir = Path(cache_dir)
    details = {"cache_dir": str(cache_dir), "files": {}}
    try:
        for name in CACHE_FILES:
            path = resolved_file(cache_dir / name)
            if not path.is_file():
                raise ValueError(f"Missing PG anchor file: {name}")
            details["files"][name] = sha256(path)
        manifest = read_json(cache_dir / "manifest.json")
        rows = _manifest_rows(manifest)
        cards = np.load(cache_dir / "cards.npy", allow_pickle=False, mmap_mode="r")
        offsets = np.load(cache_dir / "offsets.npy", allow_pickle=False, mmap_mode="r")
        if cards.ndim != 2 or cards.shape[1] < 2 or not np.isfinite(cards).all():
            raise ValueError("Invalid cards.npy shape or values")
        if offsets.ndim != 2 or offsets.shape[1] < 3 or len(offsets) != len(rows) + 1:
            raise ValueError("offsets.npy shape does not match manifest")
        first = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("query_hash"), str):
                raise TypeError("Invalid manifest row identity")
            first.setdefault(row["query_hash"], index)
        expected_ids = []
        for value in query_id:
            if not isinstance(value, str):
                raise TypeError("query_id must be a string")
            parts = value.split("/")
            if len(parts) != 3 or parts[1] != "pretrain" or not parts[2].isdigit():
                raise ValueError(f"Invalid query_id row format: {value}")
            row_index = int(parts[2]) - 1
            if row_index >= len(rows) or rows[row_index].get("id") != value:
                raise ValueError(f"query_id does not address manifest row: {value}")
            expected_ids.append(row_index)
        selected_rows = []
        first_rows = []
        for identity, value in zip(query_hash, query_id):
            if identity not in first:
                raise ValueError("NPZ query_hash absent from manifest")
            row_index = int(str(value).split("/")[2]) - 1
            if rows[row_index].get("query_hash") != identity:
                raise ValueError("query_id row query_hash does not match NPZ")
            selected_rows.append(row_index)
            first_rows.append(first[identity])
        if any(left >= right for left, right in itertools.pairwise(first_rows)):
            raise ValueError("NPZ order does not follow manifest first-occurrence order")
        canonical_rows = first_rows
        cards_true = np.asarray(cards[canonical_rows, 0], dtype=np.float64)
        np_true = np.asarray(true, dtype=np.float64)
        if not np.array_equal(cards_true, np_true):
            raise ValueError("cards[rows,0] does not equal NPZ true")
        table_values = offsets[1:, 2] - offsets[:-1, 2]
        if not np.array_equal(np.asarray(table_values[canonical_rows], dtype=np.int64),
                              np.asarray(n_tables, dtype=np.int64)):
            raise ValueError("offsets table count does not equal NPZ n_tables")
        if len(set(first_rows)) != len(first_rows):
            raise ValueError("NPZ identities do not deduplicate to unique manifest rows")
        details.update(status_reason="all checks passed", rows=canonical_rows,
                       query_id_rows=expected_ids)
        return "verified", details, np.asarray(cards[canonical_rows, 1], dtype=np.float64)
    except Exception as error:  # noqa: BLE001 - PG anchoring must degrade to a reported failure, never crash the run.
        details["error"] = str(error)
        return "failed", details, None


def load_npz(path):
    path = no_symlinks(path)
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != set(NPZ_FIELDS):
            raise ValueError(f"Unexpected NPZ fields: {path}: {sorted(values.files)}")
        result = {key: np.asarray(values[key]) for key in values.files}
    log_pred = np.asarray(result["log_pred"], dtype=np.float64).reshape(-1)
    true = np.asarray(result["true"], dtype=np.float64).reshape(-1)
    tables = np.asarray(result["n_tables"], dtype=np.int64).reshape(-1)
    hashes = np.asarray(result["query_hash"]).astype(str).reshape(-1)
    ids = np.asarray(result["query_id"]).astype(str).reshape(-1)
    if not (log_pred.shape == true.shape == tables.shape == hashes.shape == ids.shape):
        raise ValueError(f"NPZ field lengths disagree: {path}")
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"Duplicate query_hash in NPZ: {path}")
    return {"log_pred": log_pred, "true": true, "n_tables": tables,
            "query_hash": hashes, "query_id": ids, "path": path}


def _prediction_path(run, arm, db, k, run_index):
    if arm == "C_last3_last3":
        return Path(run) / arm / "predictions" / f"{db}-k{k}-run{run_index}.npz"
    return Path(run) / arm / "predictions" / f"{db}-k{k}-run{run_index}.npz"


def _plan_for(run_dir):
    path = no_symlinks(Path(run_dir) / "endpoint_plan.json")
    plan = read_json(path)
    if set(plan) != set(SOURCE_DATABASES):
        raise ValueError("Endpoint plan does not contain exactly source databases")
    for db in SOURCE_DATABASES:
        if not isinstance(plan[db], list) or len(plan[db]) != RUNS:
            raise ValueError(f"Endpoint plan must have {RUNS} runs for {db}")
    return plan, path


def _query_index(cell, plan):
    query = np.asarray(cell["query_hash"]).astype(str)
    query_id = np.asarray(cell["query_id"]).astype(str)
    if plan and not isinstance(plan[0], dict):
        # The frozen endpoint plan stores 0-based manifest row indices for the
        # query set (all unique rows minus this run's 100-row support), while the
        # NPZ row numbers embedded in query_id are 1-based.  The two therefore
        # cannot be compared element-wise.  Cell-level identity, digest and plan
        # binding was already certified per cell in the completion stage; here we
        # only re-check that the NPZ carries a well-formed, unique, complete
        # query set of the planned cardinality.
        parsed = []
        for value in query_id:
            parts = str(value).split("/")
            if len(parts) != 3 or parts[1] != "pretrain" or not parts[2].isdigit():
                raise ValueError(f"Invalid query_id row format: {value}")
            parsed.append(int(parts[2]))
        if len(parsed) != len(plan) or len(set(parsed)) != len(parsed):
            raise ValueError("Integer endpoint plan does not match NPZ query cardinality")
        return {value: index for index, value in enumerate(query)}
    expected = {str(item["query_hash"]): str(item["query_id"]) for item in plan}
    if len(expected) != len(plan):
        raise ValueError("Plan contains duplicate query identities")
    if set(query) != set(expected):
        raise ValueError("NPZ query_hash set does not match endpoint plan")
    order = {value: index for index, value in enumerate(query)}
    if any(expected[value] != query_id[order[value]] for value in expected):
        raise ValueError("NPZ query_id does not match endpoint plan")
    return order


def _ordered(cell, identities):
    index = {value: pos for pos, value in enumerate(cell["query_hash"])}
    missing = [value for value in identities if value not in index]
    if missing:
        raise ValueError("Paired query identity missing from cell")
    positions = np.asarray([index[value] for value in identities], dtype=np.int64)
    return {key: np.asarray(cell[key])[positions] for key in
            ("log_pred", "true", "n_tables", "query_id")}


def collect_cells(old_run, completion_run, cache_dir=None):
    """Read and validate all 225 source prediction cells."""
    old_plan, old_plan_path = _plan_for(old_run)
    new_plan, new_plan_path = _plan_for(completion_run)
    if old_plan != new_plan:
        raise ValueError("A/B and C endpoint plans differ")
    cells = {}
    input_hashes = {}
    for arm in ARMS:
        run_dir = completion_run if arm == "C_last3_last3" else old_run
        for db in SOURCE_DATABASES:
            run_plans = old_plan[db]
            for k in SHOTS:
                for run_index, plan in enumerate(run_plans):
                    path = no_symlinks(_prediction_path(run_dir, arm, db, k, run_index))
                    cell = load_npz(path)
                    input_hashes[str(path)] = sha256(path)
                    _query_index(cell, plan["query"])
                    cells[(arm, db, k, run_index)] = cell
    for db in SOURCE_DATABASES:
        for run_index in range(RUNS):
            query_sets = []
            for k in SHOTS:
                cell = cells[(ARMS[0], db, k, run_index)]
                query_sets.append(list(zip(cell["query_hash"], cell["query_id"])))
            verify_query_sets(query_sets, f"{db}/run{run_index}")
    for db in SOURCE_DATABASES:
        for run_index in range(RUNS):
            query_sets = []
            for k in SHOTS:
                reference = old_plan[db][run_index]["query"]
                for arm in ARMS:
                    cell = cells[(arm, db, k, run_index)]
                    _query_index(cell, reference)
            for arm in ARMS:
                query_sets.append(list(zip(cells[(arm, db, 5, run_index)]["query_hash"],
                                           cells[(arm, db, 5, run_index)]["query_id"])))
            verify_query_sets(query_sets, f"{db}/run{run_index}/arms")
    return cells, input_hashes, (old_plan_path, new_plan_path)


def _cell_log_q(cell):
    return q_error(cell["log_pred"], cell["true"])[1]


def _values(cells, arm, db, k, key):
    return np.concatenate([np.asarray(cells[(arm, db, k, run)][key])
                           for run in range(RUNS)])


def _log_qs(cells, arm, db, k):
    return np.concatenate([_cell_log_q(cells[(arm, db, k, run)])
                           for run in range(RUNS)])


def _group_metrics(log_q, masks, labels, min_n=MIN_BUCKET_N):
    log_q = np.asarray(log_q, dtype=np.float64)
    result = {}
    total = len(log_q)
    for label, mask in zip(labels, masks):
        count = int(mask.sum())
        if count < min_n:
            result[label] = {"n": count, "status": "below_min_n"}
        else:
            metrics = quantile_metrics(log_q[mask])
            metrics.update(count=count, share=float(count / total) if total else 0.0,
                           status="ok")
            result[label] = metrics
    return result


def _d3_mode(cells, db, mode, pg_by_cell=None):
    result = {}
    for k in SHOTS:
        groups = {}
        for run in range(RUNS):
            a_cell = cells[(ARMS[0], db, k, run)]
            if mode == "A":
                labels = direction_labels(a_cell["log_pred"], a_cell["true"])
            else:
                if pg_by_cell is None:
                    raise ValueError("PG direction requested without verified PG anchor")
                key = (ARMS[0], db, k, run)
                if key not in pg_by_cell:
                    raise ValueError("PG direction missing for NPZ cell")
                pg = np.asarray(pg_by_cell[key], dtype=np.float64)
                if len(pg) != len(a_cell["true"]):
                    raise ValueError("PG anchor length does not match NPZ cell")
                labels = pg_direction_labels(pg, a_cell["true"])
            for direction in ("over", "under", "exact"):
                mask = labels == direction
                if not mask.any():
                    continue
                group = groups.setdefault(direction, {"count": 0, "arms": {}})
                group["count"] += int(mask.sum())
                for arm in ARMS:
                    log_q = _cell_log_q(cells[(arm, db, k, run)])[mask]
                    group["arms"].setdefault(arm, []).append(log_q)
        for group in groups.values():
            total = sum(item["count"] for item in groups.values())
            group["share"] = float(group["count"] / total) if total else 0.0
            for arm in ARMS:
                group["arms"][arm] = quantile_metrics(np.concatenate(group["arms"][arm]))
        result[str(k)] = groups
    return result


def _d4(cells, db):
    result = {}
    for k in SHOTS:
        # Each run excludes its own 100-row support, so the query set differs
        # across runs.  Pairing is valid within a run (A/B/C share the plan) and
        # across runs only on the intersection of identities present in every
        # arm and every run; we never invent or impute a missing query.
        indices = {}
        common = None
        for arm in ARMS:
            for run in range(RUNS):
                cell = cells[(arm, db, k, run)]
                index = {str(value): position
                         for position, value in enumerate(cell["query_hash"])}
                if len(index) != len(cell["query_hash"]):
                    raise ValueError("D4 query identities are not unique")
                indices[(arm, run)] = index
                common = set(index) if common is None else (common & set(index))
        if not common:
            raise ValueError("D4 query sets share no common identities")
        hashes = sorted(common)
        ids = []
        for value in hashes:
            ids.append(str(cells[(ARMS[0], db, k, 0)]["query_id"][
                indices[(ARMS[0], 0)][value]]))
        means = {}
        for arm in ARMS:
            ordered = []
            for run in range(RUNS):
                index = indices[(arm, run)]
                positions = np.asarray([index[value] for value in hashes], dtype=np.int64)
                ordered.append(_cell_log_q(cells[(arm, db, k, run)])[positions])
            means[arm] = np.mean(np.vstack(ordered), axis=0)
        result[str(k)] = {
            "query_count": len(hashes),
            "query_set_basis": "intersection of identities present in all arms and all runs",
            "query_set_verified_across_runs": False,
            "C_minus_A": delta_summary(means[ARMS[0]], means[ARMS[2]], hashes, ids),
            "B_minus_A": delta_summary(means[ARMS[0]], means[ARMS[1]], hashes, ids),
        }
    return result


def _four_plus_q95(cell):
    mask = np.asarray(cell["n_tables"]) >= 4
    log_q = _cell_log_q(cell)[mask]
    if not log_q.size:
        raise ValueError("Empty four-plus cell")
    return float(np.exp(np.percentile(log_q, 95)))


def _d5(cells, db):
    per_k_arrays = {}
    result = {}
    for k in SHOTS:
        arrays = {arm: [_four_plus_q95(cells[(arm, db, k, run)])
                        for run in range(RUNS)] for arm in ARMS}
        per_k_arrays[str(k)] = arrays
        result[str(k)] = {
            "five_run_mean_ratio": {
                "C_over_A": float(np.mean(arrays[ARMS[2]]) / np.mean(arrays[ARMS[0]])),
                "B_over_A": float(np.mean(arrays[ARMS[1]]) / np.mean(arrays[ARMS[0]])),
            },
            "absolute_delta": {
                "C_minus_A_mean": float(np.mean(np.asarray(arrays[ARMS[2]])
                                                - np.asarray(arrays[ARMS[0]]))),
                "B_minus_A_mean": float(np.mean(np.asarray(arrays[ARMS[1]])
                                                - np.asarray(arrays[ARMS[0]]))),
            },
            "per_run": [{
                "run": run,
                **{arm: arrays[arm][run] for arm in ARMS},
                "C_over_A": four_tuple(arrays[ARMS[0]][run], arrays[ARMS[2]][run]),
                "B_over_A": four_tuple(arrays[ARMS[0]][run], arrays[ARMS[1]][run]),
            } for run in range(RUNS)],
            "main_mean_four_plus_q95": {arm: float(np.mean(arrays[arm]))
                                        for arm in ARMS},
        }
    result["leave_one_out"] = leave_one_out_stability(per_k_arrays)
    return result


def _d6(cells, db):
    result = {}
    for k in SHOTS:
        result[str(k)] = {}
        for arm in ARMS:
            all_log_q = _log_qs(cells, arm, db, k)
            four = np.concatenate([
                _cell_log_q(cells[(arm, db, k, run)])[
                    np.asarray(cells[(arm, db, k, run)]["n_tables"]) >= 4]
                for run in range(RUNS)
            ])
            result[str(k)][arm] = {
                "all": top_log_contribution(all_log_q),
                "four_plus": top_log_contribution(four),
            }
    return result


def _cross_database(d5):
    result = {}
    for db in SOURCE_DATABASES:
        a = [d5[db][str(k)]["main_mean_four_plus_q95"][ARMS[0]] for k in SHOTS]
        b = [d5[db][str(k)]["main_mean_four_plus_q95"][ARMS[1]] for k in SHOTS]
        c = [d5[db][str(k)]["main_mean_four_plus_q95"][ARMS[2]] for k in SHOTS]
        result[db] = {
            "A_four_plus_q95_mean_K": float(np.mean(a)),
            "B_four_plus_q95_mean_K": float(np.mean(b)),
            "C_four_plus_q95_mean_K": float(np.mean(c)),
            "C_over_A": four_tuple(float(np.mean(a)), float(np.mean(c))),
            "B_over_A": four_tuple(float(np.mean(a)), float(np.mean(b))),
        }
    return result


def diagnose_cells(cells, pg_by_cell=None, min_n=MIN_BUCKET_N, top_n=20):
    d1, d2, d3, d4, d5, d6 = {}, {}, {}, {}, {}, {}
    for db in SOURCE_DATABASES:
        d1[db], d2[db] = {}, {}
        for k in SHOTS:
            d1[db][str(k)], d2[db][str(k)] = {}, {}
            for arm in ARMS:
                d1[db][str(k)][arm] = table_bucket_rows(
                    _values(cells, arm, db, k, "n_tables"),
                    _values(cells, arm, db, k, "log_pred"),
                    _values(cells, arm, db, k, "true"), min_n)
                d2[db][str(k)][arm] = true_bucket_rows(
                    _values(cells, arm, db, k, "true"),
                    _values(cells, arm, db, k, "log_pred"), min_n)
        d3[db] = {"A": _d3_mode(cells, db, "A")}
        # The PG anchor is only defined for the primary database; never request a
        # PG direction layer for databases without a verified anchor.
        if pg_by_cell is not None and db == PRIMARY_DATABASE:
            d3[db]["PG"] = _d3_mode(cells, db, "PG", pg_by_cell)
        d4[db] = _d4(cells, db)
        d5[db] = _d5(cells, db)
        d6[db] = _d6(cells, db)
    return {
        "D1_table_count": d1,
        "D2_true_cardinality": d2,
        "D3_direction": d3,
        "D4_paired_query": d4,
        "D5_support_stability": d5,
        "D6_tail_concentration": d6,
        "cross_database_four_tuple": _cross_database(d5),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(result):
    rows = []
    for db, per_k in result["D1_table_count"].items():
        for k, arms in per_k.items():
            for arm, buckets in arms.items():
                for bucket, values in buckets.items():
                    rows.append({"section": "D1", "database": db, "k": k, "arm": arm,
                                 "bucket": bucket, "status": values["status"],
                                 "count": values.get("count", values.get("n")),
                                 "share": values.get("share"),
                                 "q50": values.get("q50"), "q95": values.get("q95"),
                                 "log_q95": values.get("log_q95")})
    for db, per_k in result["D2_true_cardinality"].items():
        for k, arms in per_k.items():
            for arm, buckets in arms.items():
                for bucket, values in buckets.items():
                    rows.append({"section": "D2", "database": db, "k": k, "arm": arm,
                                 "bucket": bucket, "status": values["status"],
                                 "count": values.get("count", values.get("n")),
                                 "share": values.get("share"),
                                 "q50": values.get("q50"), "q95": values.get("q95"),
                                 "log_q95": values.get("log_q95")})
    for db, modes in result["D3_direction"].items():
        for mode, per_k in modes.items():
            for k, groups in per_k.items():
                for direction, group in groups.items():
                    for arm, metrics in group["arms"].items():
                        rows.append({"section": "D3", "database": db, "k": k, "arm": arm,
                                     "bucket": f"{mode}:{direction}", "status": "ok",
                                     "count": metrics["count"], "share": group["share"],
                                     "q50": metrics["q50"], "q95": metrics["q95"],
                                     "log_q95": metrics["log_q95"]})
    for db, per_k in result["D4_paired_query"].items():
        for k, comparisons in per_k.items():
            for comparison in ("C_minus_A", "B_minus_A"):
                values = comparisons[comparison]
                rows.append({"section": "D4", "database": db, "k": k, "arm": comparison,
                             "bucket": "paired_query_delta", "status": "ok",
                             "count": values["count"], "share": values["mean"],
                             "q50": values["p50"], "q95": values["p90"]})
    for db, per_k in result["D5_support_stability"].items():
        for k, values in per_k.items():
            if not isinstance(values, dict) or "main_mean_four_plus_q95" not in values:
                continue
            for arm, q95 in values["main_mean_four_plus_q95"].items():
                rows.append({"section": "D5", "database": db, "k": k, "arm": arm,
                             "bucket": "four_plus", "status": "ok", "q95": q95,
                             "share": values["five_run_mean_ratio"].get("C_over_A")})
    for db, per_k in result["D6_tail_concentration"].items():
        for k, arms in per_k.items():
            for arm, cohorts in arms.items():
                for cohort, values in cohorts.items():
                    rows.append({"section": "D6", "database": db, "k": k, "arm": arm,
                                 "bucket": cohort, "status": "ok",
                                 "count": values["selected_count"],
                                 "share": values["contribution_ratio"]})
    return rows


def _recheck_inputs(hashes):
    for name, expected in hashes.items():
        if sha256(Path(name)) != expected:
            raise ValueError(f"Input changed during diagnosis: {name}")


def execute(args, protocol_path=None):
    out, csv_dir = prepare_outputs(args)
    protocol = no_symlinks(protocol_path or Path(__file__).resolve().parents[1]
                           / PROTOCOL_RELATIVE)
    if not protocol.is_file():
        raise ValueError("Diagnosis protocol is missing")
    protocol_hash = sha256(protocol)
    cells, input_hashes, plan_paths = collect_cells(args.old_run, args.completion_run)
    for path in plan_paths:
        input_hashes[str(path)] = sha256(path)
    pg_status = "not_attempted"
    pg_details = {}
    pg_by_cell = {}
    if getattr(args, "cache", None):
        # The PG anchor is only defined for the primary analysis database; other
        # databases have no frozen PG column in this retention and must stay
        # 'not_attempted' rather than being reported as failures.
        cache_dir = no_symlinks(args.cache)
        for name in ("cards.npy", "manifest.json", "offsets.npy"):
            path = no_symlinks(cache_dir / name)
            input_hashes[str(path)] = sha256(path)
        statuses = []
        for db in SOURCE_DATABASES:
            if db != PRIMARY_DATABASE:
                pg_details[db] = {"status": "not_attempted",
                                  "details": {"reason": "PG anchor is only defined for the primary database"}}
                pg_details[db]["pg"] = None
                continue
            last_details = None
            status = "verified"
            for k in SHOTS:
                for run in range(RUNS):
                    sample = cells[(ARMS[0], db, k, run)]
                    status, details, pg = anchor_pg(
                        cache_dir, sample["query_id"], sample["query_hash"],
                        sample["true"], sample["n_tables"])
                    if status != "verified":
                        last_details = details
                        break
                    if pg is not None:
                        pg_by_cell[(ARMS[0], db, k, run)] = pg
                    last_details = details
                if status != "verified":
                    break
            statuses.append(status)
            pg_details[db] = {"status": status, "details": last_details}
            if status == "verified":
                pg_details[db]["cells_verified"] = sum(
                    1 for key in pg_by_cell if key[0] == ARMS[0] and key[1] == db)
                pg_details[db]["pg"] = "per_cell_verified"
            else:
                pg_details[db]["pg"] = None
        pg_status = "verified" if statuses and all(value == "verified" for value in statuses) else "failed"
        if pg_status != "verified":
            pg_by_cell = {}
    diagnosis = diagnose_cells(cells, pg_by_cell if pg_status == "verified" else None)
    result = {
        "schema": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "descriptive_diagnosis_only",
        "primary_analysis_database": PRIMARY_DATABASE,
        "protocol_sha256": protocol_hash,
        "input_sha256": input_hashes,
        "pg_anchor_status": pg_status,
        "pg_anchor": pg_details,
        "explicit_confounds": [
            "A->B changes scope (all->head) and inner steps (5->3) together.",
            "B and C are independently meta-trained with different outer-loop endpoints.",
        ],
        "limitations": [
            "Single seed and repeatedly inspected source validation libraries.",
            "Descriptive only; no causal attribution, model selection or threshold change.",
            "Absolute and relative differences must be read together across databases.",
        ],
    }
    result.update(diagnosis)
    result["databases"] = {database: {"D1": diagnosis["D1_table_count"][database]}
                           for database in SOURCE_DATABASES}
    _recheck_inputs(input_hashes)
    if sha256(protocol) != protocol_hash:
        raise ValueError("Protocol changed during diagnosis")
    write_json(out, result)
    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=False)
        fields = ("section", "database", "k", "arm", "bucket", "status", "count",
                  "share", "q50", "q95", "log_q95")
        write_csv(csv_dir / "telstra_diagnosis_layers.csv", csv_rows(result), fields)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", required=True)
    parser.add_argument("--completion-run", required=True)
    parser.add_argument("--cache", help="Source cache directory for the primary database (telstra)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv-dir")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if not args.execute:
        out, csv_dir = prepare_outputs(args)
        print(json.dumps({"execute": False, "out": str(out),
                          "csv_dir": str(csv_dir) if csv_dir else None}, indent=2))
        return args
    execute(args)
    return args


def main(argv=None):
    return parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
