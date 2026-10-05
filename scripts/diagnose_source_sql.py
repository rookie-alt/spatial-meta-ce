import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATABASES = ("chembl", "legalacts", "telstra")
ARMS = ("sum", "true", "shuffled")
SEEDS = (42, 43, 44)
MIN_CELL_N = 100
FOCUS = {"chembl": (6, 100000, float("inf")), "telstra": (5, 10, 100)}
PG_BINS = ((-float("inf"), -1), (-1, 0), (0, 1), (1, float("inf")))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def degree_signature(mapping: dict) -> str:
    tables = mapping["counts"][2]
    columns = mapping["join_column_tables"]
    directed = mapping["fanout_join_columns"]
    if len(directed) % 2 or len(directed) != 2 * (tables - 1):
        raise ValueError("Expected a join tree with complete endpoint pairs")
    neighbors = [set() for _ in range(tables)]
    for a, b in zip(directed[::2], directed[1::2]):
        if a < 0 or b < 0 or a >= len(columns) or b >= len(columns):
            raise ValueError("Join endpoint outside join columns")
        left, right = columns[a], columns[b]
        if left == right or left < 0 or right < 0 or left >= tables or right >= tables:
            raise ValueError("Invalid join table endpoint")
        neighbors[left].add(right)
        neighbors[right].add(left)
    seen = {0}
    pending = [0]
    while pending:
        for neighbor in neighbors[pending.pop()] - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    if len(seen) != tables or sum(map(len, neighbors)) != 2 * (tables - 1):
        raise ValueError("Join table graph is not a tree")
    return ",".join(str(len(group)) for group in sorted(neighbors, key=len))


def predicate_class(types: list[str]) -> str:
    remaining = set(types) - {"EQ"}
    if not remaining:
        return "EQ-only"
    if remaining == {"LT"}:
        return "LT"
    if remaining == {"GTE"}:
        return "GTE"
    if remaining == {"LT", "GTE"}:
        return "LT+GTE"
    return "other"


def describe(
    mask: np.ndarray,
    truth: np.ndarray,
    pg_log: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict:
    n = int(mask.sum())
    if n < MIN_CELL_N:
        return {"n": n, "status": "below_min_cell_n"}
    log_true = np.log(np.maximum(truth[mask], 1))
    pg_error = pg_log[mask] - log_true
    a, b = (predictions[f"true-seed{seed}"][mask] for seed in (42, 43))
    shuffled = predictions["shuffled-seed43"][mask]

    def q95(pred: np.ndarray) -> float:
        return float(np.quantile(np.exp(np.abs(np.maximum(pred, 0) - log_true)), 0.95))

    return {
        "n": n,
        "pg_error_mean": float(pg_error.mean()),
        "pg_q95": q95(pg_log[mask]),
        "true42_q95": q95(a),
        "true43_q95": q95(b),
        "true43_head_mean": float(np.mean(b - pg_log[mask])),
        "true43_minus_42_mean": float(np.mean(b - a)),
        "true43_minus_shuffled43_mean": float(np.mean(b - shuffled)),
    }


def self_check() -> None:
    mapping = {
        "counts": [6, 6, 4, 2],
        "join_column_tables": [0, 1, 1, 2, 2, 3],
        "fanout_join_columns": [0, 1, 2, 3, 4, 5],
    }
    assert degree_signature(mapping) == "1,1,2,2"
    for types, expected in (
        (["EQ", "LT", "GTE"], "LT+GTE"),
        (["EQ"], "EQ-only"),
        (["EQ", "Between"], "other"),
    ):
        assert predicate_class(types) == expected
    pg = np.zeros(100)
    truth = np.ones(100)
    predictions = {
        "true-seed42": np.ones(100),
        "true-seed43": np.zeros(100),
        "shuffled-seed43": np.ones(100) * 2,
    }
    assert (
        describe(np.ones(100, dtype=bool), truth, pg, predictions)[
            "true43_minus_42_mean"
        ]
        == -1
    )
    assert describe(np.arange(100) < 99, truth, pg, predictions) == {
        "n": 99,
        "status": "below_min_cell_n",
    }
    for broken in (
        {**mapping, "fanout_join_columns": [0, 1, 2, 3]},
        {**mapping, "fanout_join_columns": [0, 1, 2, 3, 4, 99]},
    ):
        try:
            degree_signature(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid tree accepted")
    print("self-check passed")


def audit_sql(root: Path, database: str, selected: dict, source_hash: str) -> dict:
    sys.path.insert(0, str(root))
    from scripts.audit_input_mapping import query_layout
    from sqlglot import exp, parse_one

    workload = (
        root / "../PRICE-master/datas/workloads/pretrain" / database / "workloads.sql"
    )
    if sha256(workload) != source_hash:
        raise ValueError(f"Frozen source workload hash changed: {database}")
    checked = []
    with workload.open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = selected.get(line_number)
            if item is None:
                continue
            sql = line.split("||", 1)[0].strip()
            if hashlib.sha256(sql.encode()).hexdigest() != item["exact_hash"]:
                raise ValueError(
                    f"Source SQL identity mismatch: {database}:{line_number}"
                )
            parsed = query_layout(sql)
            independent = parse_one(sql, read="postgres")
            where = independent.args.get("where")
            predicates = (
                list(
                    where.find_all(
                        exp.EQ,
                        exp.NEQ,
                        exp.LT,
                        exp.LTE,
                        exp.GT,
                        exp.GTE,
                        exp.In,
                        exp.Between,
                        exp.Like,
                        exp.Is,
                    )
                )
                if where
                else []
            )
            joins = [
                p
                for p in predicates
                if isinstance(p, exp.EQ)
                and isinstance(p.this, exp.Column)
                and isinstance(p.expression, exp.Column)
            ]
            filters = [p for p in predicates if p not in joins]
            if len(joins) * 2 != item["mapping"]["counts"][1]:
                raise ValueError(
                    f"Independent SQL join count mismatch: {database}:{line_number}"
                )
            for key in (
                "counts",
                "flags",
                "predicate_types",
                "join_columns",
                "filter_columns",
                "join_column_tables",
                "fanout_join_columns",
            ):
                if parsed[key] != item["mapping"][key]:
                    raise ValueError(
                        f"SQL label mismatch: {database}:{line_number}/{key}"
                    )
            checked.append(
                {
                    "source_line": line_number,
                    "query_hash": item["query_hash"],
                    "degree_signature": degree_signature(parsed),
                    "predicate_class": predicate_class(parsed["predicate_types"]),
                    "multiple_filters": "multiple_filters" in parsed["flags"],
                    "independent_join_equalities": len(joins),
                    "independent_filter_predicates": len(filters),
                    "independent_filter_equalities": sum(
                        isinstance(p, exp.EQ) for p in filters
                    ),
                    "independent_range_predicates": sum(
                        isinstance(p, (exp.LT, exp.LTE, exp.GT, exp.GTE))
                        for p in filters
                    ),
                }
            )
    if len(checked) != len(selected):
        raise ValueError(f"SQL sentinel coverage mismatch: {database}")
    return {"workload_sha256": source_hash, "sentinels": checked}


def analyze(root: Path) -> dict:
    run_root = root / "runs/p1-mapping-ablation-20261004"
    diagnosis_root = run_root / "diagnosis-20261004"
    sidecar_root = root / "runs/mapping-sidecars-source26-20261004"
    matrix_path = run_root / "ablation_runs.json"
    prior_path = diagnosis_root / "pg_representation_verified.json"
    overlap_path = diagnosis_root / "tail_overlap.json"
    matrix = json.loads(matrix_path.read_text())
    prior = json.loads(prior_path.read_text())
    overlap = json.loads(overlap_path.read_text())
    if prior["input_sha256"]["ablation_runs.json"] != sha256(matrix_path) or prior[
        "input_sha256"
    ]["tail_overlap.json"] != sha256(overlap_path):
        raise ValueError("Frozen prior diagnosis changed")
    if len(matrix["runs"]) != 9 or {
        (record["arm"], record["seed"]) for record in matrix["runs"]
    } != {(arm, seed) for arm in ARMS for seed in SEEDS}:
        raise ValueError("Nine-run matrix changed")
    report = {
        "purpose": "source-only posthoc SQL structure and predicate label audit; not semantic certification",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "training": False,
        "inference": False,
        "target_data_read": False,
        "databases": list(DATABASES),
        "minimum_cell_n": MIN_CELL_N,
        "limitations": "Degree signature is not full topology; predicate_types is a set not counts; multiple_filters means repeated reference to one filtered column; no causal comparison or change to frozen endpoint.",
        "input_sha256": {
            "ablation_runs.json": sha256(matrix_path),
            "pg_representation_verified.json": sha256(prior_path),
            "tail_overlap.json": sha256(overlap_path),
        },
        "per_database": {},
    }
    for database in DATABASES:
        sidecar = sidecar_root / database
        sidecar_manifest_path = sidecar / "manifest.json"
        sidecar_manifest = json.loads(sidecar_manifest_path.read_text())
        mapped = [record for record in matrix["runs"] if record["arm"] != "sum"]
        expected = {
            record["mapping_provenance"][database]["manifest_hash"] for record in mapped
        }
        if (
            len(expected) != 1
            or sha256(sidecar_manifest_path) != next(iter(expected))
            or sidecar_manifest["database"] != database
            or sidecar_manifest["split"] != "pretrain"
            or sha256(sidecar / "mapping.jsonl") != sidecar_manifest["jsonl_sha256"]
        ):
            raise ValueError(f"Frozen sidecar hash changed: {database}")
        cache = root / "artifacts/cache/pretrain" / database
        if (
            sha256(cache / "manifest.json")
            != sidecar_manifest["input_sha256"]["cache/manifest.json"]
            or sha256(cache / "offsets.npy")
            != sidecar_manifest["input_sha256"]["cache/offsets.npy"]
            or sha256(cache / "cards.npy")
            != sidecar_manifest["input_sha256"]["cache/cards.npy"]
        ):
            raise ValueError(f"Frozen cache hash changed: {database}")
        cache_manifest = json.loads((cache / "manifest.json").read_text())
        if cache_manifest["content_hash"] != sidecar_manifest["cache_content_hash"] or {
            record["manifest_hashes"][database] for record in matrix["runs"]
        } != {cache_manifest["content_hash"]}:
            raise ValueError(f"Frozen cache manifest changed: {database}")
        seen = set()
        indices = []
        for index, row in enumerate(cache_manifest["rows"]):
            if row["query_hash"] not in seen:
                indices.append(index)
                seen.add(row["query_hash"])
        query_hash = np.array(
            [cache_manifest["rows"][i]["query_hash"] for i in indices]
        )
        cards = np.load(cache / "cards.npy", mmap_mode="r", allow_pickle=False)[indices]
        offsets = np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)
        counts = np.diff(offsets, axis=0)[indices]
        truth, pg_cards = cards.T
        pg_log = np.log(np.maximum(pg_cards, 1)).astype(np.float32).astype(np.float64)
        predictions = {}
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{arm}-seed{seed}"
                path = diagnosis_root / f"{name}-{database}.npz"
                if sha256(path) != overlap["npz_sha256"][path.name]:
                    raise ValueError(
                        f"Frozen prediction hash changed: {name}-{database}"
                    )
                with np.load(path, allow_pickle=False) as npz:
                    if (
                        not np.array_equal(query_hash, npz["query_hash"])
                        or not np.array_equal(truth, npz["true"])
                        or not np.array_equal(counts[:, 2], npz["n_tables"])
                    ):
                        raise ValueError(
                            f"Cache/prediction pairing failed: {name}-{database}"
                        )
                    predictions = {**predictions, name: npz["log_pred"]}
        primary = counts[:, 2] >= 4
        baseline = prior["per_database"][database]["baseline"]["4plus"]
        if int(primary.sum()) != baseline["n"] or not np.isclose(
            describe(primary, truth, pg_log, predictions)["pg_q95"],
            baseline["pg_q95"],
            rtol=1e-7,
        ):
            raise ValueError(f"Frozen PG endpoint mismatch: {database}")
        summary = {
            "primary_n": int(primary.sum()),
            "primary": describe(primary, truth, pg_log, predictions),
            "flag_counts": Counter(),
            "predicate_class_counts": Counter(),
            "degree_signature_counts": Counter(),
            "focus": {},
            "sentinels": {},
        }
        rows = {}
        with (sidecar / "mapping.jsonl").open() as stream:
            for line in stream:
                item = json.loads(line)
                key = item["query_hash"]
                if key in rows:
                    raise ValueError(f"Duplicate sidecar query_hash: {database}")
                rows[key] = item
        if len(rows) != len(query_hash) or set(rows) != set(query_hash):
            raise ValueError(
                f"Sidecar/cache unique query coverage mismatch: {database}"
            )
        flags = np.empty(len(query_hash), dtype=bool)
        predicates = np.empty(len(query_hash), dtype=object)
        signatures = np.empty(len(query_hash), dtype=object)
        for i, key in enumerate(query_hash):
            item = rows[key]
            if (
                item["mapping"]["counts"] != counts[i].tolist()
                or item["true"] != float(truth[i])
                or item["pg"] != float(pg_cards[i])
                or item["source_line"]
                != cache_manifest["rows"][indices[i]]["source_line"]
                or item["exact_hash"]
                != cache_manifest["rows"][indices[i]]["exact_hash"]
            ):
                raise ValueError(
                    f"Sidecar/cache identity or values mismatch: {database}:{i}"
                )
            mapping = item["mapping"]
            flags[i] = "multiple_filters" in mapping["flags"]
            predicates[i] = predicate_class(mapping["predicate_types"])
            signatures[i] = (
                degree_signature(mapping) if primary[i] else "out_of_primary"
            )
            if primary[i]:
                summary["flag_counts"].update(mapping["flags"])
                summary["predicate_class_counts"].update([predicates[i]])
                summary["degree_signature_counts"].update([signatures[i]])
        if sum(summary["predicate_class_counts"].values()) != int(primary.sum()) or sum(
            summary["degree_signature_counts"].values()
        ) != int(primary.sum()):
            raise ValueError(f"SQL labels do not partition primary bucket: {database}")
        if database in FOCUS:
            tables, lower, upper = FOCUS[database]
            focus = (counts[:, 2] == tables) & (truth >= lower) & (truth < upper)
            if not focus.any():
                raise ValueError(f"Predeclared focus group empty: {database}")
            cells = {}
            for dimension, values in (
                ("degree_signature", signatures),
                ("predicate_class", predicates),
                ("multiple_filters", flags),
                ("join_columns", counts[:, 0]),
                ("filter_columns", counts[:, 3]),
            ):
                for value in np.unique(values[focus]):
                    selected = focus & (values == value)
                    label = f"{dimension}={value}"
                    cells[label] = describe(selected, truth, pg_log, predictions)
                    if dimension not in ("join_columns", "filter_columns"):
                        for low, high in PG_BINS:
                            pg_error = pg_log - np.log(np.maximum(truth, 1))
                            conditioned = (
                                selected & (pg_error >= low) & (pg_error < high)
                            )
                            if conditioned.any():
                                cells[f"{label}|pg_error=[{low},{high})"] = describe(
                                    conditioned, truth, pg_log, predictions
                                )
            summary["focus"] = {
                "tables": tables,
                "true_range": [lower, "inf" if np.isinf(upper) else upper],
                "n": int(focus.sum()),
                "cells": cells,
            }
            sentinel_items = {}
            for dimension, values in (
                ("degree_signature", signatures),
                ("predicate_class", predicates),
                ("multiple_filters", flags),
            ):
                for value in np.unique(values[focus]):
                    matches = np.flatnonzero(focus & (values == value))
                    for index in np.unique(matches[[0, len(matches) // 2, -1]]):
                        item = rows[query_hash[index]]
                        sentinel_items[item["source_line"]] = item
            summary["sentinels"] = audit_sql(
                root,
                database,
                sentinel_items,
                sidecar_manifest["input_sha256"]["workload"],
            )
        report["input_sha256"] = {
            **report["input_sha256"],
            f"{database}/sidecar_manifest.json": sha256(sidecar_manifest_path),
            f"{database}/mapping.jsonl": sha256(sidecar / "mapping.jsonl"),
            f"{database}/workload.sql": sidecar_manifest["input_sha256"]["workload"],
        }
        report["per_database"] = {**report["per_database"], database: summary}
        print(
            f"verified {database}: {int(primary.sum())} primary, {len(summary['focus'].get('cells', {}))} focus cells",
            flush=True,
        )
    return {
        **report,
        "checks": {
            "sidecar_manifest_hashes": 3,
            "sidecar_jsonl_hashes": 3,
            "prediction_npz_hashes": 27,
            "sql_sentinel_count": sum(
                len(part["sentinels"].get("sentinels", []))
                for part in report["per_database"].values()
            ),
            "source_only": True,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    self_check()
    if not arguments.self_check:
        if arguments.root is None or arguments.output is None:
            parser.error("--root and --output required")
        expected_output_dir = (
            arguments.root / "runs/p1-mapping-ablation-20261004/diagnosis-20261004"
        ).resolve()
        if (
            arguments.output.parent.resolve() != expected_output_dir
            or arguments.output.is_symlink()
        ):
            parser.error(
                "Output must be a new file in the frozen source diagnosis directory"
            )
        result = analyze(arguments.root)
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        with arguments.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
        print(f"wrote {arguments.output}")
