import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATABASES = ("chembl", "legalacts", "telstra")
MIN_TABLES = 4
TOLERANCE = 1e-6
FOCUS = {"chembl": (6, 100000, float("inf")), "telstra": (5, 10, 100)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_digest(path: Path) -> str:
    value = json.loads(Path(path).read_text())
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def histogram_interval(hist, edges, low, high, table_size):
    cumulative = np.r_[0.0, np.cumsum(hist, dtype=np.float64)]
    return float(
        (np.interp(high, edges, cumulative) - np.interp(low, edges, cumulative))
        / table_size
    )


def table_aggregates(selectivities):
    if not selectivities:
        return np.ones(3)
    order = sorted(selectivities, reverse=True)[:4]
    return np.array(
        (
            np.prod(selectivities),
            min(selectivities),
            np.prod([value ** (1 / 2**i) for i, value in enumerate(order)]),
        ),
        dtype=np.float64,
    )


def repeated_filters(where):
    from collections import defaultdict

    from sqlglot import exp

    by_column = defaultdict(list)
    for predicate in where.find_all(exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE):
        if isinstance(predicate.this, exp.Column) and not isinstance(
            predicate.expression, exp.Column
        ):
            by_column[str(predicate.this)].append(
                (type(predicate).__name__, str(predicate.expression))
            )
    return {name: clauses for name, clauses in by_column.items() if len(clauses) > 1}


def self_check() -> None:
    histogram = np.array([20.0, 30.0])
    edges = np.array([0.0, 10.0, 20.0])
    assert np.isclose(histogram_interval(histogram, edges, 5.0, 15.0, 100.0), 0.25)
    assert np.isclose(histogram_interval(histogram, edges, 10.0, 10.0, 100.0), 0.0)
    assert np.isclose(histogram_interval(histogram, edges, -5.0, 30.0, 100.0), 0.5)
    assert np.isclose(histogram_interval(histogram, edges, 30.0, 40.0, 100.0), 0.0)
    np.testing.assert_allclose(
        table_aggregates([0.2, 0.4]), [0.08, 0.2, 0.4 * 0.2**0.5]
    )
    np.testing.assert_allclose(
        table_aggregates([0.5, 0.5]), [0.25, 0.5, 0.5 * 0.5**0.5]
    )
    np.testing.assert_allclose(table_aggregates([]), [1.0, 1.0, 1.0])
    print("self-check passed")


def audit_database(project: Path, source: Path, db: str, matrix: dict) -> dict:
    import pickle

    from sqlglot import exp, parse_one

    sidecar = project / "runs/mapping-sidecars-source26-20261004" / db
    cache = project / "artifacts/cache/pretrain" / db
    manifest_path = sidecar / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        row["mapping_provenance"][db]["manifest_hash"]
        for row in matrix["runs"]
        if row["arm"] in ("true", "shuffled")
    }
    if (
        len(expected) != 1
        or sha256(manifest_path) != next(iter(expected))
        or manifest["database"] != db
        or manifest["split"] != "pretrain"
        or sha256(sidecar / "mapping.jsonl") != manifest["jsonl_sha256"]
    ):
        raise ValueError(f"Frozen sidecar mismatch: {db}")
    audit_matches = sorted(
        (project / "runs/p1-input-audit-20261002").glob(f"group-*/{db}.json")
    )
    if (
        len(audit_matches) != 1
        or report_digest(audit_matches[0]) != manifest["audit_report_sha256"]
    ):
        raise ValueError(f"Frozen audit report mismatch: {db}")
    audit_report = json.loads(audit_matches[0].read_text())
    for name in (
        "manifest.json",
        "cards.npy",
        "offsets.npy",
        "group2.npy",
        "group3.npy",
    ):
        if sha256(cache / name) != manifest["input_sha256"][f"cache/{name}"]:
            raise ValueError(f"Frozen cache file mismatch: {db}/{name}")
    workload = source / "datas/workloads/pretrain" / db / "workloads.sql"
    stats = source / "datas/statistics/pretrain" / db
    sources = {
        "workload": workload,
        "statistics/histogram40.pkl": stats / "histogram40.pkl",
        "statistics/size.pkl": stats / "size.pkl",
    }
    for key, path in sources.items():
        if sha256(path) != manifest["input_sha256"][key]:
            raise ValueError(f"Frozen source input mismatch: {db}/{key}")
    with (stats / "histogram40.pkl").open("rb") as stream:
        histograms = pickle.load(stream)
    with (stats / "size.pkl").open("rb") as stream:
        sizes = pickle.load(stream)
    cache_manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["cache_content_hash"] != cache_manifest["content_hash"] or {
        row["manifest_hashes"][db] for row in matrix["runs"]
    } != {cache_manifest["content_hash"]}:
        raise ValueError(f"Frozen cache manifest mismatch: {db}")
    offsets = np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)
    filters = np.load(cache / "group3.npy", mmap_mode="r", allow_pickle=False)
    tables = np.load(cache / "group2.npy", mmap_mode="r", allow_pickle=False)
    cards = np.load(cache / "cards.npy", mmap_mode="r", allow_pickle=False)
    if offsets.shape != (len(cache_manifest["rows"]) + 1, 4):
        raise ValueError(f"Cache offsets shape mismatch: {db}")
    by_line = {}
    for i, row in enumerate(cache_manifest["rows"]):
        line_number = int(row["id"].rsplit("/", 1)[1])
        if line_number in by_line:
            raise ValueError(f"Duplicate cache source line: {db}/{line_number}")
        by_line[line_number] = i
    selected, primary = {}, 0
    all_row_flags, all_predicate_types, sidecar_rows = Counter(), Counter(), 0
    with (sidecar / "mapping.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            mapping = row["mapping"]
            sidecar_rows += 1
            all_row_flags.update(mapping["flags"])
            all_predicate_types.update(mapping["predicate_types"])
            if mapping["counts"][2] < MIN_TABLES:
                continue
            primary += 1
            if "multiple_filters" in mapping["flags"]:
                if row["source_line"] in selected:
                    raise ValueError(f"Duplicate selected source line: {db}")
                selected[row["source_line"]] = row
    if not selected:
        raise ValueError(f"No repeated filter rows: {db}")
    counts = {
        "or_queries": 0,
        "repeated_queries": 0,
        "range_columns": 0,
        "unsupported_pattern": 0,
        "empty_or_reversed": 0,
        "below_histogram_min": 0,
        "above_histogram_max": 0,
    }
    patterns = {}
    selectivity_max = 0.0
    aggregate_max = 0.0
    mismatches = 0
    focus_flagged = 0
    for line_no, line in enumerate(workload.open(), 1):
        row = selected.get(line_no)
        if row is None:
            continue
        sql = line.split("||", 1)[0].strip()
        if hashlib.sha256(sql.encode()).hexdigest() != row["exact_hash"]:
            raise ValueError(f"Source SQL identity mismatch: {db}/{line_no}")
        index = by_line[line_no]
        if (
            cache_manifest["rows"][index]["query_hash"] != row["query_hash"]
            or row["true"] != float(cards[index, 0])
            or row["pg"] != float(cards[index, 1])
            or row["mapping"]["counts"]
            != (offsets[index + 1] - offsets[index]).tolist()
        ):
            raise ValueError(f"Source/cache identity mismatch: {db}/{line_no}")
        tree = parse_one(sql, read="postgres")
        where = tree.args.get("where")
        if where is None:
            raise ValueError(f"Missing WHERE: {db}/{line_no}")
        counts["or_queries"] += int(bool(where.find(exp.Or)))
        repeated = repeated_filters(where)
        if not repeated:
            raise ValueError(f"Flag without repeated filter: {db}/{line_no}")
        counts["repeated_queries"] += 1
        if db in FOCUS:
            n_tables, lower, upper = FOCUS[db]
            focus_flagged += int(
                row["mapping"]["counts"][2] == n_tables and lower <= row["true"] < upper
            )
        for name, clauses in repeated.items():
            signature = "+".join(sorted(operator for operator, _ in clauses))
            patterns[signature] = patterns.get(signature, 0) + 1
            if sorted(operator for operator, _ in clauses) != ["GTE", "LT"]:
                counts["unsupported_pattern"] += 1
                continue
            counts["range_columns"] += 1
            low = float(next(v for op, v in clauses if op == "GTE"))
            high = float(next(v for op, v in clauses if op == "LT"))
            if low >= high:
                counts["empty_or_reversed"] += 1
            table_name, col_name = name.split(".", 1)
            histogram = histograms[table_name][col_name]
            edges = np.asarray(histogram["bin_edges"], dtype=np.float64)
            bins = np.asarray(histogram["hist"], dtype=np.float64)
            size = sizes[table_name]["size"]
            if len(edges) != len(bins) + 1 or size <= 0:
                raise ValueError(f"Invalid frozen histogram: {db}/{line_no}")
            counts["below_histogram_min"] += int(low < edges[0])
            counts["above_histogram_max"] += int(high > edges[-1])
            expected = histogram_interval(bins, edges, low, high, size)
            column_index = row["mapping"]["filter_columns"].index(name)
            observed = float(filters[offsets[index, 3] + column_index, -1])
            difference = abs(expected - observed)
            selectivity_max = max(selectivity_max, difference)
            mismatches += int(difference > TOLERANCE)
        per_table = [[] for _ in range(row["mapping"]["counts"][2])]
        start, stop = offsets[index, 3], offsets[index + 1, 3]
        mapping = row["mapping"]["filter_column_tables"]
        if len(filters[start:stop]) != len(mapping):
            raise ValueError(f"Filter token mismatch: {db}/{line_no}")
        for table_index, value in zip(mapping, filters[start:stop, -1]):
            per_table[table_index].append(float(value))
        observed_tables = tables[offsets[index, 2] : offsets[index + 1, 2], 1:4]
        for table_sels, observed in zip(per_table, observed_tables):
            aggregate_max = max(
                aggregate_max,
                float(np.max(np.abs(table_aggregates(table_sels) - observed))),
            )
    if counts["repeated_queries"] != len(selected):
        raise ValueError(f"Flagged source SQL coverage mismatch: {db}")
    if (
        mismatches
        or counts["unsupported_pattern"]
        or counts["or_queries"]
        or counts["empty_or_reversed"]
    ):
        raise ValueError(f"Repeated filter audit failed: {db}, mismatches={mismatches}")
    if aggregate_max > TOLERANCE:
        raise ValueError(f"Table aggregation mismatch: {db}")
    if (
        audit_report.get("alignment_passed") is not True
        or audit_report["workload_rows"] != audit_report["cache_rows"]
        or audit_report["unique_queries"] != sidecar_rows
        or set(audit_report["sql_flags"]) != set(all_row_flags)
    ):
        raise ValueError(f"Audit/sidecar full-coverage mismatch: {db}")
    return {
        "primary_n": primary,
        "flagged_unique_n": len(selected),
        "focus_flagged_n": focus_flagged if db in FOCUS else None,
        **counts,
        "patterns": patterns,
        "selectivity_max_abs_diff": selectivity_max,
        "selectivity_mismatch_gt_tolerance": mismatches,
        "table_aggregate_max_abs_diff": aggregate_max,
        "sidecar_manifest_sha256": sha256(manifest_path),
        "sidecar_jsonl_sha256": manifest["jsonl_sha256"],
        "workload_sha256": sha256(workload),
        "audit_report_sha256_verified": True,
        "all_unique_rows": sidecar_rows,
        "all_row_flags": dict(all_row_flags),
        "all_row_predicate_types": dict(all_predicate_types),
    }


def analyze(project: Path, source: Path) -> dict:
    matrix_path = project / "runs/p1-mapping-ablation-20261004/ablation_runs.json"
    matrix = json.loads(matrix_path.read_text())
    if len(matrix["runs"]) != 9 or {(r["arm"], r["seed"]) for r in matrix["runs"]} != {
        (arm, seed) for arm in ("sum", "true", "shuffled") for seed in (42, 43, 44)
    }:
        raise ValueError("Frozen run matrix changed")
    return {
        "purpose": "source-only repeated-range boolean and frozen histogram feature consistency audit",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "training": False,
        "inference": False,
        "target_data_read": False,
        "databases": list(DATABASES),
        "minimum_tables": MIN_TABLES,
        "tolerance_abs": TOLERANCE,
        "input_sha256": {"ablation_runs.json": sha256(matrix_path)},
        "limitations": "Only the flagged 4+ source unique queries are re-parsed. Histogram selectivity is not observed database selectivity; cached table aggregation is checked on flagged rows only. No causality or endpoint change.",
        "per_database": {
            db: audit_database(project, source, db, matrix) for db in DATABASES
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if args.project is None or args.source is None or args.output is None:
            parser.error("--project, --source and --output required")
        expected_output = (
            args.project / "runs/p1-mapping-ablation-20261004/diagnosis-20261004"
        ).resolve()
        if args.output.parent.resolve() != expected_output or args.output.is_symlink():
            parser.error(
                "Output must be a new file in the frozen source diagnosis directory"
            )
        report = analyze(args.project, args.source)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
        print(f"wrote {args.output}")
