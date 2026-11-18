#!/usr/bin/env python3
"""Certify isolated official mapping inputs on CPU without model predictions.

Legacy source sidecars are copied after current source/cache bindings and all
workload rows pass verification. New sidecars require full frozen-extractor
reconstruction, mandatory for explicitly included test databases. This certifies
frozen feature encoding, not SQL semantics or selectivity truth.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sqlglot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_input_mapping import feature_groups_match, query_layout, write_report  # noqa: E402
from scripts.build_mapping_sidecar import (  # noqa: E402
    build_record,
    merge_record,
    persist_records,
    summarize_shuffle,
    validate_row,
)
from spatial_meta_ce.data import FeatureStore  # noqa: E402
from spatial_meta_ce.features import COUNT_KEYS, FIELDS, extractor, lower_except_quotes  # noqa: E402
from spatial_meta_ce.manifests import canonical_sql, digest, file_hash  # noqa: E402
from spatial_meta_ce.mappings import MappingStore  # noqa: E402
from spatial_meta_ce.official_mappings import (  # noqa: E402
    CACHE_ARRAYS,
    OFFICIAL_MAPPING_SCOPE,
    OfficialMappingStore,
    builder_compatibility,
    plain_path,
    validate_official_pair,
)
from spatial_meta_ce.official_split import TEST_DATABASES, validate_official_splits  # noqa: E402
from spatial_meta_ce.price_capacity import OFFICIAL_TRAIN_DATABASES  # noqa: E402
from vendor.price_features.features_tool import Sql2Feature  # noqa: E402

STATISTICS = (
    "abbrev_col_type.pkl",
    "fanout40.pkl",
    "histogram40.pkl",
    "size.pkl",
    "summary40.pkl",
)
INTERPRETATION = (
    "Query-local token structure and frozen extractor/source/cache encoding only; "
    "no SQL selectivity/semantic truth certification, model prediction, training, "
    "adaptation, checkpoint selection or target-label optimization."
)


class InputPreparationError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        failed = ", ".join(report["failed_databases"])
        super().__init__(f"Official mapping input certification failed: {failed}")


def _configuration(data_config) -> tuple[dict, dict]:
    binding = {}
    if isinstance(data_config, (str, Path)):
        path = plain_path(data_config)
        binding = {"data_config_path": str(path), "data_config_sha256": file_hash(path)}
        data_config = json.loads(path.read_text())
    if not isinstance(data_config, dict):
        raise ValueError("Official preparation requires a data configuration object")
    validate_official_splits(data_config.get("splits", {}))
    binding["data_config_hash"] = digest(data_config)
    data = copy.deepcopy(data_config)
    for name in ("data_root", "cache_root"):
        path = Path(data[name])
        data[name] = str(plain_path(path if path.is_absolute() else ROOT / path))
    binding["resolved_data_config_hash"] = digest(data)
    return data, binding


def _output_path(output, data, source_mapping_root) -> Path:
    path = plain_path(output)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite official mapping output: {path}")
    roots = [Path(data["data_root"]), Path(data["cache_root"])]
    if source_mapping_root is not None:
        roots.append(source_mapping_root)
    for root in roots:
        if path.is_relative_to(root) or root.is_relative_to(path):
            raise ValueError("Official mapping output overlaps a read-only input root")
    return path


def _input_paths(data, split, database) -> dict[str, Path]:
    validate_official_pair(split, database)
    source, cache = Path(data["data_root"]), Path(data["cache_root"])
    folder = plain_path(cache / split / database, cache)
    stats = plain_path(source / "datas/statistics" / split / database, source)
    paths = {
        "workload": source / "datas/workloads" / split / database / "workloads.sql",
        "source_features": source
        / "setup/features"
        / split
        / database
        / "features40.pkl",
        "extractor": ROOT / "vendor/price_features/features_tool.py",
        **{f"statistics/{name}": stats / name for name in STATISTICS},
        **{f"statistics/{path.name}": path for path in sorted(stats.glob("*.pkl"))},
        **{f"cache/{name}": folder / name for name in ("manifest.json", *CACHE_ARRAYS)},
        **{f"cache/{path.name}": path for path in sorted(folder.glob("*.npy"))},
    }
    if (folder / "struct_manifest.json").exists():
        paths["cache/struct_manifest.json"] = folder / "struct_manifest.json"
    for name, path in paths.items():
        base = (
            folder
            if name.startswith("cache/")
            else (ROOT if name == "extractor" else source)
        )
        paths[name] = plain_path(path, base)
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    return paths


def _source_provenance(store, hashes, split, database) -> None:
    provenance = store.manifest.get("provenance", {})
    source = provenance.get("source", {})
    if provenance.get("excluded"):
        raise ValueError("Official preparation rejects cache exclusions")
    if source.get("database") != database or source.get("split") != split:
        raise ValueError("Source cache provenance database/split mismatch")
    for key, source_key in (
        ("workload", "workload_sha256"),
        ("source_features", "feature_sha256"),
        ("extractor", "extractor_sha256"),
    ):
        if hashes[key] != source.get(source_key):
            raise ValueError(f"Current cache/source provenance hash mismatch: {key}")
    statistics = source.get("statistics")
    actual = {
        key.removeprefix("statistics/"): value
        for key, value in hashes.items()
        if key.startswith("statistics/")
    }
    if statistics != actual or source.get("sqlglot") != "18.17.0":
        raise ValueError("Current statistics/parser provenance mismatch")


def _legacy_records(path, store, hashes, database, compatibility=None):
    legacy = MappingStore(
        path,
        expected_database=database,
        expected_split="pretrain",
        expected_cache_hash=store.manifest["content_hash"],
    )
    legacy.bind_cache(store)
    manifest = legacy.manifest
    if (
        manifest.get("sqlglot") != "18.17.0"
        or type(manifest.get("shuffle_seed")) is not int
        or manifest["shuffle_seed"] != 42
        or manifest.get("builder_sha256")
        != (
            compatibility["historical_sha256"]
            if compatibility is not None
            else file_hash(ROOT / "scripts/build_mapping_sidecar.py")
        )
    ):
        raise ValueError(
            "Legacy source sidecar parser/shuffle/builder provenance mismatch"
        )
    declared = manifest.get("input_sha256", {})
    required = set(hashes) - {"cache/struct_log.npy", "cache/struct_manifest.json"}
    if not required.issubset(declared) or any(
        hashes.get(key) != value for key, value in declared.items()
    ):
        raise ValueError("Legacy source sidecar current input provenance mismatch")
    with (path / "mapping.jsonl").open() as stream:
        records = {record["query_hash"]: record for record in map(json.loads, stream)}
    return records, {
        "manifest_path": str(path / "manifest.json"),
        "manifest_sha256": file_hash(path / "manifest.json"),
        "jsonl_path": str(path / "mapping.jsonl"),
        "jsonl_sha256": file_hash(path / "mapping.jsonl"),
        "builder_sha256": manifest["builder_sha256"],
        **(
            {"builder_compatibility": compatibility}
            if compatibility is not None
            else {}
        ),
    }


def _workload_rows(path, database, split):
    """Retain a position for invalid rows so every failure stays in the denominator."""
    rows, failures = [], {}
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                fields = line.rstrip().split("||")
                if len(fields) < 3:
                    raise ValueError("Malformed workload row")
                sql, true, pg = fields[0].strip(), float(fields[1]), float(fields[2])
                if any(not math.isfinite(value) or value < 0 for value in (true, pg)):
                    raise ValueError("Invalid workload cardinality")
                rows.append(
                    {
                        "id": f"{database}/{split}/{line_number}",
                        "source_line": line_number,
                        "sql": sql,
                        "true": true,
                        "pg": pg,
                        "query_hash": hashlib.sha256(
                            canonical_sql(sql).encode()
                        ).hexdigest(),
                        "exact_hash": hashlib.sha256(sql.encode()).hexdigest(),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - audit all unsupported rows, then refuse the database.
                failures[len(rows)] = f"{type(exc).__name__}: {exc}"
                rows.append(None)
    if not rows:
        raise ValueError("Empty workload")
    return rows, failures


def _verify_token_order(sql, layout):
    columns, tables, joins, aliases = Sql2Feature.parse_sql(
        None, lower_except_quotes(sql)
    )
    endpoints = [column.strip() for join in joins for column in join.split("=")]
    join_set = set(endpoints)
    join_columns = [
        column
        for table in tables
        for column in columns
        if column.split(".")[0] == table and column in join_set
    ]
    filter_columns = [
        column
        for table in tables
        for column in columns
        if column.split(".")[0] == table and column not in join_set
    ]
    if (
        layout["tables"] != tables
        or layout["table_names"] != [aliases[table] for table in tables]
        or layout["join_columns"] != join_columns
        or layout["filter_columns"] != filter_columns
        or layout["fanout_join_columns"]
        != [join_columns.index(column) for column in endpoints]
    ):
        raise ValueError("SQL mapping/extractor token order mismatch")


def _check_row(row, index, store, original, layout, engine):
    validate_row(row, store, index, layout)
    _verify_token_order(row["sql"], layout)
    counts = [int(original[key][index]) for key in COUNT_KEYS]
    if counts != layout["counts"]:
        raise ValueError("Source/cache token counts mismatch")
    source_cards = np.asarray(
        [original[key][index] for key in ("true_cards", "pg_est_cards")],
        dtype=np.float32,
    )
    if not np.array_equal(
        source_cards, np.asarray([row["true"], row["pg"]], dtype=np.float32)
    ):
        raise ValueError("Source/workload cardinality sequence mismatch")
    cached = [
        group[store.offsets[index, i] : store.offsets[index + 1, i]]
        for i, group in enumerate(store.groups)
    ]
    if not feature_groups_match(original["data_features"][index], cached):
        raise ValueError("Source/cache feature values or order mismatch")
    if engine is not None:
        with contextlib.redirect_stdout(io.StringIO()):
            recreated = engine.create_sql_features(lower_except_quotes(row["sql"]))
        if recreated is None:
            raise ValueError("Frozen extractor does not support this SQL")
        if list(recreated[1:]) != counts:
            raise ValueError("Recreated token counts mismatch")
        if not feature_groups_match(recreated[0], cached) or not feature_groups_match(
            recreated[0],
            original["data_features"][index],
        ):
            raise ValueError("Recreated/source/cache feature values or order mismatch")


def _collect(rows, parse_errors, store, original, engine, imported):
    records, totals, errors, flags, failures = {}, Counter(), Counter(), Counter(), []
    seen_cards = {}
    for index, row in enumerate(rows):
        totals["checked_rows"] += 1
        try:
            if row is None:
                raise ValueError(parse_errors[index])
            cards = (row["true"], row["pg"])
            if seen_cards.setdefault(row["query_hash"], cards) != cards:
                totals["dedup_conflict_count"] += 1
                raise ValueError("duplicate_query_cardinality_conflict")
            layout = query_layout(row["sql"])
            flags.update(layout["flags"])
            totals["sql_mapping_recoverable"] += int(layout["mapping_recoverable"])
            _check_row(row, index, store, original, layout, engine)
            record = build_record(row, seed=42, layout=layout)
            if imported is not None:
                previous = imported.get(row["query_hash"])
                if previous is None or any(
                    previous[key] != record[key]
                    for key in (
                        "mapping",
                        "shuffled_mapping",
                        "true",
                        "pg",
                    )
                ):
                    raise ValueError(
                        "Imported source mapping/workload layout or label conflict"
                    )
            try:
                records[record["query_hash"]] = merge_record(records, record)
            except ValueError:
                totals["dedup_conflict_count"] += 1
                raise
            totals["mapping_recoverable"] += 1
            totals["recreated_rows"] += int(engine is not None)
        except Exception as exc:  # noqa: BLE001 - report every row failure; never certify a filtered subset.
            totals["unsupported_count"] += 1
            message = f"{type(exc).__name__}: {exc}"
            errors[message] += 1
            if len(failures) < 20:
                failures.append(
                    {
                        "row_index": index,
                        "id": row["id"] if row else None,
                        "error": message,
                    }
                )
    return list(records.values()), {
        **{
            key: totals[key]
            for key in (
                "checked_rows",
                "sql_mapping_recoverable",
                "mapping_recoverable",
                "recreated_rows",
                "unsupported_count",
                "dedup_conflict_count",
            )
        },
        "unsupported_count_complete": True,
        "row_errors": dict(errors),
        "first_failures": failures,
        "sql_flags": dict(flags),
    }


def _database(
    data, split, database, paths, output, legacy_path, target_files, compatibility=None
):
    if split == "test":
        target_files.extend(
            str(path) for name, path in paths.items() if name != "extractor"
        )
    hashes = {key: file_hash(path) for key, path in paths.items()}
    store = FeatureStore(paths["cache/manifest.json"].parent)
    _source_provenance(store, hashes, split, database)
    if (
        not np.issubdtype(store.offsets.dtype, np.integer)
        or np.any(store.offsets[0] != 0)
        or np.any(np.diff(store.offsets, axis=0) < 0)
    ):
        raise ValueError("Invalid cache offsets")
    rows, parse_errors = _workload_rows(paths["workload"], database, split)
    with paths["source_features"].open("rb") as stream:
        original = pickle.load(stream)
    if any(
        key not in original or len(original[key]) != len(rows) for key in FIELDS
    ) or len(store) != len(rows):
        raise ValueError("Workload/source features/cache row coverage mismatch")
    imported, import_binding = None, None
    if legacy_path is not None:
        imported, import_binding = _legacy_records(
            legacy_path, store, hashes, database, compatibility
        )
    engine = extractor(data["data_root"], database, split) if imported is None else None
    records, checks = _collect(rows, parse_errors, store, original, engine, imported)
    valid_rows = [row for row in rows if row is not None]
    unique = len({row["query_hash"] for row in valid_rows})
    result = {
        "database": database,
        "split": split,
        "rows": len(rows),
        "unique_queries": unique,
        "duplicate_rows": len(valid_rows) - unique,
        "parse_failed_rows": len(parse_errors),
        "input_sha256": hashes,
        "array_sha256": {name: hashes[f"cache/{name}"] for name in CACHE_ARRAYS},
        "workload_sha256": hashes["workload"],
        "source_feature_sha256": hashes["source_features"],
        "cache_manifest_sha256": hashes["cache/manifest.json"],
        "cache_content_hash": store.manifest["content_hash"],
        "source_sidecar": import_binding,
        **checks,
    }
    if any(file_hash(path) != hashes[key] for key, path in paths.items()):
        raise ValueError("Inputs changed during official input certification")
    if import_binding is not None and (
        file_hash(legacy_path / "manifest.json") != import_binding["manifest_sha256"]
        or file_hash(legacy_path / "mapping.jsonl") != import_binding["jsonl_sha256"]
    ):
        raise ValueError("Legacy source sidecar changed during certification")
    result["inputs_unchanged"] = True
    if checks["unsupported_count"]:
        return {**result, "state": "failed"}
    if imported is not None:
        if imported != {record["query_hash"]: record for record in records}:
            raise ValueError(
                "Imported sidecar full identity/source line coverage mismatch"
            )
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(legacy_path / "mapping.jsonl", output / "mapping.jsonl")
    else:
        persist_records(output, records, rows)
    manifest = {
        "schema": 1,
        "scope": OFFICIAL_MAPPING_SCOPE,
        "database": database,
        "split": split,
        "shuffle_seed": 42,
        "sqlglot": sqlglot.__version__,
        "cache_content_hash": store.manifest["content_hash"],
        "input_sha256": hashes,
        "input_paths": {key: str(path) for key, path in paths.items()},
        "array_sha256": result["array_sha256"],
        "source_sidecar": import_binding,
        "builder_sha256": file_hash(Path(__file__)),
        "layout_sha256": file_hash(ROOT / "scripts/audit_input_mapping.py"),
        "record_builder_sha256": file_hash(ROOT / "scripts/build_mapping_sidecar.py"),
        "reader_sha256": file_hash(ROOT / "spatial_meta_ce/official_mappings.py"),
        "jsonl_sha256": file_hash(output / "mapping.jsonl"),
        "certification": checks,
        "interpretation": INTERPRETATION,
    }
    write_report(output / "manifest.json", manifest)
    mapping = OfficialMappingStore(
        output,
        expected_database=database,
        expected_split=split,
        expected_cache_hash=store.manifest["content_hash"],
    )
    mapping.bind_cache(store)
    return {
        **result,
        "state": "completed",
        "shuffle": summarize_shuffle(records),
        "manifest_sha256": file_hash(output / "manifest.json"),
        "jsonl_sha256": manifest["jsonl_sha256"],
    }


def prepare_inputs(
    data_config,
    output,
    source_mapping_root=None,
    include_test=False,
    legacy_builder_source=None,
):
    """Build fresh <output>/<split>/<db> mappings and a bound root report.

    ``include_test`` enables structural preprocessing only. False never reads
    test inputs. A supplied source root uses legacy <root>/<db> sidecars. Without
    it all 26 sources also undergo full frozen-extractor reconstruction.
    """
    data, config_binding = _configuration(data_config)
    if type(include_test) is not bool:
        raise ValueError("include_test must be an explicit boolean")
    if sqlglot.__version__ != "18.17.0":
        raise ValueError("Official mapping preparation requires sqlglot==18.17.0")
    source_root = (
        plain_path(source_mapping_root) if source_mapping_root is not None else None
    )
    if legacy_builder_source is not None and source_root is None:
        raise ValueError(
            "Historical builder proof requires a legacy source mapping root"
        )
    compatibility = (
        builder_compatibility(legacy_builder_source)
        if legacy_builder_source is not None
        else None
    )
    output = _output_path(output, data, source_root)
    pairs = [("pretrain", db) for db in OFFICIAL_TRAIN_DATABASES]
    if include_test:
        pairs += [("test", db) for db in TEST_DATABASES]
    # Preflight all required files and paths before creating output parents.
    inventory = {(split, db): _input_paths(data, split, db) for split, db in pairs}
    legacy = {}
    if source_root is not None:
        for db in OFFICIAL_TRAIN_DATABASES:
            path = plain_path(source_root / db, source_root)
            for name in ("manifest.json", "mapping.jsonl"):
                file = plain_path(path / name, source_root)
                if not file.is_file():
                    raise FileNotFoundError(file)
            legacy[db] = path
    code_paths = (
        Path(__file__),
        ROOT / "spatial_meta_ce/official_mappings.py",
        ROOT / "scripts/audit_input_mapping.py",
        ROOT / "scripts/build_mapping_sidecar.py",
        ROOT / "spatial_meta_ce/data.py",
        ROOT / "spatial_meta_ce/features.py",
        ROOT / "spatial_meta_ce/manifests.py",
        ROOT / "spatial_meta_ce/mappings.py",
        ROOT / "spatial_meta_ce/official_split.py",
        ROOT / "spatial_meta_ce/price_capacity.py",
        ROOT / "vendor/price_features/features_tool.py",
    )
    code_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in code_paths}
    report = {
        "schema": 1,
        "scope": OFFICIAL_MAPPING_SCOPE,
        "state": "running",
        **config_binding,
        "mapping_root": str(output),
        "cache_root": data["cache_root"],
        "data_root": data["data_root"],
        "splits": data["splits"],
        "include_test": include_test,
        "source_mapping_root": str(source_root) if source_root is not None else None,
        "databases": {"pretrain": {}, "test": {}},
        "failed_databases": [],
        "unsupported_count": 0,
        "dedup_conflict_count": 0,
        "unsupported_count_complete": True,
        "target_data_read": [],
        "target_files_read": [],
        "model_predictions_run": False,
        "training_run": False,
        "code_sha256": code_hashes,
        "interpretation": INTERPRETATION,
    }
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "official_module_inputs.json"
    write_report(report_path, report)
    for split, db in pairs:
        if split == "test":
            report["target_data_read"].append(db)
        try:
            result = _database(
                data,
                split,
                db,
                inventory[(split, db)],
                output / split / db,
                legacy.get(db) if split == "pretrain" else None,
                report["target_files_read"],
                compatibility if split == "pretrain" else None,
            )
        except Exception as exc:  # noqa: BLE001 - retain each database failure and continue the fixed population.
            result = {
                "database": db,
                "split": split,
                "state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "unsupported_count": 0,
                "dedup_conflict_count": 0,
                "unsupported_count_complete": False,
            }
        report["databases"][split][db] = result
        if result["state"] != "completed":
            report["failed_databases"].append(f"{split}/{db}")
        report["unsupported_count"] += result["unsupported_count"]
        report["dedup_conflict_count"] += result["dedup_conflict_count"]
        report["unsupported_count_complete"] &= result["unsupported_count_complete"]
        write_report(report_path, report)
    if any(file_hash(ROOT / name) != value for name, value in code_hashes.items()):
        report["failed_databases"].append("code_changed_during_preparation")
    report["target_files_read"] = list(dict.fromkeys(report["target_files_read"]))
    report["state"] = "failed" if report["failed_databases"] else "completed"
    write_report(report_path, report)
    if report["failed_databases"]:
        raise InputPreparationError(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--data-config", dest="config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-mapping-root")
    parser.add_argument("--legacy-builder-source")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Explicit CPU-only structural preparation of all four test databases",
    )
    args = parser.parse_args()
    report = prepare_inputs(
        args.config,
        args.output,
        args.source_mapping_root,
        args.include_test,
        args.legacy_builder_source,
    )
    print(
        json.dumps(
            {
                "state": report["state"],
                "mapping_root": report["mapping_root"],
                "target_data_read": report["target_data_read"],
                "model_predictions_run": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
