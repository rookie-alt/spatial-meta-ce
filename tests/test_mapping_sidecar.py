import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_input_audit import ROOT, audit_fixture

from scripts.audit_input_mapping import query_layout, source_databases
from scripts.build_mapping_sidecar import (
    build_database,
    build_record,
    build_records,
    load_final_audit,
    main,
    shuffle_mapping,
    summarize_shuffle,
    validate_marginals,
    validate_output_path,
    validate_row,
)
from spatial_meta_ce.data import FeatureStore
from spatial_meta_ce.manifests import digest, file_hash


def row(sql: str, query_hash: str = "hash-x", line: int = 1) -> dict:
    return {
        "id": f"a/pretrain/{line}",
        "query_hash": query_hash,
        "exact_hash": f"exact-{line}",
        "source_line": line,
        "true": 10.0,
        "pg": 2.0,
        "sql": sql,
    }


class MappingSidecarTests(unittest.TestCase):
    def test_record_preserves_query_identity_and_fanout_direction(self) -> None:
        record = build_record(
            row(
                "select count(*) from beta b, alpha a "
                "where a.id=b.aid and b.v>2 and a.x<9",
                line=7,
            )
        )
        self.assertEqual(record["query_hash"], "hash-x")
        self.assertEqual(record["source_line"], 7)
        mapping = record["mapping"]
        self.assertEqual(mapping["tables"], ["b", "a"])
        self.assertEqual(mapping["join_columns"], ["b.aid", "a.id"])
        self.assertEqual(mapping["join_column_tables"], [0, 1])
        self.assertEqual(mapping["filter_column_tables"], [0, 1])
        self.assertEqual(mapping["fanout_join_columns"], [1, 0])
        self.assertEqual(mapping["counts"], [2, 2, 2, 2])

    def test_shuffle_is_deterministic_immutable_and_preserves_marginals(self) -> None:
        mapping = build_record(
            row(
                "select count(*) from a,b,c,d where a.id=b.aid and b.cid=c.id "
                "and c.did=d.id and a.x>1 and b.y>2 and c.z>3"
            )
        )["mapping"]
        original = copy.deepcopy(mapping)
        first = shuffle_mapping(mapping, seed=42)
        self.assertEqual(first, shuffle_mapping(mapping, seed=42))
        self.assertEqual(mapping, original)
        self.assertEqual(first["counts"], mapping["counts"])
        for key in ("join_column_tables", "filter_column_tables"):
            self.assertCountEqual(first[key], mapping[key])
        for direction in (0, 1):
            self.assertCountEqual(
                first["fanout_join_columns"][direction::2],
                mapping["fanout_join_columns"][direction::2],
            )
        self.assertEqual(first["tables"], mapping["tables"])
        self.assertEqual(first["join_columns"], mapping["join_columns"])
        self.assertNotEqual(first, mapping)

    def test_duplicate_hash_deduplicates_but_preserves_all_source_lines(self) -> None:
        sql = "select count(*) from a,b where a.id=b.id"
        records = build_records([row(sql), row(sql.upper(), line=2)])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_lines"], [1, 2])
        summary = summarize_shuffle(records)
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(summary["unique_queries"], 1)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["changed_rows"] + summary["unchanged_rows"], 2)

    def test_duplicate_hash_with_different_mapping_or_labels_is_rejected(self) -> None:
        first = row("select count(*) from a,b where a.id=b.id")
        for second in (
            row("select count(*) from a,b where a.id=b.other_id", line=2),
            {**first, "source_line": 2, "true": 11.0},
            {**first, "source_line": 2, "pg": 3.0},
        ):
            with (
                self.subTest(second=second),
                self.assertRaisesRegex(ValueError, "query_hash"),
            ):
                build_records([first, second])

    def test_unsupported_layout_is_not_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "mapping"):
            build_records([row("select count(*) from a join b on a.id=b.id")])

    def test_risk_flags_remain_visible(self) -> None:
        record = build_record(
            row("select count(*) from a,b where a.id=b.id and a.x>1 and a.x<9")
        )
        self.assertIn("multiple_filters", record["mapping"]["flags"])

    def test_constant_marginals_are_reported_as_unchangeable(self) -> None:
        mapping = {
            "join_column_tables": [0, 0],
            "filter_column_tables": [],
            "fanout_join_columns": [0, 1],
            "counts": [2, 2, 1, 0],
        }
        record = {
            "mapping": mapping,
            "shuffled_mapping": shuffle_mapping(mapping, 42),
            "source_lines": [1, 2],
        }
        summary = summarize_shuffle([record])
        self.assertEqual(summary["unchangeable_rows"], 2)
        self.assertEqual(summary["changed_rows"], 0)
        self.assertEqual(summary["random_unchanged_rows"], 0)

    def test_output_rejects_original_cache_and_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"data_root": str(root / "price"), "cache_root": str(root / "cache")}
            for name in ("price", "cache", "existing"):
                (root / name).mkdir()
            for output in (
                root / "price/sidecar",
                root / "cache/sidecar",
                root / "price",
            ):
                with self.subTest(output=output), self.assertRaises(ValueError):
                    validate_output_path(output, data)
            with self.assertRaises(FileExistsError):
                validate_output_path(root / "existing", data)
            self.assertEqual(
                validate_output_path(root / "sidecar", data),
                (root / "sidecar").resolve(),
            )

    def test_direction_swaps_and_join_filter_mixing_are_rejected(self) -> None:
        mapping = build_record(
            row(
                "select count(*) from a,b,c where a.id=b.id and a.id=c.id "
                "and a.x>1 and b.x>2"
            )
        )["mapping"]
        swapped = {
            **mapping,
            "fanout_join_columns": list(reversed(mapping["fanout_join_columns"])),
        }
        with self.assertRaisesRegex(ValueError, "marginal"):
            validate_marginals(mapping, swapped)
        mixed = {**mapping, "filter_column_tables": mapping["join_column_tables"]}
        with self.assertRaisesRegex(ValueError, "marginal"):
            validate_marginals(mapping, mixed)

    def test_source_only_and_symlink_boundaries(self) -> None:
        data = {"splits": {"train": ["a"], "validation": ["v"], "test": ["stats"]}}
        for selected in (["stats"], ["missing"], ["a", "a"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                source_databases(data, selected)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            (root / "link").symlink_to(root / "source", target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output_path(
                    root / "link/new",
                    {
                        "data_root": str(root / "source"),
                        "cache_root": str(root / "cache"),
                    },
                )


def fixture(root: Path) -> tuple[dict, dict, list[dict], Path]:
    data, _, rows = audit_fixture(root)
    source = Path(data["data_root"])
    report = {
        "database": "a",
        "alignment_passed": True,
        "inputs_unchanged": True,
        "provenance_errors": [],
        "recreate_all_features": True,
        "workload_rows": 2,
        "source_feature_rows": 2,
        "cache_rows": 2,
        "totals": {
            "checked_rows": 2,
            "recreated_rows": 2,
            "mapping_recoverable": 2,
            "sql_mapping_recoverable": 2,
            "rows_with_errors": 0,
        },
        "sql_flags": {},
        "input_hashes": {
            "workload": file_hash(source / "datas/workloads/pretrain/a/workloads.sql"),
            "source_features": file_hash(
                source / "setup/features/pretrain/a/features40.pkl"
            ),
            "extractor": file_hash(ROOT / "vendor/price_features/features_tool.py"),
        },
    }
    audit = root / "final-audit"
    audit.mkdir()
    summary = {
        "expected_databases": ["a", "v"],
        "reported_databases": ["a", "v"],
        "missing_databases": [],
        "duplicate_databases": [],
        "failed_databases": [],
        "checked_rows": 4,
        "groups_completed": 1,
        "reports": [report, {**report, "database": "v"}],
    }
    (audit / "summary.json").write_text(json.dumps(summary))
    (audit / "status.txt").write_text("completed time=2026-10-03T02:00:00+08:00")
    (audit / "group-0").mkdir()
    protocol = {
        "databases": ["a", "v"],
        "recreate_all_features": True,
        "script_sha256": file_hash(ROOT / "scripts/audit_input_mapping.py"),
        "sqlglot": "18.17.0",
    }
    (audit / "group-0/protocol.json").write_text(json.dumps(protocol))
    raw_reports = []
    for database in ("a", "v"):
        raw = {
            **report,
            "database": database,
            "by_tables": {},
            "by_predicate": {},
            "by_flag": {},
            "mapping_examples": [],
        }
        raw_reports.append(raw)
        (audit / f"group-0/{database}.json").write_text(json.dumps(raw))
    stripped = [
        {
            k: v
            for k, v in raw.items()
            if k not in ("mapping_examples", "by_tables", "by_predicate", "by_flag")
        }
        for raw in raw_reports
    ]
    group_summary = {
        **protocol,
        "databases_completed": 2,
        "checked_rows": 4,
        "mapping_recoverable": 4,
        "sql_mapping_recoverable": 4,
        "failed_databases": [],
        "reports": stripped,
    }
    (audit / "group-0/summary.json").write_text(json.dumps(group_summary))
    return data, report, rows, audit


class MappingSidecarIntegrationTests(unittest.TestCase):
    def test_roundtrip_verifies_all_rows_and_keeps_input_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, report, _, audit = fixture(root)
            reports = load_final_audit(audit, data)
            self.assertEqual(set(reports), {"a", "v"})
            inputs = {
                p: file_hash(p)
                for name in ("source", "cache")
                for p in (root / name).rglob("*")
                if p.is_file()
            }
            result = build_database(data, "a", report, root / "sidecar", seed=42)
            records = [
                json.loads(line)
                for line in (root / "sidecar/mapping.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["source_lines"], [1, 2])
            self.assertEqual(records[0]["mapping"]["fanout_join_columns"], [0, 1])
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["unique_queries"], 1)
            self.assertTrue(result["roundtrip_passed"])
            manifest = json.loads((root / "sidecar/manifest.json").read_text())
            self.assertEqual(manifest["audit_report_sha256"], digest(report))
            self.assertEqual(
                manifest["jsonl_sha256"], file_hash(root / "sidecar/mapping.jsonl")
            )
            self.assertEqual(inputs, {p: file_hash(p) for p in inputs})

    def test_every_identity_label_and_count_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, _, rows, _ = fixture(Path(directory))
            store = FeatureStore(Path(data["cache_root"]) / "pretrain/a")
            layout = query_layout(rows[0]["sql"])
            for key in ("id", "query_hash", "exact_hash", "source_line"):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_row({**rows[0], key: None}, store, 0, layout)
            for key in ("true", "pg"):
                for value in (None, float("nan"), -1.0, 99.0):
                    with (
                        self.subTest(key=key, value=value),
                        self.assertRaises(ValueError),
                    ):
                        validate_row({**rows[0], key: value}, store, 0, layout)
            with self.assertRaises(ValueError):
                validate_row(rows[0], store, 0, {**layout, "counts": [1, 2, 2, 0]})

    def test_original_report_failure_cannot_be_hidden_by_top_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, _, _, audit = fixture(Path(directory))
            original = json.loads((audit / "group-0/a.json").read_text())
            (audit / "group-0/a.json").write_text(
                json.dumps({**original, "alignment_passed": False})
            )
            with self.assertRaises(ValueError):
                load_final_audit(audit, data)

    def test_trusted_summary_hash_rejects_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, _, _, audit = fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "hash"):
                load_final_audit(audit, data, expected_sha256="0" * 64)

    def test_cli_requires_external_audit_summary_hash(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["build_mapping_sidecar.py", "--audit", "audit", "--output", "out"],
            ),
            self.assertRaises(SystemExit),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            main()

    def test_incomplete_final_audit_or_changed_input_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, report, _, audit = fixture(root)
            final = json.loads((audit / "summary.json").read_text())
            for change in (
                {"checked_rows": 3},
                {"failed_databases": ["a"]},
                {"missing_databases": ["a"]},
                {"reported_databases": ["a"]},
            ):
                (audit / "summary.json").write_text(json.dumps({**final, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    load_final_audit(audit, data)
            bad = {
                **report,
                "input_hashes": {**report["input_hashes"], "workload": "bad"},
            }
            with self.assertRaises(ValueError):
                build_database(data, "a", bad, root / "bad-sidecar")
            self.assertFalse((root / "bad-sidecar/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
