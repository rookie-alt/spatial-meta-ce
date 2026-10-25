import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

AVAILABLE = importlib.util.find_spec("spatial_meta_ce.train") is not None
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

    import measure_estimation_latency as mel
    from spatial_meta_ce.checkpoints import save_checkpoint
    from spatial_meta_ce.features import write_cache
    from spatial_meta_ce.model import SetEstimator

MODEL_CONFIG = {"width": 8, "use_pg": True}
DATABASE = "t"
MULTI = "select count(*) from a join b on a.id = b.id"
MULTI_ALT = "select count(*) from a join b on a.id = b.id and 1=1"
MULTI_THREE = "select count(*) from a join b on a.id = b.id join c on c.id = b.id"
SINGLE = "select count(*) from a"
PACKED = (np.ones((2, 40)), np.ones((1, 40)), np.ones((1, 4)), None)


def digest(sql):
    return hashlib.sha256(sql.encode()).hexdigest()


class FakeEngine:
    """Stand-in for PRICE's Sql2Feature: returns PACKED features for known SQL."""

    def __init__(self, mapping):
        self.mapping = mapping

    def create_sql_features(self, sql):
        if sql not in self.mapping:
            return None
        return [self.mapping[sql], 2, 1, 1]


def build_cache(cache_root, sqls):
    records = [
        {
            "id": f"{DATABASE}/test/{position}",
            "query_hash": f"h{position}",
            "exact_hash": digest(sql),
            "true": float(20 + position),
            "pg": 2.0,
        }
        for position, sql in enumerate(sqls)
    ]
    data = {
        "data_features": [PACKED for _ in records],
        "true_cards": [torch.tensor(record["true"]) for record in records],
        "pg_est_cards": [torch.tensor(2.0) for _ in records],
        "n_join_cols": [2] * len(records),
        "n_fanouts": [1] * len(records),
        "n_tables": [1] * len(records),
        "n_filter_cols": [0] * len(records),
    }
    write_cache(
        Path(cache_root) / "test" / DATABASE, records, data, {"source": "fixture"}
    )


def build_workload(data_root, lines):
    path = Path(data_root) / "datas/workloads/test" / DATABASE / "workloads_all.sql"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def make_checkpoint(path, metadata=None):
    model = SetEstimator(**MODEL_CONFIG)
    save_checkpoint(
        path,
        model,
        None,
        0,
        0.0,
        0,
        {
            "method": "unit-fixture",
            "train_databases": ["a"],
            "validation_databases": ["v"],
            **(metadata or {}),
        },
    )
    return path


def argv(cache_root, data_root, checkpoint, output, **overrides):
    values = {
        "cache_root": str(cache_root),
        "data_root": str(data_root),
        "checkpoint": str(checkpoint),
        "databases": [DATABASE],
        "allowed_targets": [DATABASE],
        "threads": 1,
        "verification_sample": 2,
        "output": str(output),
    }
    values.update(overrides)
    args = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, list):
            args += [flag] + [str(item) for item in value]
        else:
            args += [flag, str(value)]
    return args


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_parses_and_orders_subqueries(self):
        with tempfile.TemporaryDirectory() as d:
            path = build_workload(
                d, [f"{MULTI}||10||@0-0@", f"{SINGLE}||5||@0-1@", f"{MULTI}||9||@1-0@"]
            )
            rows = mel.parse_workload_all(path)
            groups = mel.group_queries(rows)
        self.assertEqual([len(group) for group in groups], [2, 1])
        self.assertEqual([row["subquery"] for row in groups[0]], [0, 1])
        self.assertEqual(groups[1][0]["query"], 1)
        self.assertEqual(rows[0]["true"], 10.0)

    def test_rejects_wrong_field_count(self):
        with tempfile.TemporaryDirectory() as d:
            path = build_workload(d, ["select 1||10||2||@0-0@"])
            with self.assertRaises(ValueError):
                mel.parse_workload_all(path)

    def test_rejects_missing_tag(self):
        with tempfile.TemporaryDirectory() as d:
            path = build_workload(d, ["select 1||10||nope"])
            with self.assertRaises(ValueError):
                mel.parse_workload_all(path)

    def test_table_count_separates_single_from_multi(self):
        self.assertEqual(mel.table_count(SINGLE), 1)
        self.assertEqual(mel.table_count(MULTI), 2)


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")
        self.cached = [torch.as_tensor(part) for part in PACKED[:3]] + [
            torch.zeros(0, 43)
        ]

    def test_identical_features_pass(self):
        mel.compare(self.cached, PACKED, "ok")

    def test_changed_value_is_rejected(self):
        changed = (np.full((2, 40), 2.0),) + PACKED[1:]
        with self.assertRaises(ValueError):
            mel.compare(self.cached, changed, "changed")

    def test_changed_shape_is_rejected(self):
        changed = (np.ones((3, 40)),) + PACKED[1:]
        with self.assertRaises(ValueError):
            mel.compare(self.cached, changed, "reshaped")


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")
        torch.set_num_threads(1)
        self.original = mel.extractor
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def tearDown(self):
        mel.extractor = self.original

    def prepare(self, lines, mapping, cache_sqls):
        build_cache(self.root, cache_sqls)
        build_workload(self.root, lines)
        mel.extractor = lambda *args, **kwargs: FakeEngine(mapping)

    def run_cli(self, **overrides):
        checkpoint = make_checkpoint(self.root / "check.pt")
        output = self.root / "estimation.json"
        code = mel.main(argv(self.root, self.root, checkpoint, output, **overrides))
        return code, json.loads(output.read_text())

    def test_aggregates_per_query_and_counts_single_table(self):
        self.prepare(
            lines=[
                f"{MULTI}||10||@0-0@",
                f"{SINGLE}||5||@0-1@",
                f"{MULTI}||11||@0-2@",
                f"{MULTI}||9||@1-0@",
            ],
            mapping={MULTI: PACKED},
            cache_sqls=[MULTI, MULTI_ALT],
        )
        code, report = self.run_cli()
        database = report["databases"][DATABASE]
        self.assertEqual(code, 0)
        self.assertFalse(report["includes_database_execution"])
        self.assertTrue(report["includes_feature_extraction"])
        self.assertEqual(database["queries_measured"], 2)
        self.assertEqual(database["population"]["single_table_true_card"], 1)
        self.assertEqual(database["population"]["not_in_cache"], 1)
        self.assertEqual(database["failures"], [])
        self.assertEqual(database["per_query_rows"][0]["subqueries"], 3)
        self.assertEqual(database["verification"]["checked"], 2)
        self.assertGreater(database["extraction_ms"]["mean"], 0.0)
        self.assertGreater(database["ours_total_ms"]["mean"], 0.0)
        self.assertGreaterEqual(database["extraction_share_of_ours_total"], 0.0)
        self.assertLessEqual(database["extraction_share_of_ours_total"], 1.0)

    def test_multi_table_without_pg_estimate_is_a_failure(self):
        self.prepare(
            lines=[f"{MULTI}||10||@0-0@", f"{MULTI_THREE}||11||@0-1@"],
            mapping={MULTI: PACKED},
            cache_sqls=[MULTI],
        )
        _, report = self.run_cli()
        failures = report["databases"][DATABASE]["failures"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["n_tables"], 3)
        self.assertIn("PostgreSQL estimate", failures[0]["reason"])

    def test_extractor_returning_none_is_a_failure(self):
        self.prepare(lines=[f"{MULTI}||10||@0-0@"], mapping={}, cache_sqls=[MULTI])
        _, report = self.run_cli()
        failures = report["databases"][DATABASE]["failures"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["reason"], "extractor returned None")

    def test_feature_drift_aborts_the_run(self):
        drifted = (np.full((2, 40), 3.0),) + PACKED[1:]
        self.prepare(
            lines=[f"{MULTI}||10||@0-0@"], mapping={MULTI: drifted}, cache_sqls=[MULTI]
        )
        checkpoint = make_checkpoint(self.root / "check.pt")
        with self.assertRaises(ValueError):
            mel.main(
                argv(self.root, self.root, checkpoint, self.root / "estimation.json")
            )

    def test_limit_queries_truncates(self):
        self.prepare(
            lines=[f"{MULTI}||10||@0-0@", f"{MULTI}||11||@1-0@"],
            mapping={MULTI: PACKED},
            cache_sqls=[MULTI, MULTI_ALT],
        )
        _, report = self.run_cli(limit_queries=1)
        self.assertEqual(report["databases"][DATABASE]["queries_measured"], 1)

    def test_unapproved_database_is_refused(self):
        self.prepare(
            lines=[f"{MULTI}||10||@0-0@"], mapping={MULTI: PACKED}, cache_sqls=[MULTI]
        )
        with self.assertRaises(SystemExit):
            self.run_cli(allowed_targets=["u"])

    def test_exposed_checkpoint_is_refused(self):
        self.prepare(
            lines=[f"{MULTI}||10||@0-0@"], mapping={MULTI: PACKED}, cache_sqls=[MULTI]
        )
        checkpoint = make_checkpoint(
            self.root / "exposed.pt", {"train_databases": [DATABASE]}
        )
        with self.assertRaises(SystemExit):
            mel.main(
                argv(self.root, self.root, checkpoint, self.root / "estimation.json")
            )


if __name__ == "__main__":
    unittest.main()
