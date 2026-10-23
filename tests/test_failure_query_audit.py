import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parents[1] / "scripts/failure_query_audit.py"
_SPEC = importlib.util.spec_from_file_location("failure_query_audit", _SOURCE)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def row(qid, true, pred, pg, n_tables, db="chembl"):
    return {
        "database": db,
        "query_hash": qid,
        "true": true,
        "log_pred": pred,
        "pg_log": pg,
        "n_tables": n_tables,
    }


class DeteriorationTests(unittest.TestCase):
    """A query 'worsened' when its adapted log error exceeds the PG anchor's."""

    def test_log_q_is_absolute_residual_in_log_space(self):
        # true=20 -> log(20); residual from log_pred is absolute in log space.
        self.assertAlmostEqual(audit.log_q(5.0, 20.0), abs(5.0 - np.log(20.0)))
        self.assertAlmostEqual(audit.log_q(1.0, 20.0), abs(1.0 - np.log(20.0)))

    def test_worsened_flag_is_strict(self):
        anchor = np.log(10.0)
        # pred far from truth while the anchor is exact -> worsened.
        self.assertTrue(audit.is_worsened(row("a", 10.0, anchor + 3.0, anchor, 3)))
        # pred closer to truth than the anchor -> not worsened.
        self.assertFalse(
            audit.is_worsened(row("b", 10.0, anchor + 0.5, anchor + 2.0, 3))
        )
        # pred equal to the anchor -> not worsened.
        self.assertFalse(audit.is_worsened(row("c", 10.0, anchor, anchor, 3)))

    def test_tie_tolerance_ignores_machine_roundoff(self):
        logged = np.log(10.0)
        self.assertFalse(audit.is_worsened(row("d", 10.0, logged + 1e-12, logged, 3)))

    def test_invalid_rows_rejected(self):
        for bad in (
            row("e", -1.0, 1.0, 1.0, 3),
            row("f", 10.0, float("nan"), 1.0, 3),
            row("g", 10.0, 1.0, 1.0, 0),
        ):
            with self.subTest(qid=bad["query_hash"]), self.assertRaises(ValueError):
                audit.is_worsened(bad)


class BucketTests(unittest.TestCase):
    def test_pg_direction_buckets(self):
        self.assertEqual(
            audit.pg_direction(row("a", 10.0, 1.0, np.log(100.0), 3)), "over"
        )
        self.assertEqual(
            audit.pg_direction(row("b", 100.0, 1.0, np.log(10.0), 3)), "under"
        )
        self.assertEqual(
            audit.pg_direction(row("c", 10.0, 1.0, np.log(10.0), 3)), "tie"
        )

    def test_table_bucket_boundaries(self):
        self.assertEqual(audit.table_bucket(1), "1")
        self.assertEqual(audit.table_bucket(2), "2-3")
        self.assertEqual(audit.table_bucket(3), "2-3")
        self.assertEqual(audit.table_bucket(4), "4-5")
        self.assertEqual(audit.table_bucket(6), "6+")


class ReportTests(unittest.TestCase):
    def setUp(self):
        # q1/q2 worsen vs their anchor; q3 is worse than anchor too but is kept
        # as a pg-over case; q4 improves because its anchor is far from truth.
        self.rows = [
            row("q1", 100.0, 5.0, np.log(100.0), 4),  # worsened, pg tie
            row("q2", 100.0, 9.0, np.log(10.0), 4),  # worsened, pg under
            row("q3", 100.0, 1.0, np.log(1000.0), 2),  # worsened, pg over
            row("q4", 100.0, np.log(100.0), np.log(1e6), 6),  # improved
        ]

    def test_summary_counts_and_fractions(self):
        report = audit.audit_rows(self.rows, "chembl")
        self.assertEqual(report["queries"], 4)
        self.assertEqual(report["worsened"], 3)
        self.assertAlmostEqual(report["worsened_fraction"], 0.75)

    def test_profile_by_pg_direction_and_tables(self):
        report = audit.audit_rows(self.rows, "chembl")
        self.assertEqual(report["worsened_by_pg_direction"]["tie"], 1)
        self.assertEqual(report["worsened_by_pg_direction"]["under"], 1)
        self.assertEqual(report["worsened_by_pg_direction"]["over"], 1)
        self.assertEqual(report["worsened_by_table_bucket"]["4-5"], 2)
        self.assertEqual(report["worsened_by_table_bucket"]["2-3"], 1)
        self.assertEqual(report["worsened_by_table_bucket"]["6+"], 0)

    def test_deterioration_is_concentrated_or_diffuse(self):
        stats = audit.concentration(self.rows)
        self.assertEqual(stats["worsened"], 3)
        self.assertGreater(stats["top1pct_share"], 0)
        self.assertLessEqual(stats["top1pct_share"], 1.0)
        self.assertIn(stats["shape"], ("concentrated", "diffuse", "inconclusive"))

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            audit.audit_rows([], "chembl")


class CliTests(unittest.TestCase):
    def test_write_is_atomic_and_json_safe(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.json"
            audit.write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertFalse(Path(str(path) + ".tmp").exists())
            with self.assertRaises(ValueError):
                audit.write_json(path, {"a": float("nan")})

    def test_target_databases_are_refused(self):
        for db in ("stats", "genome", "ergastf1", "imdb"):
            with self.subTest(db=db), self.assertRaises(ValueError):
                audit.assert_source_database(db)
        audit.assert_source_database("chembl")


class MultiRunTests(unittest.TestCase):
    def test_mean_across_runs_does_not_pool_identities(self):
        r1 = audit.audit_rows(
            [row("a", 100.0, np.log(100.0) + 3.0, np.log(100.0), 4)],
            "chembl",
        )
        r2 = audit.audit_rows(
            [row("a", 100.0, np.log(100.0) + 1.0, np.log(100.0), 4)],
            "chembl",
        )
        merged = audit._mean_across_runs({"run0": r1, "run1": r2})
        self.assertEqual(merged["runs"], 2)
        self.assertAlmostEqual(merged["worsened_fraction"], 1.0)

    def test_missing_n_tables_still_audits_but_flags_unavailable(self):
        rows = [
            {
                "query_hash": "x",
                "true": 100.0,
                "log_pred": np.log(100.0) + 2.0,
                "pg_log": np.log(100.0),
            },
            {
                "query_hash": "y",
                "true": 100.0,
                "log_pred": np.log(100.0),
                "pg_log": np.log(100.0),
            },
        ]
        report = audit.audit_rows(rows, "chembl")
        self.assertFalse(report["table_buckets_available"])
        self.assertNotIn("worsened_by_table_bucket", report)
        self.assertEqual(report["worsened"], 1)
        self.assertEqual(report["worsened_by_pg_direction"]["tie"], 1)


if __name__ == "__main__":
    unittest.main()
