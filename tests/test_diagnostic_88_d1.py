import unittest

import numpy as np

from scripts import diagnostic_88_d1 as d1


def cells(n=1000, seed=0, head3_shift=0.0):
    rng = np.random.default_rng(seed)
    true = rng.integers(1, 500, n).astype(np.float64)
    tables = rng.integers(1, 8, n).astype(np.int64)
    base = np.log(np.maximum(true, 1)) + rng.normal(0, 0.5, n)
    hashes = np.array([f"h{i}" for i in range(n)])
    ids = np.array([f"chembl/pretrain/{i + 1}" for i in range(n)])
    return {
        "none": {
            "log_pred": base,
            "true": true,
            "n_tables": tables,
            "query_hash": hashes,
            "query_id": ids,
        },
        "head3": {
            "log_pred": base + head3_shift,
            "true": true.copy(),
            "n_tables": tables.copy(),
            "query_hash": hashes.copy(),
            "query_id": ids.copy(),
        },
    }


class Diagnostic88D1Tests(unittest.TestCase):
    def test_low4_definition_and_bucket_denominator(self):
        data = cells()
        result = d1.run_analysis(data, data["none"]["true"])
        low4 = (data["none"]["n_tables"] >= 4) & (data["none"]["true"] < 100)
        self.assertEqual(result["low4_count"], int(low4.sum()))
        covered = sum(
            result["buckets"][name]["count"]
            for name in ("n_tables=4", "n_tables=5", "n_tables=6+")
        )
        self.assertEqual(covered, result["low4_count"])
        for prefix in ("true_tercile_", "pg_tercile_"):
            self.assertEqual(
                sum(result["buckets"][f"{prefix}{i}"]["count"] for i in (1, 2, 3)),
                result["low4_count"],
            )

    def test_concentration_matches_manual_calculation(self):
        data = cells(seed=4, head3_shift=0.2)
        result = d1.run_analysis(data, data["none"]["true"])
        low4 = (data["none"]["n_tables"] >= 4) & (data["none"]["true"] < 100)
        delta = d1.log_errors(data["head3"]) - d1.log_errors(data["none"])
        positive = delta[low4][delta[low4] > 0]
        top_count = max(1, int(np.ceil(0.01 * int(low4.sum()))))
        expected = np.sort(positive)[::-1][:top_count].sum() / positive.sum()
        self.assertAlmostEqual(
            result["concentration_low4"]["share"], expected, places=12
        )
        self.assertEqual(result["concentration_low4"]["top_count"], top_count)

    def test_concentrated_synthetic_reaches_threshold(self):
        n = 10000
        true = np.full(n, 50.0)
        tables = np.full(n, 5)
        base = np.full(n, np.log(50))
        head3 = base.copy()
        head3[:100] += 5.0
        hashes = np.arange(n).astype(str)
        data = {
            arm: {
                "log_pred": values,
                "true": true.copy(),
                "n_tables": tables.copy(),
                "query_hash": hashes.copy(),
                "query_id": hashes.copy(),
            }
            for arm, values in (("none", base), ("head3", head3))
        }
        result = d1.run_analysis(data, true)
        self.assertGreaterEqual(result["concentration_low4"]["share"], 0.99)
        verdict = d1.decide({"chembl": [dict(result, run=run) for run in range(5)]})
        self.assertEqual(verdict["verdict"], "concentrated")

    def test_balanced_diffuse_synthetic_reaches_diffuse(self):
        n = 10000
        true = np.full(n, 50.0)
        tables = np.full(n, 5)
        # none has a constant error equal to the median of |head3 noise|, so the
        # paired improvement share sits near 50% while deltas stay diffuse.
        base = np.full(n, np.log(50) + 0.674)
        rng = np.random.default_rng(7)
        head3 = np.log(50) + rng.normal(0, 1.0, n)
        hashes = np.arange(n).astype(str)
        data = {
            arm: {
                "log_pred": values,
                "true": true.copy(),
                "n_tables": tables.copy(),
                "query_hash": hashes.copy(),
                "query_id": hashes.copy(),
            }
            for arm, values in (("none", base), ("head3", head3))
        }
        result = d1.run_analysis(data, true)
        self.assertLess(result["concentration_low4"]["share"], d1.DIFFUSE_THRESHOLD)
        verdict = d1.decide({"chembl": [dict(result, run=run) for run in range(5)]})
        self.assertEqual(verdict["verdict"], "diffuse")

    def test_unbalanced_improvement_is_inconclusive(self):
        n = 10000
        true = np.full(n, 50.0)
        tables = np.full(n, 5)
        none = np.full(n, np.log(50) + 1.0)
        head3 = np.full(n, np.log(50))
        hashes = np.arange(n).astype(str)
        data = {
            arm: {
                "log_pred": values,
                "true": true.copy(),
                "n_tables": tables.copy(),
                "query_hash": hashes.copy(),
                "query_id": hashes.copy(),
            }
            for arm, values in (("none", none), ("head3", head3))
        }
        result = d1.run_analysis(data, true)
        self.assertEqual(result["paired_low4"]["better_share"], 1.0)
        verdict = d1.decide({"chembl": [dict(result, run=run) for run in range(5)]})
        self.assertEqual(verdict["verdict"], "inconclusive")

    def test_small_low4_database_is_ineligible(self):
        data = cells(seed=9)
        result = d1.run_analysis(data, data["none"]["true"])
        small = dict(result, low4_count=10)
        verdict = d1.decide({"chembl": [dict(small, run=run) for run in range(5)]})
        self.assertFalse(verdict["eligible"]["chembl"])
        self.assertEqual(verdict["verdict"], "inconclusive")

    def test_guard_rejects_forbidden_and_unexpected_databases(self):
        for name in d1.FORBIDDEN_DATABASES:
            with self.assertRaises(ValueError):
                d1.guard_database(name)
        with self.assertRaises(ValueError):
            d1.guard_database("accidents")

    def test_unpaired_identity_is_rejected(self):
        data = cells()
        data["head3"]["true"][0] += 1
        with self.assertRaises(ValueError):
            d1.run_analysis(data, data["none"]["true"])

    def test_nonfinite_prediction_is_rejected(self):
        data = cells()
        data["head3"]["log_pred"][0] = np.nan
        with self.assertRaises(ValueError):
            d1.run_analysis(data, data["none"]["true"])

    def test_tercile_ids_cover_every_row(self):
        values = np.arange(100, dtype=np.float64)
        ids = d1.tercile_ids(values)
        self.assertEqual(set(ids.tolist()), {0, 1, 2})
        self.assertTrue((np.diff(ids) >= 0).all())

    def test_empty_bucket_reports_null(self):
        report = d1.bucket_report(
            np.zeros(4, dtype=bool), np.zeros(4), np.zeros(4), np.ones(4), np.ones(4)
        )
        self.assertEqual(report["count"], 0)
        self.assertIsNone(report["better_share"])
        self.assertIsNone(report["q95_ratio"])


if __name__ == "__main__":
    unittest.main()
