import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location(
    "error_structure_diagnosis", _ROOT / "scripts/error_structure_diagnosis.py"
)
diagnosis = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(diagnosis)


class PredictionStatsTests(unittest.TestCase):
    def test_exact_prediction_has_zero_bias_and_unit_error(self):
        truth = np.log(np.array([10.0, 100.0, 1000.0]))
        stats = diagnosis.prediction_stats(truth.copy(), truth)
        self.assertAlmostEqual(stats["bias_med"], 0.0)
        self.assertAlmostEqual(stats["q50"], 1.0)
        self.assertAlmostEqual(stats["q95"], 1.0)
        self.assertAlmostEqual(stats["corr_truth"], 1.0)

    def test_constant_underestimate_is_flagged_as_low(self):
        # Predict a tenth of every true cardinality: log offset -log(10).
        truth = np.log(np.array([10.0, 100.0, 1000.0, 10000.0]))
        pred = truth - np.log(10.0)
        stats = diagnosis.prediction_stats(pred, truth)
        self.assertAlmostEqual(stats["bias_med"], -np.log(10.0), places=9)
        self.assertAlmostEqual(stats["over_pct"], 0.0)
        self.assertAlmostEqual(stats["q50"], 10.0)
        self.assertAlmostEqual(stats["log_q50"], np.log(10.0), places=9)

    def test_constant_overestimate_is_flagged_as_high(self):
        truth = np.log(np.array([10.0, 100.0, 1000.0]))
        pred = truth + np.log(2.0)
        stats = diagnosis.prediction_stats(pred, truth)
        self.assertAlmostEqual(stats["over_pct"], 100.0)
        self.assertAlmostEqual(stats["bias_med"], np.log(2.0), places=9)

    def test_centred_estimator_reports_about_fifty_percent(self):
        truth = np.zeros(100)
        pred = np.concatenate([np.full(50, 0.1), np.full(50, -0.1)])
        stats = diagnosis.prediction_stats(pred, truth)
        self.assertAlmostEqual(stats["over_pct"], 50.0)
        self.assertAlmostEqual(stats["bias_med"], 0.0)

    def test_constant_prediction_has_undefined_correlation(self):
        truth = np.log(np.array([10.0, 100.0, 1000.0]))
        stats = diagnosis.prediction_stats(np.full(3, 2.0), truth)
        self.assertIsNone(stats["corr_truth"])

    def test_empty_population_is_refused(self):
        with self.assertRaises(ValueError):
            diagnosis.prediction_stats(np.zeros(0), np.zeros(0))

    def test_nonfinite_is_refused(self):
        with self.assertRaises(ValueError):
            diagnosis.prediction_stats(np.array([1.0, np.nan]), np.zeros(2))

    def test_misaligned_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            diagnosis.prediction_stats(np.zeros(3), np.zeros(2))


class FourPlusMaskTests(unittest.TestCase):
    def test_boundary_is_four_tables(self):
        mask = diagnosis.four_plus_mask([1, 3, 4, 5, 7])
        np.testing.assert_array_equal(mask, [False, False, True, True, True])


class PgDirectionTests(unittest.TestCase):
    def test_under_over_and_ties_are_separated(self):
        truth = np.log(np.array([100.0, 100.0, 100.0]))
        pg = np.log(np.array([10.0, 1000.0, 100.0]))
        splits = diagnosis.pg_direction_splits(pg, truth)
        np.testing.assert_array_equal(splits["pg_under"], [True, False, False])
        np.testing.assert_array_equal(splits["pg_over"], [False, True, False])
        np.testing.assert_array_equal(splits["pg_equal"], [False, False, True])

    def test_ties_are_not_forced_into_a_side(self):
        truth = np.zeros(3)
        splits = diagnosis.pg_direction_splits(truth.copy(), truth)
        self.assertEqual(int(splits["pg_under"].sum()), 0)
        self.assertEqual(int(splits["pg_over"].sum()), 0)
        self.assertEqual(int(splits["pg_equal"].sum()), 3)


class MaximaTests(unittest.TestCase):
    def test_parses_four_values(self):
        self.assertEqual(diagnosis.parse_maxima("12,14,8,10").tolist(), [12, 14, 8, 10])

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            diagnosis.parse_maxima("12,14,8")

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            diagnosis.parse_maxima("12,0,8,10")


class SubgroupTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.log(np.array([10.0, 100.0, 1000.0, 10000.0]))
        self.preds = {
            "ours_k0": self.truth.copy(),
            "price_k0": self.truth.copy(),
            "pg": self.truth.copy(),
        }

    def test_small_subgroup_is_not_reported_as_a_percentile(self):
        entry = diagnosis.subgroup_stats(
            self.preds, self.truth, [True, False, False, False], minimum=30
        )
        self.assertEqual(entry["n"], 1)
        self.assertFalse(entry["reported"])
        self.assertNotIn("ours_k0", entry)

    def test_large_enough_subgroup_reports_all_three(self):
        entry = diagnosis.subgroup_stats(
            self.preds, self.truth, [True, True, True, True], minimum=3
        )
        self.assertTrue(entry["reported"])
        self.assertEqual(entry["n"], 4)
        for method in ("ours_k0", "price_k0", "pg"):
            self.assertIn(method, entry)


class DiagnoseDatabaseTests(unittest.TestCase):
    def test_report_keeps_overview_four_plus_and_direction(self):
        n = 60
        truth = np.log(np.linspace(10.0, 100000.0, n))
        preds = {
            "ours_k0": truth - 0.5,
            "price_k0": truth - 0.2,
            "pg": truth + 0.9,
        }
        n_tables = np.array([2] * 40 + [5] * 20)
        report = diagnosis.diagnose_database(preds, truth, n_tables, minimum=10)
        self.assertEqual(report["overview"]["n"], n)
        self.assertEqual(report["four_plus"]["n"], 20)
        self.assertTrue(report["four_plus"]["reported"])
        # pg is always above truth, so every row lands in pg_over.
        self.assertEqual(report["pg_direction"]["pg_over"]["n"], n)
        self.assertFalse(report["pg_direction"]["pg_under"]["reported"])
        self.assertEqual(report["tie_count"], 0)

    def test_bias_matches_the_shift_we_applied(self):
        n = 50
        truth = np.log(np.linspace(10.0, 10000.0, n))
        preds = {
            "ours_k0": truth - 0.7,
            "price_k0": truth,
            "pg": truth,
        }
        report = diagnosis.diagnose_database(preds, truth, np.full(n, 2), minimum=10)
        self.assertAlmostEqual(
            report["overview"]["ours_k0"]["bias_med"], -0.7, places=9
        )
        self.assertAlmostEqual(
            report["overview"]["price_k0"]["bias_med"], 0.0, places=9
        )


class PairwiseReasoningTests(unittest.TestCase):
    def _report(self, ours_q95, price_q95, ours_corr, price_corr):
        return {
            "overview": {
                "ours_k0": {"q95": ours_q95, "corr_truth": ours_corr},
                "price_k0": {"q95": price_q95, "corr_truth": price_corr},
            }
        }

    def test_counts_split_by_q95_and_by_correlation(self):
        reports = {
            "a": self._report(10.0, 5.0, 0.90, 0.95),  # price better on both
            "b": self._report(5.0, 10.0, 0.95, 0.90),  # we better on both
            "c": self._report(5.0, 10.0, 0.95, 0.99),  # we better, price ranks
        }
        cross = diagnosis.pairwise_reasoning(reports)
        self.assertEqual(cross["q95_better_price"]["databases"], ["a"])
        self.assertEqual(cross["q95_better_ours"]["databases"], ["b", "c"])
        self.assertEqual(cross["corr_better_price"]["databases"], ["a", "c"])
        self.assertEqual(cross["corr_better_ours"]["databases"], ["b"])

    def test_missing_correlation_is_treated_as_zero_not_crash(self):
        reports = {"a": self._report(10.0, 5.0, None, 0.5)}
        cross = diagnosis.pairwise_reasoning(reports)
        self.assertEqual(cross["corr_better_price"]["databases"], ["a"])


if __name__ == "__main__":
    unittest.main()
