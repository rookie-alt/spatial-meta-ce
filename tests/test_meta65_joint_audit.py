import json
import unittest

import numpy as np

from scripts import meta65_joint_audit as audit


def cells(qs=(1, 2, 4, 8), tables=(1, 2, 4, 5), true=(1, 100, 1000, 1e6)):
    return {
        arm: {
            "query_hash": np.array([f"h{i}" for i in range(4)]),
            "query_id": np.array([f"telstra/pretrain/{i + 1}" for i in range(4)]),
            "true": np.array(true, dtype=float),
            "n_tables": np.array(tables),
            "log_pred": np.log(np.array(true) * np.array(qs) * (index + 1)),
        }
        for index, arm in enumerate(audit.ARMS)
    }


class JointAuditTests(unittest.TestCase):
    def test_full_joint_denominator_and_one_table(self):
        result = audit.analyze_run(cells(), min_n=30)
        rows = result["joint"]
        self.assertEqual(sum(row["count"] for row in rows.values()), 4)
        self.assertEqual(rows["<100|1"]["count"], 1)
        self.assertEqual(rows["100-1e3|2"]["count"], 1)
        self.assertEqual(rows["<100|1"]["status"], "below_min_n")
        self.assertIn("q95", rows["<100|1"]["arms"][audit.ARMS[0]])
        self.assertIsNone(rows[">=1e6|4"]["arms"][audit.ARMS[0]]["q95"])

    def test_pairing_aligns_reordered_rows(self):
        original = cells()
        expected = audit.analyze_run(original, 1)
        reordered = cells()
        reordered[audit.ARMS[1]] = {
            key: value[::-1] for key, value in reordered[audit.ARMS[1]].items()
        }
        self.assertEqual(expected, audit.analyze_run(reordered, 1))

    def test_pairing_rejects_mismatch_duplicate_and_nonfinite(self):
        for field, value in (
            ("true", 9),
            ("n_tables", 9),
            ("query_hash", "h1"),
            ("log_pred", np.nan),
        ):
            data = cells()
            data[audit.ARMS[1]][field][0] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.analyze_run(data)

    def test_standard_q95_not_log_interpolation(self):
        result = audit.analyze_run(cells(), 1)["marginal"]["all"]["arms"][audit.ARMS[0]]
        self.assertAlmostEqual(result["q95"], float(np.percentile([1, 2, 4, 8], 95)))
        self.assertNotEqual(
            result["q95"], float(np.exp(np.percentile(np.log([1, 2, 4, 8]), 95)))
        )

    def test_mean_of_run_quantiles_and_missing_runs(self):
        first = audit.analyze_run(cells(), 1)["marginal"]["all"]
        second = audit.analyze_run(cells(qs=(100, 100, 100, 100)), 1)["marginal"]["all"]
        result = audit.aggregate_runs([first, second])
        self.assertAlmostEqual(
            result["arms"][audit.ARMS[0]]["q95_mean"],
            (first["arms"][audit.ARMS[0]]["q95"] + 100) / 2,
        )
        empty = audit.analyze_run(cells(), 30)["joint"]["<100|4"]
        result = audit.aggregate_runs([first, empty])
        self.assertEqual(result["status"], "incomplete_nonempty_runs")
        self.assertIsNone(result["paired"]["C-A"]["q95_ratio_of_means"])

    def test_paired_query_changes_and_stability(self):
        row = audit.analyze_run(cells(), 1)["marginal"]["all"]
        self.assertEqual(row["paired"]["C-A"]["worse_count"], 4)
        result = audit.aggregate_runs([row] * 5)
        self.assertEqual(result["paired"]["C-A"]["q95_worse_runs"], 5)
        self.assertEqual(result["paired"]["C-A"]["leave_one_out_positive"], 5)
        json.dumps(result, allow_nan=False)

    def test_source_allowlist(self):
        self.assertEqual(audit.SOURCE_DATABASES, ("chembl", "legalacts", "telstra"))
        self.assertEqual(audit.TABLE_BUCKETS, ("1", "2", "3", "4", "5+"))
