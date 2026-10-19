import unittest

import numpy as np

from scripts.audit_p2_semantics import (
    frozen_summary,
    numeric_bounds,
    numeric_histogram_selectivity,
)


class P2SemanticsTests(unittest.TestCase):
    def test_null_and_range_intersection_without_legacy_operator_order(self):
        self.assertEqual(
            numeric_bounds(["x >= 3", "x >= 5", "x < 9"], "x", 0, 10), (5.0, 9.0)
        )
        self.assertEqual(numeric_bounds(["x = 2", "x = 3"], "x", 0, 10), (3.0, 2.0))
        self.assertEqual(numeric_bounds(["x < 1", "x >= 3"], "x", 0, 10), (3.0, 1.0))

    def test_histogram_boundaries_and_empty_range(self):
        bins = np.array([20.0, 30.0])
        edges = np.array([0.0, 10.0, 20.0])
        self.assertAlmostEqual(
            numeric_histogram_selectivity(bins, edges, 5.0, 15.0, 100), 0.25
        )
        self.assertEqual(
            numeric_histogram_selectivity(bins, edges, 12.0, 12.0, 100), 0.0
        )
        self.assertEqual(
            numeric_histogram_selectivity(bins, edges, -3.0, 40.0, 100), 0.5
        )

    def test_aggregation_uses_all_columns_but_ebo_only_top_four(self):
        expected = np.array(
            [0.1 * 0.2 * 0.3 * 0.4 * 0.5, 0.1, 0.5 * 0.4**0.5 * 0.3**0.25 * 0.2**0.125]
        )
        np.testing.assert_allclose(frozen_summary([0.1, 0.2, 0.3, 0.4, 0.5]), expected)
        np.testing.assert_allclose(frozen_summary([]), np.ones(3))


if __name__ == "__main__":
    unittest.main()
