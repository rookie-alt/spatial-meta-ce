"""Synthetic feasibility tests, not production acceptance tests."""

import math
import unittest
from fractions import Fraction as F

import verified_arithmetic as v


class ArithmeticTests(unittest.TestCase):
    def test_exact_binary64_and_nonfinite(self):
        self.assertEqual(v.exact(0.1), F(*(0.1).as_integer_ratio()))
        for bad in [math.inf, -math.inf, math.nan, True]:
            with self.assertRaises((TypeError, ValueError)):
                v.exact(bad)

    def test_sqrt_enclosure(self):
        for x in [F(0), F(4), F(2), F(1, 10**200), F(10**200), F(5, 7)]:
            lo, hi = v.sqrt_interval(x)
            self.assertLessEqual(lo * lo, x)
            self.assertGreaterEqual(hi * hi, x)
        self.assertEqual(v.sqrt_interval(F(4)), (F(2), F(2)))
        with self.assertRaises(ValueError):
            v.sqrt_interval(F(-1))

    def test_cancellation_is_not_hidden_by_float_matmul(self):
        # Exact 1e16 + 1 - 1e16 is 1, even if a float fold loses it.
        self.assertEqual(v.dot([F(10**16), F(1), F(-(10**16))], [F(1)] * 3), 1)

    def test_rank_zero_redundant_and_fullrank(self):
        for a, s, rank in [
            ([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0], 0),
            ([[1.0, 0.0], [0.0, 0.0]], [1.0, 0.0], 1),
            ([[1.0, 0.0], [0.0, 0.5]], [1.0, 0.5], 2),
        ]:
            result = v.certify_rank(
                a, [[1.0, 0.0], [0.0, 1.0]], s, [[1.0, 0.0], [0.0, 1.0]], rank
            )
            self.assertEqual(result["rank_status"], "verified")
            self.assertNotIn("pass", result)
            self.assertFalse(result["LS_or_subspace_or_G1_certified"])

    def test_cutoff_equality_unknown_and_wrong_rank(self):
        rho = 1e-12
        result = v.certify_rank(
            [[1.0, 0.0], [0.0, rho]],
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, rho],
            [[1.0, 0.0], [0.0, 1.0]],
            1,
        )
        self.assertEqual(result["rank_status"], "unknown")
        result = v.certify_rank(
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [1.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            2,
        )
        self.assertEqual(result["rank_status"], "unknown")

    def test_forged_factors_and_orthogonality_failure(self):
        result = v.certify_rank(
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [2.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            1,
        )
        self.assertEqual(result["rank_status"], "unknown")
        with self.assertRaises(ValueError):
            v.certify_rank(
                [[1.0, 0.0], [0.0, 0.0]],
                [[2.0, 0.0], [0.0, 2.0]],
                [1.0, 0.0],
                [[1.0, 0.0], [0.0, 1.0]],
                1,
            )

    def test_shapes_order_and_size_budget(self):
        for a, u, s, vh in [
            ([[1.0]], [[1.0]], [-1.0], [[1.0]]),
            ([[1.0]], [[1.0, 0.0]], [1.0], [[1.0]]),
            ([[0.0] for _ in range(17)], [[0.0] for _ in range(17)], [0.0], [[1.0]]),
        ]:
            with self.assertRaises(ValueError):
                v.certify_rank(a, u, s, vh, 0)


if __name__ == "__main__":
    unittest.main()
