import unittest

import numpy as np

from scripts import diagnostic_88_d2 as d2


def design(n=4000, dimension=64, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, dimension))
    z = np.concatenate([features, np.ones((n, 1))], axis=1)
    return features, z


class Diagnostic88D2Tests(unittest.TestCase):
    def test_linear_delta_has_unit_r2_linear(self):
        features, z = design()
        rng = np.random.default_rng(1)
        delta = z @ rng.normal(size=z.shape[1])
        result = d2.solve(delta, features)
        self.assertAlmostEqual(result["R2_linear"], 1.0, places=10)

    def test_rank_four_delta_reaches_unit_r2_four(self):
        features, z = design(seed=2)
        _, _, vt = np.linalg.svd(z, full_matrices=False)
        rng = np.random.default_rng(3)
        delta = z @ vt[:4].T @ rng.normal(size=4)
        result = d2.solve(delta, features)
        self.assertAlmostEqual(result["R2_4_protocol"], 1.0, places=8)
        self.assertAlmostEqual(result["R2_4_altbasis"], 1.0, places=8)

    def test_delta_outside_top_four_gets_zero(self):
        features, z = design(seed=4)
        u, _, _ = np.linalg.svd(z, full_matrices=False)
        delta = u[:, 30] * 3.0
        result = d2.solve(delta, features)
        self.assertAlmostEqual(result["R2_4_protocol"], 0.0, places=8)
        self.assertAlmostEqual(result["R2_4_altbasis"], 0.0, places=8)

    def test_altbasis_is_never_negative_beyond_rounding(self):
        features, z = design(seed=5)
        rng = np.random.default_rng(6)
        result = d2.solve(z @ rng.normal(size=z.shape[1]), features)
        self.assertGreater(result["R2_4_altbasis"], -1e-9)

    def test_nonfinite_inputs_are_rejected(self):
        features, _ = design(n=100)
        delta = np.zeros(100)
        delta[3] = np.nan
        with self.assertRaises(ValueError):
            d2.solve(delta, features)

    def test_zero_delta_is_rejected(self):
        features, _ = design(n=100)
        with self.assertRaises(ValueError):
            d2.solve(np.zeros(100), features)

    def test_verdict_thresholds_and_range(self):
        def status(left, right):
            return d2.decide(
                {
                    "a": [{"R2_4_protocol": left}] * 5,
                    "b": [{"R2_4_protocol": right}] * 5,
                }
            )["status"]

        self.assertEqual(status(0.90, 0.85), "expressible")
        self.assertEqual(status(0.90, 0.60), "partial")
        self.assertEqual(status(0.90, 0.55), "unstable")
        self.assertEqual(status(0.79, 0.60), "partial")
        self.assertEqual(status(0.49, 0.45), "insufficient")

    def test_verdict_uses_median_of_five_runs(self):
        cells = [{"R2_4_protocol": value} for value in (0.10, 0.20, 0.90, 0.95, 0.99)]
        result = d2.decide({"a": cells})
        self.assertAlmostEqual(result["medians"]["a"], 0.90)

    def test_guard_rejects_forbidden_and_unexpected_databases(self):
        for name in d2.FORBIDDEN_DATABASES:
            with self.assertRaises(ValueError):
                d2.guard_database(name)
        for name in ("hepatitis", "sakila", "accidents"):
            with self.assertRaises(ValueError):
                d2.guard_database(name)
        d2.guard_database("chembl")

    def test_median_handles_none(self):
        self.assertIsNone(d2.median([None, None]))
        self.assertAlmostEqual(d2.median([0.2, None, 0.4]), 0.3)


if __name__ == "__main__":
    unittest.main()
