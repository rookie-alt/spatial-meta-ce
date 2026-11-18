import unittest

import numpy as np

from spatial_meta_ce.holdout import (
    holdout_digest,
    holdout_validation_plan,
    split_holdout,
)


class SplitHoldoutTests(unittest.TestCase):
    def test_split_is_disjoint_and_complete(self):
        unique = np.arange(20000, dtype=np.int64)
        train, held = split_holdout(unique, 0.02, 1729, "demo")
        self.assertEqual(len(train) + len(held), len(unique))
        self.assertEqual(np.intersect1d(train, held).size, 0)
        self.assertTrue(np.array_equal(np.sort(np.concatenate([train, held])), unique))
        self.assertGreaterEqual(len(held), 200)

    def test_split_is_deterministic(self):
        unique = np.arange(20000, dtype=np.int64)
        first = split_holdout(unique, 0.02, 1729, "demo")
        second = split_holdout(unique, 0.02, 1729, "demo")
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))

    def test_split_depends_on_database_and_seed(self):
        unique = np.arange(20000, dtype=np.int64)
        a = split_holdout(unique, 0.02, 1729, "alpha")[1]
        b = split_holdout(unique, 0.02, 1729, "beta")[1]
        c = split_holdout(unique, 0.02, 7, "alpha")[1]
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_fraction_bounds_rejected(self):
        unique = np.arange(20000, dtype=np.int64)
        for fraction in (0, 1, -0.1, 1.5):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_holdout(unique, fraction, 1729, "demo")

    def test_small_database_rejected(self):
        with self.assertRaises(ValueError):
            split_holdout(np.arange(100, dtype=np.int64), 0.02, 1729, "demo")

    def test_repeated_indices_rejected(self):
        unique = np.array([1, 1, 2, 3] * 100, dtype=np.int64)
        with self.assertRaises(ValueError):
            split_holdout(unique, 0.02, 1729, "demo")


class ValidationPlanTests(unittest.TestCase):
    def setUp(self):
        unique = np.arange(2000, dtype=np.int64)
        train, held = split_holdout(unique, 0.2, 1729, "demo")
        self.entries = {
            "demo": {
                "content_hash": "hash",
                "held": held,
                "held_hashes": [str(i) for i in held],
            }
        }

    def test_plan_uses_only_held_queries(self):
        episodes = holdout_validation_plan(self.entries, [5, 20], 2, 64, 1729)
        held = set(self.entries["demo"]["held"].tolist())
        self.assertEqual(len(episodes), 2 * 2)
        for episode in episodes:
            self.assertTrue(set(episode["support"]) <= held)
            self.assertTrue(set(episode["query"]) <= held)
            self.assertFalse(set(episode["support"]) & set(episode["query"]))

    def test_plan_is_deterministic(self):
        first = holdout_validation_plan(self.entries, [5], 1, 64, 1729)
        second = holdout_validation_plan(self.entries, [5], 1, 64, 1729)
        self.assertEqual(first, second)

    def test_plan_rejects_oversized_support(self):
        with self.assertRaises(ValueError):
            holdout_validation_plan(self.entries, [500], 1, 64, 1729)

    def test_digest_changes_with_holdout_content(self):
        first = holdout_digest(self.entries)
        changed = {
            "demo": {
                **self.entries["demo"],
                "held_hashes": ["different"],
            }
        }
        self.assertNotEqual(first, holdout_digest(changed))


if __name__ == "__main__":
    unittest.main()
