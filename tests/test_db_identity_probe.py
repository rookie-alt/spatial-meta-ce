"""Tests for the database-identity probe's statistics.

The probe itself runs on a server with real caches; these tests exercise the
three statistics on synthetic data whose answer is known, so a wrong number
cannot slip through into the report.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "db_identity_probe", ROOT / "scripts" / "db_identity_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DbIdentityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probe = _load()

    def _clustered(self, n_classes=4, per=200, dim=8, sep=6.0, seed=0):
        rng = np.random.default_rng(seed)
        centres = rng.normal(scale=sep, size=(n_classes, dim))
        x = np.concatenate(
            [
                centres[c] + rng.normal(scale=1.0, size=(per, dim))
                for c in range(n_classes)
            ]
        )
        y = np.repeat(np.arange(n_classes), per)
        return x, y

    def test_k_way_separates_clusters(self):
        x, y = self._clustered()
        acc = self.probe.k_way(x, y, 4, 5, np.random.default_rng(1))
        self.assertGreater(acc, 0.9)

    def test_permutation_control_drops_to_chance(self):
        x, y = self._clustered()
        acc = self.probe.negative_control(x, y, 4, 5, np.random.default_rng(2))
        # chance is 0.25; allow generous slack for the finite sample
        self.assertLess(acc, 0.4)

    def test_k_way_on_overlapping_clusters_is_near_chance(self):
        x, y = self._clustered(sep=0.15)
        acc = self.probe.k_way(x, y, 4, 5, np.random.default_rng(3))
        self.assertLess(acc, 0.45)

    def test_centroid_ratio_is_small_when_clusters_overlap(self):
        x, y = self._clustered(sep=0.15)
        rows, _ = self.probe.lodo_centroid_ratio(x, y)
        ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
        self.assertTrue(all(r < 3.0 for r in ratios), ratios)

    def test_centroid_ratio_is_large_when_clusters_are_far(self):
        x, y = self._clustered(sep=8.0)
        rows, _ = self.probe.lodo_centroid_ratio(x, y)
        ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
        self.assertTrue(all(r > 3.0 for r in ratios), ratios)

    def test_standardise_fits_on_train_only(self):
        train = np.array([[0.0], [0.0], [0.0]])
        test = np.array([[100.0]])
        _, scaled_test = self.probe.standardise(train, test)
        # train has zero variance, so scale is forced to 1 and the test value
        # must carry its raw distance from the train mean
        self.assertAlmostEqual(float(scaled_test[0, 0]), 100.0)

    def test_forbidden_targets_are_declared(self):
        for name in ("imdb", "stats", "genome", "ergastf1"):
            self.assertIn(name, self.probe.FORBIDDEN)


if __name__ == "__main__":
    unittest.main()
