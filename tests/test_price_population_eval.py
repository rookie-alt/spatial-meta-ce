import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parents[1] / "scripts/price_population_eval.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_SPEC = importlib.util.spec_from_file_location("price_population_eval", _SOURCE)
population = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(population)

HEADER = [
    "database",
    "run",
    "k",
    "query_id",
    "query_hash",
    "true_cardinality",
    "log_prediction",
]


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return Path(path)


class DecodeTests(unittest.TestCase):
    """PRICE emits log1p(card)+1; the shared space is log(max(card, 1))."""

    def test_exact_cardinality_round_trips(self):
        for card in (10.0, 100.0, 1_956_303_104.0):
            encoded = np.log1p(card) + 1.0
            with self.subTest(card=card):
                self.assertAlmostEqual(
                    float(population.decode_price_output(np.array([encoded]))[0]),
                    np.log(card),
                    places=9,
                )

    def test_cardinality_below_one_decodes_to_zero(self):
        # log(max(card, 1)) is 0 for card in [0, 1]; the decoder must not
        # produce a negative log there.  card=1 lands on the rounding edge of
        # the decoder's exp/log branch, so the equality is checked numerically.
        for card in (0.0, 1.0):
            encoded = np.log1p(card) + 1.0
            with self.subTest(card=card):
                self.assertAlmostEqual(
                    float(population.decode_price_output(np.array([encoded]))[0]),
                    0.0,
                    places=12,
                )

    def test_decoded_scale_is_comparable_with_log_true(self):
        card = 1234.0
        encoded = np.log1p(card) + 1.0
        decoded = float(population.decode_price_output(np.array([encoded]))[0])
        self.assertAlmostEqual(decoded, np.log(max(card, 1.0)), places=9)


class TargetGuardTests(unittest.TestCase):
    def test_target_databases_are_refused(self):
        for db in ("imdb", "stats", "genome", "ergastf1"):
            with self.subTest(db=db), self.assertRaises(ValueError):
                population.assert_source_database(db)

    def test_source_databases_are_accepted(self):
        for db in ("chembl", "legalacts", "telstra"):
            population.assert_source_database(db)

    def test_prediction_file_containing_a_target_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(
                Path(directory) / "p.csv",
                [["stats", 0, 0, "x/1", "deadbeef", 10.0, 2.3]],
            )
            with self.assertRaises(ValueError):
                population.read_k0_predictions(path, ["chembl"])


class ReadPredictionsTests(unittest.TestCase):
    def test_only_k0_rows_are_collected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(
                Path(directory) / "p.csv",
                [
                    ["chembl", 0, 0, "c/1", "h1", 10.0, 2.3],
                    ["chembl", 1, 0, "c/1", "h1", 10.0, 2.3],
                    ["chembl", 0, 5, "c/1", "h1", 10.0, 2.9],
                    ["legalacts", 0, 0, "l/1", "h2", 20.0, 3.0],
                ],
            )
            anchors = population.read_k0_predictions(path, ["chembl"])
        self.assertEqual(anchors, {("chembl", "h1"): (2.3, 10.0)})

    def test_conflicting_k0_across_runs_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(
                Path(directory) / "p.csv",
                [
                    ["chembl", 0, 0, "c/1", "h1", 10.0, 2.3],
                    ["chembl", 1, 0, "c/1", "h1", 10.0, 2.4],
                ],
            )
            with self.assertRaises(ValueError):
                population.read_k0_predictions(path, ["chembl"])

    def test_file_without_k0_rows_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(
                Path(directory) / "p.csv", [["chembl", 0, 5, "c/1", "h1", 10.0, 2.9]]
            )
            with self.assertRaises(ValueError):
                population.read_k0_predictions(path, ["chembl"])

    def test_nonfinite_prediction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(
                Path(directory) / "p.csv",
                [["chembl", 0, 0, "c/1", "h1", 10.0, float("nan")]],
            )
            with self.assertRaises(ValueError):
                population.read_k0_predictions(path, ["chembl"])


class PopulationMetricTests(unittest.TestCase):
    def test_exact_predictions_have_unit_error(self):
        metrics = population.population_metrics(
            [np.log(10.0), np.log(100.0)], [10.0, 100.0]
        )
        self.assertAlmostEqual(metrics["q50"], 1.0)
        self.assertAlmostEqual(metrics["q95"], 1.0)
        self.assertAlmostEqual(metrics["log_q95"], 0.0)

    def test_one_bad_prediction_moves_the_percentiles(self):
        metrics = population.population_metrics(
            [np.log(10.0), np.log(1000.0)], [10.0, 100.0]
        )
        self.assertAlmostEqual(metrics["q50"], 5.5)
        self.assertAlmostEqual(metrics["q95"], 9.55)
        self.assertAlmostEqual(metrics["log_q95"], 0.95 * np.log(10.0))

    def test_empty_population_has_no_metrics(self):
        self.assertIsNone(population.population_metrics([], []))


class BucketTests(unittest.TestCase):
    def setUp(self):
        self.n_tables = np.array([1, 2, 3, 4, 5, 6, 7])
        self.true = np.full(7, 100.0)
        self.log_preds = {
            "ours_k0": np.log(np.full(7, 100.0)),
            "price_k0": np.log(np.full(7, 100.0)),
            "pg": np.log(np.full(7, 100.0)),
        }

    def test_table_buckets_follow_the_shared_definition(self):
        report = population.database_report(self.log_preds, self.true, self.n_tables)
        self.assertEqual(report["n"], 7)
        self.assertEqual(report["buckets"]["1"]["n"], 1)
        self.assertEqual(report["buckets"]["2-3"]["n"], 2)
        self.assertEqual(report["buckets"]["4-5"]["n"], 2)
        self.assertEqual(report["buckets"]["6+"]["n"], 2)

    def test_four_plus_bucket_pools_everything_above_three_tables(self):
        report = population.database_report(self.log_preds, self.true, self.n_tables)
        self.assertEqual(report["bucket_4plus"]["n"], 4)
        self.assertIn("n_tables >= 4", report["bucket_4plus"]["definition"])

    def test_bucket_metrics_use_only_that_bucket(self):
        log_preds = {key: value.copy() for key, value in self.log_preds.items()}
        log_preds["ours_k0"][0] = np.log(1.0)  # only the single-table query is wrong
        report = population.database_report(log_preds, self.true, self.n_tables)
        # The single-table bucket holds only that query, so its Q95 is its
        # own error; every other bucket stays exact.
        self.assertAlmostEqual(report["buckets"]["1"]["ours_k0"]["q95"], 100.0)
        self.assertAlmostEqual(report["buckets"]["2-3"]["ours_k0"]["q95"], 1.0)
        # The population Q95 is a tail statistic: one bad query out of seven
        # moves it to 70.3 rather than staying at 1.
        self.assertAlmostEqual(report["ours_k0"]["q95"], 70.3)

    def test_report_keeps_the_three_estimators(self):
        report = population.database_report(self.log_preds, self.true, self.n_tables)
        for method in ("ours_k0", "price_k0", "pg"):
            self.assertIn(method, report)
            self.assertIn(method, report["bucket_4plus"])


class FakeStore:
    """Minimal stand-in for a derived cache: identities, cards and offsets."""

    def __init__(self, rows, cards, offsets):
        self.rows = rows
        self.cards = np.asarray(cards, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def unique_indices(self):
        return np.array([0, 1, 2], dtype=np.int64)

    def batch(self, indices, device="cpu"):
        import torch

        return {"pg_log": torch.zeros(len(indices), dtype=torch.float32)}


class FakeModel:
    """Returns a fixed PRICE-encoded output for every batch."""

    def __init__(self, encoded):
        self.encoded = encoded

    def __call__(self, batch):
        import torch

        size = int(batch["pg_log"].shape[0])
        return torch.tensor(
            [self.encoded] * size, dtype=torch.float32, device=batch["pg_log"].device
        )


class CollectDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "c/1", "query_hash": "h1"},
            {"id": "c/2", "query_hash": "h2"},
            {"id": "c/3", "query_hash": "h3"},
        ]
        # cards[:, 0] is the truth, cards[:, 1] the raw PG estimate.
        self.store = FakeStore(
            self.rows,
            [[10.0, 5.0], [20.0, 40.0], [30.0, 1.0]],
            [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        )
        self.anchors = {
            ("chembl", "h1"): (np.log(10.0), 10.0),
            ("chembl", "h2"): (np.log(20.0), 20.0),
            ("chembl", "h3"): (np.log(30.0), 30.0),
        }

    def _collect(self, encoded):
        model = FakeModel(encoded)
        return population.collect_database(
            self.store, self.anchors, "chembl", model, "cpu", 8
        )

    def test_three_estimators_are_aligned_with_the_cache(self):
        encoded = np.log1p(100.0) + 1.0
        log_preds, true, n_tables = self._collect(encoded)
        np.testing.assert_allclose(true, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(log_preds["ours_k0"], np.log([10.0, 20.0, 30.0]))
        np.testing.assert_allclose(log_preds["price_k0"], np.log(100.0))
        np.testing.assert_allclose(log_preds["pg"], np.log([5.0, 40.0, 1.0]))
        np.testing.assert_array_equal(n_tables, [1, 1, 1])

    def test_truth_disagreement_between_cache_and_export_fails(self):
        self.anchors[("chembl", "h2")] = (np.log(20.0), 21.0)
        with self.assertRaises(ValueError):
            self._collect(np.log1p(100.0) + 1.0)

    def test_missing_prediction_for_a_cache_query_fails(self):
        del self.anchors[("chembl", "h3")]
        with self.assertRaises(ValueError):
            self._collect(np.log1p(100.0) + 1.0)


class WriteJsonTests(unittest.TestCase):
    def test_write_is_atomic_and_json_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.json"
            population.write_json(path, {"a": 1})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})
            self.assertFalse(Path(str(path) + ".tmp").exists())
            with self.assertRaises(ValueError):
                population.write_json(path, {"a": float("nan")})


if __name__ == "__main__":
    unittest.main()
