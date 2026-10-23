import argparse
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location(
    "target_population_eval", _ROOT / "scripts/target_population_eval.py"
)
target = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(target)


class AllowedTargetsTests(unittest.TestCase):
    def test_target_must_be_listed_explicitly(self):
        target.assert_target_allowed("imdb", ["imdb"])

    def test_unlisted_database_is_refused(self):
        with self.assertRaises(ValueError):
            target.assert_target_allowed("stats", ["imdb"])

    def test_source_database_is_refused_as_target(self):
        # chembl is a source, never a target: it cannot be authorised here.
        with self.assertRaises(ValueError):
            target.validate_allowed_targets(["imdb", "chembl"])

    def test_path_traversal_name_is_refused(self):
        for bad in ("../imdb", "imdb/", "", "."):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    target.validate_allowed_targets([bad])


class MaximaTests(unittest.TestCase):
    def test_parses_comma_list(self):
        self.assertEqual(target.parse_maxima("12,14,8,10").tolist(), [12, 14, 8, 10])

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            target.parse_maxima("12,14,8")

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            target.parse_maxima("12,14,8,x")

    def test_rejects_nonpositive(self):
        with self.assertRaises(ValueError):
            target.parse_maxima("12,0,8,10")

    def test_capacity_guard_refuses_cache_exceeding_maxima(self):
        with self.assertRaises(ValueError):
            target.check_price_capacity(np.array([11, 12, 7, 10]), [12, 12, 7, 10], "x")

    def test_capacity_guard_accepts_cache_within_maxima(self):
        target.check_price_capacity(np.array([12, 14, 8, 10]), [5, 8, 5, 4], "imdb")


class FrozenCheckpointTests(unittest.TestCase):
    def _state(self, train, valid):
        return {
            "metadata": {
                "train_databases": train,
                "validation_databases": valid,
                "seed": 42,
                "method": "baseline",
            },
            "model_config": {"width": 64},
            "transform": "price40-group-logfanout-v1",
        }

    def test_checkpoint_exposed_on_target_is_refused(self):
        state = self._state(["stats", "genome"], ["chembl"])
        with self.assertRaises(ValueError):
            target.assert_checkpoint_frozen(state, ["stats"])

    def test_checkpoint_not_exposed_on_target_passes(self):
        state = self._state(["accidents"], ["chembl"])
        frozen = target.assert_checkpoint_frozen(state, ["imdb"])
        self.assertEqual(frozen["seed"], 42)
        self.assertEqual(frozen["model_config"], {"width": 64})


class FakeStore:
    """Minimal stand-in for a derived target cache."""

    def __init__(self, rows, cards, offsets):
        self.rows = rows
        self.cards = np.asarray(cards, dtype=np.float64)
        self.offsets = np.asarray(offsets, dtype=np.int64)

    def unique_indices(self):
        return np.array([0, 1, 2], dtype=np.int64)

    def batch(self, indices, device="cpu"):
        import torch

        return {"pg_log": torch.zeros(len(indices), dtype=torch.float32)}


class FakeOursModel:
    """Returns a fixed log-space value for every batch."""

    def __init__(self, log_value):
        self.log_value = log_value

    def __call__(self, batch):
        import torch

        size = int(batch["pg_log"].shape[0])
        return torch.full((size,), self.log_value, dtype=torch.float32)


class FakePricePopulation:
    """Stands in for the loaded price_population_eval module."""

    POPULATION_METRIC = "unique-query population Q95 (single pass, NOT episode mean)"
    TABLE_BUCKETS = ("1", "2-3", "4-5", "6+")

    def price_predictions(self, model, store, indices, device, batch_size):
        return np.full(len(indices), float(np.log(100.0)))


class CollectTargetTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": "i/1", "query_hash": "h1"},
            {"id": "i/2", "query_hash": "h2"},
            {"id": "i/3", "query_hash": "h3"},
        ]
        self.offsets = [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
        self.store = FakeStore(
            self.rows, [[10.0, 5.0], [20.0, 40.0], [30.0, 1.0]], self.offsets
        )

    def test_three_estimators_share_one_query_population(self):
        log_preds, true, n_tables = target.collect_target(
            self.store,
            FakeOursModel(float(np.log(10.0))),
            None,
            FakePricePopulation(),
            "cpu",
            8,
        )
        np.testing.assert_allclose(true, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(log_preds["ours_k0"], np.log(10.0))
        np.testing.assert_allclose(log_preds["price_k0"], np.log(100.0))
        np.testing.assert_allclose(log_preds["pg"], np.log([5.0, 40.0, 1.0]))
        np.testing.assert_array_equal(n_tables, [1, 1, 1])
        self.assertEqual(len(log_preds["ours_k0"]), 3)

    def test_nonpositive_truth_is_rejected(self):
        store = FakeStore(
            self.rows, [[0.0, 5.0], [20.0, 40.0], [30.0, 1.0]], self.offsets
        )
        with self.assertRaises(ValueError):
            target.collect_target(
                store, FakeOursModel(1.0), None, FakePricePopulation(), "cpu", 8
            )

    def test_nonfinite_prediction_is_rejected(self):
        class BadOurs(FakeOursModel):
            def __call__(self, batch):
                import torch

                size = int(batch["pg_log"].shape[0])
                return torch.full((size,), float("nan"), dtype=torch.float32)

        with self.assertRaises(ValueError):
            target.collect_target(
                self.store, BadOurs(1.0), None, FakePricePopulation(), "cpu", 8
            )


class BuildProtocolTests(unittest.TestCase):
    class FakePool:
        def get(self, split, db):
            return type(
                "S",
                (),
                {"manifest": {"content_hash": "abc", "rows": [0]}},
            )()

    def test_target_recorded_not_empty_and_maxima_present(self):
        with tempfile.TemporaryDirectory() as directory:
            ckpt = Path(directory) / "last.pt"
            ckpt.write_bytes(b"x")
            price = Path(directory) / "params.pth"
            price.write_bytes(b"y")
            args = argparse.Namespace(
                checkpoint=str(ckpt),
                price_checkpoint=str(price),
                split="test",
                cache_root=str(Path(directory) / "cache"),
                maxima_rule="historical target-table rule: test x {test,finetune} max",
            )
            reports = {"imdb": {"n": 521}}
            protocol = target.build_protocol(
                args,
                np.array([12, 14, 8, 10]),
                ["imdb"],
                ["imdb"],
                {
                    "seed": 42,
                    "method": "baseline",
                    "train_databases": ["accidents"],
                    "validation_databases": ["chembl"],
                    "model_config": {},
                    "transform": "price40-group-logfanout-v1",
                },
                {"imdb": [5, 8, 5, 4]},
                reports,
                10709330,
                FakePricePopulation(),
                self.FakePool(),
            )
        self.assertEqual(protocol["target_databases_read"], ["imdb"])
        self.assertNotEqual(protocol["target_databases_read"], [])
        self.assertEqual(protocol["price"]["maxima"], [12, 14, 8, 10])
        self.assertIn("historical target-table rule", protocol["price"]["maxima_rule"])
        self.assertEqual(
            protocol["checkpoint"]["sha256"],
            hashlib.sha256(b"x").hexdigest(),
        )
        self.assertEqual(protocol["inputs"]["caches"]["imdb"]["unique_queries"], 521)


if __name__ == "__main__":
    unittest.main()
