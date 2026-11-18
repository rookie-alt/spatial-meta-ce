import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.learning_curve_eval import (
    _assert_same_run,
    assert_source_databases,
    evaluate_checkpoint,
    geometric_macro_ratio,
    macro,
    summarize_population,
    table_counts,
)
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.train import train


class Store:
    manifest = {"content_hash": "curve-fixture"}
    rows = [{"id": str(i), "query_hash": str(i)} for i in range(20)]
    offsets = np.zeros((21, 4), dtype=np.int64)
    # Rows 0-9 hold one table, rows 10-19 hold four tables (group 2 width).
    offsets[:, 2] = np.concatenate(([0], np.cumsum([1] * 10 + [4] * 10)))

    def unique_indices(self):
        return np.arange(20)

    def batch(self, indices, device):
        from spatial_meta_ce.data import batch_samples

        samples = [
            {
                "groups": [
                    torch.ones(1, 40),
                    torch.ones(1, 40),
                    torch.ones(3, 4),
                    torch.ones(0, 43),
                ],
                "true": 20.0 + int(i),
                "pg": 2.0,
                "id": str(i),
                "hash": str(i),
            }
            for i in indices
        ]
        return batch_samples(samples, device)


class Pool:
    def __init__(self, root, **kwargs):
        pass

    def get(self, split, db):
        return Store()


def config(**changes):
    return {
        "model": {"width": 8, "use_pg": True},
        "method": "erm",
        "seed": 7,
        "steps": 4,
        "meta_batch": 1,
        "accumulation": 1,
        "outer_lr": 0.001,
        "inner_lr": 0.01,
        "inner_steps": 0,
        "adapt_scope": "all",
        "support_sizes": [2],
        "query_size": 4,
        "loss": "smooth_l1",
        "validate_every": 2,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 50,
        "min_delta": 0.0,
        **changes,
    }


class SnapshotTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {
            "cache_root": "unused",
            "splits": {"train": ["a"], "validation": ["v"], "test": ["t"]},
        }
        for target, replacement in (
            ("StorePool", Pool),
            ("validation_plan", lambda *args: []),
            ("validate", lambda *args: (1.0, [])),
        ):
            patcher = patch("spatial_meta_ce.train." + target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_snapshot_steps_write_one_checkpoint_per_rung(self):
        out = self.root / "curve"
        result = train(config(snapshot_steps=[2, 4]), self.data, out)
        self.assertEqual(result["last_step"], 4)
        for step in (2, 4):
            path = out / f"step-{step}.pt"
            self.assertTrue(path.is_file(), path)
            _, state = load_checkpoint(path)
            self.assertEqual(state["next_step"], step)
        self.assertTrue((out / "last.pt").is_file())

    def test_absent_snapshot_steps_write_no_extra_checkpoint(self):
        out = self.root / "plain"
        train(config(), self.data, out)
        self.assertEqual(
            sorted(p.name for p in out.glob("*.pt")), ["best.pt", "last.pt"]
        )

    def test_snapshot_steps_validation_errors(self):
        cases = [
            {"snapshot_steps": "4"},
            {"snapshot_steps": [0]},
            {"snapshot_steps": [2, 2]},
            {"snapshot_steps": [4, 2]},
            {"snapshot_steps": [6]},
            {"snapshot_steps": [True]},
        ]
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    train(config(**changes), self.data, self.root / str(index))


class LearningCurveMetricTests(unittest.TestCase):
    def test_table_counts_read_the_table_group(self):
        store = Store()
        counts = table_counts(store, np.array([0, 9, 10, 19]))
        self.assertEqual(counts.tolist(), [1, 1, 4, 4])

    def test_four_plus_bucket_absent_reports_null(self):
        predictions = np.zeros(4)
        truths = np.array([10.0, 10.0, 10.0, 10.0])
        result = summarize_population(predictions, truths, np.array([1, 2, 3, 3]))
        self.assertEqual(result["four_plus_n"], 0)
        self.assertIsNone(result["four_plus_q95"])
        self.assertEqual(result["n"], 4)

    def test_four_plus_bucket_pools_only_wide_queries(self):
        predictions = np.array([0.0, 0.0, 0.0, 0.0])
        truths = np.array([100.0, 100.0, 100.0, 1000.0])
        result = summarize_population(predictions, truths, np.array([1, 2, 4, 6]))
        self.assertEqual(result["four_plus_n"], 2)
        # Q-error is exact: predictions are 1, truths are 100 and 1000.
        self.assertAlmostEqual(
            result["four_plus_q95"], float(np.percentile([100.0, 1000.0], 95))
        )

    def test_macro_is_equal_weight_over_databases(self):
        rows = [
            {"database": "chembl", "k": 0, "q95": 4.0},
            {"database": "legalacts", "k": 0, "q95": 8.0},
            {"database": "telstra", "k": 0, "q95": 12.0},
        ]
        self.assertAlmostEqual(macro(rows, "q95"), 8.0)

    def test_target_databases_are_rejected(self):
        for db in ("imdb", "stats", "genome", "ergastf1"):
            with self.subTest(db=db), self.assertRaises(ValueError):
                assert_source_databases(["chembl", db])
        assert_source_databases(["chembl", "legalacts", "telstra"])

    def test_macro_skips_null_buckets(self):
        rows = [
            {"database": "chembl", "k": 0, "four_plus_q95": None},
            {"database": "legalacts", "k": 0, "four_plus_q95": 10.0},
            {"database": "telstra", "k": 0, "four_plus_q95": 20.0},
        ]
        self.assertAlmostEqual(macro(rows, "four_plus_q95"), 15.0)
        self.assertIsNone(
            macro(
                [{"database": "chembl", "k": 0, "four_plus_q95": None}], "four_plus_q95"
            )
        )

    def test_ratio_is_geometric_over_databases_not_pooled(self):
        short = {
            "chembl": {"q95": 100.0},
            "legalacts": {"q95": 10.0},
            "telstra": {"q95": 10.0},
        }
        long = {
            "chembl": {"q95": 50.0},
            "legalacts": {"q95": 10.0},
            "telstra": {"q95": 10.0},
        }
        ratio, per_database = geometric_macro_ratio(short, long, "q95")
        # Arithmetic macro of Q95 would give 70/120 = 0.583; the frozen P0 rule
        # is the geometric mean of per-database ratios.
        self.assertAlmostEqual(ratio, (0.5 * 1.0 * 1.0) ** (1 / 3))
        self.assertAlmostEqual(per_database["chembl"], 0.5)

    def test_ratio_skips_undefined_bucket(self):
        short = {"chembl": {"q95": 10.0}, "legalacts": {"q95": None}}
        long = {"chembl": {"q95": 5.0}, "legalacts": {"q95": None}}
        ratio, per_database = geometric_macro_ratio(short, long, "q95")
        self.assertAlmostEqual(ratio, 0.5)
        self.assertNotIn("legalacts", per_database)

    def test_same_run_rejects_different_metadata(self):
        state = {
            "model_config": {"width": 8},
            "metadata": {"seed": 42, "train_databases": ["a"]},
        }
        _assert_same_run([(4000, state), (8000, state)])
        other = {
            "model_config": {"width": 8},
            "metadata": {"seed": 43, "train_databases": ["a"]},
        }
        with self.assertRaisesRegex(ValueError, "different metadata"):
            _assert_same_run([(4000, state), (8000, other)])

    def test_few_shot_rung_requires_positive_inner_steps(self):
        adaptation = {
            "inner_lr": 0.001,
            "inner_steps": 0,
            "adapt_scope": "all",
            "loss": "smooth_l1",
        }
        with self.assertRaisesRegex(ValueError, "inner_steps > 0"):
            evaluate_checkpoint(
                "unused.pt",
                plan={"databases": {}},
                stores={},
                device="cpu",
                batch_size=1,
                ks=[0, 5],
                blocks=1,
                adaptation=adaptation,
            )


if __name__ == "__main__":
    unittest.main()
