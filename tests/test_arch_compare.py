import unittest
from pathlib import Path

import numpy as np

from scripts.arch_compare_summary import ALL_ARCHS, paired_verdict, select_cells
from scripts.arch_select import four_plus_bucket, macro_four_plus


def rows_of(spec):
    """spec: list of (n_tables, log_pred, true) -> endpoint rows."""
    return [{"n_tables": n, "log_pred": lp, "true": t} for n, lp, t in spec]


class FourPlusBucketTests(unittest.TestCase):
    def test_only_four_or_more_tables_are_merged(self):
        rows = rows_of(
            [
                (2, 0.0, 10.0),
                (3, 0.0, 10.0),
                (4, np.log(4.0), 10.0),
                (5, np.log(2.0), 10.0),
            ]
        )
        bucket = four_plus_bucket(rows)
        self.assertEqual(bucket["queries"], 2)
        self.assertTrue(bucket["reported"])

    def test_merging_is_not_averaging_of_per_bucket_q95(self):
        # Two buckets with very different tails. Pooled Q95 must sit on the
        # pooled distribution, not be the mean of the two bucket Q95s.
        rows = rows_of(
            [
                (4, 0.0, 10.0),
                (4, 0.0, 10.0),
                (4, 0.0, 10.0),
                (4, 0.0, 10.0),
                (5, np.log(1000.0), 1.0),
            ]
        )
        pooled = four_plus_bucket(rows)["q95"]
        per_bucket = [
            four_plus_bucket(rows_of([(4, 0.0, 10.0)]))["q95"],
            four_plus_bucket(rows_of([(5, np.log(1000.0), 1.0)]))["q95"],
        ]
        self.assertNotAlmostEqual(pooled, float(np.mean(per_bucket)), places=6)

    def test_empty_bucket_is_unreported_not_zero(self):
        rows = rows_of([(2, 0.0, 10.0), (3, 0.0, 10.0)])
        self.assertIsNone(four_plus_bucket(rows))

    def test_macro_skips_unreported_databases(self):
        per_db = {
            "a": {"q95": 10.0, "reported": True},
            "b": {"q95": 30.0, "reported": True},
            "c": None,
        }
        self.assertAlmostEqual(macro_four_plus(per_db), 20.0)

    def test_macro_is_unreported_when_no_database_has_wide_queries(self):
        # A database set with no 4+ table query is a "not reported" state, not
        # an error: aborting here would kill a whole comparison over a source
        # set that simply has no wide queries.
        self.assertIsNone(macro_four_plus({"a": None, "b": None}))


class PairedVerdictTests(unittest.TestCase):
    def test_inconclusive_when_delta_is_within_its_own_spread(self):
        verdict = paired_verdict([0.03, -0.02, 0.01, -0.01])
        self.assertEqual(verdict["verdict"], "inconclusive")

    def test_decisive_when_left_is_lower_on_every_seed(self):
        verdict = paired_verdict([-0.30, -0.32, -0.28, -0.31])
        self.assertEqual(verdict["verdict"], "lower_is_better_left")
        self.assertEqual(verdict["wins_left"], 4)

    def test_decisive_when_right_is_lower_on_every_seed(self):
        verdict = paired_verdict([0.30, 0.32, 0.28, 0.31])
        self.assertEqual(verdict["verdict"], "lower_is_better_right")
        self.assertEqual(verdict["wins_left"], 0)

    def test_zero_variance_zero_mean_is_inconclusive(self):
        self.assertEqual(paired_verdict([0.0, 0.0, 0.0])["verdict"], "inconclusive")

    def test_zero_variance_nonzero_mean_is_decisive(self):
        self.assertEqual(
            paired_verdict([-0.2, -0.2, -0.2])["verdict"], "lower_is_better_left"
        )

    def test_too_few_pairs_cannot_decide(self):
        self.assertEqual(paired_verdict([0.5])["verdict"], "inconclusive")


class CellSelectionTests(unittest.TestCase):
    def _cell(self, arch, seed, value=10.0):
        return {
            "arch": arch,
            "seed": seed,
            "four_plus_q95": {"macro": value, "per_database": {}},
        }

    def test_complete_matrix_is_accepted(self):
        cells = [self._cell(a, s) for a in ALL_ARCHS for s in (42, 43)]
        grid = select_cells(cells)
        self.assertEqual(sorted(grid), sorted(ALL_ARCHS))
        self.assertEqual(sorted(grid["sum"]), [42, 43])

    def test_missing_arch_is_rejected(self):
        cells = [self._cell(a, 42) for a in ALL_ARCHS if a != "attn"]
        with self.assertRaises(ValueError):
            select_cells(cells)

    def test_duplicate_seed_is_rejected(self):
        cells = [self._cell(a, s) for a in ALL_ARCHS for s in (42, 43)]
        cells.append(self._cell("sum", 42))
        with self.assertRaises(ValueError):
            select_cells(cells)

    def test_unreported_macro_is_rejected(self):
        cells = [self._cell(a, s) for a in ALL_ARCHS for s in (42, 43)]
        cells[0]["four_plus_q95"] = {"macro": None, "per_database": {}}
        with self.assertRaises(ValueError):
            select_cells(cells)


class ArchConfigTests(unittest.TestCase):
    def test_every_arch_loads_its_frozen_config(self):
        from scripts.run_arch_compare import ARCHS, arch_config

        for arch in ARCHS:
            config = arch_config(arch, 45)
            self.assertEqual(config["seed"], 45)
            self.assertEqual(config["steps"], 4000)

    def test_set_keeps_the_historical_two_key_model_config(self):
        from scripts.run_arch_compare import arch_config

        self.assertEqual(arch_config("set", 42)["model"], {"width": 64, "use_pg": True})

    def test_unknown_arch_is_rejected(self):
        from scripts.run_arch_compare import arch_config

        with self.assertRaises(ValueError):
            arch_config("attn_pair_v2", 42)

    def test_matrix_covers_every_cell_once(self):
        from scripts.run_arch_compare import ARCHS, SEEDS, matrix

        plan = matrix(Path("/tmp/arch"))
        self.assertEqual(len(plan), len(ARCHS) * len(SEEDS))
        self.assertEqual(
            {(p["arch"], p["seed"]) for p in plan},
            {(a, s) for a in ARCHS for s in SEEDS},
        )
        self.assertTrue(all(p["checkpoint"].endswith("last.pt") for p in plan))

    def test_patience_is_raised_to_the_full_budget(self):
        from scripts.run_arch_compare import STEPS, arch_config

        self.assertEqual(arch_config("sum", 42)["patience"], STEPS)


class RunnerVerificationTests(unittest.TestCase):
    def _record(self, arch, seed, **overrides):
        base = {
            "arch": arch,
            "seed": seed,
            "checkpoint": f"/tmp/{arch}-{seed}/last.pt",
            "checkpoint_kind": "last",
            "optimizer_steps": 4000,
            "train_databases": ["accidents"],
            "validation_databases": ["chembl"],
            "parameters": 132545,
        }
        return {**base, **overrides}

    def test_clean_matrix_is_accepted(self):
        from scripts.run_arch_compare import ARCHS, verify

        records = [self._record(a, s) for a in ARCHS for s in (42, 43)]
        report = verify(records, len(records))
        self.assertEqual(report["cells"], 8)
        self.assertEqual(report["checkpoint_kind"], "last")

    def test_target_database_in_training_is_rejected(self):
        from scripts.run_arch_compare import verify

        records = [self._record("sum", 42, train_databases=["imdb"])]
        with self.assertRaises(ValueError):
            verify(records, 1)

    def test_unfinished_budget_is_rejected(self):
        from scripts.run_arch_compare import verify

        records = [self._record("sum", 42, optimizer_steps=2000)]
        with self.assertRaises(ValueError):
            verify(records, 1)

    def test_best_checkpoint_is_rejected(self):
        from scripts.run_arch_compare import verify

        records = [self._record("sum", 42, checkpoint_kind="best")]
        with self.assertRaises(ValueError):
            verify(records, 1)

    def test_duplicate_cell_is_rejected(self):
        from scripts.run_arch_compare import verify

        records = [self._record("sum", 42), self._record("sum", 42)]
        with self.assertRaises(ValueError):
            verify(records, 2)

    def test_missing_cell_is_rejected(self):
        from scripts.run_arch_compare import verify

        with self.assertRaises(ValueError):
            verify([self._record("sum", 42)], 2)


class EvaluateCheckpointFourPlusTests(unittest.TestCase):
    def test_wide_queries_produce_a_reported_bucket(self):
        """Positive path: a 5-table database must report a 4+ bucket.

        The empty-bucket case is covered above; without this test a bug that
        always returns None would still pass.
        """
        import contextlib
        import io
        import sys
        import tempfile

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from spatial_meta_ce.checkpoints import load_checkpoint
        from spatial_meta_ce.train import train, validation_plan
        from tests.test_mapping_ablation import build_store

        from scripts.arch_select import checkpoint_pool, evaluate_checkpoint

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for db in ("a", "v"):
                build_store(root, db, n_rows=16, n_tables=5)
            cfg = {
                "model": {"width": 64, "use_pg": True},
                "method": "erm",
                "seed": 42,
                "steps": 10,
                "meta_batch": 1,
                "accumulation": 1,
                "outer_lr": 0.0003,
                "inner_lr": 0.001,
                "inner_steps": 0,
                "adapt_scope": "all",
                "support_sizes": [2, 4],
                "query_size": 4,
                "loss": "smooth_l1",
                "validate_every": 100,
                "validation_runs": 1,
                "validation_seed": 1729,
                "patience": 100,
                "min_delta": 0.0001,
                "cache_root": str(root / "cache"),
                "data_config": "configs/data.json",
            }
            data = {
                "cache_root": str(root / "cache"),
                "splits": {"train": ["a"], "validation": ["v"], "test": ["t"]},
            }
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, data, root / "trained")
            model, state = load_checkpoint(root / "trained/last.pt")
            pool = checkpoint_pool(state, data)
            result = evaluate_checkpoint(
                model, cfg, pool, validation_plan(pool, ["v"], cfg), "cpu"
            )
            bucket = result["four_plus_q95"]["zero_shot"]["per_database"]["v"]
            self.assertIsNotNone(bucket)
            self.assertTrue(bucket["reported"])
            self.assertGreater(bucket["queries"], 0)
            self.assertIsNotNone(result["four_plus_q95"]["zero_shot"]["macro"])


if __name__ == "__main__":
    unittest.main()
