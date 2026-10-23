import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from test_protocol import fixtures
from torch import nn

from spatial_meta_ce.baselines import price_log_predictions
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import FeatureStore, batch_samples
from spatial_meta_ce.evaluate import support_plan
from spatial_meta_ce.train import train, validation_plan


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_support_is_nested_and_excludes_test_hashes(self):
        class Store:
            def __init__(self, hashes):
                self.rows = [{"query_hash": h} for h in hashes]

            unique_indices = FeatureStore.unique_indices

        s = Store(["a", "a", "b", "c", "d", "e", "f"])
        t = Store(["b", "g"])
        chosen, report = support_plan(s, t, 5, 2026, 0)
        hashes = [s.rows[i]["query_hash"] for i in chosen]
        self.assertNotIn("b", hashes)
        self.assertEqual(len(set(hashes)), 5)
        self.assertEqual(report["overlapping_unique_removed"], 1)
        chosen2, _ = support_plan(s, t, 5, 2026, 0)
        self.assertTrue(np.array_equal(chosen, chosen2))

    def test_validation_plan_is_fixed_and_uses_all_shots(self):
        from spatial_meta_ce.data import StorePool

        with tempfile.TemporaryDirectory() as d:
            cfg, dc = fixtures(Path(d))
            cfg["support_sizes"] = [1, 5]
            a = validation_plan(StorePool(dc["cache_root"]), ["v"], cfg)
            b = validation_plan(StorePool(dc["cache_root"]), ["v"], cfg)
            self.assertEqual(a, b)
            self.assertEqual({ep["k"] for ep in a}, {1, 5})
            for ep in a:
                self.assertFalse(set(ep["support_ids"]) & set(ep["query_ids"]))

    def test_resume_rejects_changed_hyperparameters(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, dc = fixtures(root)
            train(cfg, dc, root / "run")
            cfg["inner_lr"] = 0.5
            with self.assertRaises(ValueError):
                train(cfg, dc, root / "run", resume=root / "run/last.pt")

    def test_price_output_transform(self):
        cards = np.array([1.0, 10.0, 1e10])
        pred = price_log_predictions(np.log1p(cards) + 1)
        self.assertTrue(np.allclose(pred, np.log(cards)))
        self.assertEqual(price_log_predictions(np.array([-5.0]))[0], 0.0)

    def test_adapted_checkpoint_cannot_be_reused_as_meta_initialization(self):
        from spatial_meta_ce.checkpoints import save_checkpoint
        from spatial_meta_ce.evaluate import evaluate

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, dc = fixtures(root)
            train(cfg, dc, root / "train")
            model, state = load_checkpoint(root / "train/best.pt")
            meta = dict(state["metadata"], adapted_database="t")
            save_checkpoint(root / "adapted.pt", model, None, 0, 1.0, 0, meta)
            ec = {
                "shots": [0, 5],
                "support_runs": 1,
                "support_seed": 1,
                "inner_lr": 0.01,
                "inner_steps": 1,
                "adapt_scope": "all",
                "loss": "smooth_l1",
                "batch_size": 8,
                "latency_repetitions": 1,
                "model": cfg["model"],
            }
            with self.assertRaises(ValueError):
                evaluate(ec, dc, root / "eval", checkpoint=root / "adapted.pt")

    def test_price_validation_decodes_legacy_log_scale(self):
        from spatial_meta_ce.baselines import PriceAdapter
        from spatial_meta_ce.data import StorePool
        from spatial_meta_ce.train import validate

        class KnownPrice(PriceAdapter):
            def __init__(self):
                nn.Module.__init__(self)
                self.bias = nn.Parameter(torch.zeros(()))

            def forward(self, batch):
                return torch.log1p(batch["true"]).float() + 1 + self.bias

        with tempfile.TemporaryDirectory() as d:
            cfg, dc = fixtures(Path(d))
            cfg["inner_steps"] = 0
            pool = StorePool(dc["cache_root"])
            episodes = validation_plan(pool, ["v"], cfg)
            score, _ = validate(KnownPrice(), pool, episodes, cfg, "cpu")
            self.assertLess(abs(score), 1e-5)

    def test_cli_help(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "spatial_meta_ce.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("prepare", result.stdout)

    def test_baseline_forward_padding_matches_original(self):
        from spatial_meta_ce.baselines import PriceAdapter
        from vendor.price_baseline.encoder import RegressionModel

        with tempfile.TemporaryDirectory() as d:
            original = RegressionModel(
                2, 2, 2, 2, 40, 4, 43, 512, 1024, 1, 256, 6, 8, 0.2
            )
            original.eval()
            path = Path(d) / "p.pt"
            torch.save(
                {"module." + k: v for k, v in original.state_dict().items()}, path
            )
            adapter = PriceAdapter(path, [2, 2, 2, 2])
            groups = [torch.rand(1, dim) for dim in (40, 40, 4, 43)]
            s = {"groups": groups, "true": 10.0, "pg": 2.0, "id": "x"}
            batch = batch_samples([s])
            parts = []
            for j, (g, dim) in enumerate(zip(groups, (40, 40, 4, 43))):
                padding = torch.full((1, dim), -1000.0)
                if j == 2:
                    padding[:, 0] = 0
                parts.append(torch.cat([g, padding]).flatten())
            x = torch.cat(parts)[None, :]
            mask = torch.tensor([[1, 1, 0, 1, 0, 1, 0, 1, 0]])
            c = torch.ones(1, 1)
            with torch.no_grad():
                expected = original(
                    x,
                    torch.tensor([[np.log(3) + 1]], dtype=torch.float32),
                    mask,
                    c,
                    c,
                    c,
                    c,
                )
                actual = adapter(batch)
            self.assertTrue(
                torch.allclose(actual, expected.flatten(), atol=1e-5, rtol=1e-5)
            )


if __name__ == "__main__":
    unittest.main()
