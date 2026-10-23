import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.train import train


class Store:
    manifest = {"content_hash": "p1-fixture"}
    rows = [{"id": str(i), "query_hash": str(i)} for i in range(20)]

    def unique_indices(self):
        return np.arange(20)

    def batch(self, indices, device):
        samples = [
            {
                "groups": [
                    torch.ones(1, 40),
                    torch.ones(1, 40),
                    torch.ones(3 + int(i) % 2, 4),
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
        "steps": 2,
        "meta_batch": 2,
        "accumulation": 1,
        "outer_lr": 0.001,
        "inner_lr": 0.01,
        "inner_steps": 0,
        "adapt_scope": "all",
        "support_sizes": [2],
        "query_size": 4,
        "loss": "smooth_l1",
        "validate_every": 1,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 20,
        "min_delta": 0.0,
        **changes,
    }


class TailAlignedTrainingTests(unittest.TestCase):
    """P1-1/P1-2 wiring in train() (doc 93 §3)."""

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

    def test_pinball_requires_registered_tau(self):
        cases = [
            {"loss": "pinball_tau"},
            {"loss": "pinball_tau", "pinball_tau": 0.0},
            {"loss": "pinball_tau", "pinball_tau": 1.0},
            {"loss": "pinball_tau", "pinball_tau": True},
            {"loss": "pinball_tau", "pinball_tau": "0.9"},
            {"pinball_tau": 0.9},
        ]
        with patch("spatial_meta_ce.train.seed_all") as seed:
            for i, changes in enumerate(cases):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    train(config(**changes), self.data, self.root / str(i))
                self.assertFalse((self.root / str(i)).exists())
            seed.assert_not_called()

    def test_log_l1_trains_and_records_signature(self):
        result = train(config(loss="log_l1"), self.data, self.root / "log")
        self.assertEqual(result["last_step"], 2)
        _, state = load_checkpoint(self.root / "log/last.pt")
        self.assertEqual(state["metadata"]["training_signature"]["loss"], "log_l1")
        self.assertNotIn("pinball_tau", state["metadata"]["training_signature"])

    def test_pinball_trains_and_records_tau(self):
        train(
            config(loss="pinball_tau", pinball_tau=0.9),
            self.data,
            self.root / "pinball",
        )
        _, state = load_checkpoint(self.root / "pinball/last.pt")
        signature = state["metadata"]["training_signature"]
        self.assertEqual(signature["loss"], "pinball_tau")
        self.assertEqual(signature["pinball_tau"], 0.9)

    def test_legacy_path_still_passes_three_positional_arguments(self):
        from spatial_meta_ce.meta import loss_value

        with patch("spatial_meta_ce.train.loss_value", wraps=loss_value) as spy:
            train(config(), self.data, self.root / "legacy")
        self.assertTrue(spy.call_count > 0)
        self.assertTrue(
            all(len(c.args) == 3 and not c.kwargs for c in spy.call_args_list)
        )

    def test_pinball_reaches_the_training_step(self):
        from spatial_meta_ce.meta import loss_value

        with patch("spatial_meta_ce.train.loss_value", wraps=loss_value) as spy:
            train(
                config(loss="pinball_tau", pinball_tau=0.8),
                self.data,
                self.root / "tau",
            )
        self.assertTrue(
            all(c.kwargs == {"tau": 0.8} for c in spy.call_args_list),
            spy.call_args_list,
        )

    def test_s5_and_s6_reject_the_new_losses(self):
        cases = [
            {"complex_query_weight": 2, "loss": "log_l1"},
            {"tail_risk_mode": "mean", "loss": "pinball_tau", "pinball_tau": 0.9},
        ]
        for i, changes in enumerate(cases):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                train(config(**changes), self.data, self.root / f"s{i}")

    def test_gate_model_config_roundtrips_through_checkpoint(self):
        model_config = {"width": 8, "use_pg": True, "gate": "sigmoid"}
        train(
            config(model=model_config, loss="log_l1"),
            self.data,
            self.root / "gate",
        )
        model, state = load_checkpoint(self.root / "gate/last.pt")
        self.assertEqual(state["model_config"]["gate"], "sigmoid")
        self.assertTrue(hasattr(model, "gate_values"))


if __name__ == "__main__":
    unittest.main()
