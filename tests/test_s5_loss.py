import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.meta import loss_value
from spatial_meta_ce.train import train


class Store:
    manifest = {"content_hash": "s5-fixture"}
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
        "steps": 3,
        "meta_batch": 4,
        "accumulation": 2,
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


class S5LossTests(unittest.TestCase):
    def test_uniform_weights_preserve_loss_and_gradient(self):
        for size in (1, 8, 10001):
            for weight in (1.0, 2.0):
                with self.subTest(size=size, weight=weight):
                    pred = torch.linspace(-2, 2, size, requires_grad=True)
                    label = torch.zeros(size)
                    reference = loss_value(pred, label)
                    actual = loss_value(
                        pred, label, weights=torch.full_like(pred, weight)
                    )
                    torch.testing.assert_close(actual, reference)
                    torch.testing.assert_close(
                        torch.autograd.grad(actual, pred)[0],
                        torch.autograd.grad(reference, pred)[0],
                    )

    def test_mixed_three_four_tables_loss_and_gradient(self):
        pred = torch.tensor([0.5, -2.0], requires_grad=True)
        counts = torch.tensor([[0, 0, 3, 0], [0, 0, 4, 0]])
        weights = torch.where(counts[:, 2] >= 4, 2.0, 1.0)
        actual = loss_value(pred, torch.zeros(2), weights=weights)
        self.assertAlmostEqual(actual.item(), (0.125 + 2 * 1.5) / 3, places=6)
        actual.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([0.5 / 3, -2 / 3]))

    def test_invalid_weights_and_shapes(self):
        pred = torch.zeros(2)
        for weights in (
            None,
            [],
            "2",
            torch.ones(2, 1),
            torch.ones(1),
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1.0, 1.0]),
            torch.tensor([float("nan"), 1.0]),
            torch.tensor([float("inf"), 1.0]),
        ):
            if weights is None:  # None is the explicitly supported legacy path.
                self.assertEqual(loss_value(pred, pred, weights=None).item(), 0)
                continue
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                loss_value(pred, pred, weights=weights)
        with self.assertRaises(ValueError):
            loss_value(pred, torch.zeros(1), weights=torch.ones(2))
        with self.assertRaises(ValueError):
            loss_value(pred, pred, "mse", weights=torch.ones(2))
        with self.assertRaises(ValueError):
            loss_value(torch.zeros(0), torch.zeros(0), weights=torch.zeros(0))
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(FloatingPointError):
                loss_value(torch.tensor([bad, 0.0]), pred, weights=torch.ones(2))


class S5TrainingTests(unittest.TestCase):
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

    def test_config_rejected_before_training_side_effects(self):
        cases = [
            {"complex_query_weight": v}
            for v in (True, False, None, "2", [], 0, 3, float("nan"), float("inf"))
        ]
        cases += [
            {"complex_query_weight": w, key: value}
            for w in (1, 2)
            for key, value in (("method", "fomaml"), ("loss", "mse"))
        ]
        with patch("spatial_meta_ce.train.seed_all") as seed:
            for i, changes in enumerate(cases):
                output = self.root / str(i)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    train(config(**changes), self.data, output)
                self.assertFalse(output.exists())
            seed.assert_not_called()

    def test_legacy_and_a_keep_original_loss_call(self):
        for index, changes in enumerate(({}, {"complex_query_weight": 1})):
            with patch("spatial_meta_ce.train.loss_value", wraps=loss_value) as spy:
                result = train(config(**changes), self.data, self.root / str(index))
            self.assertEqual(spy.call_count, 12)
            self.assertTrue(
                all(len(c.args) == 3 and not c.kwargs for c in spy.call_args_list)
            )
            if not changes:
                self.assertNotIn("gradient_stats", result)
                _, state = load_checkpoint(self.root / str(index) / "last.pt")
                self.assertNotIn("gradient_stats", state["extra"])
            else:
                self.assertEqual(result["gradient_stats"]["count"], 2)
                _, state = load_checkpoint(self.root / str(index) / "last.pt")
                self.assertEqual(
                    state["metadata"]["training_signature"]["complex_query_weight"], 1
                )
        legacy, _ = load_checkpoint(self.root / "0/last.pt")
        arm_a, _ = load_checkpoint(self.root / "1/last.pt")
        for name in legacy.state_dict():
            self.assertTrue(
                torch.equal(legacy.state_dict()[name], arm_a.state_dict()[name]), name
            )

    def test_real_s5_resume_matches_uninterrupted(self):
        cfg = config(complex_query_weight=2, accumulation=1)
        full = train(cfg, self.data, self.root / "full")
        train(dict(cfg, steps=1), self.data, self.root / "split")
        resumed = train(
            cfg, self.data, self.root / "split", resume=self.root / "split/last.pt"
        )
        self.assertEqual(full["gradient_stats"], resumed["gradient_stats"])
        a, ac = load_checkpoint(self.root / "full/last.pt")
        b, bc = load_checkpoint(self.root / "split/last.pt")
        self.assertEqual(ac["extra"]["counts"], bc["extra"]["counts"])
        for name in a.state_dict():
            self.assertTrue(
                torch.equal(a.state_dict()[name], b.state_dict()[name]), name
            )

    def test_b_erm_counts_and_episode_normalization(self):
        original = Store.batch
        batches = []

        def batch(store, indices, device):
            value = original(store, indices, device)
            batches.append(value)
            return value

        with (
            patch.object(Store, "batch", batch),
            patch("spatial_meta_ce.train.loss_value", wraps=loss_value) as spy,
        ):
            train(config(complex_query_weight=2), self.data, self.root / "b")
        self.assertEqual(len(batches), 12)
        for call, batch in zip(spy.call_args_list, batches):
            weights = call.kwargs["weights"]
            self.assertEqual(weights.dtype, torch.float32)
            torch.testing.assert_close(
                weights, torch.where(batch["counts"][:, 2] >= 4, 2.0, 1.0)
            )
            self.assertEqual(weights.shape, batch["label"].shape)

    def test_signature_gradient_stats_and_resume(self):
        cfg = config(complex_query_weight=2)
        with patch(
            "spatial_meta_ce.train.torch.nn.utils.clip_grad_norm_",
            side_effect=[torch.tensor(1.0), torch.tensor(2.5)],
        ):
            result = train(cfg, self.data, self.root / "run")
        expected = {"count": 2, "total_norm": 3.5, "max_norm": 2.5, "clipped_count": 1}
        self.assertEqual(result["gradient_stats"], expected)
        _, state = load_checkpoint(self.root / "run/last.pt")
        self.assertEqual(state["extra"]["gradient_stats"], expected)
        self.assertEqual(
            state["metadata"]["training_signature"]["complex_query_weight"], 2
        )
        with self.assertRaisesRegex(ValueError, "Resume configuration"):
            train(
                config(complex_query_weight=1, steps=4),
                self.data,
                self.root / "run",
                resume=self.root / "run/last.pt",
            )
        with patch(
            "spatial_meta_ce.train.torch.nn.utils.clip_grad_norm_",
            return_value=torch.tensor(0.5),
        ):
            resumed = train(
                config(complex_query_weight=2, steps=4),
                self.data,
                self.root / "run",
                resume=self.root / "run/last.pt",
            )
        self.assertEqual(
            resumed["gradient_stats"],
            {"count": 3, "total_norm": 4.0, "max_norm": 2.5, "clipped_count": 1},
        )
        _, state = load_checkpoint(self.root / "run/last.pt")
        state["extra"] = {
            k: v for k, v in state["extra"].items() if k != "gradient_stats"
        }
        torch.save(state, self.root / "missing.pt")
        with self.assertRaisesRegex(ValueError, "gradient_stats"):
            train(
                config(complex_query_weight=2, steps=5),
                self.data,
                self.root / "missing",
                resume=self.root / "missing.pt",
            )


if __name__ == "__main__":
    unittest.main()
