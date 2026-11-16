import importlib
import tempfile
import unittest
from pathlib import Path

import torch


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.modes = []

    def forward(self, batch):
        self.modes.append(self.training)
        return self.weight * batch["x"]


def batch(x, encoded_target):
    return {
        "x": torch.tensor(x),
        "true": torch.expm1(torch.tensor(encoded_target, dtype=torch.float64) - 1),
    }


class NativeGradientTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta123_model")

    def test_first_order_gradient_matches_independent_scalar_calculation(self):
        module = self.module()
        model = Scalar()
        support, query = batch([1.0], [3.0]), batch([2.0], [5.0])
        alpha = 0.1
        loss, _ = module.fomaml_loss(model, support, query, alpha)
        loss.backward()
        fast = 2.0
        for _ in range(5):
            fast -= alpha * 2 * (fast - 3)
        expected_gradient = 2 * (fast * 2 - 5) * 2
        self.assertAlmostEqual(float(model.weight.grad), expected_gradient, places=5)
        self.assertEqual(float(model.weight), 2.0)

    def test_eval_fast_weights_do_not_mutate_base_and_restore_modes(self):
        module = self.module()
        model = Scalar().train()
        params, audit = module.adapt_native(model, batch([1.0], [3.0]), 0.1)
        self.assertTrue(model.training)
        self.assertFalse(any(model.modes))
        self.assertEqual(float(model.weight), 2.0)
        self.assertIsNone(params["weight"].grad_fn)
        self.assertEqual(len(audit["gradient_norms"]), 5)

    def test_zero_steps_and_nonfinite_rejection(self):
        module = self.module()
        model = Scalar()
        params, _ = module.adapt_native(model, batch([1.0], [3.0]), 0.1, steps=0)
        self.assertEqual(float(params["weight"]), float(model.weight))
        with self.assertRaises(FloatingPointError):
            module.adapt_native(model, batch([float("inf")], [3.0]), 0.1)

    def test_wrapper_matches_existing_price_adapter(self):
        module = self.module()
        from spatial_meta_ce.baselines import PriceAdapter
        from spatial_meta_ce.data import batch_samples

        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = module.NativePriceModel().eval()
        sample = {
            "groups": [
                torch.ones(1, 40),
                torch.ones(1, 40) * 2,
                torch.tensor([[3.0, 1.0, 0.5, 0.1]]),
                torch.empty(0, 43),
            ],
            "true": 12.0,
            "pg": 5.0,
            "id": "source/1",
        }
        features = batch_samples([sample])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "price.pt"
            torch.save(model.price.state_dict(), checkpoint)
            old = PriceAdapter(checkpoint, module.CAPACITY)
            with torch.no_grad():
                torch.testing.assert_close(
                    model(features), old(features), rtol=0, atol=0
                )
            with self.assertRaisesRegex(ValueError, "SHA|checkpoint"):
                module.build_model(checkpoint)
            from spatial_meta_ce.manifests import file_hash

            restored = module.build_model(
                checkpoint, expected_sha=file_hash(checkpoint)
            )
            with torch.no_grad():
                torch.testing.assert_close(
                    restored(features), model(features), rtol=0, atol=0
                )
            loss, audit = module.fomaml_loss(restored, features, features, 1e-7)
            loss.backward()
            self.assertEqual(audit["steps"], 5)
            self.assertTrue(
                all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in restored.parameters()
                )
            )
        features["masks"][0] = torch.ones(1, 13, dtype=torch.bool)
        features["groups"][0] = torch.ones(1, 13, 40)
        with self.assertRaisesRegex(ValueError, "capacity"):
            model(features)


if __name__ == "__main__":
    unittest.main()
