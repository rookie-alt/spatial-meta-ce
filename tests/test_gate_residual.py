import unittest

import torch
from torch import nn

from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.meta import meta_loss
from spatial_meta_ce.model import SetEstimator


def sample(n=2, empty=False, y=10.0, pg=2.0):
    return {
        "groups": [
            torch.rand(n, 40),
            torch.rand(n, 40),
            torch.rand(n, 4),
            torch.rand(0 if empty else n, 43),
        ],
        "true": y,
        "pg": pg,
        "id": str(n),
        "hash": str(n),
    }


class GateResidualTests(unittest.TestCase):
    """P1-2: learnable residual gate g(q) in [0, 1] (doc 93 §3)."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_gate_is_opt_in_and_absent_by_default(self):
        model = SetEstimator()
        self.assertNotIn("gate", model.config)
        self.assertFalse(any("gate" in name for name, _ in model.named_parameters()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 99777)

    def test_unknown_gate_mode_rejected(self):
        for mode in ("hard", 1, True, "sign"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                SetEstimator(gate=mode)

    def test_initial_prediction_is_pg_with_gate(self):
        batch = batch_samples([sample(empty=True), sample(4)], "cpu")
        model = SetEstimator(gate="sigmoid")
        self.assertTrue(torch.allclose(model(batch), batch["pg_log"]))

    def test_gate_is_bounded_and_reported(self):
        batch = batch_samples([sample(3), sample(2, empty=True), sample(5)], "cpu")
        model = SetEstimator(gate="sigmoid")
        nn.init.normal_(model.head[-1].weight, std=0.01)
        nn.init.normal_(model.gate[-1].weight, std=1.0)
        with torch.no_grad():
            model.gate[-1].bias.fill_(3.0)
        gate = model.gate_values(batch)
        self.assertEqual(gate.shape, (3,))
        self.assertTrue(((gate >= 0.0) & (gate <= 1.0)).all())
        self.assertTrue(torch.all(gate > 0.9))
        residual = model(batch) - model.output_anchor(batch)
        explicit = model.residual_values(batch) * gate
        torch.testing.assert_close(residual, explicit)

    def test_gate_zero_reproduces_anchor(self):
        batch = batch_samples([sample(4)], "cpu")
        model = SetEstimator(gate="sigmoid")
        nn.init.normal_(model.head[-1].weight, std=0.1)
        with torch.no_grad():
            model.gate[-1].bias.fill_(-60.0)
        self.assertTrue(
            torch.allclose(model(batch), model.output_anchor(batch), atol=1e-5)
        )

    def test_gate_survives_meta_gradient(self):
        model = SetEstimator(gate="sigmoid", arch="sum")
        support = batch_samples([sample(2, y=5.0), sample(3, y=50.0)], "cpu")
        query = batch_samples([sample(4, y=20.0)], "cpu")
        meta_loss(model, support, query, 1e-3, 2).backward()
        self.assertIsNotNone(model.gate[-1].weight.grad)
        self.assertIsNotNone(model.gate[-1].bias.grad)
        self.assertTrue(torch.isfinite(model.gate[-1].weight.grad).all())

    def test_gate_masked_padding_invariance(self):
        model = SetEstimator(gate="sigmoid").eval()
        nn.init.normal_(model.gate[-1].weight, std=0.5)
        alone = model(batch_samples([sample(3)], "cpu"))
        padded_batch = batch_samples([sample(3), sample(7, empty=True)], "cpu")
        for group, mask in zip(padded_batch["groups"], padded_batch["masks"]):
            group[~mask] = 1e8
        self.assertTrue(torch.isfinite(model(padded_batch)).all())
        self.assertTrue(torch.allclose(alone, model(padded_batch)[:1], atol=1e-5))

    def test_gate_parameter_count_is_reported_in_config(self):
        model = SetEstimator(gate="sigmoid")
        self.assertEqual(model.config["gate"], "sigmoid")
        default = SetEstimator()
        extra = sum(p.numel() for p in model.parameters()) - sum(
            p.numel() for p in default.parameters()
        )
        # One Linear(64 -> 1) over the shared trunk output.
        self.assertEqual(extra, 64 + 1)


if __name__ == "__main__":
    unittest.main()
