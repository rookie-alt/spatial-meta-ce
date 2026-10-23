import copy
import unittest

import torch
from torch import nn

from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.meta import adapt, adapt_trajectory, forward, meta_loss
from spatial_meta_ce.model import SetEstimator


def sample(i, y=200.0):
    return {
        "groups": [
            torch.rand(2, 40),
            torch.rand(2, 40),
            torch.rand(2, 4),
            torch.rand(2, 43),
        ],
        "true": y,
        "pg": 2.0,
        "id": str(i),
    }


class LastLayerAdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(37)
        self.model = SetEstimator(arch="sum", width=64)
        self.support = batch_samples([sample(i) for i in range(5)])
        self.query = batch_samples([sample(20 + i, 300.0) for i in range(7)])

    def test_sum_parameter_counts(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 132545)
        self.assertEqual(sum(p.numel() for p in self.model.head.parameters()), 107521)
        self.assertEqual(sum(p.numel() for p in self.model.head[-1].parameters()), 65)

    def test_last_updates_only_final_linear_without_mutating_model(self):
        original = copy.deepcopy(self.model.state_dict())
        params = adapt(self.model, self.support, lr=0.01, steps=3, scope="last")
        changed = {
            name
            for name, p in self.model.named_parameters()
            if not torch.equal(p, params[name])
        }
        self.assertEqual(changed, {"head.4.weight", "head.4.bias"})
        for name, value in original.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]))
        self.assertFalse(
            torch.equal(self.model(self.query), forward(self.model, params, self.query))
        )

    def test_last_evaluation_trajectory_matches_independent_adaptation(self):
        trajectory = adapt_trajectory(
            self.model, self.support, 0.01, [0, 1, 3], scope="last"
        )
        for steps in (0, 1, 3):
            expected = adapt(
                self.model, self.support, 0.01, steps, "last", training=False
            )
            for name in expected:
                self.assertTrue(torch.equal(expected[name], trajectory[steps][name]))

    def test_last_episode_restarts_and_query_labels_do_not_affect_support_update(self):
        first = adapt(self.model, self.support, 0.01, 3, "last", training=False)
        self.query["label"] += 100
        second = adapt(self.model, self.support, 0.01, 3, "last", training=False)
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))
            self.assertTrue(second[name].is_leaf)

    def test_last_inner_loop_keeps_encoder_outer_gradient(self):
        objective = meta_loss(self.model, self.support, self.query, 0.01, 3, "last")
        objective.backward()
        encoder_grads = [p.grad for p in self.model.encoders.parameters()]
        self.assertTrue(
            all(g is not None and torch.isfinite(g).all() for g in encoder_grads)
        )
        self.assertTrue(any(torch.count_nonzero(g).item() > 0 for g in encoder_grads))

    def test_last_scope_requires_linear_output_head(self):
        invalid = nn.Linear(2, 1)
        with self.assertRaises(ValueError):
            adapt(
                invalid, {"x": torch.ones(1, 2), "label": torch.ones(1)}, scope="last"
            )
        self.model.head[-1] = nn.ReLU()
        with self.assertRaises(ValueError):
            adapt(self.model, self.support, scope="last")

    def test_unknown_scope_rejected(self):
        with self.assertRaises(ValueError):
            adapt(self.model, self.support, scope="encoder")


if __name__ == "__main__":
    unittest.main()
