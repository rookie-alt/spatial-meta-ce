import copy
import unittest

import torch
from torch import nn

from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.lowdim_adaptation import ridge_last
from spatial_meta_ce.meta import forward
from spatial_meta_ce.model import SetEstimator


def sample(i, y=200.0):
    return {'groups': [torch.rand(2, 40), torch.rand(2, 40),
                       torch.rand(2, 4), torch.rand(2, 43)],
            'true': y, 'pg': 2.0, 'id': str(i)}


class RidgeLastAdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        self.model = SetEstimator(arch='sum', width=64)
        with torch.no_grad():
            nn.init.normal_(self.model.head[-1].weight, std=0.01)
            nn.init.normal_(self.model.head[-1].bias, std=0.01)
        self.support = batch_samples([sample(i) for i in range(20)])
        self.query = batch_samples([sample(20 + i, 300.0) for i in range(7)])

    def test_forward_is_anchor_plus_head_and_features_are_read_only(self):
        before = copy.deepcopy(self.model.state_dict())
        features = self.model.head_features(self.query)
        expected = (self.model.output_anchor(self.query)
                    + self.model.head[-1](features).squeeze(-1))
        self.assertTrue(torch.equal(expected, self.model(self.query)))
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]))

    def test_only_final_linear_changes_and_model_is_not_mutated(self):
        before = copy.deepcopy(self.model.state_dict())
        params = ridge_last(self.model, self.support, weight_decay=0.5)
        changed = {name for name, value in self.model.named_parameters()
                   if not torch.equal(value, params[name])}
        self.assertEqual(changed, {'head.4.weight', 'head.4.bias'})
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]))

    def test_solution_matches_independent_proximal_ridge(self):
        weight_decay, bias_decay = 0.7, 1.3
        params, audit = ridge_last(
            self.model, self.support, weight_decay=weight_decay,
            bias_decay=bias_decay, training=False, return_audit=True)
        x = self.model.head_features(self.support).to(torch.float64)
        y = (self.support['label'] - self.model.output_anchor(self.support)).to(torch.float64)
        before = self.model.head[-1]
        w0 = before.weight.reshape(-1).to(torch.float64)
        b0 = before.bias.reshape(-1)[0].to(torch.float64)
        design = torch.cat([x, torch.ones(x.shape[0], 1, dtype=torch.float64)], dim=1)
        penalty = torch.tensor([weight_decay] * x.shape[1] + [bias_decay],
                               dtype=torch.float64)
        solution = torch.linalg.solve(
            design.T @ design + torch.diag(penalty),
            design.T @ y + penalty * torch.cat([w0, b0.reshape(1)]))
        self.assertTrue(torch.allclose(
            params['head.4.weight'].reshape(-1).to(torch.float64),
            solution[:-1], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(
            params['head.4.bias'].reshape(-1).to(torch.float64),
            solution[-1:], atol=1e-6, rtol=1e-6))
        self.assertFalse(audit['training'])
        self.assertTrue(audit['finite'])

    def test_large_penalty_stays_near_meta_initialization(self):
        original = self.model.head[-1].weight.detach().clone()
        params = ridge_last(self.model, self.support, weight_decay=1e8)
        self.assertTrue(torch.allclose(params['head.4.weight'], original, atol=1e-6))

    def test_training_solution_keeps_encoder_gradient(self):
        params = ridge_last(self.model, self.support, weight_decay=0.1, training=True)
        torch.nn.functional.mse_loss(forward(self.model, params, self.query),
                                     self.query['label']).backward()
        grads = [p.grad for p in self.model.encoders.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.count_nonzero(g).item() > 0 for g in grads))

    def test_invalid_head_and_penalty_are_rejected(self):
        with self.assertRaises(ValueError):
            ridge_last(self.model, self.support, weight_decay=0)
        with self.assertRaises(ValueError):
            ridge_last(self.model, self.support, bias_decay=float('nan'))
        self.model.head[-1] = nn.Linear(64, 2)
        with self.assertRaises(ValueError):
            ridge_last(self.model, self.support)
        self.model.head[-1] = nn.ReLU()
        with self.assertRaises(ValueError):
            ridge_last(self.model, self.support)

    def test_zero_anchor_and_attention_paths_remain_supported(self):
        for kwargs in ({'use_pg': False}, {'pg_output_scale': 0.0},
                       {'arch': 'attn'}):
            model = SetEstimator(width=64, **kwargs)
            with torch.no_grad():
                nn.init.normal_(model.head[-1].weight, std=0.01)
            params, audit = ridge_last(
                model, self.support, weight_decay=0.5,
                training=False, return_audit=True)
            prediction = forward(model, params, self.query)
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertGreater(audit['condition_number'], 0)
            expect_zero_anchor = (not kwargs.get('use_pg', True)
                                  or kwargs.get('pg_output_scale', 1.0) == 0.0)
            if expect_zero_anchor:
                self.assertFalse(torch.count_nonzero(
                    model.output_anchor(self.support)).item())


if __name__ == '__main__':
    unittest.main()
