import copy
import importlib.util
import math
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn

AVAILABLE = importlib.util.find_spec('spatial_meta_ce') is not None
if AVAILABLE:
    from spatial_meta_ce.data import batch_samples
    from spatial_meta_ce.model import SetEstimator
    from spatial_meta_ce.meta import adapt, forward, meta_loss
    from spatial_meta_ce.metrics import summarize
    from spatial_meta_ce.checkpoints import save_checkpoint, load_checkpoint


def sample(n=2, empty=False, y=10.0):
    return {'groups': [torch.rand(n, 40), torch.rand(n, 40), torch.rand(n, 4),
                       torch.rand(0 if empty else n, 43)], 'true': y, 'pg': 2.0,
            'id': str(n), 'hash': str(n)}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, 'spatial_meta_ce package has not been implemented')
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in SetEstimator().parameters()), 99777)

    def test_initial_pg_and_empty_groups(self):
        b = batch_samples([sample(empty=True), sample(4)], 'cpu')
        m = SetEstimator()
        self.assertTrue(torch.allclose(m(b), b['pg_log']))
        self.assertTrue(torch.isfinite(m(b)).all())

    def test_padding_and_permutation_invariance(self):
        s = sample(3)
        m = SetEstimator()
        nn.init.normal_(m.head[-1].weight, std=0.01)
        reference = m(batch_samples([s], 'cpu'))
        changed = copy.deepcopy(s)
        changed['groups'] = [g[torch.arange(len(g)-1, -1, -1)] for g in s['groups']]
        actual = m(batch_samples([changed, sample(9)], 'cpu'))[:1]
        self.assertTrue(torch.allclose(reference, actual, atol=1e-6))

    def test_mask_hides_large_padding(self):
        m = SetEstimator()
        nn.init.normal_(m.head[-1].weight, std=0.01)
        b = batch_samples([sample(1), sample(4)], 'cpu')
        expected = m(b).detach()
        for g, mask in zip(b['groups'], b['masks']):
            g[~mask] = 1e8
        self.assertTrue(torch.allclose(expected, m(b), atol=1e-6))

    def test_bad_values_rejected(self):
        s = sample()
        s['groups'][0][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            batch_samples([s], 'cpu')
        s = sample()
        s['true'] = -3
        with self.assertRaises(ValueError):
            batch_samples([s], 'cpu')

    def test_no_pg_ablation(self):
        b = batch_samples([sample()], 'cpu')
        m = SetEstimator(use_pg=False)
        nn.init.normal_(m.head[-1].weight, std=0.01)
        a = m(b).detach()
        b['pg_log'] += 100
        self.assertTrue(torch.equal(a, m(b)))

    def test_fomaml_scalar_gradient_and_isolation(self):
        class Scalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.tensor(2.0))
            def forward(self, b):
                return self.w * b['x']
        m = Scalar()
        b = {'x': torch.ones(2), 'label': torch.ones(2)}
        p = adapt(m, b, lr=0.1, steps=1, loss='mse')
        self.assertAlmostEqual(p['w'].item(), 1.8, places=6)
        nn.functional.mse_loss(forward(m, p, b), b['label']).backward()
        self.assertAlmostEqual(m.w.grad.item(), 1.6, places=6)
        self.assertEqual(m.w.item(), 2.0)
        p2 = adapt(m, b, lr=0.1, steps=1, loss='mse')
        self.assertTrue(torch.equal(p['w'], p2['w']))

    def test_adaptation_and_outer_update_change_predictor(self):
        m = SetEstimator()
        support = batch_samples([sample(2, y=200)], 'cpu')
        query = batch_samples([sample(3, y=200)], 'cpu')
        before = m(query).detach().clone()
        params = adapt(m, support, lr=0.01, steps=2)
        self.assertFalse(torch.equal(before, forward(m, params, query)))
        self.assertTrue(torch.equal(before, m(query)))
        opt = torch.optim.Adam(m.parameters(), lr=0.001)
        meta_loss(m, support, query, 0.01, 2).backward()
        opt.step()
        self.assertFalse(torch.equal(before, m(query)))

    def test_query_labels_do_not_change_adaptation(self):
        m = SetEstimator()
        b = batch_samples([sample()], 'cpu')
        q = batch_samples([sample()], 'cpu')
        p = adapt(m, b, lr=0.01, steps=2)
        q['label'] += 100
        p2 = adapt(m, b, lr=0.01, steps=2)
        for key in p:
            self.assertTrue(torch.equal(p[key], p2[key]))

    def test_head_only_and_zero_step(self):
        m = SetEstimator()
        b = batch_samples([sample()], 'cpu')
        p = adapt(m, b, lr=0.01, steps=1, scope='head')
        for name, param in m.named_parameters():
            if not name.startswith('head.'):
                self.assertTrue(torch.equal(param, p[name]))
        self.assertTrue(torch.equal(m(b), forward(m, adapt(m, b, steps=0), b)))

    def test_trajectory_matches_adapt(self):
        from spatial_meta_ce.meta import adapt_trajectory
        m = SetEstimator()
        nn.init.normal_(m.head[-1].weight, std=0.01)
        b = batch_samples([sample(2, y=50), sample(3, y=500)], 'cpu')
        for scope in ('head', 'all'):
            traj = adapt_trajectory(m, b, 0.01, [0, 1, 3], scope)
            self.assertEqual(sorted(traj), [0, 1, 3])
            for steps in (0, 1, 3):
                ref = adapt(m, b, lr=0.01, steps=steps, scope=scope, training=False)
                for key in ref:
                    self.assertTrue(torch.allclose(ref[key], traj[steps][key], atol=1e-6))

    def test_negative_predictions_keep_gradient(self):
        m = SetEstimator()
        with torch.no_grad():
            m.head[-1].bias.fill_(-10)
        b = batch_samples([sample()], 'cpu')
        p = adapt(m, b, lr=0.01, steps=1)
        self.assertGreater(p['head.4.bias'].item(), -10)

    def test_metrics_standard_and_shifted(self):
        r = summarize(np.array([0.0]), np.array([10.0]))
        self.assertAlmostEqual(r['q50'], 10.0)
        self.assertAlmostEqual(r['price_shifted_q50'], 5.5)
        with self.assertRaises(ValueError):
            summarize(np.array([np.nan]), np.array([1.0]))

    def test_checkpoint_roundtrip_strict(self):
        with tempfile.TemporaryDirectory() as d:
            m = SetEstimator()
            opt = torch.optim.Adam(m.parameters())
            b = batch_samples([sample()], 'cpu')
            metadata = {'method': 'fomaml', 'seed': 3, 'train_databases': ['a'],
                        'validation_databases': ['v'], 'manifest_hashes': {'a': 'abc'}}
            save_checkpoint(Path(d)/'ok.pt', m, opt, 12, 2.3, 4, metadata)
            clone, ckpt = load_checkpoint(Path(d)/'ok.pt')
            self.assertTrue(torch.equal(m(b), clone(b)))
            self.assertEqual(ckpt['next_step'], 12)
            self.assertEqual(ckpt['bad_validations'], 4)
            torch.save({'model_state_dict': m.state_dict()}, Path(d)/'bad.pt')
            with self.assertRaises(ValueError):
                load_checkpoint(Path(d)/'bad.pt')


if __name__ == '__main__':
    unittest.main()
