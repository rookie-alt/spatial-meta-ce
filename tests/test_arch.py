import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import ARCHES, SetEstimator
from spatial_meta_ce.meta import meta_loss
from spatial_meta_ce.checkpoints import save_checkpoint, load_checkpoint

VARIANTS = [a for a in ARCHES if a != 'set']


def sample(n=2, empty=False, y=10.0):
    return {'groups': [torch.rand(n, 40), torch.rand(2 * n, 40), torch.rand(n, 4),
                       torch.rand(0 if empty else n, 43)], 'true': y, 'pg': 2.0,
            'id': str(n), 'hash': str(n)}


def live(arch):
    m = SetEstimator(arch=arch)
    nn.init.normal_(m.head[-1].weight, std=0.01)
    return m


class ArchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_set_config_unchanged(self):
        self.assertEqual(SetEstimator().config, {'width': 64, 'use_pg': True})
        self.assertEqual(SetEstimator(arch='set').config, {'width': 64, 'use_pg': True})

    def test_unknown_arch(self):
        with self.assertRaises(ValueError):
            SetEstimator(arch='nope')

    def test_initial_prediction_is_pg(self):
        b = batch_samples([sample(empty=True), sample(4)], 'cpu')
        for arch in VARIANTS:
            m = SetEstimator(arch=arch)
            self.assertTrue(torch.allclose(m(b), b['pg_log']), arch)

    def test_mask_hides_large_padding(self):
        for arch in VARIANTS:
            m = live(arch).eval()
            b = batch_samples([sample(1), sample(4, empty=True), sample(3)], 'cpu')
            expected = m(b).detach()
            for g, mask in zip(b['groups'], b['masks']):
                g[~mask] = 1e8
            self.assertTrue(torch.isfinite(m(b)).all(), arch)
            self.assertTrue(torch.allclose(expected, m(b), atol=1e-5), arch)

    def test_batch_composition_invariance(self):
        s = sample(3)
        for arch in VARIANTS:
            m = live(arch).eval()
            alone = m(batch_samples([s], 'cpu'))
            padded = m(batch_samples([s, sample(7)], 'cpu'))[:1]
            self.assertTrue(torch.allclose(alone, padded, atol=1e-5), arch)

    def test_permutation_invariance(self):
        s = sample(3)
        for arch in VARIANTS:
            m = live(arch).eval()
            ref = m(batch_samples([s], 'cpu'))
            changed = copy.deepcopy(s)
            groups = [g.flip(0) for g in s['groups']]
            if arch == 'attn_pair':
                # Join order may be permuted as whole pairs; within-pair order is kept.
                f = s['groups'][1].view(-1, 2, 40)
                groups[1] = f.flip(0).reshape(-1, 40)
            changed['groups'] = groups
            got = m(batch_samples([changed], 'cpu'))
            self.assertTrue(torch.allclose(ref, got, atol=1e-5), arch)

    def test_pair_embedding_is_order_sensitive(self):
        s = sample(3)
        m = live('attn_pair').eval()
        nn.init.normal_(m.join_embedding.weight, std=1.0)
        ref = m(batch_samples([s], 'cpu'))
        changed = copy.deepcopy(s)
        # Break pairs: shift fanouts by one position.
        changed['groups'][1] = torch.roll(s['groups'][1], 1, 0)
        got = m(batch_samples([changed], 'cpu'))
        self.assertFalse(torch.allclose(ref, got, atol=1e-5))

    def test_meta_gradient_reaches_all_parameters(self):
        for arch in VARIANTS:
            m = live(arch)
            support = batch_samples([sample(2, y=5.0), sample(3, y=50.0)], 'cpu')
            query = batch_samples([sample(4, y=20.0)], 'cpu')
            meta_loss(m, support, query, 1e-3, 2).backward()
            missing = [n for n, p in m.named_parameters() if p.grad is None]
            self.assertEqual(missing, [], arch)

    def test_checkpoint_roundtrip(self):
        b = batch_samples([sample(2), sample(3)], 'cpu')
        for arch in VARIANTS:
            m = live(arch).eval()
            opt = torch.optim.Adam(m.parameters())
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / 'm.pt'
                save_checkpoint(path, m, opt, 1, 0.0, 0, {}, {})
                loaded, _ = load_checkpoint(path, 'cpu')
            self.assertEqual(loaded.config['arch'], arch)
            self.assertTrue(torch.allclose(m(b), loaded.eval()(b), atol=1e-6), arch)


if __name__ == '__main__':
    unittest.main()
