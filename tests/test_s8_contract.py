"""Synthetic S8 boundary/representation tests; no source/target data read."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce.data import GROUP_DIMS, batch_samples
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.s8_contract import (
    HELD_OUT,
    TRAIN,
    VALIDATION,
    extract_representations,
    panel_indices,
    partition_databases,
)


class SourceContractTests(unittest.TestCase):
    def test_identity_split_is_fixed_and_keeps_original_train_order(self):
        from scripts.s2_source_replay import TRAIN as ORIGINAL

        split = partition_databases(list(ORIGINAL))
        self.assertEqual(split['holdout'], list(HELD_OUT))
        self.assertEqual(split['train'], list(TRAIN))
        self.assertEqual(split['validation'], list(VALIDATION))
        self.assertEqual(len(TRAIN), 19)
        for bad in (list(ORIGINAL)[::-1], list(ORIGINAL) + ['imdb'], list(ORIGINAL)[:-1]):
            with self.assertRaises(ValueError):
                partition_databases(bad)

    def test_panel_identities_are_order_independent_and_disjoint(self):
        hashes = [f'q{i}' for i in range(1224)]
        p, q = panel_indices(TRAIN[0], hashes)
        self.assertEqual(len(p), 1024)
        self.assertEqual(len(q), 200)
        self.assertFalse(set(p) & set(q))
        self.assertEqual(set(p) | set(q), set(range(1224)))
        reversed_hashes = hashes[::-1]
        p2, _ = panel_indices(TRAIN[0], reversed_hashes)
        self.assertEqual([hashes[i] for i in p], [reversed_hashes[i] for i in p2])
        for db in (*HELD_OUT, *VALIDATION, 'imdb', 'unknown'):
            with self.assertRaises(ValueError):
                panel_indices(db, hashes)
        for bad in (hashes[:-1], hashes[:-1] + [hashes[0]], hashes[:-1] + ['']):
            with self.assertRaises(ValueError):
                panel_indices(TRAIN[0], bad)


class RepresentationTests(unittest.TestCase):
    def batch(self):
        samples = []
        for seed in (1, 2, 3):
            rng = np.random.default_rng(seed)
            groups = [rng.random((seed, dim), dtype=np.float32) for dim in GROUP_DIMS]
            samples.append({'id': f'synthetic/{seed}', 'query_hash': f'q{seed}',
                            'groups': groups, 'true': 10.0, 'pg': 4.0})
        return batch_samples(samples)

    def test_hook_capture_is_exact_and_does_not_change_forward_or_weights(self):
        torch.manual_seed(42)
        model = SetEstimator(arch='sum', width=64).eval()
        with torch.no_grad():
            model.head[-1].weight.fill_(0.1)
        batch = self.batch()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        expected = model(batch)
        result = extract_representations(model, batch)
        self.assertEqual(result['x'].shape, (3, 774))
        self.assertEqual(result['h'].shape, (3, 64))
        self.assertTrue(torch.equal(result['raw_log_pred'], expected))
        rebuilt = batch['pg_log'] + model.head[-1](result['h']).squeeze(-1)
        self.assertTrue(torch.equal(result['raw_log_pred'], rebuilt))
        self.assertTrue(torch.equal(result['h'], model.head[:4](result['x'])))
        self.assertFalse(result['x'].requires_grad)
        self.assertFalse(model.head[0]._forward_pre_hooks)
        self.assertFalse(model.head[-1]._forward_pre_hooks)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_nonfrozen_model_and_invalid_architecture_are_rejected(self):
        batch = self.batch()
        for model in (SetEstimator(arch='sum'), SetEstimator(arch='set').eval(),
                      SetEstimator(arch='sum', use_pg=False).eval(),
                      SetEstimator(arch='sum', pg_output_scale=0).eval()):
            with self.assertRaises(ValueError):
                extract_representations(model, batch)

    def test_hooks_are_removed_after_forward_exception(self):
        model = SetEstimator(arch='sum').eval()
        with self.assertRaises((ValueError, KeyError)):
            extract_representations(model, {})
        self.assertFalse(model.head[0]._forward_pre_hooks)
        self.assertFalse(model.head[-1]._forward_pre_hooks)


    def test_first_hook_is_removed_if_second_registration_fails(self):
        model = SetEstimator(arch='sum').eval()
        with patch.object(model.head[-1], 'register_forward_pre_hook', side_effect=RuntimeError('registration')), self.assertRaisesRegex(RuntimeError, 'registration'):
            extract_representations(model, self.batch())
        self.assertFalse(model.head[0]._forward_pre_hooks)


if __name__ == '__main__':
    unittest.main()
