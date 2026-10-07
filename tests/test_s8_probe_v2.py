"""Approved engineering split: synthetic matrices only, no source data."""

import unittest
from unittest.mock import patch

import numpy as np

from spatial_meta_ce import s8_probe as probe


class ProbeV2Tests(unittest.TestCase):
    def certificate(self, design):
        target = np.zeros(len(design))
        adam = {'theta': np.zeros(design.shape[1]), 'objective': 0.0, 'updates': 0}
        ls = probe.solve_ls(design, target)
        return probe.certify_pair(design, target, adam, ls)

    def test_large_null_direction_engineering_does_not_promote_old_pass(self):
        design = np.zeros((19456, 2))
        design[0, 0] = 1.0
        cert = self.certificate(design)
        self.assertTrue(cert['engineering_verified'])
        self.assertTrue(cert['observed_rank_agreement'])
        self.assertEqual(cert['observed_ranks'], {'reported_ls': 1, 'svd': 1, 'reference_ls': 1})
        self.assertEqual(cert['rank_stability'], 'unknown')
        self.assertFalse(cert['pass'])
        self.assertFalse(cert['solver_verified'])
        self.assertTrue(cert['numerical_rank_limited'])
        self.assertFalse(cert['fullrank_numeric_solver'])
        self.assertFalse(cert['full_linear_optimality'])
        self.assertEqual(cert['strict_certificate'], {
            'status': 'unknown',
            'scopes': ['true_rank', 'rigorous_retained_subspace', 'strict_linear_optimality'],
            'reason': 'no_verified_arithmetic_certificate',
        })

    def test_small_boundary_preserves_old_gate(self):
        cert = self.certificate(np.diag([1.0, probe.RCOND]))
        self.assertTrue(cert['engineering_verified'])
        self.assertEqual(cert['rank_stability'], 'unknown')
        self.assertFalse(cert['pass'])

    def test_rank_disagreement_is_hard_engineering_failure(self):
        design = np.diag([1.0, probe.RCOND])
        target = np.zeros(2)
        adam = {'theta': np.zeros(2), 'objective': 0., 'updates': 0}
        ls = probe.solve_ls(design, target)
        reported = dict(ls, rank=2, singular_values=np.array([1., probe.RCOND + 1e-16]))
        cert = probe.certify_pair(design, target, adam, reported)
        self.assertFalse(cert['observed_rank_agreement'])
        self.assertFalse(cert['engineering_verified'])
        self.assertFalse(cert['pass'])

    def test_stable_engineering_is_still_not_strict_science(self):
        cert = self.certificate(np.eye(2))
        self.assertTrue(cert['engineering_verified'])
        self.assertTrue(cert['pass'])
        self.assertEqual(cert['strict_certificate']['status'], 'unknown')
        self.assertFalse(cert['full_linear_optimality_verified'])

    def test_all_existing_numeric_hard_flags_remain_in_conjunction(self):
        design = np.eye(2)
        target = np.zeros(2)
        adam = {'theta': np.zeros(2), 'objective': 0., 'updates': 0}
        ls = probe.solve_ls(design, target)
        forged = dict(ls, singular_values=ls['singular_values'] * 2)
        cert = probe.certify_pair(design, target, adam, forged)
        self.assertFalse(cert['engineering_verified'])
        for key, bad in [('objective', 1.), ('normalized_normal_residual', 1.),
                         ('theta', np.ones(2))]:
            with self.subTest(key=key):
                cert = probe.certify_pair(design, target, adam, dict(ls, **{key: bad}))
                self.assertFalse(cert['engineering_verified'])
        with patch.object(np.linalg, 'svd', return_value=(2 * design, np.ones(2), design)):
            cert = probe.certify_pair(design, target, adam, ls)
        self.assertFalse(cert['svd_backend_pass'])
        self.assertFalse(cert['engineering_verified'])

    def test_matching_forged_solvers_cannot_bypass_retained_projection(self):
        design, target = np.diag([1., 1e-11]), np.array([0., 1.])
        adam = {'theta': np.zeros(2), 'objective': .5, 'updates': 0}
        ls = probe.solve_ls(design, target)
        forged = dict(ls, theta=np.zeros(2), objective=.5, normalized_normal_residual=1e-11)
        with patch.object(np.linalg, 'lstsq', return_value=(np.zeros(2), np.array([]), 2, ls['singular_values'])):
            cert = probe.certify_pair(design, target, adam, forged)
        self.assertTrue(cert['reference_prediction_pass'])
        self.assertFalse(cert['retained_subspace_pass'])
        self.assertFalse(cert['engineering_verified'])


if __name__ == '__main__':
    unittest.main()
