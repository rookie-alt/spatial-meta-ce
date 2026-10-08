"""Retained-projection tolerance revision: derivation-based RED/GREEN tests.

Only the retained-projection consistency bound changes. rcond, rank definition,
LS, the other gates and the strict certificate stay untouched. The bound is
derived a priori from the frozen cutoff and problem scale; it never consults an
observed residual.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce import s8_probe as probe


class RetainedToleranceTests(unittest.TestCase):
    def test_bound_derived_from_frozen_cutoff(self):
        singular = np.array([100.0, 10.0, 1.0, 1e-10])
        projection = np.array([3.0, -4.0])
        info = probe.retained_projection_bound(singular, 4, projection)
        eps = np.finfo(np.float64).eps
        self.assertAlmostEqual(info['sigma1'], 100.0, places=12)
        self.assertAlmostEqual(info['sigma_r'], 1e-10, places=22)
        # kappa is sigma_1 / sigma_cutoff = 1/RCOND (an a-priori upper bound).
        self.assertAlmostEqual(info['kappa_cutoff'], 1.0 / probe.RCOND,
                               delta=(1.0 / probe.RCOND) * 1e-9)
        self.assertAlmostEqual(info['projection_norm'], 4.0, places=12)
        scale = max(info['target_scale'], info['projection_norm'])
        expected = probe.RETAINED_PROJECTION_ACCUMULATION * eps * (1.0 / probe.RCOND) * scale
        self.assertAlmostEqual(info['reachable_bound'], expected,
                               delta=expected * 1e-9)
        self.assertEqual(info['tol'], max(info['engineering_floor'], info['reachable_bound']))
        self.assertEqual(info['accumulation_constant'],
                         probe.RETAINED_PROJECTION_ACCUMULATION)

    def test_bound_never_tighter_than_frozen_floor(self):
        # Well-conditioned panel: the old 1e-12 allclose must still be honoured.
        singular = np.array([2.0, 2.0, 2.0])
        projection = np.array([1.0, 0.0, 0.0])
        info = probe.retained_projection_bound(singular, 3, projection, target_scale=1.0)
        self.assertGreaterEqual(info['tol'], info['engineering_floor'])
        self.assertAlmostEqual(info['engineering_floor'],
                               probe.RETAINED_PROJECTION_ATOL
                               + probe.RETAINED_PROJECTION_RTOL * 1.0, delta=1e-20)

    def test_rank_zero_uses_engineering_floor(self):
        singular = np.array([1.0, 0.0])
        projection = np.array([0.0, 0.0])
        info = probe.retained_projection_bound(singular, 0, projection)
        self.assertEqual(info['tol'], probe.RETAINED_PROJECTION_ATOL)
        self.assertEqual(info['reachable_bound'], 0.0)
        ok, residual, _ = probe.retained_projection_check(
            np.array([0.0, 0.0]), projection, singular, 0)
        self.assertTrue(ok)
        self.assertEqual(residual, 0.0)
        ok2, residual2, _ = probe.retained_projection_check(
            np.array([1e-11, 0.0]), projection, singular, 0)
        self.assertFalse(ok2)
        self.assertGreater(residual2, probe.RETAINED_PROJECTION_ATOL)

    def test_residual_slightly_above_bound_still_fails(self):
        singular = np.array([100.0, 1e-10])
        projection = np.array([3.0, -4.0])
        _, _, info = probe.retained_projection_check(
            projection.copy(), projection, singular, 2)
        bound = info['tol']
        over = projection.copy()
        over[0] += bound * 1.5
        ok, residual, info2 = probe.retained_projection_check(
            over, projection, singular, 2)
        self.assertFalse(ok)
        self.assertGreater(residual, info2['tol'])

    def test_residual_at_bound_passes(self):
        singular = np.array([100.0, 1e-10])
        projection = np.array([3.0, -4.0])
        _, _, info = probe.retained_projection_check(
            projection.copy(), projection, singular, 2)
        bound = info['tol']
        at = projection.copy()
        at[0] += bound * (1.0 - 1e-9)
        ok, residual, info2 = probe.retained_projection_check(
            at, projection, singular, 2)
        self.assertTrue(ok)
        self.assertLessEqual(residual, info2['tol'])

    def test_condition_number_anomaly_keeps_strict_unknown(self):
        design = np.diag([1.0, 1e-12])
        target = np.array([1.0, 1.0])
        ls = probe.solve_ls(design, target)
        cert = probe.certify_pair(design, target,
                                  {'theta': ls['theta'], 'objective': ls['objective'],
                                   'updates': 0}, ls)
        self.assertEqual(cert['strict_certificate']['status'], 'unknown')
        self.assertIn(cert['rank_stability'], ('stable', 'unknown'))
        self.assertFalse(cert['full_linear_optimality_verified'])
        self.assertFalse(cert['full_linear_optimality'])

    def test_rcond_and_rank_definition_unchanged(self):
        self.assertEqual(probe.RCOND, 1e-12)
        singular = np.array([1.0, 1e-12, 1e-13])
        expected = int(np.count_nonzero(singular > probe.RCOND * singular[0]))
        rank, _ = probe._rank_and_singular(expected, singular, (3, 3))
        self.assertEqual(rank, expected)

    def test_strict_certificate_fields_preserved(self):
        design = np.eye(3)
        target = np.array([1.0, 2.0, 3.0])
        ls = probe.solve_ls(design, target)
        cert = probe.certify_pair(design, target,
                                  {'theta': ls['theta'], 'objective': ls['objective'],
                                   'updates': 0}, ls)
        self.assertEqual(cert['strict_certificate'], {
            'status': 'unknown',
            'scopes': ['true_rank', 'rigorous_retained_subspace', 'strict_linear_optimality'],
            'reason': 'no_verified_arithmetic_certificate'})
        self.assertFalse(cert['full_linear_optimality'])
        self.assertFalse(cert['full_linear_optimality_verified'])

    def test_bound_does_not_use_observed_residual(self):
        info = probe.retained_projection_bound(
            np.array([5.0, 1.0, 1e-11]), 3, np.array([1.0, 1.0]))
        self.assertIn('eps', info)
        self.assertEqual(info['eps'], np.finfo(np.float64).eps)
        for banned in ('observed_residual', 'calibrated', 'residual'):
            self.assertNotIn(banned, info)

    def test_nonfinite_or_out_of_range_inputs_raise(self):
        with self.assertRaises(ValueError):
            probe.retained_projection_bound(np.array([1.0, np.nan]), 2, np.array([1.0]))
        with self.assertRaises(ValueError):
            probe.retained_projection_bound(np.array([1.0]), 5, np.array([1.0]))
        with self.assertRaises(ValueError):
            probe.retained_projection_check(np.array([1.0, 2.0]), np.array([1.0]),
                                            np.array([1.0, 1.0]), 1)

    def test_strongly_rank_deficient_pair_passes(self):
        rng = np.random.default_rng(7)
        m, n = 64, 8
        u, _ = np.linalg.qr(rng.standard_normal((m, n)))
        v, _ = np.linalg.qr(rng.standard_normal((n, n)))
        singular = np.array([10.0, 5.0, 2.0, 1.0, 1e-3, 1e-6, 1e-9, 1e-13])
        design = (u * singular) @ v.T
        target = rng.standard_normal(m) * 5.0
        ls = probe.solve_ls(design, target)
        cert = probe.certify_pair(design, target,
                                  {'theta': ls['theta'], 'objective': ls['objective'],
                                   'updates': 0}, ls)
        self.assertTrue(cert['retained_subspace_pass'])
        self.assertTrue(cert['reference_retained_subspace_pass'])
        self.assertIn('retained_subspace_check', cert)
        self.assertEqual(cert['retained_subspace_check']['scope'],
                         'fixed_cutoff_fp64_reachable_bound')

    def test_well_conditioned_panel_still_passes_as_before(self):
        design = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = np.array([1.0, 2.0, 3.0])
        ls = probe.solve_ls(design, target)
        cert = probe.certify_pair(design, target,
                                  {'theta': ls['theta'], 'objective': ls['objective'],
                                   'updates': 0}, ls)
        self.assertTrue(cert['retained_subspace_pass'])
        self.assertTrue(cert['reference_retained_subspace_pass'])


if __name__ == '__main__':
    unittest.main()
