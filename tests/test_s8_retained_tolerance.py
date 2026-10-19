"""Retained-projection tolerance revision: real retained condition number.

Only the retained-projection consistency bound changes. rcond, rank definition,
LS, the other gates and the strict certificate stay untouched. The bound is
derived a priori from the retained spectrum (kappa_retained = sigma_1/sigma_r)
and the problem scale; it never consults an observed residual.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce import s8_probe as probe  # noqa: E402


class RetainedToleranceTests(unittest.TestCase):
    def test_bound_uses_real_retained_condition_number(self):
        # kappa_retained = sigma_1 / sigma_r = 100 / 10 = 10, NOT 1/RCOND.
        singular = np.array([100.0, 10.0, 1e-10])
        projection = np.array([3.0, -4.0])
        info = probe.retained_projection_bound(singular, 2, projection)
        eps = np.finfo(np.float64).eps
        self.assertAlmostEqual(info['sigma1'], 100.0, places=12)
        self.assertAlmostEqual(info['sigma_r'], 10.0, places=12)
        self.assertAlmostEqual(info['kappa_retained'], 10.0, delta=1e-9)
        self.assertLess(info['kappa_retained'], 1.0 / probe.RCOND)
        self.assertAlmostEqual(info['projection_norm'], 4.0, places=12)
        scale = max(info['target_scale'], info['projection_norm'])
        expected = probe.RETAINED_PROJECTION_ACCUMULATION * eps * 10.0 * scale
        self.assertAlmostEqual(info['reachable_bound'], expected, delta=expected * 1e-9)
        self.assertEqual(info['tol'],
                         max(info['engineering_floor'], info['reachable_bound']))
        self.assertEqual(info['accumulation_constant'],
                         probe.RETAINED_PROJECTION_ACCUMULATION)

    def test_well_conditioned_panel_uses_tight_bound_not_rcond(self):
        # A perfectly conditioned retained block must NOT inherit the constant
        # 1/RCOND amplification; the reachable bound collapses and the frozen
        # 1e-12 floor dominates.
        singular = np.array([2.0, 2.0, 2.0])
        projection = np.array([1.0, 0.0, 0.0])
        info = probe.retained_projection_bound(
            singular, 3, projection, target_scale=1.0)
        eps = np.finfo(np.float64).eps
        tight = probe.RETAINED_PROJECTION_ACCUMULATION * eps * 1.0 * 1.0
        self.assertAlmostEqual(info['kappa_retained'], 1.0, delta=1e-12)
        self.assertAlmostEqual(info['reachable_bound'], tight, delta=tight * 1e-9)
        self.assertAlmostEqual(info['engineering_floor'],
                               probe.RETAINED_PROJECTION_ATOL
                               + probe.RETAINED_PROJECTION_RTOL * 1.0, delta=1e-20)
        self.assertEqual(info['tol'], info['engineering_floor'])
        self.assertLess(info['reachable_bound'], info['engineering_floor'])

    def test_bound_never_wider_than_frozen_cutoff_bound(self):
        # sigma_r > sigma_cutoff = RCOND*sigma_1 by the frozen rank definition,
        # so kappa_retained <= 1/RCOND and the new bound is never wider.
        singular = np.array([161.161, 4.0, 1.3104])
        projection = np.array([4.875, -1.0])
        info = probe.retained_projection_bound(
            singular, 3, projection, target_scale=8.334347248077393)
        eps = np.finfo(np.float64).eps
        frozen = probe.RETAINED_PROJECTION_ACCUMULATION * eps * (
            1.0 / probe.RCOND) * info['amplification_scale']
        self.assertLessEqual(info['reachable_bound'], frozen)
        self.assertLessEqual(info['kappa_retained'], 1.0 / probe.RCOND)

    def test_strongly_rank_deficient_panel_uses_wider_bound(self):
        # kappa_retained = 1e6 amplifies the reachable bound; a residual above
        # the frozen floor but within the a-priori bound must pass.
        singular = np.array([1.0, 1e-6, 1e-13])
        projection = np.array([1.0, 0.0, 0.0])
        info = probe.retained_projection_bound(
            singular, 2, projection, target_scale=1.0)
        self.assertAlmostEqual(info['kappa_retained'], 1e6, delta=1e-3)
        self.assertGreater(info['reachable_bound'], info['engineering_floor'])
        self.assertEqual(info['tol'], info['reachable_bound'])
        # Residual above the frozen floor but well inside the a-priori bound.
        over_floor = 1e-11
        self.assertGreater(over_floor, info['engineering_floor'])
        self.assertLess(over_floor, info['reachable_bound'])
        predicted = projection + np.array([over_floor, 0.0, 0.0])
        ok, residual, check = probe.retained_projection_check(
            predicted, projection, singular, 2, target_scale=1.0)
        self.assertGreater(residual, info['engineering_floor'])
        self.assertTrue(ok)
        self.assertEqual(check['tol'], info['tol'])

    def test_rank_zero_uses_engineering_floor(self):
        singular = np.array([1.0, 0.0])
        projection = np.array([0.0, 0.0])
        info = probe.retained_projection_bound(singular, 0, projection)
        self.assertEqual(info['tol'], probe.RETAINED_PROJECTION_ATOL)
        self.assertEqual(info['reachable_bound'], 0.0)
        self.assertEqual(info['kappa_retained'], 0.0)
        ok, residual, _ = probe.retained_projection_check(
            np.array([0.0, 0.0]), projection, singular, 0)
        self.assertTrue(ok)
        self.assertEqual(residual, 0.0)
        ok2, residual2, _ = probe.retained_projection_check(
            np.array([1e-11, 0.0]), projection, singular, 0)
        self.assertFalse(ok2)
        self.assertGreater(residual2, probe.RETAINED_PROJECTION_ATOL)

    def test_sigma_r_at_or_below_frozen_cutoff_is_fail_closed(self):
        # An inconsistent retained block (sigma_r not strictly above the frozen
        # cutoff) must fail closed instead of inflating the tolerance.
        singular = np.array([1.0, 1e-13])
        projection = np.array([1.0, 0.0])
        with self.assertRaises(ValueError):
            probe.retained_projection_bound(singular, 2, projection)

    def test_nonfinite_sigma_r_or_inputs_raise(self):
        with self.assertRaises(ValueError):
            probe.retained_projection_bound(
                np.array([1.0, np.nan]), 2, np.array([1.0]))
        with self.assertRaises(ValueError):
            probe.retained_projection_bound(
                np.array([1.0, np.inf]), 2, np.array([1.0]))
        with self.assertRaises(ValueError):
            probe.retained_projection_bound(np.array([1.0]), 5, np.array([1.0]))
        with self.assertRaises(ValueError):
            probe.retained_projection_check(
                np.array([1.0, 2.0]), np.array([1.0]),
                np.array([1.0, 1.0]), 1)

    def test_residual_slightly_above_new_bound_still_fails(self):
        singular = np.array([100.0, 10.0, 1e-13])
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

    def test_residual_at_new_bound_passes(self):
        # kappa_retained = 100/1e-5 = 1e7 drives the reachable term far above
        # the frozen floor, so a residual just inside the bound must pass.
        singular = np.array([100.0, 1e-5, 1e-13])
        projection = np.array([3.0, -4.0])
        _, _, info = probe.retained_projection_check(
            projection.copy(), projection, singular, 2)
        self.assertAlmostEqual(info['kappa_retained'], 1e7, delta=1e-3)
        self.assertEqual(info['tol'], info['reachable_bound'])
        self.assertGreater(info['reachable_bound'], info['engineering_floor'])
        bound = info['tol']
        at = projection.copy()
        at[0] += bound * (1.0 - 1e-6)
        ok, residual, info2 = probe.retained_projection_check(
            at, projection, singular, 2)
        self.assertTrue(ok)
        self.assertLessEqual(residual, info2['tol'])
        self.assertGreater(residual, bound * 0.5)

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

    def test_anomalous_condition_number_does_not_falsely_pass(self):
        # Near-singular retained block (sigma_r just above the frozen cutoff,
        # kappa_retained ~ 1e11): the bound widens, yet a grossly wrong solve
        # must still fail the retained check and stay strict_certificate=unknown
        # -- an anomalous condition number widens the tolerance, it never turns
        # a false solve into a pass.
        design = np.diag([1.0, 1e-11])
        target = np.array([1.0, 1.0])
        ls = probe.solve_ls(design, target)
        self.assertEqual(ls['rank'], 2)
        wrong = dict(ls, theta=ls['theta'] + np.array(
            [0.0, ls['theta'][1] * 1e-3]))
        cert = probe.certify_pair(
            design, target,
            {'theta': ls['theta'], 'objective': ls['objective'], 'updates': 0},
            wrong)
        self.assertFalse(cert['retained_subspace_pass'])
        self.assertFalse(cert['engineering_verified'])
        self.assertFalse(cert['pass'])
        self.assertEqual(cert['strict_certificate']['status'], 'unknown')
        self.assertFalse(cert['full_linear_optimality_verified'])
        info = cert['retained_subspace_check']['reported_bound']
        self.assertAlmostEqual(info['kappa_retained'], 1e11, delta=1e6)
        self.assertGreater(info['reachable_bound'], info['engineering_floor'])

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
        singular = np.array([5.0, 1.0, 1e-11])
        projection = np.array([1.0, 1.0])
        info = probe.retained_projection_bound(singular, 3, projection)
        self.assertIn('eps', info)
        self.assertEqual(info['eps'], np.finfo(np.float64).eps)
        for banned in ('observed_residual', 'calibrated', 'residual'):
            self.assertNotIn(banned, info)
        # Two different observed solve errors must yield the identical bound.
        _, _, info_a = probe.retained_projection_check(
            projection + 1e-9, projection, singular, 3)
        _, _, info_b = probe.retained_projection_check(
            projection - 1e-7, projection, singular, 3)
        self.assertEqual(info_a['tol'], info_b['tol'])
        self.assertEqual(info_a['tol'], info['tol'])

    def test_bound_signature_and_outputs_exclude_residual_inputs(self):
        # Structural guard: the bound must not accept or emit a residual.
        import inspect

        params = set(inspect.signature(
            probe.retained_projection_bound).parameters)
        self.assertEqual(params,
                         {'singular', 'rank', 'projection', 'target_scale'})
        self.assertFalse(params & {'residual', 'observed_residual',
                                   'predicted', 'calibrated'})
        singular = np.array([9.0, 3.0, 1e-11])
        projection = np.array([1.0, 2.0])
        info = probe.retained_projection_bound(singular, 2, projection)
        self.assertFalse(set(info) & {'residual', 'observed_residual',
                                      'predicted', 'calibrated'})

    def test_bound_is_insensitive_to_unretained_spectrum(self):
        # Only the retained block may drive the bound: perturbing discarded
        # singular values (rank:) must not move tol, while moving sigma_r must
        # (kappa_retained is chosen so the reachable term dominates the floor).
        singular = np.array([1e3, 1.0, 1e-6, 1e-13, 1e-15])
        projection = np.array([1.0, -0.5])
        base = probe.retained_projection_bound(
            singular, 3, projection, target_scale=1.0)
        self.assertEqual(base['tol'], base['reachable_bound'])
        self.assertGreater(base['reachable_bound'], base['engineering_floor'])
        perturbed = singular.copy()
        perturbed[3:] = np.array([7e-14, 3e-16])
        other = probe.retained_projection_bound(
            perturbed, 3, projection, target_scale=1.0)
        self.assertEqual(base['tol'], other['tol'])
        self.assertEqual(base['reachable_bound'], other['reachable_bound'])
        moved = singular.copy()
        moved[2] = 2e-6
        tighter = probe.retained_projection_bound(
            moved, 3, projection, target_scale=1.0)
        self.assertEqual(tighter['kappa_retained'], base['kappa_retained'] / 2.0)
        self.assertAlmostEqual(tighter['tol'], base['tol'] / 2.0, delta=base['tol'] * 1e-9)

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
        check = cert['retained_subspace_check']
        self.assertEqual(check['scope'],
                         'real_retained_condition_number_fp64_reachable_bound')
        self.assertLessEqual(check['reported_bound']['kappa_retained'],
                             1.0 / probe.RCOND)

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
