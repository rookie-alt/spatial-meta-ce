"""Synthetic engineering certification only; no cached data or training."""

import unittest
from unittest.mock import patch

import numpy as np

from spatial_meta_ce import s8_probe as probe


class NumericalCertificateTests(unittest.TestCase):
    def pair(self, design, target=None):
        if target is None:
            target = np.zeros(design.shape[0])
        adam = {
            "theta": np.zeros(design.shape[1]),
            "objective": float(np.mean(target**2)),
            "updates": 0,
        }
        return adam, probe.solve_ls(design, target)

    def envelope(self, design):
        eps = np.finfo(np.float64).eps
        d = max(design.shape)
        return d * eps / (1 - d * eps) * np.linalg.svd(design, compute_uv=False)[0]

    def test_zero_redundant_and_constant_columns(self):
        z = np.column_stack((np.arange(8.0), np.arange(8.0), np.ones(8), np.zeros(8)))
        design, normalizer = probe.standardize_fit(z)
        adam, ls = self.pair(design, np.arange(8.0))
        cert = probe.certify_pair(design, np.arange(8.0), adam, ls)
        self.assertTrue(cert["solver_verified"])
        self.assertEqual(cert["rank_stability"], "stable")
        self.assertTrue(cert["retained_subspace_pass"])
        self.assertTrue(cert["rank_limited"])
        self.assertFalse(cert["full_linear_optimality"])
        self.assertEqual(int(normalizer["zero_mask"].sum()), 2)

    def test_zero_matrix_has_exact_stable_rank_zero(self):
        design, target = np.zeros((4, 3)), np.arange(4.0)
        adam, ls = self.pair(design, target)
        cert = probe.certify_pair(design, target, adam, ls)
        self.assertTrue(cert["pass"])
        self.assertEqual(cert["rank_stability"], "stable")
        self.assertEqual(cert["spectrum_error_bound"], 0.0)
        forged = dict(ls, singular_values=np.array([1e-200, 0.0, 0.0]), rank=1)
        self.assertFalse(probe.certify_pair(design, target, adam, forged)["pass"])

    def test_roundoff_tail_is_checked_not_ignored(self):
        design = np.diag([1.0, 0.0, 0.0])
        adam, ls = self.pair(design)
        eta = self.envelope(design)
        for multiple, expected in [(0.5, True), (2.0, True), (3.0, False)]:
            with self.subTest(multiple=multiple):
                forged = dict(ls, singular_values=np.array([1.0, multiple * eta, 0.0]))
                cert = probe.certify_pair(design, np.zeros(3), adam, forged)
                self.assertEqual(cert["spectrum_pass"], expected)
                self.assertEqual(cert["pass"], expected)
                self.assertEqual(cert["spectrum_error_bound"], eta)

    def test_reference_tail_uses_same_bound(self):
        design = np.diag([1.0, 0.0])
        adam, ls = self.pair(design)
        eta = self.envelope(design)
        answer = np.linalg.lstsq(design, np.zeros(2), rcond=probe.RCOND)
        for multiple, expected in [(0.5, True), (3.0, False)]:
            result = (*answer[:3], np.array([1.0, multiple * eta]))
            with patch.object(np.linalg, "lstsq", return_value=result):
                cert = probe.certify_pair(design, np.zeros(2), adam, ls)
            self.assertEqual(cert["reference_spectrum_pass"], expected)
            self.assertEqual(cert["pass"], expected)

    def test_cutoff_sides_and_ambiguous_boundary(self):
        eta = self.envelope(np.eye(2))
        for value, expected in [
            (probe.RCOND - 4 * eta, "stable"),
            (probe.RCOND + 4 * eta, "stable"),
            (probe.RCOND - eta / 2, "unknown"),
            (probe.RCOND + eta / 2, "unknown"),
            (probe.RCOND, "unknown"),
        ]:
            with self.subTest(value=value):
                design = np.diag([1.0, value])
                adam, ls = self.pair(design)
                cert = probe.certify_pair(design, np.zeros(2), adam, ls)
                self.assertEqual(cert["rank_stability"], expected)
                self.assertEqual(cert["pass"], expected == "stable")

    def test_cross_algorithm_rank_disagreement_is_unknown_not_success(self):
        design = np.diag([1.0, probe.RCOND])
        adam, ls = self.pair(design)
        # Internally consistent reported rank can disagree near the cutoff.
        forged = dict(ls, rank=2, singular_values=np.array([1.0, probe.RCOND + 1e-16]))
        cert = probe.certify_pair(design, np.zeros(2), adam, forged)
        self.assertEqual(cert["rank_stability"], "unknown")
        self.assertFalse(cert["pass"])

    def test_scale_covariant_not_fixed_atol(self):
        for scale in [1e-150, 1e-20, 1.0, 1e100]:
            with self.subTest(scale=scale):
                design = scale * np.diag([1.0, 1e-10, 0.0])
                adam, ls = self.pair(design)
                eta = self.envelope(design)
                valid = dict(ls, singular_values=np.array([scale, scale * 1e-10, eta]))
                cert = probe.certify_pair(design, np.zeros(3), adam, valid)
                self.assertTrue(cert["pass"])
                self.assertEqual(cert["spectrum_error_bound"], eta)
                bad = dict(
                    ls, singular_values=np.array([scale, scale * 1e-10, 3 * eta])
                )
                self.assertFalse(
                    probe.certify_pair(design, np.zeros(3), adam, bad)["pass"]
                )

    def test_tiny_nonzero_standardization_is_not_zero_column(self):
        design, normalizer = probe.standardize_fit(np.array([[0.0], [2e-14], [4e-14]]))
        self.assertTrue(normalizer["tiny_mask"][0])
        self.assertFalse(normalizer["zero_mask"][0])
        adam, ls = self.pair(design)
        self.assertEqual(ls["rank"], 2)
        self.assertTrue(probe.certify_pair(design, np.zeros(3), adam, ls)["pass"])

    def test_nullspace_coefficients_need_not_match(self):
        design = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        target = np.array([1.0, 2.0, 3.0])
        adam, ls = self.pair(design, target)
        alternative = dict(ls, theta=ls["theta"] + np.array([1.0, -1.0]))
        cert = probe.certify_pair(design, target, adam, alternative)
        self.assertTrue(cert["pass"])
        self.assertTrue(cert["retained_subspace_pass"])

    def test_fake_main_spectrum_rank_prediction_and_objective_rejected(self):
        design, target = np.eye(3), np.arange(3.0)
        adam, ls = self.pair(design, target)
        for forged in [
            dict(ls, singular_values=ls["singular_values"] * 1.001),
            dict(
                ls,
                theta=np.zeros(3),
                objective=float(np.mean(target**2)),
                normalized_normal_residual=float(np.max(target) / 5.0),
            ),
            dict(ls, objective=100.0),
        ]:
            self.assertFalse(probe.certify_pair(design, target, adam, forged)["pass"])
        with self.assertRaises(ValueError):
            probe.certify_pair(design, target, adam, dict(ls, rank=2))

    def test_backend_reconstruction_and_orthogonality_failure(self):
        design = np.eye(3)
        adam, ls = self.pair(design)
        u, s, vh = np.linalg.svd(design, full_matrices=False)
        for result in [(u * 2, s, vh), (u, s * 1.01, vh), (u, s, vh * 2)]:
            with patch.object(np.linalg, "svd", return_value=result):
                cert = probe.certify_pair(design, np.zeros(3), adam, ls)
            self.assertFalse(cert["svd_backend_pass"])
            self.assertFalse(cert["pass"])

    def test_retained_projection_catches_two_matching_forged_solvers(self):
        design = np.diag([1.0, 1e-11])
        target = np.array([0.0, 1.0])
        adam, ls = self.pair(design, target)
        forged = dict(
            ls, theta=np.zeros(2), objective=0.5, normalized_normal_residual=1e-11
        )
        answer = (np.zeros(2), np.array([]), ls["rank"], ls["singular_values"])
        with patch.object(np.linalg, "lstsq", return_value=answer):
            cert = probe.certify_pair(design, target, adam, forged)
        self.assertTrue(cert["reference_prediction_pass"])
        self.assertFalse(cert["retained_subspace_pass"])
        self.assertFalse(cert["pass"])

    def test_full_p_dimension_with_null_direction_stays_unknown(self):
        design = np.zeros((19456, 2))
        design[:, 0] = 1.0
        adam, ls = self.pair(design)
        cert = probe.certify_pair(design, np.zeros(len(design)), adam, ls)
        self.assertGreater(cert["spectrum_gamma"], probe.RCOND)
        self.assertEqual(cert["rank_stability"], "unknown")
        self.assertFalse(cert["pass"])

    def test_envelope_underflow_is_not_an_absolute_tolerance_floor(self):
        design = np.array([[np.nextafter(0.0, 1.0)]])
        adam, ls = self.pair(design)
        with self.assertRaisesRegex(ValueError, "underflow"):
            probe.certify_pair(design, np.zeros(1), adam, ls)

    def test_zero_backend_spectrum_cannot_authenticate_nonzero_design(self):
        design = np.eye(2)
        adam, ls = self.pair(design)
        with (
            patch.object(np.linalg, "svd", return_value=(design, np.zeros(2), design)),
            self.assertRaisesRegex(ValueError, "nonzero"),
        ):
            probe.certify_pair(design, np.zeros(2), adam, ls)

    def test_bad_vector_shapes_or_finite_values_are_rejected(self):
        design = np.eye(2)
        adam, ls = self.pair(design)
        for u, s, vh in [
            (np.ones((1, 2)), np.ones(2), design),
            (design * np.nan, np.ones(2), design),
        ]:
            with (
                patch.object(np.linalg, "svd", return_value=(u, s, vh)),
                self.assertRaises(ValueError),
            ):
                probe.certify_pair(design, np.zeros(2), adam, ls)

    def test_orthogonality_rms_envelope_is_separate_from_reconstruction(self):
        design = np.zeros((3, 3))
        adam, ls = self.pair(design)
        eps = np.finfo(np.float64).eps
        orth_bound = 6 * eps / (1 - 6 * eps)
        for multiple, expected in [(0.5, True), (4.0, False)]:
            u = np.eye(3) * (1 + multiple * orth_bound / 2)
            with patch.object(
                np.linalg, "svd", return_value=(u, np.zeros(3), np.eye(3))
            ):
                cert = probe.certify_pair(design, np.zeros(3), adam, ls)
            self.assertEqual(cert["svd_normalized_orthogonality_bound"], orth_bound)
            self.assertEqual(cert["svd_backend_pass"], expected)
            self.assertEqual(cert["pass"], expected)

    def test_empty_or_nonfinite_backend_spectrum_is_rejected(self):
        design = np.eye(2)
        adam, ls = self.pair(design)
        for values in [np.array([]), np.array([np.nan, 0.0])]:
            with (
                patch.object(np.linalg, "svd", return_value=(design, values, design)),
                self.assertRaises(ValueError),
            ):
                probe.certify_pair(design, np.zeros(2), adam, ls)

    def shifted_cutoff_case(self):
        eta = self.envelope(np.eye(2))
        # The tail interval clears the independent cutoff by one FP64 ULP,
        # but overlaps the shifted caller/reference cutoff. All ranks stay 2.
        tail = np.nextafter(probe.RCOND * (1 + eta) + eta, np.inf)
        design = np.diag([1.0, tail])
        target = np.zeros(2)
        adam, ls = self.pair(design, target)
        shifted = np.array([1.0 + 2 * eta, tail])
        self.assertLessEqual(shifted[0] - 1.0, 2 * eta)
        self.assertGreater(tail - eta, probe.RCOND * (1 + eta))
        self.assertLessEqual(tail - eta, probe.RCOND * (shifted[0] + eta))
        return design, target, adam, ls, shifted

    def test_caller_own_cutoff_interval_overlap_is_unknown(self):
        design, target, adam, ls, shifted = self.shifted_cutoff_case()
        cert = probe.certify_pair(
            design, target, adam, dict(ls, singular_values=shifted)
        )
        self.assertTrue(cert["spectrum_pass"])
        self.assertEqual(cert["rank_stability"], "unknown")
        self.assertFalse(cert["pass"])

    def test_reference_own_cutoff_interval_overlap_is_unknown(self):
        design, target, adam, ls, shifted = self.shifted_cutoff_case()
        result = (ls["theta"], np.array([]), ls["rank"], shifted)
        with patch.object(np.linalg, "lstsq", return_value=result):
            cert = probe.certify_pair(design, target, adam, ls)
        self.assertTrue(cert["reference_spectrum_pass"])
        self.assertEqual(cert["rank_stability"], "unknown")
        self.assertFalse(cert["pass"])


if __name__ == "__main__":
    unittest.main()
