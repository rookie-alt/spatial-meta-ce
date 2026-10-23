"""Tiny synthetic-only S8 numerical contracts; no data/model/runner access."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce import s8_probe as probe


class S8ProbeTests(unittest.TestCase):
    def fixture(self):
        z = np.array([[0, 1], [1, 0], [2, 1], [3, -1]], dtype=np.float32)
        design, _ = probe.standardize_fit(z)
        theta = np.array([0.7, -0.4, 0.2], dtype=np.float64)
        return design, design @ theta, theta

    def test_standardize_population_bias_and_fp64(self):
        z = np.array([[1, 5, 0], [3, 5, 1e-14]], dtype=np.float32)
        design, norm = probe.standardize_fit(z)
        self.assertEqual(design.dtype, np.float64)
        self.assertEqual(norm["mean"].dtype, np.float64)
        self.assertEqual(norm["std"].dtype, np.float64)
        np.testing.assert_allclose(norm["mean"], z.astype(np.float64).mean(axis=0))
        np.testing.assert_array_equal(norm["scale"][:2], [1, 1])
        self.assertEqual(norm["scale"][2], 1e-12)
        np.testing.assert_allclose(
            norm["std"], z.astype(np.float64).std(axis=0, ddof=0)
        )
        np.testing.assert_array_equal(norm["zero_mask"], [False, True, False])
        np.testing.assert_array_equal(norm["tiny_mask"], [False, False, True])
        np.testing.assert_allclose(design[:, 0], [-1, 1])
        np.testing.assert_array_equal(design[:, 1], [0, 0])
        np.testing.assert_array_equal(design[:, -1], [1, 1])
        self.assertEqual(design.shape, (2, 4))

    def test_apply_uses_p_only_without_mutating_normalizer(self):
        p = np.array([[1, 4], [3, 4]], dtype=np.float32)
        _, norm = probe.standardize_fit(p)
        snapshot = {k: v.copy() for k, v in norm.items()}
        q = np.array([[100, -20], [200, 30]], dtype=np.float64)
        applied = probe.standardize_apply(q, norm)
        np.testing.assert_array_equal(applied, [[98, -24, 1], [198, 26, 1]])
        for k, value in snapshot.items():
            np.testing.assert_array_equal(norm[k], value)
        np.testing.assert_array_equal(p, [[1, 4], [3, 4]])

    def test_residual_casts_before_subtracting(self):
        label = np.array([1e8, -np.finfo(np.float32).max], dtype=np.float32)
        pg = np.array([1, np.finfo(np.float32).max], dtype=np.float32)
        residual = probe.residual_targets(label, pg)
        self.assertEqual(residual.dtype, np.float64)
        np.testing.assert_array_equal(
            residual, label.astype("float64") - pg.astype("float64")
        )
        self.assertEqual(residual[0], 99999999)
        self.assertTrue(np.isfinite(residual).all())

    def test_residual_rejects_wrong_dtype_shape_and_nonfinite(self):
        good = np.ones(2, dtype=np.float32)
        bads = [
            good.astype("float64"),
            good.astype("int32"),
            [1.0, 1.0],
            np.ones((2, 1), dtype="float32"),
            good[:0],
            good[:1],
            np.array([np.nan, 1], dtype="float32"),
            np.array([np.inf, 1], dtype="float32"),
        ]
        for bad in bads:
            for pair in [(bad, good), (good, bad)]:
                with self.subTest(pair=pair), self.assertRaises(ValueError):
                    probe.residual_targets(*pair)

    def test_standardize_rejects_shape_dtype_nonfinite_and_overflow(self):
        bads = [
            np.ones(2),
            np.ones((0, 2)),
            np.ones((2, 0)),
            np.array([[np.nan]]),
            np.array([[np.inf]]),
            np.array([[True]]),
            np.array([["1"]]),
            np.array([[1j]]),
            np.array([[1e308], [1e308]]),
        ]
        for z in bads:
            with self.subTest(z=z), self.assertRaises(ValueError):
                probe.standardize_fit(z)
        norm = {"mean": np.zeros(1), "scale": np.ones(1)}
        for z in bads[:-1]:
            with self.subTest(z=z), self.assertRaises(ValueError):
                probe.standardize_apply(z, norm)
        with self.assertRaises(ValueError):
            probe.standardize_apply(
                np.array([[1e308]]), {"mean": np.array([-1e308]), "scale": np.ones(1)}
            )

    def test_apply_rejects_bad_normalizers(self):
        z = np.ones((2, 1))
        bads = [
            {},
            {"mean": np.zeros(2), "scale": np.ones(2)},
            {"mean": np.zeros((1, 1)), "scale": np.ones(1)},
            {"mean": np.zeros(1), "scale": np.zeros(1)},
            {"mean": np.zeros(1), "scale": np.array([1e-14])},
            {"mean": np.array([np.inf]), "scale": np.ones(1)},
            {"mean": np.zeros(1), "scale": np.array([np.nan])},
        ]
        for norm in bads:
            with self.subTest(norm=norm), self.assertRaises(ValueError):
                probe.standardize_apply(z, norm)

    def test_known_linear_oracle_and_independent_normal_equation(self):
        design, target, theta = self.fixture()
        result = probe.solve_ls(design, target)
        np.testing.assert_allclose(result["theta"], theta, atol=1e-13)
        self.assertEqual(result["theta"].dtype, np.float64)
        self.assertEqual(result["rank"], 3)
        self.assertLess(result["objective"], 1e-25)
        residual = design @ result["theta"] - target
        normal = np.max(np.abs(design.T @ residual)) / max(
            1, np.linalg.norm(design, "fro") * np.linalg.norm(target)
        )
        self.assertAlmostEqual(result["normalized_normal_residual"], normal, places=16)
        np.testing.assert_allclose(
            result["singular_values"], np.linalg.svd(design, compute_uv=False)
        )
        certified = probe.certify_pair(
            design, target, probe.solve_adam(design, target, 4), result
        )
        self.assertTrue(certified["pass"])
        self.assertFalse(certified["rank_limited"])

    def test_rank_deficient_minimum_norm_and_cutoff(self):
        design = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1]], dtype=np.float64)
        target = np.array([3, 5, 7], dtype=np.float64)
        ls = probe.solve_ls(design, target)
        np.testing.assert_allclose(ls["theta"], [1, 1, 1], atol=1e-13)
        self.assertEqual(ls["rank"], 2)
        self.assertTrue(
            probe.certify_pair(design, target, probe.solve_adam(design, target, 0), ls)[
                "rank_limited"
            ]
        )
        tiny = np.diag([1.0, 1e-14])
        ls = probe.solve_ls(tiny, np.ones(2))
        self.assertEqual(ls["rank"], 1)
        np.testing.assert_array_equal(ls["theta"], [1, 0])
        self.assertEqual(ls["objective"], 0.5)
        cert = probe.certify_pair(
            tiny, np.ones(2), probe.solve_adam(tiny, np.ones(2), 0), ls
        )
        self.assertTrue(cert["pass"])
        self.assertTrue(cert["rank_limited"])

    def test_rank_zero_and_underdetermined_are_explicitly_limited(self):
        for design, target in [
            (np.zeros((2, 2)), np.ones(2)),
            (np.array([[1.0, 0, 0], [0, 1, 0]]), np.array([2.0, 3])),
        ]:
            ls = probe.solve_ls(design, target)
            cert = probe.certify_pair(
                design, target, probe.solve_adam(design, target, 0), ls
            )
            self.assertTrue(cert["pass"])
            self.assertTrue(cert["rank_limited"])

    def test_row_permutation_keeps_normalizer_and_ls(self):
        design, target, _ = self.fixture()
        order = np.array([3, 0, 2, 1])
        a, na = probe.standardize_fit(design[:, :-1])
        b, nb = probe.standardize_fit(design[order, :-1])
        for k in na:
            np.testing.assert_allclose(na[k], nb[k], atol=1e-15)
        np.testing.assert_allclose(b, a[order], atol=1e-15)
        sa, sb = probe.solve_ls(a, target), probe.solve_ls(b, target[order])
        np.testing.assert_allclose(sa["theta"], sb["theta"], atol=1e-13)
        aa = probe.solve_adam(a, target, 5)
        ab = probe.solve_adam(b, target[order], 5)
        np.testing.assert_allclose(aa["theta"], ab["theta"], atol=1e-15)

    def test_adam_zero_init_pg_and_exact_update_count(self):
        design, target, _ = self.fixture()
        initial = probe.solve_adam(design, target, steps=0)
        np.testing.assert_array_equal(initial["theta"], np.zeros(3))
        self.assertEqual(initial["objective"], float(np.mean(target**2)))
        self.assertEqual(initial["updates"], 0)
        pg = np.array([4.0, 5, 6, 7])
        np.testing.assert_array_equal(pg + design @ initial["theta"], pg)
        updated = probe.solve_adam(design, target, steps=3)
        self.assertEqual(updated["updates"], 3)
        self.assertLess(updated["objective"], initial["objective"])

    def test_adam_matches_fixed_torch_reference_and_is_deterministic(self):
        design, target, _ = self.fixture()
        x, y = torch.tensor(design), torch.tensor(target)
        theta = torch.zeros(design.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam(
            [theta], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
        )
        for _ in range(7):
            opt.zero_grad()
            ((x @ theta - y) ** 2).mean().backward()
            opt.step()
        a = probe.solve_adam(design, target, 7)
        b = probe.solve_adam(design, target, 7)
        np.testing.assert_array_equal(a["theta"], theta.detach().numpy())
        np.testing.assert_array_equal(a["theta"], b["theta"])
        self.assertEqual(a["objective"], b["objective"])
        self.assertEqual(a["theta"].dtype, np.float64)

    def test_solvers_reject_shape_dtype_nonfinite_overflow(self):
        for solve in [probe.solve_ls, lambda x, y: probe.solve_adam(x, y, 1)]:
            for design, target in [
                (np.ones(3), np.ones(3)),
                (np.ones((0, 2)), np.ones(0)),
                (np.ones((2, 0)), np.ones(2)),
                (np.ones((2, 2)), np.ones((2, 1))),
                (np.ones((2, 2)), np.ones(3)),
                (np.array([[np.inf]]), np.ones(1)),
                (np.ones((1, 1)), np.array([np.nan])),
                (np.ones((1, 1)), np.array([1e308])),
                (np.array([["1"]]), np.ones(1)),
                (np.ones((1, 1)), np.array([1j])),
            ]:
                with (
                    self.subTest(solve=solve, design=design),
                    self.assertRaises(ValueError),
                ):
                    solve(design, target)
        for steps in [-1, True, 1.5, "2"]:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                probe.solve_adam(np.ones((1, 1)), np.ones(1), steps)

    def test_ls_failure_and_invalid_solver_outputs_never_fallback(self):
        design, target, _ = self.fixture()
        with patch.object(
            np.linalg, "lstsq", side_effect=np.linalg.LinAlgError("failed")
        ) as solver:
            with self.assertRaises(ValueError):
                probe.solve_ls(design, target)
            self.assertEqual(solver.call_count, 1)
            self.assertEqual(solver.call_args.kwargs["rcond"], 1e-12)
        for theta, rank, singular in [
            (np.full(3, np.nan), 3, np.ones(3)),
            (np.zeros(3), 4, np.ones(3)),
            (np.zeros(3), -1, np.ones(3)),
            (np.zeros(3), 3, np.array([1.0, np.inf, 0])),
            (np.zeros(2), 3, np.ones(3)),
        ]:
            with (
                patch.object(
                    np.linalg,
                    "lstsq",
                    return_value=(theta, np.array([]), rank, singular),
                ),
                self.subTest(rank=rank),
                self.assertRaises(ValueError),
            ):
                probe.solve_ls(design, target)

    def test_certification_recomputes_objective_and_normal_residual(self):
        design, target, _ = self.fixture()
        adam = probe.solve_adam(design, target, 0)
        ls = probe.solve_ls(design, target)
        for which, key, value in [
            ("ls", "objective", 1.0),
            ("adam", "objective", 0.0),
            ("ls", "normalized_normal_residual", 0.5),
            ("ls", "theta", np.zeros(3)),
        ]:
            aa, ll = dict(adam), dict(ls)
            (aa if which == "adam" else ll)[key] = value
            with self.subTest(which=which, key=key):
                self.assertFalse(probe.certify_pair(design, target, aa, ll)["pass"])
        bad = dict(ls, theta=np.zeros(3), objective=adam["objective"])
        bad["normalized_normal_residual"] = np.max(np.abs(design.T @ -target)) / max(
            1, np.linalg.norm(design, "fro") * np.linalg.norm(target)
        )
        self.assertFalse(probe.certify_pair(design, target, adam, bad)["pass"])

    def test_certification_objective_inequality_is_not_assumed(self):
        design = np.diag([1.0, 1e-13])
        target = np.array([0.0, 1.0])
        ls = probe.solve_ls(design, target)
        adam = {"theta": np.array([0.0, 1e13]), "objective": 0.0, "updates": 1}
        cert = probe.certify_pair(design, target, adam, ls)
        self.assertFalse(cert["pass"])
        self.assertTrue(cert["rank_limited"])

    def test_certification_rejects_bad_rank_shape_and_nonfinite(self):
        design, target, _ = self.fixture()
        adam, ls = probe.solve_adam(design, target, 0), probe.solve_ls(design, target)
        for key, value in [
            ("rank", -1),
            ("rank", 4),
            ("rank", 1),
            ("rank", True),
            ("objective", np.nan),
            ("theta", np.array([np.inf] * 3)),
            ("theta", np.zeros(2)),
            ("singular_values", np.array([1.0])),
            ("normalized_normal_residual", np.inf),
        ]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                probe.certify_pair(design, target, adam, dict(ls, **{key: value}))

    def test_nonzero_underflow_variance_does_not_become_constant(self):
        z = np.array([[0.0, 7.0], [1e-200, 7.0]])
        design, norm = probe.standardize_fit(z)
        np.testing.assert_array_equal(norm["scale"], [1e-12, 1.0])
        np.testing.assert_allclose(norm["std"], [5e-201, 0.0], rtol=1e-14, atol=0)
        np.testing.assert_array_equal(norm["zero_mask"], [False, True])
        np.testing.assert_array_equal(norm["tiny_mask"], [True, False])
        np.testing.assert_allclose(design[:, 0], [-5e-189, 5e-189], rtol=1e-14, atol=0)

    def test_certification_normal_threshold_inclusive(self):
        design, target = np.ones((1, 1)), np.zeros(1)
        adam = {"theta": np.zeros(1), "objective": 0.0, "updates": 0}
        for coefficient, expected in [
            (1e-8, True),
            (np.nextafter(1e-8, np.inf), False),
        ]:
            ls = {
                "theta": np.array([coefficient]),
                "objective": coefficient**2,
                "rank": 1,
                "singular_values": np.ones(1),
                "normalized_normal_residual": coefficient,
            }
            cert = probe.certify_pair(design, target, adam, ls)
            self.assertEqual(cert["normal_residual_pass"], expected)
            self.assertFalse(cert["reference_prediction_pass"])
            self.assertFalse(cert["pass"])

    def test_certification_objective_threshold_inclusive(self):
        design, target = np.array([[1e-10]]), np.zeros(1)
        adam = {"theta": np.zeros(1), "objective": 0.0, "updates": 0}
        for coefficient, expected in [(1e6, True), (np.nextafter(1e6, np.inf), False)]:
            prediction = 1e-10 * coefficient
            ls = {
                "theta": np.array([coefficient]),
                "objective": prediction**2,
                "rank": 1,
                "singular_values": np.array([1e-10]),
                "normalized_normal_residual": 1e-10 * prediction,
            }
            cert = probe.certify_pair(design, target, adam, ls)
            self.assertEqual(cert["objective_pass"], expected)
            self.assertFalse(cert["reference_prediction_pass"])
            self.assertFalse(cert["pass"])

    def test_adam_rejects_moment_overflow_with_finite_loss_and_gradient(self):
        # Initial loss=1e200, gradient=-2e200 are finite; its square is not.
        with self.assertRaises(ValueError):
            probe.solve_adam(np.array([[1e100]]), np.array([1e100]), 1)

    def test_adam_does_not_mutate_inputs_or_rng(self):
        design, target, _ = self.fixture()
        x, y = design.copy(), target.copy()
        torch_state = torch.random.get_rng_state().clone()
        np_state = np.random.get_state()
        probe.solve_adam(design, target, 2)
        np.testing.assert_array_equal(design, x)
        np.testing.assert_array_equal(target, y)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        after = np.random.get_state()
        self.assertEqual(after[0], np_state[0])
        np.testing.assert_array_equal(after[1], np_state[1])
        self.assertEqual(after[2:], np_state[2:])

    def test_noisy_linear_oracle_has_known_nonzero_mse(self):
        design = np.array([[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
        target = np.array([0.0, 2.0, 1.0])
        ls = probe.solve_ls(design, target)
        np.testing.assert_allclose(ls["theta"], [0.5, 1.0], atol=1e-14)
        self.assertAlmostEqual(ls["objective"], 0.5, places=14)
        np.testing.assert_allclose(
            design.T @ (design @ ls["theta"] - target), [0.0, 0.0], atol=1e-14
        )

    def test_certification_scales_objective_slack_when_adam_mse_exceeds_one(self):
        design, target = np.array([[1e-12]]), np.array([2.0])
        adam = {"theta": np.zeros(1), "objective": 4.0, "updates": 0}
        for coefficient, expected in [(-1000.0, True), (-100000.0, False)]:
            residual = 1e-12 * coefficient - 2.0
            ls = {
                "theta": np.array([coefficient]),
                "objective": residual**2,
                "rank": 1,
                "singular_values": np.array([1e-12]),
                "normalized_normal_residual": abs(1e-12 * residual),
            }
            cert = probe.certify_pair(design, target, adam, ls)
            self.assertEqual(cert["objective_pass"], expected)
            self.assertFalse(cert["reference_prediction_pass"])
            self.assertFalse(cert["pass"])

    def test_certification_rejects_forged_singular_values_and_svd_failure(self):
        design, target, _ = self.fixture()
        adam, ls = probe.solve_adam(design, target, 0), probe.solve_ls(design, target)
        forged = dict(ls, singular_values=ls["singular_values"] * 2.0)
        self.assertFalse(probe.certify_pair(design, target, adam, forged)["pass"])
        with patch.object(
            np.linalg, "svd", side_effect=np.linalg.LinAlgError("failed")
        ) as svd:
            with self.assertRaises(ValueError):
                probe.certify_pair(design, target, adam, ls)
            self.assertEqual(svd.call_count, 1)

    def test_truncated_solver_certification_is_not_full_linear_optimality(self):
        design = np.array([[-1e-14, 1.0], [1e-14, 1.0]])
        target = np.array([-1.0, 1.0])
        ls = probe.solve_ls(design, target)
        # Tiny 2x2 synthetic counterexample only, not a real 1000-step panel.
        adam = probe.solve_adam(design, target, steps=1000)
        self.assertAlmostEqual(ls["objective"], 1.0)
        self.assertAlmostEqual(adam["objective"], 1.0)
        oracle = np.array([1e14, 0.0])
        np.testing.assert_array_equal(design @ oracle - target, [0.0, 0.0])
        self.assertLessEqual(ls["normalized_normal_residual"], 1e-8)
        cert = probe.certify_pair(design, target, adam, ls)
        self.assertTrue(cert["pass"])
        self.assertTrue(cert["solver_verified"])
        self.assertTrue(cert["rank_limited"])
        self.assertTrue(cert["numerical_rank_limited"])
        self.assertFalse(cert["full_linear_optimality_verified"])
        self.assertFalse(cert["full_linear_optimality"])

    def test_exact_redundancy_is_conservatively_unknown_for_full_optimality(self):
        design, target = np.array([[0.0, 1.0], [0.0, 1.0]]), np.ones(2)
        ls = probe.solve_ls(design, target)
        cert = probe.certify_pair(
            design, target, probe.solve_adam(design, target, 0), ls
        )
        self.assertTrue(cert["solver_verified"])
        self.assertTrue(cert["numerical_rank_limited"])
        self.assertFalse(cert["full_linear_optimality_verified"])

    def test_full_rank_numeric_verification_is_not_scientific_optimality(self):
        design, target, _ = self.fixture()
        adam, ls = probe.solve_adam(design, target, 0), probe.solve_ls(design, target)
        certified = probe.certify_pair(design, target, adam, ls)
        self.assertTrue(certified["fullrank_numeric_solver"])
        self.assertFalse(certified["full_linear_optimality_verified"])
        self.assertFalse(certified["numerical_rank_limited"])
        uncertified = probe.certify_pair(design, target, adam, dict(ls, objective=1.0))
        self.assertFalse(uncertified["solver_verified"])
        self.assertFalse(uncertified["fullrank_numeric_solver"])
        self.assertFalse(uncertified["full_linear_optimality_verified"])

    def test_apply_uses_scale_not_std_or_p_masks(self):
        _, norm = probe.standardize_fit(np.array([[1.0, 7.0], [3.0, 7.0]]))
        # Masks describe P only. Even a formerly constant column can vary on Q.
        q = np.array([[5.0, 20.0]])
        np.testing.assert_array_equal(
            probe.standardize_apply(q, norm), [[3.0, 13.0, 1.0]]
        )
        irrelevant = dict(
            norm,
            std=np.array([999.0, 0.0]),
            zero_mask=np.array([True, False]),
            tiny_mask=np.array([True]),
        )
        np.testing.assert_array_equal(
            probe.standardize_apply(q, irrelevant), [[3.0, 13.0, 1.0]]
        )
        with self.assertRaises(ValueError):
            probe.standardize_apply(q, {"mean": norm["mean"], "std": norm["std"]})

    def test_fullrank_ill_conditioned_forgery_needs_reference_not_normal_gate(self):
        design = np.array([[-1e-11, 1.0], [1e-11, 1.0]])
        target = np.array([-1.0, 1.0])
        singular = np.linalg.svd(design, compute_uv=False)
        forged = {
            "theta": np.zeros(2),
            "objective": 1.0,
            "rank": 2,
            "singular_values": singular,
            "normalized_normal_residual": 1e-11,
        }
        adam = {"theta": np.zeros(2), "objective": 1.0, "updates": 0}
        actual = probe.solve_ls(design, target)
        self.assertEqual(actual["rank"], 2)
        self.assertLess(actual["objective"], 1e-25)
        np.testing.assert_allclose(actual["theta"], [1e11, 0.0], atol=1e-5)
        cert = probe.certify_pair(design, target, adam, forged)
        self.assertTrue(cert["metadata_pass"])
        self.assertTrue(cert["normal_residual_pass"])
        self.assertTrue(cert["objective_pass"])
        self.assertFalse(cert["pass"])
        self.assertFalse(cert["solver_verified"])
        self.assertFalse(cert["full_linear_optimality_verified"])
        self.assertFalse(cert["fullrank_numeric_solver"])
        self.assertFalse(cert["reference_objective_pass"])
        self.assertFalse(cert["reference_prediction_pass"])
        self.assertLess(cert["reference_objective"], 1e-25)
        valid = probe.certify_pair(design, target, adam, actual)
        self.assertTrue(valid["solver_verified"])
        self.assertTrue(valid["fullrank_numeric_solver"])
        self.assertFalse(valid["full_linear_optimality_verified"])

    def test_reference_lstsq_fixed_cutoff_failure_and_call_cost(self):
        design, target, _ = self.fixture()
        adam = probe.solve_adam(design, target, 0)
        with (
            patch.object(np.linalg, "lstsq", wraps=np.linalg.lstsq) as lstsq,
            patch.object(np.linalg, "svd", wraps=np.linalg.svd) as svd,
        ):
            ls = probe.solve_ls(design, target)
            cert = probe.certify_pair(design, target, adam, ls)
            self.assertTrue(cert["solver_verified"])
            self.assertEqual(lstsq.call_count, 2)
            self.assertEqual(svd.call_count, 1)
            self.assertTrue(
                all(c.kwargs["rcond"] == 1e-12 for c in lstsq.call_args_list)
            )
            self.assertEqual(cert["certification_linalg_calls"], 2)
        with patch.object(
            np.linalg, "lstsq", side_effect=np.linalg.LinAlgError("reference failed")
        ) as lstsq:
            with self.assertRaises(ValueError):
                probe.certify_pair(design, target, adam, ls)
            self.assertEqual(lstsq.call_count, 1)

    def test_reference_objective_alone_does_not_authenticate_predictions(self):
        design, target = np.array([[-1.0, 1.0], [1.0, 1.0]]), np.zeros(2)
        adam = {"theta": np.zeros(2), "objective": 0.0, "updates": 0}
        ls = probe.solve_ls(design, target)
        coefficient = 1e-9
        forged = dict(
            ls,
            theta=np.array([coefficient, 0.0]),
            objective=coefficient**2,
            normalized_normal_residual=2 * coefficient,
        )
        cert = probe.certify_pair(design, target, adam, forged)
        self.assertTrue(cert["normal_residual_pass"])
        self.assertTrue(cert["objective_pass"])
        self.assertTrue(cert["reference_objective_pass"])
        self.assertFalse(cert["reference_prediction_pass"])
        self.assertFalse(cert["solver_verified"])


if __name__ == "__main__":
    unittest.main()
