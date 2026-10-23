import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.meta import loss_value
from spatial_meta_ce.tail_risk import capture_loss, snapshot_step, tail_terms
from spatial_meta_ce.train import train, validate, validation_plan


class StoreMock:
    manifest = {"content_hash": "synthetic-s6"}
    rows = [{"id": str(i), "query_hash": f"hash-查询-{i}"} for i in range(180)]

    def unique_indices(self):
        return np.arange(len(self.rows))

    def batch(self, indices, device):
        return batch_samples(
            [
                {
                    "groups": [
                        torch.ones(1, 40),
                        torch.ones(1, 40),
                        torch.ones(3 + int(i) % 2, 4),
                        torch.ones(0, 43),
                    ],
                    "true": 20.0 + int(i),
                    "pg": 2.0,
                    "id": self.rows[int(i)]["id"],
                    "hash": self.rows[int(i)]["query_hash"],
                }
                for i in indices
            ],
            device,
        )


class PoolMock:
    def __init__(self, root, *, max_open, mapping_root, mapping_mode):
        pass

    def get(self, split, db):
        return StoreMock()


def config(**changes):
    return {
        "model": {"width": 8, "use_pg": True},
        "method": "erm",
        "seed": 7,
        "steps": 2,
        "meta_batch": 4,
        "accumulation": 1,
        "outer_lr": 0.001,
        "inner_lr": 0.01,
        "inner_steps": 0,
        "adapt_scope": "all",
        "support_sizes": [5],
        "query_size": 64,
        "loss": "smooth_l1",
        "validate_every": 1,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 20,
        "min_delta": 0.0,
        **changes,
    }


class TailTermsTests(unittest.TestCase):
    def test_mixed_hand_gradient_and_detached_coefficients(self):
        pred = torch.tensor(
            [-0.5, 2.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True
        )
        terms = tail_terms(pred, torch.zeros_like(pred), "mean_tail10")
        expected = torch.tensor([0.05] * 3 + [0.55] + [0.05] * 6)
        torch.testing.assert_close(terms["coefficients"], expected)
        self.assertFalse(terms["coefficients"].requires_grad)
        self.assertEqual(
            (terms["n"], terms["m"], terms["h"], terms["e"]), (10, 1, 0, 1)
        )
        self.assertEqual(float(terms["tau"]), 2.5)
        np.testing.assert_allclose(
            float(terms["objective"]),
            0.05 * (0.125 + 1.5 + 0.5) + 0.55 * 2.5,
            rtol=1e-6,
            atol=1e-6,
        )
        terms["objective"].backward()
        torch.testing.assert_close(pred.grad, expected * pred.detach().clamp(-1, 1))

    def test_threshold_ties_permutation_and_all_equal(self):
        pred = torch.tensor([4.0, 3.0, 3.0] + [0.0] * 17, requires_grad=True)
        t = tail_terms(pred, torch.zeros_like(pred), "mean_tail10")
        self.assertEqual((t["m"], t["h"], t["e"]), (2, 1, 2))
        expected = torch.tensor([0.275, 0.15, 0.15] + [0.025] * 17)
        torch.testing.assert_close(t["coefficients"], expected)
        order = torch.arange(19, -1, -1)
        perm = tail_terms(pred[order], torch.zeros_like(pred), "mean_tail10")
        torch.testing.assert_close(perm["coefficients"], expected[order])
        torch.testing.assert_close(perm["objective"], t["objective"])
        for n in (1, 69, 74, 84, 114, 164, 10001):
            for mode in ("mean", "mean_tail10"):
                p = torch.ones(n, requires_grad=True)
                terms = tail_terms(p, torch.zeros(n), mode)
                self.assertEqual(terms["m"], (n + 9) // 10)
                torch.testing.assert_close(
                    terms["coefficients"], torch.full((n,), 1 / n)
                )
                torch.testing.assert_close(
                    terms["objective"], loss_value(p, torch.zeros(n))
                )
                torch.testing.assert_close(
                    torch.autograd.grad(terms["objective"], p)[0],
                    torch.full((n,), 1 / n),
                )

    def test_mean_matches_original_and_rng_unchanged(self):
        p = torch.linspace(-3, 2, 69, requires_grad=True)
        rng = torch.get_rng_state().clone()
        terms = tail_terms(p, torch.zeros_like(p), "mean")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(
            torch.equal(terms["objective"], loss_value(p, torch.zeros_like(p)))
        )
        torch.testing.assert_close(
            torch.autograd.grad(terms["objective"], p)[0],
            torch.autograd.grad(loss_value(p, torch.zeros_like(p)), p)[0],
            rtol=0,
            atol=0,
        )

    def test_underflow_and_threshold_neighbours(self):
        p = torch.tensor([1e-23, -1e-23], requires_grad=True)
        t = tail_terms(p, torch.zeros_like(p), "mean_tail10")
        self.assertTrue(torch.equal(t["losses"], torch.zeros(2)))
        t["objective"].backward()
        self.assertTrue(bool((p.grad != 0).all()))
        x = torch.tensor(2.0)
        p = torch.stack(
            [
                torch.nextafter(x, torch.tensor(3.0)),
                x,
                torch.nextafter(x, torch.tensor(1.0)),
            ]
        )
        t = tail_terms(p, torch.zeros_like(p), "mean_tail10")
        self.assertEqual((t["h"], t["e"]), (0, 1))

    def test_invalid_inputs_modes_and_overflow(self):
        good = torch.zeros(2)
        cases = [
            (None, good),
            ([], good),
            ("x", good),
            (good, None),
            (torch.zeros(0), torch.zeros(0)),
            (torch.zeros(2, 1), good),
            (good, torch.zeros(1)),
            (good.double(), good),
            (good, good.double()),
            (torch.ones(2, dtype=torch.int64), good),
        ]
        for p, y in cases:
            with self.subTest(p=p, y=y), self.assertRaises(ValueError):
                tail_terms(p, y, "mean")
        for mode in (None, [], "", "tail10", True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                tail_terms(good, good, mode)
        for bad in (float("nan"), float("inf"), -float("inf")):
            for p, y in (
                (torch.tensor([bad, 0.0]), good),
                (good, torch.tensor([bad, 0.0])),
            ):
                with self.assertRaises(FloatingPointError):
                    tail_terms(p, y, "mean")
        with self.assertRaises(FloatingPointError):
            tail_terms(torch.tensor([3e38]), torch.tensor([-3e38]), "mean_tail10")

    def test_fp32_coefficients_independent_of_default_dtype(self):
        default = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, default)
        torch.set_default_dtype(torch.float64)
        p = torch.arange(69, dtype=torch.float32)
        terms = tail_terms(p, torch.zeros_like(p), "mean_tail10")
        self.assertEqual(terms["coefficients"].dtype, torch.float32)
        self.assertEqual(terms["objective"].dtype, torch.float32)

    def test_capture_append_mass_directions_and_non_snapshot(self):
        p = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
        batch = {
            "label": torch.zeros(3),
            "counts": torch.tensor([[0, 0, 3], [0, 0, 4], [0, 0, 5]]),
        }
        terms = tail_terms(p, batch["label"], "mean_tail10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (2, 3):
                selected = capture_loss(
                    root,
                    step=step,
                    episode_index=step,
                    database="a",
                    k=1,
                    mode="mean_tail10",
                    sample_plan_digest="digest",
                    terms=terms,
                    objective=terms["objective"],
                    pred=p,
                    batch=batch,
                    hashes=["one", "two", "three"],
                    steps=4000,
                )
                self.assertEqual(selected, 0)
                rows = [
                    json.loads(line)
                    for line in (root / "loss-audit.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(rows), step - 1)
                c = terms["coefficients"].double().tolist()
                self.assertEqual(rows[-1]["under_mass"], c[0])
                self.assertEqual(rows[-1]["exact_mass"], c[1])
                self.assertEqual(rows[-1]["over_mass"], c[2])
                self.assertEqual(rows[-1]["low_mass"], c[0])
                self.assertEqual(rows[-1]["high_mass"], c[1] + c[2])
            self.assertFalse((root / "loss-snapshots.jsonl").exists())

    def test_fixed_snapshot_selection_4000(self):
        self.assertEqual(
            [s for s in range(1, 4001) if snapshot_step(s, 4000)], [1, 1000, 2000, 4000]
        )
        self.assertEqual([s for s in range(1, 3) if snapshot_step(s, 2)], [1, 2])


class S6TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {
            "cache_root": "unused",
            "splits": {"train": ["a"], "validation": ["v"], "test": ["t"]},
        }
        for target, replacement in (
            ("StorePool", PoolMock),
            ("validation_plan", lambda *a: []),
            ("validate", lambda *a: (1.0, [])),
        ):
            patcher = patch("spatial_meta_ce.train." + target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_restrictions_and_no_resume_before_side_effects(self):
        cases = [{"tail_risk_mode": v} for v in (None, [], True, "", "bad")]
        cases += [
            {"tail_risk_mode": "mean", key: value}
            for key, value in [
                ("method", "fomaml"),
                ("loss", "mse"),
                ("accumulation", 2),
                ("meta_batch", 3),
                ("complex_query_weight", 1),
                ("complex_query_weight", None),
            ]
        ]
        with patch("spatial_meta_ce.train.seed_all") as seed:
            for i, changes in enumerate(cases):
                out = self.root / str(i)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    train(config(**changes), self.data, out)
                self.assertFalse(out.exists())
            for mode in ("mean", "mean_tail10"):
                with self.assertRaisesRegex(ValueError, "S6.*resume"):
                    train(
                        config(tail_risk_mode=mode),
                        self.data,
                        self.root / mode,
                        resume="nonexistent.pt",
                    )
                self.assertFalse((self.root / mode).exists())
            seed.assert_not_called()

    def test_a_actual_legacy_loss_and_model_state_equivalence(self):
        train(config(), self.data, self.root / "legacy")
        with patch("spatial_meta_ce.train.loss_value", wraps=loss_value) as spy:
            result = train(config(tail_risk_mode="mean"), self.data, self.root / "a")
        self.assertEqual(spy.call_count, 8)
        self.assertTrue(
            all(len(c.args) == 3 and not c.kwargs for c in spy.call_args_list)
        )
        legacy, _ = load_checkpoint(self.root / "legacy/last.pt")
        a, state = load_checkpoint(self.root / "a/last.pt")
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, a.state_dict()[key]), key)
        self.assertFalse((self.root / "legacy/loss-audit.jsonl").exists())
        self.assertEqual(
            state["metadata"]["training_signature"]["tail_risk_mode"], "mean"
        )
        self.assertEqual(result["loss_evidence"], state["extra"]["loss_evidence"])

    def test_a_ignores_helper_objective_and_audits_actual_loss(self):
        def wrong_objective(pred, label, mode):
            terms = tail_terms(pred, label, mode)
            return {**terms, "objective": terms["objective"] * 0}

        with patch("spatial_meta_ce.train.tail_terms", side_effect=wrong_objective):
            train(config(tail_risk_mode="mean"), self.data, self.root / "a")
        for row in self.read_rows("a", "loss-audit.jsonl"):
            self.assertGreater(row["mixed_loss"], 0)
            self.assertEqual(row["mixed_loss"], row["mean_loss"])

    def test_real_validation_adaptation_uses_legacy_loss_only(self):
        with (
            patch("spatial_meta_ce.train.validation_plan", validation_plan),
            patch("spatial_meta_ce.train.validate", validate),
            patch("spatial_meta_ce.train.tail_terms", wraps=tail_terms) as helper,
            patch("spatial_meta_ce.meta.loss_value", wraps=loss_value) as adaptation,
        ):
            train(
                config(tail_risk_mode="mean_tail10", inner_steps=1),
                self.data,
                self.root / "validation",
            )
        self.assertEqual(helper.call_count, 8)
        self.assertEqual(adaptation.call_count, 2)
        self.assertTrue(
            all(len(c.args) == 3 and not c.kwargs for c in adaptation.call_args_list)
        )

    def test_b_uses_helper_objective_for_backward(self):
        objectives = []

        def tracked(pred, label, mode):
            terms = tail_terms(pred, label, mode)
            terms["objective"].retain_grad()
            objectives.append(terms["objective"])
            return terms

        with (
            patch("spatial_meta_ce.train.tail_terms", side_effect=tracked),
            patch(
                "spatial_meta_ce.train.loss_value",
                side_effect=AssertionError("B must use helper"),
            ),
        ):
            train(config(tail_risk_mode="mean_tail10"), self.data, self.root / "b")
        self.assertEqual(len(objectives), 8)
        for objective in objectives:
            self.assertEqual(float(objective.grad), 0.25)
        audit = self.read_rows("b", "loss-audit.jsonl")
        self.assertEqual(
            [r["mixed_loss"] for r in audit], [float(o.detach()) for o in objectives]
        )

    def read_rows(self, arm, name):
        return [
            json.loads(line)
            for line in (self.root / arm / name).read_text().splitlines()
        ]

    def test_audit_snapshots_independent_reconstruction_and_counts(self):
        digests = []
        for mode in ("mean", "mean_tail10"):
            with patch(
                "spatial_meta_ce.train.torch.nn.utils.clip_grad_norm_",
                side_effect=[torch.tensor(0.5), torch.tensor(2.0)],
            ):
                result = train(config(tail_risk_mode=mode), self.data, self.root / mode)
            self.assertEqual(
                result["loss_evidence"],
                {"audit_rows": 8, "snapshot_rows": 8, "mode": mode},
            )
            self.assertEqual(
                result["gradient_stats"],
                {"count": 2, "total_norm": 2.5, "max_norm": 2.0, "clipped_count": 1},
            )
            _, state = load_checkpoint(self.root / mode / "last.pt")
            self.assertEqual(state["extra"]["gradient_stats"], result["gradient_stats"])
            audits = self.read_rows(mode, "loss-audit.jsonl")
            snapshots = self.read_rows(mode, "loss-snapshots.jsonl")
            chain = hashlib.sha256(b"").hexdigest()
            for i, (row, snapshot) in enumerate(zip(audits, snapshots), 1):
                self.assertEqual(snapshot["audit"], row)
                self.assertEqual(
                    set(row),
                    set(
                        "step episode_index database k n m tau h e mode mean_loss tail_mean mixed_loss coefficient_sum coefficient_square_sum low_mass high_mass under_mass over_mass exact_mass sample_plan_digest".split()
                    ),
                )
                self.assertEqual(
                    (row["step"], row["episode_index"], row["n"], row["m"]),
                    ((i - 1) // 4 + 1, i, 69, 7),
                )
                samples = snapshot["samples"]
                self.assertEqual(len(samples), 69)
                self.assertEqual(
                    [s["role"] for s in samples], ["support"] * 5 + ["query"] * 64
                )
                self.assertEqual(len({s["query_hash"] for s in samples}), 69)
                chain = digest(
                    {
                        "previous": chain,
                        "episode": {
                            "db": "a",
                            "support": [s["query_hash"] for s in samples[:5]],
                            "query": [s["query_hash"] for s in samples[5:]],
                        },
                    }
                )
                self.assertEqual(chain, row["sample_plan_digest"])
                values = {
                    key: np.array([s[key] for s in samples])
                    for key in ("log_pred", "label", "loss", "coefficient", "n_tables")
                }
                for key in ("log_pred", "label", "loss", "coefficient"):
                    np.testing.assert_array_equal(
                        values[key], values[key].astype(np.float32).astype(np.float64)
                    )
                error = values["log_pred"] - values["label"]
                independent = np.where(abs(error) < 1, 0.5 * error**2, abs(error) - 0.5)
                np.testing.assert_allclose(
                    values["loss"], independent, rtol=1e-6, atol=1e-7
                )
                tau = np.sort(values["loss"])[-7]
                above, equal = values["loss"] > tau, values["loss"] == tau
                self.assertEqual(
                    (row["tau"], row["h"], row["e"]),
                    (tau, int(above.sum()), int(equal.sum())),
                )
                a = np.where(
                    above,
                    1 / 7,
                    np.where(equal, (7 - above.sum()) / (7 * equal.sum()), 0),
                )
                c = np.full(69, 1 / 69) if mode == "mean" else 0.5 / 69 + 0.5 * a
                np.testing.assert_allclose(
                    values["coefficient"], c, rtol=1e-6, atol=1e-7
                )
                self.assertAlmostEqual(
                    row["coefficient_sum"], values["coefficient"].sum(), places=12
                )
                self.assertAlmostEqual(
                    row["coefficient_square_sum"],
                    (values["coefficient"] ** 2).sum(),
                    places=12,
                )
                for field, mask in [
                    ("low_mass", values["n_tables"] < 4),
                    ("high_mass", values["n_tables"] >= 4),
                    ("under_mass", error < 0),
                    ("over_mass", error > 0),
                    ("exact_mass", error == 0),
                ]:
                    self.assertAlmostEqual(
                        row[field], values["coefficient"][mask].sum(), places=12
                    )
                np.testing.assert_allclose(
                    row["mixed_loss"],
                    (values["coefficient"] * values["loss"]).sum(),
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    row["tail_mean"], (a * values["loss"]).sum(), rtol=1e-6, atol=1e-6
                )
                self.assertAlmostEqual(
                    row["mean_loss"], values["loss"].mean(), places=6
                )
            self.assertEqual(chain, state["extra"]["sample_plan_digest"])
            digests.append(chain)
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
