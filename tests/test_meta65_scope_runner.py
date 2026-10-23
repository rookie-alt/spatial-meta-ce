import copy
import os
from contextlib import ExitStack
from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import meta65_scope_contract as contract
from scripts import meta65_scope_runner as runner
from scripts import run_meta65_screen as screen
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.meta import meta_loss
from spatial_meta_ce.model import SetEstimator

ROOT = Path(__file__).resolve().parents[1]


def identity(n: int = 140) -> dict:
    return {
        "query_hash": np.array([f"h{i:03}" for i in range(n)]),
        "query_id": np.array([f"telstra/pretrain/{i + 1}" for i in range(n)]),
        "true": np.ones(n, dtype=np.float64),
        "n_tables": np.full(n, 4, dtype=np.int16),
    }


def sample(value: float = 40.0) -> dict:
    return batch_samples(
        [
            {
                "groups": [
                    np.ones((1, dim), dtype=np.float32) for dim in (40, 40, 4, 43)
                ],
                "true": value,
                "pg": 2.0,
                "id": "telstra/pretrain/1",
            }
        ]
    )


class ScopeContractTests(unittest.TestCase):
    def test_frozen_config_and_training_only_scope_changes(self) -> None:
        self.assertEqual(
            contract.load_config(ROOT / "configs/meta65_scope.json"), contract.FROZEN
        )
        reference = contract.arm_config(contract.FROZEN, "R_head3", "formal")
        treatment = contract.arm_config(contract.FROZEN, "T_all3", "formal")
        self.assertEqual(reference["steps"], 4000)
        self.assertEqual(reference["seed"], 42)
        self.assertEqual(reference["inner_steps"], 3)
        self.assertEqual({**reference, "adapt_scope": "all"}, treatment)
        old = screen.arm_config(screen.FROZEN, "B_head3_head3")
        self.assertEqual(
            {k: v for k, v in reference.items() if k != "purpose"},
            {k: v for k, v in old.items() if k != "purpose"},
        )
        for mode in ("cpu_preflight", "gpu_preflight"):
            self.assertEqual(
                contract.arm_config(contract.FROZEN, "R_head3", mode)["steps"], 2
            )
        for field, value in (("seed", 43), ("steps", 2), ("meta_batch", True)):
            bad = copy.deepcopy(contract.FROZEN)
            bad["training"][field] = value
            with self.assertRaises(ValueError):
                contract.validate_config(bad)
        with self.assertRaises(ValueError):
            contract.validate_config({**contract.FROZEN, "budget": 1})
        with self.assertRaises(ValueError):
            contract.arm_config(contract.FROZEN, "imdb", "formal")

    def test_matrices_and_preflight_not_formal(self) -> None:
        self.assertEqual(len(contract.matrix("formal")), 60)
        self.assertEqual(len(contract.matrix("cpu_preflight")), 12)
        self.assertEqual(len(contract.matrix("gpu_preflight")), 12)
        self.assertTrue(
            all(run == 0 for _, _, _, run in contract.matrix("cpu_preflight"))
        )
        with self.assertRaises(ValueError):
            contract.matrix("smoke")

    def test_strict_json_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            p = root / "json"
            for text in ('{"a":1,"a":2}', '{"v":NaN}', '{"v":Infinity}'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    contract.read_json(p)
            retained = root / "retained"
            retained.mkdir()
            (root / "link").symlink_to(retained)
            with self.assertRaises(ValueError):
                contract.checked_path(root / "link" / "new")
            for output in (retained / "new", root, retained):
                with self.assertRaises(ValueError):
                    contract.safe_output(output, [retained])
            self.assertEqual(
                contract.safe_output(root / "new", [retained]), root / "new"
            )
            p.write_text("original")
            bound = {str(p): contract.sha256(p)}
            contract.recheck(bound)
            p.write_text("modified")
            with self.assertRaises(ValueError):
                contract.recheck(bound)

    def test_anchor_discovery_only_b_k20_exact_15(self) -> None:
        names = {
            str(
                contract.RETAINED
                / "B_head3_head3"
                / "predictions"
                / f"{db}-k20-run{run}.npz"
            ): "a" * 64
            for db in contract.DATABASES
            for run in range(5)
        }
        names[
            str(contract.RETAINED / "A_all5_head3/predictions/chembl-k20-run0.npz")
        ] = "b" * 64
        names[
            str(contract.RETAINED / "B_head3_head3/predictions/chembl-k5-run0.npz")
        ] = "c" * 64
        anchors = contract.discover_anchors({"input_sha256": names})
        self.assertEqual(len(anchors), 15)
        self.assertTrue(
            all(
                a["path"].startswith(str(contract.RETAINED / "B_head3_head3"))
                for a in anchors.values()
            )
        )
        for key in (next(k for k in names if "B_head3_head3" in k and "-k20-" in k),):
            with self.assertRaises(ValueError):
                contract.discover_anchors(
                    {"input_sha256": {k: v for k, v in names.items() if k != key}}
                )
        with self.assertRaises(ValueError):
            contract.discover_anchors(
                {
                    "input_sha256": {
                        **names,
                        str(
                            contract.RETAINED
                            / "B_head3_head3/predictions/imdb-k20-run0.npz"
                        ): "a" * 64,
                    }
                }
            )

    def test_formal_and_direct_worker_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve() / "new"
            for extra in ([], ["--worker"]):
                with self.assertRaisesRegex(ValueError, "authoriz|approval"):
                    runner.main(
                        ["--out", str(out), "--mode", "formal", "--execute", *extra]
                    )
            with patch.dict("sys.modules", {"scripts.meta65_scope_launch": None}):
                with self.assertRaises((ImportError, NotImplementedError)):
                    runner.formal_gate(out, out, out, out, worker=False)
            self.assertFalse(out.exists())


class ScopeRunnerTests(unittest.TestCase):
    def test_full_plan_preserves_100_exclusion(self) -> None:
        raw = identity()
        plan = screen.support_plan(
            "telstra",
            list(range(140)),
            raw["query_hash"].tolist(),
            seed=2718,
            runs=5,
            shots=list(screen.SHOTS),
        )[0]
        selected = runner.certify_plan(plan, raw, "telstra", 0, expected_n=40)
        self.assertEqual(selected["support"], plan["support_max"][:20])
        self.assertFalse(set(selected["query"]) & set(plan["support_max"]))
        with self.assertRaises(ValueError):
            runner.certify_plan(
                {**plan, "query": plan["query"] + plan["support_max"][20:]},
                raw,
                "telstra",
                0,
                expected_n=120,
            )

    def test_low4_is_direct_mask_not_average(self) -> None:
        raw = identity(8)
        raw = {
            **raw,
            "n_tables": np.array([4, 5, 5, 5, 5, 3, 4, 5], dtype=np.int16),
            "true": np.array([1, 1, 1, 1, 1, 1, 100, 100], dtype=np.float64),
        }
        values = np.log([2, 4, 8, 16, 32, 100, 100, 100])
        metrics = runner.cell_metrics(values, raw)
        self.assertEqual(metrics["low4"]["count"], 5)
        self.assertAlmostEqual(
            metrics["low4"]["q95"], float(np.percentile([2, 4, 8, 16, 32], 95))
        )
        self.assertEqual(len(metrics["joint"]), 30)
        self.assertNotEqual(
            metrics["low4"]["q95"],
            (metrics["joint"]["<100|4"]["q95"] + metrics["joint"]["<100|5+"]["q95"])
            / 2,
        )
        with self.assertRaises(ValueError):
            runner.cell_metrics(np.full(8, np.nan), raw)

    def test_real_head_none_reset_and_both_scopes_outer_encoder_gradient(self) -> None:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = SetEstimator(arch="sum", use_pg=True)
        before = runner.state_digest(model)
        with patch(
            "spatial_meta_ce.meta.adapt", side_effect=AssertionError("none adapted")
        ):
            _, record = runner.operator_params(model, None, "none")
        self.assertEqual(record["steps"], 0)
        self.assertEqual(record["selected_parameter_count"], 0)
        params, record = runner.operator_params(model, sample(), "head3")
        again, _ = runner.operator_params(model, sample(), "head3")
        self.assertEqual(record["selected_parameter_count"], 107521)
        self.assertEqual(
            set(record["selected_parameter_names"]),
            {f"head.{i}.{kind}" for i in (0, 2, 4) for kind in ("weight", "bias")},
        )
        self.assertEqual(before, runner.state_digest(model))
        self.assertTrue(all(torch.equal(params[k], again[k]) for k in params))
        with self.assertRaises(ValueError):
            runner.operator_params(model, sample(), "last3")
        # Zero-initialized last layer needs an outer update before encoder gradients exist.
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        meta_loss(model, sample(), sample(80), 0.001, 3, "head").backward()
        optimizer.step()
        for scope in ("head", "all"):
            model.zero_grad(set_to_none=True)
            meta_loss(model, sample(), sample(80), 0.001, 3, scope).backward()
            encoder = [
                p.grad
                for name, p in model.named_parameters()
                if not name.startswith("head.")
            ]
            self.assertTrue(
                any(g is not None and bool(torch.any(g != 0)) for g in encoder)
            )

    def test_preflight_environment_requires_preimport_determinism(self) -> None:
        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with patch.dict(os.environ, env):
            result = runner.environment("cpu_preflight")
            self.assertEqual(result["threads"], 1)
            self.assertFalse(result["tf32"])
            self.assertTrue(result["deterministic_algorithms"])
            for name in env:
                with patch.dict(os.environ, {name: "bad"}):
                    with self.assertRaises(ValueError):
                        runner.environment("cpu_preflight")

    def test_decision_normalizes_without_mutating_raw_cells(self) -> None:
        cell = {
            "arm": "R_head3",
            "operator": "head3",
            "database": "telstra",
            "run": 0,
            "k20": 20,
            "metrics": runner.cell_metrics(np.zeros(8), identity(8)),
            "identity_file": "identities/telstra-run0.npz",
        }
        before = copy.deepcopy(cell)
        normalized = runner.normalize_cell(cell)
        self.assertEqual(
            set(normalized), {"arm", "operator", "database", "run", "k", "metrics"}
        )
        self.assertEqual(set(normalized["metrics"]), {"all", "4+", "low4"})
        self.assertEqual(set(normalized["metrics"]["low4"]), {"count", "q50", "q95"})
        self.assertEqual(cell, before)

    def test_default_dry_run_no_input_loading_or_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve() / "new"
            with patch.object(
                contract, "input_binding", side_effect=AssertionError("raw input read")
            ):
                with patch("builtins.print") as printed:
                    runner.main(["--out", str(out)])
            self.assertIn("prepared_not_executed", printed.call_args.args[0])
            self.assertFalse(out.exists())

    def test_check_cell_peaks_uses_scope_frozen_limits(self) -> None:
        limits = contract.FROZEN["limits"]
        self.assertEqual(
            limits["rss_bytes"], 12 * 1024**3
        )  # Probe-informed re-freeze, progress §72.
        valid = {
            "peak_rss_bytes": limits["rss_bytes"],
            "peak_vram_bytes": limits["vram_bytes"],
        }
        self.assertEqual(contract.check_cell_peaks(valid), valid)
        for field, budget in (
            ("peak_rss_bytes", limits["rss_bytes"]),
            ("peak_vram_bytes", limits["vram_bytes"]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.check_cell_peaks({**valid, field: budget + 1})
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.check_cell_peaks({**valid, field: True})


class ScopeIntegrationTests(unittest.TestCase):
    def test_input_binding_all_182_cache_26_historical_and_b_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            retained = root / "retained"
            retained.mkdir()
            history = {}
            before = {}
            for db in (*contract.TRAIN, *contract.DATABASES):
                history[db] = {"database": db}
                folder = root / "cache/pretrain" / db
                folder.mkdir(parents=True)
                values = {}
                for name in contract.CACHE_FILES:
                    p = folder / name
                    p.write_text(db + name)
                    values[name] = contract.sha256(p)
                before[db] = {"file_sha256": values, "manifest_hash": "1" * 64}
            for relative in (
                *contract.CODE_FILES,
                *contract.PROVENANCE,
                contract.PROTOCOL,
            ):
                p = root / relative
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(relative)
            for db, p in contract.historical_paths(root).items():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(db)
            config = root / "config.json"
            config.write_text("{}")
            data_config = root / "data.json"
            data = {
                "data_root": str(root / "price"),
                "cache_root": str(root / "cache"),
                "splits": {
                    "train": list(contract.TRAIN),
                    "validation": list(contract.DATABASES),
                    "test": list(contract.TARGETS),
                },
            }
            contract.write_json(data_config, data)
            plan_path = retained / "endpoint_plan.json"
            contract.write_json(plan_path, {db: [{}] * 5 for db in contract.DATABASES})
            old = retained / "B_head3_head3/metadata.json"
            old.parent.mkdir()
            old.write_text("oldB")
            (root / contract.PROVENANCE[3]).write_text("oldB")
            diagnosis = {}
            for db in contract.DATABASES:
                for run in range(5):
                    p = (
                        retained
                        / "B_head3_head3/predictions"
                        / f"{db}-k20-run{run}.npz"
                    )
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(db + str(run))
                    diagnosis[str(p)] = contract.sha256(p)
            contract.write_json(
                root / contract.PROVENANCE[1], {"input_sha256": diagnosis}
            )
            contract.write_json(
                root / contract.PROVENANCE[2],
                {"endpoint_plan_sha256": contract.sha256(plan_path)},
            )
            with ExitStack() as stack:
                stack.enter_context(patch.object(contract, "RETAINED", retained))
                stack.enter_context(patch.object(contract, "OLD_B_METADATA", old))
                stack.enter_context(
                    patch.object(contract, "OLD_B_METADATA_SHA", contract.sha256(old))
                )
                stack.enter_context(
                    patch.object(contract, "PLAN_SHA", contract.sha256(plan_path))
                )
                stack.enter_context(
                    patch.object(contract, "load_config", return_value=contract.FROZEN)
                )
                stack.enter_context(
                    patch(
                        "scripts.precision_screen_evidence.historical_sources",
                        return_value=history,
                    )
                )
                stack.enter_context(
                    patch(
                        "scripts.precision_screen_evidence.source_evidence",
                        return_value=before,
                    )
                )
                binding, plans, anchors, observed, _, bound_data = (
                    contract.input_binding(
                        root, config_path=config, data_config_path=data_config
                    )
                )
                self.assertEqual(
                    len([p for p in binding if "/cache/pretrain/" in p]), 182
                )
                self.assertEqual(
                    len([p for p in binding if "p1-mapping-sidecar-" in p]), 26
                )
                self.assertEqual(len(anchors), 15)
                self.assertEqual(observed, before)
                self.assertEqual(bound_data, data)
                self.assertEqual(len(plans), 3)
                self.assertIn(str(old), binding)
                (root / contract.CODE_FILES[0]).write_text("drift")
                with self.assertRaises(ValueError):
                    contract.recheck(binding)

    def test_real_evaluate_cell_fp64_reset_and_prediction_storage(self) -> None:
        torch.manual_seed(42)
        model = SetEstimator(arch="sum", use_pg=True)
        samples = [sample(16777217.25 + i) for i in range(4)]
        store = SimpleNamespace(
            batch=lambda indices, device: {
                **samples[0],
                "true": torch.tensor([16777217.25 + indices[0]], dtype=torch.float64),
            }
        )
        store.cards = np.array(
            [[16777217.25 + i, 2] for i in range(4)], dtype=np.float64
        )
        before = runner.state_digest(model)
        for operator in ("none", "head3"):
            pred, true, record = runner.evaluate_cell(
                model, store, [0], [1], operator, "cpu", lambda: None
            )
            self.assertEqual(pred.dtype, np.float64)
            self.assertEqual(true[0], 16777217.25)
            self.assertEqual(record["state_before"], before)
            self.assertEqual(record["state_after"], before)

    def test_train_sequence_reproduction_gate_and_preflight_not_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            before = {
                db: {"manifest_hash": "a" * 64}
                for db in (*contract.TRAIN, *contract.DATABASES)
            }
            old = contract.read_json(ROOT / contract.PROVENANCE[3])
            events = []

            def train(
                config: dict, data: dict, armout: Path, device: str, deadline: object
            ) -> tuple:
                armout.mkdir()
                events.append(armout.name + "-train")
                return object(), {
                    **old,
                    "config": config,
                    "optimizer_steps": 2,
                    "sample_plan_episodes": 8,
                    "support_sample_count": 100,
                    "metadata": {
                        **old["metadata"],
                        "manifest_hashes": {
                            db: v["manifest_hash"] for db, v in before.items()
                        },
                    },
                    "counts": {
                        "optimizer_steps": 2,
                        "query_presentations": 512,
                        "backward_calls": 32,
                        "support_presentations": 300,
                    },
                }

            def evaluate(*args: object, **kwargs: object) -> list:
                events.append(str(kwargs["arm"]) + "-" + str(kwargs["operator"]))
                return [
                    *kwargs["cells"],
                    {"arm": kwargs["arm"], "operator": kwargs["operator"]},
                ]

            def reproduce(*args: object, **kwargs: object) -> dict:
                events.append("reproduce")
                self.assertFalse(kwargs["formal"])
                return {"status": "not_applicable", "reproduction_verified": False}

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        runner, "prepare_identities", return_value=({}, {}, {})
                    )
                )
                stack.enter_context(
                    patch.object(contract, "read_json", return_value=old)
                )
                stack.enter_context(patch.object(contract, "verify_reference_training"))
                stack.enter_context(
                    patch.object(screen, "train_arm", side_effect=train)
                )
                stack.enter_context(
                    patch.object(runner, "evaluate_operator", side_effect=evaluate)
                )
                checker = stack.enter_context(
                    patch(
                        "scripts.meta65_scope_verify.verify_reproduction",
                        side_effect=reproduce,
                    )
                )
                runner.train_and_evaluate(
                    out,
                    contract.FROZEN,
                    {},
                    "cpu_preflight",
                    {},
                    {},
                    before,
                    lambda: None,
                )
                self.assertEqual(
                    events,
                    [
                        "R_head3-train",
                        "R_head3-head3",
                        "reproduce",
                        "R_head3-none",
                        "T_all3-train",
                        "T_all3-none",
                        "T_all3-head3",
                    ],
                )
                checker.side_effect = ValueError("reproduction_blocked")
                events.clear()
                failed = out / "gate-failure"
                failed.mkdir()
                with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
                    runner.train_and_evaluate(
                        failed,
                        contract.FROZEN,
                        {},
                        "cpu_preflight",
                        {},
                        {},
                        before,
                        lambda: None,
                    )
                self.assertNotIn("T_all3-train", events)

    def test_certification_preflight_and_formal_provisional_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            contract.write_json(
                out / "reproduction.json", {"reproduction_verified": True}
            )
            cells = [
                {
                    "arm": "R_head3",
                    "operator": "head3",
                    "database": "telstra",
                    "run": 0,
                    "k20": 20,
                    "metrics": runner.cell_metrics(np.zeros(32), identity(32)),
                }
            ]
            before = {"same": "evidence"}
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "scripts.precision_screen_evidence.source_evidence",
                        return_value=before,
                    )
                )
                stack.enter_context(
                    patch(
                        "scripts.precision_screen_evidence.historical_sources",
                        return_value=before,
                    )
                )
                stack.enter_context(patch.object(contract, "recheck"))
                verified = stack.enter_context(
                    patch(
                        "scripts.meta65_scope_verify.verify",
                        return_value={"all_metrics_verified": True, "summary": None},
                    )
                )
                decision = stack.enter_context(
                    patch(
                        "scripts.meta65_scope_decision.decide",
                        return_value={"status": "no_candidate", "selected_arm": None},
                    )
                )
                runner.certify(out, "cpu_preflight", cells, {}, {}, before, before)
                self.assertFalse(decision.called)
                self.assertEqual(
                    contract.read_json(out / "decision.provisional.json")["status"],
                    "preflight_not_candidate",
                )
                verified.return_value = {
                    "all_metrics_verified": True,
                    "summary": {"status": "no_candidate"},
                }
                runner.certify(out, "formal", cells, {}, {}, before, before)
                self.assertTrue(decision.called)
                self.assertFalse((out / "decision.json").exists())
                verified.return_value = {
                    "all_metrics_verified": True,
                    "summary": {"status": "promising_single_seed_K20"},
                }
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    runner.certify(out, "formal", cells, {}, {}, before, before)

    def test_shared_full_identity_prepare_and_micro_prediction_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            raw = identity(140)
            stores = SimpleNamespace(unique_indices=lambda: np.arange(140))
            plans = {
                db: screen.support_plan(
                    db,
                    list(range(140)),
                    raw["query_hash"].tolist(),
                    seed=2718,
                    runs=5,
                    shots=list(screen.SHOTS),
                )
                for db in contract.DATABASES
            }
            anchors = {}
            for db in contract.DATABASES:
                for run in range(5):
                    query = plans[db][run]["query"]
                    p = out / f"anchor-{db}-{run}.npz"
                    np.savez(
                        p,
                        **{k: v[query] for k, v in raw.items()},
                        log_pred=np.zeros(40),
                    )
                    anchors[(db, run)] = {"path": str(p)}
            with ExitStack() as stack:
                stack.enter_context(
                    patch("spatial_meta_ce.data.FeatureStore", return_value=stores)
                )
                validator = stack.enter_context(patch.object(screen, "validate_store"))
                stack.enter_context(
                    patch.object(
                        joint := runner.joint, "store_identity", return_value=raw
                    )
                )
                original = runner.certify_plan
                stack.enter_context(
                    patch.object(
                        runner,
                        "certify_plan",
                        side_effect=lambda p, d, db, run: original(
                            p, d, db, run, expected_n=40
                        ),
                    )
                )
                prepared, identities, selected = runner.prepare_identities(
                    out, {"cache_root": directory}, plans, anchors, lambda: None
                )
                self.assertEqual(validator.call_count, 26)
                self.assertEqual(len(identities), 15)
                cell_record = {
                    "steps": 0,
                    "state_before": "a" * 64,
                    "state_after": "a" * 64,
                }
                stack.enter_context(
                    patch.object(
                        runner,
                        "evaluate_cell",
                        return_value=(
                            np.zeros(32),
                            np.ones(32, dtype=np.float64),
                            cell_record,
                        ),
                    )
                )
                cells = runner.evaluate_operator(
                    out,
                    None,
                    prepared,
                    identities,
                    selected,
                    arm="R_head3",
                    operator="none",
                    mode="cpu_preflight",
                    device="cpu",
                    deadline=lambda: None,
                    cells=[],
                )
                self.assertEqual(len(cells), 3)
                self.assertEqual(cells[0]["query_positions"], list(range(32)))
                self.assertEqual(cells[0]["count"], 32)
                self.assertEqual(len(identities[("telstra", 0)][0]["true"]), 40)
                with np.load(
                    out / cells[0]["prediction_file"], allow_pickle=False
                ) as archive:
                    self.assertEqual(archive["log_pred"].dtype, np.float64)
                self.assertEqual(
                    joint.encode_identity(raw)["query_id"].dtype, np.dtype("S128")
                )

    def test_save_binding_and_training_count_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            config = out / "config.json"
            data = out / "data.json"
            config.write_text("config")
            data.write_text("data")
            binding = {str(ROOT / p): "a" * 64 for p in contract.CODE_FILES}
            anchors = {
                (db, run): {
                    "database": db,
                    "run": run,
                    "path": "/source",
                    "sha256": "b" * 64,
                }
                for db in contract.DATABASES
                for run in range(5)
            }
            runner.save_binding(
                out, "cpu_preflight", binding, {}, {}, {}, anchors, config, data
            )
            self.assertEqual(
                len(contract.read_json(out / "input-binding.json")["anchors"]), 15
            )
            self.assertEqual(
                len(contract.read_json(out / "reproduction.json")["anchors"]), 3
            )
            old = contract.read_json(ROOT / contract.PROVENANCE[3])
            r = {**old, "arm": "R_head3"}
            t = {**old, "arm": "T_all3"}
            contract.verify_pairing([r, t])
            contract.verify_reference_training(r, old, formal=True)
            with self.assertRaises(ValueError):
                contract.verify_pairing([{**r, "arm": "T_all3"}])
            for field in ("counts", "sample_plan_digest", "initialization_digest"):
                altered = {**t, field: "bad"}
                with self.assertRaises(ValueError):
                    contract.verify_pairing([r, altered])
            with self.assertRaises(ValueError):
                contract.verify_reference_training(
                    {**r, "optimizer_steps": 1}, old, formal=True
                )
            with self.assertRaises(ValueError):
                contract.verify_reference_training(
                    {**r, "metadata": {"manifest_hashes": {}}}, old, formal=False
                )
            with patch.object(contract, "input_binding", return_value=(binding,)):
                self.assertEqual(contract.bind_inputs(ROOT, config), binding)

    def test_main_supervisor_cli_and_formal_handshake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve() / "output"
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(contract, "safe_output", return_value=out)
                )
                supervised = stack.enter_context(
                    patch(
                        "scripts.meta65_scope_supervisor.supervise",
                        return_value={"status": "preflight_completed"},
                    )
                )
                stack.enter_context(patch("builtins.print"))
                runner.main(["--out", str(out), "--execute"])
                self.assertEqual(supervised.call_args.kwargs["mode"], "cpu_preflight")
                self.assertIn("--worker", supervised.call_args.args[0])
                supervised.return_value = {"status": "failed"}
                with self.assertRaises(SystemExit):
                    runner.main(["--out", str(out), "--execute"])
                work = stack.enter_context(patch.object(runner, "worker"))
                runner.main(["--out", str(out), "--execute", "--worker"])
                self.assertTrue(work.called)
                for mode in ("gpu_preflight",):
                    stack.enter_context(patch.object(runner.joint, "verify_gpu_idle"))
                    supervised.return_value = {"status": "preflight_completed"}
                    runner.main(["--out", str(out), "--mode", mode, "--execute"])
                with self.assertRaises(ValueError):
                    runner.main(["--out", str(out), "--approval", str(out)])
                launch = SimpleNamespace(
                    authorize_formal=lambda *a, **kw: {"status": "consumed"},
                    verify_worker_approval=lambda *a, **kw: {
                        "status": "authorized_worker"
                    },
                )
                stack.enter_context(
                    patch.dict("sys.modules", {"scripts.meta65_scope_launch": launch})
                )
                stack.enter_context(patch.object(runner, "require_supervisor"))
                self.assertEqual(
                    runner.formal_gate(out, out, out, out, worker=False)["status"],
                    "consumed",
                )
                self.assertEqual(
                    runner.formal_gate(out, out, out, out, worker=True)["status"],
                    "authorized_worker",
                )

    def test_worker_success_and_failure_never_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = {
                "data_root": str(root / "price"),
                "cache_root": str(root / "cache"),
                "splits": {
                    "train": list(contract.TRAIN),
                    "validation": list(contract.DATABASES),
                    "test": list(contract.TARGETS),
                },
            }
            config = root / "config.json"
            contract.write_json(config, contract.FROZEN)
            datap = root / "data.json"
            contract.write_json(datap, data)
            result = {"all_metrics_verified": True, "formal_result": False}
            env = {
                "META65_SCOPE_SUPERVISED_MODE": "cpu_preflight",
                "META65_SCOPE_SUPERVISOR_PID": str(os.getppid()),
            }
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, env))
                stack.enter_context(
                    patch.object(contract, "load_config", return_value=contract.FROZEN)
                )
                stack.enter_context(
                    patch.object(
                        runner, "environment", return_value={"mode": "cpu_preflight"}
                    )
                )
                stack.enter_context(
                    patch.object(
                        contract,
                        "input_binding",
                        return_value=({}, {}, {}, {}, {}, data),
                    )
                )
                stack.enter_context(patch.object(runner, "save_binding"))
                train = stack.enter_context(
                    patch.object(runner, "train_and_evaluate", return_value=[{}] * 12)
                )
                stack.enter_context(
                    patch.object(runner, "certify", return_value=result)
                )
                stack.enter_context(
                    patch(
                        "scripts.precision_screen_evidence.source_evidence",
                        return_value={},
                    )
                )
                out = root / "worker"
                self.assertEqual(
                    runner.worker(out, "cpu_preflight", config, data_config_path=datap),
                    result,
                )
                status = contract.read_json(out / "status.json")
                self.assertEqual(status["status"], "worker_complete")
                self.assertEqual(status["micro_cells"], 12)
                self.assertFalse(status["formal_result"])
                train.side_effect = ValueError("nonfinite")
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    runner.worker(
                        root / "failure",
                        "cpu_preflight",
                        config,
                        data_config_path=datap,
                    )
                self.assertEqual(
                    contract.read_json(root / "failure/status.json")["status"], "failed"
                )
                self.assertFalse((out / "decision.json").exists())
            with self.assertRaisesRegex(ValueError, "supervisor"):
                runner.require_supervisor("cpu_preflight")

    def test_reproduction_failure_is_named_and_diagnosed(self) -> None:
        """A blocked R reproduction must not be reported as a generic failure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = {
                "data_root": str(root / "price"),
                "cache_root": str(root / "cache"),
                "splits": {
                    "train": list(contract.TRAIN),
                    "validation": list(contract.DATABASES),
                    "test": list(contract.TARGETS),
                },
            }
            config = root / "config.json"
            contract.write_json(config, contract.FROZEN)
            datap = root / "data.json"
            contract.write_json(datap, data)
            env = {
                "META65_SCOPE_SUPERVISED_MODE": "cpu_preflight",
                "META65_SCOPE_SUPERVISOR_PID": str(os.getppid()),
            }
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, env))
                stack.enter_context(
                    patch.object(contract, "load_config", return_value=contract.FROZEN)
                )
                stack.enter_context(
                    patch.object(
                        runner, "environment", return_value={"mode": "cpu_preflight"}
                    )
                )
                stack.enter_context(
                    patch.object(
                        contract,
                        "input_binding",
                        return_value=({}, {}, {}, {}, {}, data),
                    )
                )
                stack.enter_context(patch.object(runner, "save_binding"))
                train = stack.enter_context(patch.object(runner, "train_and_evaluate"))
                train.side_effect = runner.ReproductionBlocked(
                    "ValueError: Reproduction Q95 tolerance failed"
                )
                out = root / "blocked"
                with self.assertRaises(runner.ReproductionBlocked):
                    runner.worker(out, "cpu_preflight", config, data_config_path=datap)
                status = contract.read_json(out / "status.json")
                self.assertEqual(status["status"], "reproduction_blocked")
                self.assertFalse(status["formal_result"])
                self.assertIn("tolerance", status["failure"])
                # An unrelated error stays a generic failure.
                train.side_effect = RuntimeError("unrelated crash")
                other = root / "unrelated"
                with self.assertRaisesRegex(RuntimeError, "unrelated"):
                    runner.worker(
                        other, "cpu_preflight", config, data_config_path=datap
                    )
                self.assertEqual(
                    contract.read_json(other / "status.json")["status"], "failed"
                )

    def test_reproduction_block_is_raised_by_the_real_gate(self) -> None:
        """The real evaluation gate names its own failure type."""
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory).resolve()
            out.mkdir(exist_ok=True)
            before = {
                db: {"manifest_hash": "a" * 64}
                for db in (*contract.TRAIN, *contract.DATABASES)
            }
            old = contract.read_json(ROOT / contract.PROVENANCE[3])

            def train(
                config: dict, data: dict, armout: Path, device: str, deadline: object
            ) -> tuple:
                armout.mkdir(parents=True, exist_ok=True)
                return object(), {
                    **old,
                    "config": config,
                    "metadata": {
                        **old["metadata"],
                        "manifest_hashes": {
                            db: v["manifest_hash"] for db, v in before.items()
                        },
                    },
                }

            def evaluate(*args: object, **kwargs: object) -> list:
                return [
                    *kwargs["cells"],
                    {"arm": kwargs["arm"], "operator": kwargs["operator"]},
                ]

            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(
                        runner, "prepare_identities", return_value=({}, {}, {})
                    )
                )
                stack.enter_context(
                    patch.object(contract, "read_json", return_value=old)
                )
                stack.enter_context(patch.object(contract, "verify_reference_training"))
                stack.enter_context(patch.object(contract, "verify_pairing"))
                stack.enter_context(
                    patch.object(contract, "sha256", return_value="0" * 64)
                )
                stack.enter_context(
                    patch.object(screen, "train_arm", side_effect=train)
                )
                stack.enter_context(
                    patch.object(runner, "evaluate_operator", side_effect=evaluate)
                )
                stack.enter_context(patch.object(runner, "phase"))
                reproduction = stack.enter_context(
                    patch(
                        "scripts.meta65_scope_verify.verify_reproduction",
                        side_effect=ValueError("Reproduction identity differs: true"),
                    )
                )
                with self.assertRaises(runner.ReproductionBlocked) as caught:
                    runner.train_and_evaluate(
                        out,
                        contract.FROZEN,
                        {},
                        "cpu_preflight",
                        {},
                        {},
                        before,
                        lambda: None,
                    )
                self.assertIn("identity", str(caught.exception))
                self.assertIsInstance(caught.exception, ValueError)
                # An unrelated error must keep its own type and stay unflagged.
                reproduction.side_effect = OSError("cache read failed")
                with self.assertRaises(OSError):
                    runner.train_and_evaluate(
                        out,
                        contract.FROZEN,
                        {},
                        "cpu_preflight",
                        {},
                        {},
                        before,
                        lambda: None,
                    )


if __name__ == "__main__":
    unittest.main()
