"""Synthetic P0/A integration checks, never a source/target experiment."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce.data import TRANSFORM_VERSION
from spatial_meta_ce.manifests import digest
from spatial_meta_ce.model import create_model
from spatial_meta_ce.plans import build_source_eval_plan, validate_source_eval_plan
from spatial_meta_ce.price_capacity import (
    SOURCE_DATABASES,
    TRAIN_DATABASES,
    build_capacity,
    scan_source_capacity,
)
from spatial_meta_ce.price_protocol import a_model_config, summarize_a, validate_a_plan
from spatial_meta_ce.train import _validate_fixed_budget
from test_source_eval import data_config as small_data_config, make_cache

ROOT = Path(__file__).resolve().parents[1]


def manifests(root, overflow=False):
    for i, db in enumerate(TRAIN_DATABASES + SOURCE_DATABASES):
        counts = [1 + int(i == j) for j in range(4)]
        if db in SOURCE_DATABASES:
            counts = [3 if overflow else 2, 2, 2, 2]
        payload = {
            "schema": 1,
            "transform": TRANSFORM_VERSION,
            "inspection": {"max_counts": counts},
            "rows": [{"id": f"{db}/pretrain/1", "query_hash": digest(db)}],
        }
        entry = {**payload, "content_hash": digest(payload)}
        directory = root / "pretrain" / db
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps(entry))
        np.save(directory / "offsets.npy", np.array([[0, 0, 0, 0], counts]))
    return {
        "cache_root": str(root),
        "splits": {
            "train": list(TRAIN_DATABASES),
            "validation": list(SOURCE_DATABASES),
            "test": ["imdb", "stats", "genome", "ergastf1"],
        },
    }


class PriceProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_source_manifest_underreports_offsets_must_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = manifests(root)
            binding = build_capacity(data)
            np.save(
                root / "pretrain/chembl/offsets.npy",
                np.array([[0, 0, 0, 0], [3, 2, 2, 2]]),
            )
            with self.assertRaisesRegex(ValueError, "maxima mismatch"):
                scan_source_capacity(data, binding)

    def test_corrupt_manifest_leaves_a_failed_preflight_record(self):
        from scripts.price_meta_p0 import prepare

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = manifests(root / "cache")
            path = root / "cache/pretrain/accidents/manifest.json"
            manifest = json.loads(path.read_text())
            manifest["inspection"]["max_counts"][0] += 1
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare(data, root / "plans", root / "out")
            record = json.loads((root / "out/preflight.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "ValueError")
            self.assertFalse(record["p0_gate_passed"])
            self.assertFalse(record["formal_training_authorized"])

    def test_malformed_a_checkpoint_is_rejected_with_value_error(self):
        from spatial_meta_ce.price_protocol import validate_a_checkpoint

        for metadata in (
            {},
            None,
            [],
            {"training_signature": {}},
            {"training_signature": None},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_a_checkpoint({"metadata": metadata}, "A1", 42)

    def test_null_or_nonstring_stage_is_a_clear_configuration_error(self):
        from spatial_meta_ce.train import train

        base = json.loads((ROOT / "configs/fomaml_joint_sum_4k.json").read_text())
        data = json.loads((ROOT / "configs/data.json").read_text())
        for stage in (None, 7):
            config = {
                **base,
                "method": "erm",
                "meta_objective": "post",
                "model": {"arch": "price_lite"},
                "fixed_stage": stage,
            }
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                train(config, data, "unused")

    def test_capacity_uses_exactly_train_maxima_and_detects_source_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = manifests(root, overflow=True)
            binding = build_capacity(config)
            self.assertEqual(binding["max_counts"], [2, 2, 2, 2])
            scan = scan_source_capacity(config, binding)
            self.assertFalse(scan["full_population_supported"])
            self.assertEqual(len(scan["unsupported"]), 3)
            self.assertEqual(scan["target_databases_read"], [])
            self.assertEqual(
                scan["unsupported"][0],
                {
                    "database": "chembl",
                    "query_hash": digest("chembl"),
                    "group": "join_hist",
                    "actual": 3,
                    "capacity": 2,
                },
            )
            changed = {
                **config,
                "splits": {**config["splits"], "train": ["imdb", *TRAIN_DATABASES[1:]]},
            }
            with self.assertRaisesRegex(ValueError, "23 training|leakage"):
                build_capacity(changed)
            path = root / "pretrain" / TRAIN_DATABASES[0] / "manifest.json"
            entry = json.loads(path.read_text())
            entry["inspection"]["max_counts"][0] = 8
            path.write_text(json.dumps(entry))
            with self.assertRaisesRegex(ValueError, "hash"):
                build_capacity(config)

    def test_capacity_rejects_incomplete_sources_invalid_maxima_and_hash_bindings(self):
        from spatial_meta_ce.price_capacity import (
            read_source_manifest,
            validate_capacity,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = manifests(root)
            original = build_capacity(data)
            self.assertTrue(
                scan_source_capacity(data, original)["full_population_supported"]
            )
            for change in ("source", "counts", "missing", "maxima", "hash", "table"):
                invalid = copy.deepcopy(original)
                if change == "source":
                    invalid["databases"] = ["imdb", *TRAIN_DATABASES[1:]]
                elif change == "counts":
                    invalid["max_counts"] = [-1, 2, 2, 2]
                elif change == "missing":
                    invalid["database_max_counts"].pop(TRAIN_DATABASES[0])
                elif change == "maxima":
                    invalid["max_counts"] = [3, 2, 2, 2]
                elif change == "hash":
                    invalid["manifest_sha256"][TRAIN_DATABASES[0]] = "bad"
                else:
                    invalid["max_counts"][2] = 0
                    invalid["database_max_counts"] = {
                        db: [2, 2, 0, 2] for db in TRAIN_DATABASES
                    }
                invalid["capacity_sha256"] = digest(
                    {k: v for k, v in invalid.items() if k != "capacity_sha256"}
                )
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_capacity(invalid)
            with self.assertRaises(ValueError):
                validate_capacity(None)
            with self.assertRaises(ValueError):
                read_source_manifest(root, "imdb")
            with self.assertRaises(ValueError):
                validate_capacity(original, {TRAIN_DATABASES[0]: "0" * 64})

    def test_budget_stage_binding_accepts_only_frozen_a_model_scope_method(self):
        base = json.loads((ROOT / "configs/fomaml_joint_sum_4k.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            binding = build_capacity(manifests(Path(directory)))
            for arm in ("A0-ref", "A0", "A1", "A2"):
                config = {
                    **base,
                    "method": "erm",
                    "meta_objective": "post",
                    "fixed_stage": f"122-{arm}",
                    "model": a_model_config(arm, binding),
                }
                _validate_fixed_budget(config)
                for field, value in (
                    ("adapt_scope", "head"),
                    ("adapt_scope", "lora"),
                    ("method", "fomaml"),
                    ("steps", 3999),
                    ("outer_lr", 0.001),
                    ("fixed_stage", "122-B"),
                ):
                    with (
                        self.subTest(arm=arm, field=field),
                        self.assertRaises(ValueError),
                    ):
                        _validate_fixed_budget({**config, field: value})
                bad = {**config["model"], "arch": "attn"}
                with self.assertRaises(ValueError):
                    _validate_fixed_budget({**config, "model": bad})
            with self.assertRaises(ValueError):
                _validate_fixed_budget({**base, "model": a_model_config("A1", binding)})

    def test_122_plan_binds_ids_and_uses_nested_five_twenty_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            data = small_data_config(cache, root)
            with patch("spatial_meta_ce.price_protocol.validate_source_splits"):
                plan = build_source_eval_plan(data, expected_unique=None, protocol=122)
                validate_a_plan(plan, data)
            self.assertEqual(plan["analysis_role"], "population_primary")
            for entry in plan["databases"].values():
                hashes = [
                    h for b in entry["support_blocks"] for h in b["support_hashes"]
                ]
                self.assertEqual(len(set(hashes)), 100)
                self.assertFalse(set(hashes) & set(entry["query_hashes"]))
                self.assertEqual(entry["excluded_unique"], 100)
                for block in entry["support_blocks"]:
                    self.assertEqual(len(block["support_ids"]), 20)
                    self.assertEqual(
                        block["support_ids"][:5],
                        [
                            f"{block['support_ids'][0].split('/')[0]}/pretrain/{i}"
                            for i in block["support"][:5]
                        ],
                    )
            bad = {**plan, "plan_seed": 2718}
            with patch("spatial_meta_ce.price_protocol.validate_source_splits"):
                with self.assertRaisesRegex(ValueError, "diagnostic|population"):
                    validate_a_plan(bad, data)
            bad = copy.deepcopy(plan)
            bad["databases"]["chembl"]["support_blocks"][0]["support_ids"].pop()
            with self.assertRaises(ValueError):
                validate_source_eval_plan(bad, data)

    def test_a_decision_uses_log_q95_three_seed_sd_and_per_database_guard(self):
        units = []
        for arm, delta in (("A0-ref", 0.3), ("A0", 0), ("A1", -0.2), ("A2", -0.15)):
            for seed in (42, 43, 44):
                for db in SOURCE_DATABASES:
                    units.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "database": db,
                            "k": 0,
                            "run": 0,
                            "metrics": {"log_q95": 2 + delta + 0.01 * (seed - 42)},
                            "source_eval_plan_sha256": digest("synthetic shared plan"),
                        }
                    )
        result = summarize_a(units)
        self.assertEqual(result["status"], "a_passed_source_screen")
        differences = np.array([1.8, 1.81, 1.82]) - np.array([2.0, 2.01, 2.02])
        self.assertAlmostEqual(
            result["comparisons"]["A1-A0"]["mean_delta"], differences.mean()
        )
        self.assertAlmostEqual(
            result["comparisons"]["A1-A0"]["sd_delta"], differences.std(ddof=1)
        )
        guarded = [
            (
                {**u, "metrics": {"log_q95": 2.2}}
                if u["arm"] == "A2" and u["database"] == "telstra"
                else u
            )
            for u in units
        ]
        self.assertEqual(summarize_a(guarded)["status"], "a_no_candidate")
        with self.assertRaisesRegex(ValueError, "complete|duplicate"):
            summarize_a(units[:-1])
        with self.assertRaises(ValueError):
            summarize_a(units + [units[0]])

    def test_training_preflight_missing_or_changed_hash_is_rejected(self):
        from spatial_meta_ce.price_protocol import validate_training_preflight

        with self.assertRaisesRegex(ValueError, "source_preflight"):
            validate_training_preflight({}, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preflight.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "source_preflight"):
                validate_training_preflight(
                    {
                        "source_preflight": str(path),
                        "source_preflight_sha256": "0" * 64,
                    },
                    {},
                )

    def test_unit_diagnostics_match_independent_numpy_and_empty_bucket(self):
        from spatial_meta_ce.source_eval import _unit_diagnostics
        from spatial_meta_ce.data import StorePool

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = make_cache(root)
            store = StorePool(cache).get("pretrain", "chembl")
            prediction = np.array([-3.0, 2.0, 9.0, 3.0])
            base = np.array([1.0, 1.0, 1.0, 1.0])
            truth = np.array(store.cards[:4, 0])
            result = _unit_diagnostics(store, np.arange(4), prediction, base)
            error = np.abs(np.maximum(prediction, 0) - np.log(np.maximum(truth, 1)))
            reference = np.abs(np.maximum(base, 0) - np.log(np.maximum(truth, 1)))
            self.assertAlmostEqual(
                result["metrics"]["log_q95"], np.percentile(error, 95)
            )
            self.assertAlmostEqual(
                result["metrics"]["q95"], np.percentile(np.exp(error), 95)
            )
            self.assertEqual(
                result["worsened_fraction_vs_same_model_k0"],
                float(np.mean(error > reference)),
            )
            self.assertIsNone(result["four_plus"])

    def test_rejected_122_checkpoint_creates_no_target_output(self):
        from types import SimpleNamespace
        from spatial_meta_ce.cli import adapt_one
        from spatial_meta_ce.evaluate import evaluate

        config = {
            "model": {"arch": "sum"},
            "shots": [0],
            "support_runs": 1,
            "latency_repetitions": 1,
        }
        data = {
            "cache_root": "unused",
            "splits": {"train": ["a"], "validation": ["v"], "test": ["imdb"]},
        }
        state = {
            "model_type": "grouped_set_residual",
            "metadata": {"fixed_stage": "122-A0-ref"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "spatial_meta_ce.evaluate.load_checkpoint", return_value=(None, state)
            ):
                with self.assertRaisesRegex(ValueError, "not authorized"):
                    evaluate(config, data, root / "eval", checkpoint="synthetic.pt")
            args = SimpleNamespace(
                checkpoint="synthetic.pt", database="imdb", k=5, device="cpu"
            )
            with patch(
                "spatial_meta_ce.checkpoints.load_checkpoint",
                return_value=(None, state),
            ):
                with self.assertRaisesRegex(ValueError, "not authorized"):
                    adapt_one(config, data, root / "adapt", args)
            self.assertFalse((root / "eval").exists())
            self.assertFalse((root / "adapt").exists())

    def test_target_entry_rejects_before_opening_stores(self):
        from spatial_meta_ce.evaluate import evaluate

        with patch(
            "spatial_meta_ce.evaluate.StorePool",
            side_effect=AssertionError("target read"),
        ):
            with self.assertRaisesRegex(ValueError, "source-only"):
                evaluate({"model": {"arch": "price_lite"}}, {}, "unused")

    def test_p0_missing_cache_records_incomplete_without_authorizing_training(self):
        from scripts.price_meta_p0 import prepare

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = json.loads((ROOT / "configs/data.json").read_text())
            data = {**data, "cache_root": str(root / "missing")}
            result = prepare(data, root / "plans", root / "out")
            self.assertEqual(result["status"], "incomplete_missing_evidence")
            self.assertFalse(result["p0_gate_passed"])
            self.assertFalse(result["formal_training_authorized"])
            self.assertEqual(result["target_databases_read"], [])
            self.assertTrue((root / "out/preflight.json").is_file())
            self.assertFalse((root / "out/A1").exists())

    def test_smoke_cannot_exceed_two_updates_and_uses_cpu_supervision(self):
        from scripts.price_meta_p0 import run_smoke, synthetic_smoke

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "2 updates"):
                synthetic_smoke(root / "bad", updates=3)
            with patch(
                "scripts.price_meta_p0.supervise",
                return_value={"status": "preflight_completed"},
            ) as supervised:
                run_smoke(root / "smoke", updates=2)
            call = supervised.call_args
            self.assertEqual(call.kwargs["mode"], "cpu_preflight")
            self.assertEqual(call.kwargs["budget"]["total_seconds"], 300)
            self.assertEqual(call.kwargs["budget"]["rss_bytes"], 4 * 1024**3)
            self.assertEqual(call.kwargs["budget"]["vram_bytes"], 0)
            self.assertNotIn("--device", call.args[0])

    def test_synthetic_a_train_checkpoint_eval_end_to_end(self):
        """Two-update synthetic ERM fixture, not the formal 4000-step A run."""
        from spatial_meta_ce import train as training
        from spatial_meta_ce.checkpoints import load_checkpoint
        from spatial_meta_ce.data import StorePool
        from spatial_meta_ce.features import write_cache
        from spatial_meta_ce.plans import build_training_plan
        from spatial_meta_ce.price_protocol import validate_a_checkpoint
        from spatial_meta_ce.source_eval import evaluate_source
        from spatial_meta_ce.manifests import file_hash

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            for db in TRAIN_DATABASES + SOURCE_DATABASES:
                rows = [
                    {
                        "id": f"{db}/pretrain/{i}",
                        "query_hash": digest([db, i]),
                        "source_line": i + 1,
                        "true": float(20 + i),
                        "pg": 2.0,
                    }
                    for i in range(170)
                ]
                features = {
                    "data_features": [
                        (torch.ones(40), torch.ones(40), torch.ones(4), None)
                        for _ in rows
                    ],
                    "true_cards": [torch.tensor(r["true"]) for r in rows],
                    "pg_est_cards": [torch.tensor(2.0) for _ in rows],
                    "n_join_cols": [1] * 170,
                    "n_fanouts": [1] * 170,
                    "n_tables": [1] * 170,
                    "n_filter_cols": [0] * 170,
                }
                write_cache(
                    cache / "pretrain" / db,
                    rows,
                    features,
                    {"purpose": "smoke-not-research"},
                )
            data = {
                "cache_root": str(cache),
                "splits": {
                    "train": list(TRAIN_DATABASES),
                    "validation": list(SOURCE_DATABASES),
                    "test": ["imdb", "stats", "genome", "ergastf1"],
                },
            }
            binding = build_capacity(data)
            config = {
                **json.loads((ROOT / "configs/fomaml_joint_sum_4k.json").read_text()),
                "steps": 2,
                "method": "erm",
                "meta_objective": "post",
                "fixed_stage": "122-A1",
                "model": a_model_config("A1", binding),
            }
            plan = build_training_plan(
                data,
                seed=42,
                steps=2,
                meta_batch=4,
                accumulation=1,
                support_sizes=[5, 10, 20, 50, 100],
                query_size=64,
            )
            plan_path = root / "training.json"
            plan_path.write_text(json.dumps(plan))
            source_plan = build_source_eval_plan(
                data, expected_unique=None, protocol=122
            )
            source_path = root / "source.json"
            source_path.write_text(json.dumps(source_plan))
            preflight_path = root / "preflight.json"
            preflight = {
                "protocol": 122,
                "status": "data_preflight_passed",
                "capacity": binding,
                "p0_gate_passed": True,
                "purpose": "smoke-not-research synthetic fixture",
                "coverage": scan_source_capacity(data, binding),
                "target_databases_read": [],
                "training_plans": {"42": {"file_sha256": file_hash(plan_path)}},
            }
            preflight_path.write_text(json.dumps(preflight))
            config = {
                **config,
                "training_plan": str(plan_path),
                "source_preflight": str(preflight_path),
                "source_preflight_sha256": file_hash(preflight_path),
            }
            original_get = StorePool.get

            def train_only(pool, split, db):
                if db not in TRAIN_DATABASES or split != "pretrain":
                    raise AssertionError("Training opened a non-training shard")
                return original_get(pool, split, db)

            with (
                patch.object(training, "FIXED_STEPS", 2),
                patch.object(StorePool, "get", train_only),
            ):
                result = training.train(config, data, root / "run", "cpu")
            self.assertEqual(result["last_step"], 2)
            self.assertFalse((root / "run/best.pt").exists())
            model, state = load_checkpoint(root / "run/last.pt")
            self.assertEqual(state["metadata"]["validation_databases"], [])
            with patch.object(training, "FIXED_STEPS", 2):
                validate_a_checkpoint(state, "A1", 42)
                evaluation = evaluate_source(
                    root / "run/last.pt",
                    "A1",
                    42,
                    source_path,
                    data,
                    root / "eval",
                    batch_size=35,
                )
            self.assertEqual(len(evaluation["units"]), 33)
            self.assertEqual(evaluation["target_databases_read"], [])
            self.assertTrue(all(u["four_plus"] is None for u in evaluation["units"]))
            self.assertTrue(
                all(
                    len(u["support_ids"]) == u["k"]
                    for u in evaluation["units"]
                    if u["k"]
                )
            )
            from scripts.price_meta_p0 import prepare

            plans_dir = root / "sealed"
            plans_dir.mkdir()
            for seed in (42, 43, 44):
                seed_plan = build_training_plan(
                    data,
                    seed=seed,
                    steps=2,
                    meta_batch=4,
                    accumulation=1,
                    support_sizes=[5, 10, 20, 50, 100],
                    query_size=64,
                )
                (plans_dir / f"training_plan_seed{seed}.json").write_text(
                    json.dumps(seed_plan)
                )
            legacy_source = {
                k: v
                for k, v in source_plan.items()
                if k not in ("protocol", "analysis_role", "plan_sha256")
            }
            legacy_source["plan_sha256"] = digest(legacy_source)
            (plans_dir / "source_eval_plan.json").write_text(json.dumps(legacy_source))
            with (
                patch.object(training, "FIXED_STEPS", 2),
                patch("scripts.price_meta_p0.ROOT", root),
            ):
                (root / "configs").mkdir()
                base_path = root / "configs/fomaml_joint_sum_4k.json"
                base_path.write_text(json.dumps({**config, "steps": 2}))
                prepared = prepare(data, plans_dir, root / "prepared")
            self.assertEqual(prepared["status"], "data_preflight_passed")
            self.assertEqual(len(list((root / "prepared").glob("A*-seed*.json"))), 12)
            loaded_plan = json.loads(
                (root / "prepared/source_eval_plan_122.json").read_text()
            )
            for db in SOURCE_DATABASES:
                self.assertEqual(
                    loaded_plan["databases"][db]["query_hashes"],
                    source_plan["databases"][db]["query_hashes"],
                )
            self.assertFalse(prepared["p0_gate_passed"])
            for field, value in (
                ("fixed_stage", "122-A2"),
                ("purpose", "smoke-not-research"),
                ("training_plan_sha256", "bad"),
                ("validation_databases", ["chembl"]),
            ):
                with (
                    patch.object(training, "FIXED_STEPS", 2),
                    self.assertRaises(ValueError),
                ):
                    validate_a_checkpoint(
                        {**state, "metadata": {**state["metadata"], field: value}},
                        "A1",
                        42,
                    )
            for bad_arm, bad_seed in (("B", 42), ("A1", 7)):
                with self.assertRaises(ValueError):
                    validate_a_checkpoint(state, bad_arm, bad_seed)
            config = {
                **config,
                "source_preflight": str(root / "prepared/preflight.json"),
                "source_preflight_sha256": file_hash(root / "prepared/preflight.json"),
            }
            from spatial_meta_ce.price_protocol import validate_training_preflight

            with self.assertRaisesRegex(ValueError, "P0 gate"):
                validate_training_preflight(config, data)
            for key, value in (("next_step", 1), ("model_config", {"arch": "sum"})):
                with (
                    patch.object(training, "FIXED_STEPS", 2),
                    self.assertRaises(ValueError),
                ):
                    validate_a_checkpoint({**state, key: value}, "A1", 42)

    def test_price_model_cannot_silently_select_another_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            binding = build_capacity(manifests(Path(directory)))
            for arm, expected in (("A1", 293905), ("A2", 86769)):
                config = a_model_config(arm, binding)
                self.assertEqual(
                    sum(p.numel() for p in create_model(config).parameters()), expected
                )
                with self.assertRaises(ValueError):
                    create_model({**config, "d_model": 48})


if __name__ == "__main__":
    unittest.main()
