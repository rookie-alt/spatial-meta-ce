import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from spatial_meta_ce import train as train_module
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.plans import build_training_plan, verify_plan_against_stores
from scripts.audit_fomaml_joint_gradient import audit_run


def fixtures(root):
    cache = root / "cache"
    for split, db in [
        ("pretrain", "a"),
        ("pretrain", "v"),
        ("finetune", "t"),
        ("test", "t"),
    ]:
        rows = [
            {
                "id": f"{db}/{split}/{i}",
                "query_hash": hashlib.sha256(f"{split}-{i}".encode()).hexdigest(),
                "source_line": i + 1,
                "true": float(20 + i),
                "pg": 2.0,
            }
            for i in range(20)
        ]
        data = {
            "data_features": [
                (torch.ones(40), torch.ones(40), torch.ones(4), None) for _ in rows
            ],
            "true_cards": [torch.tensor(row["true"]) for row in rows],
            "pg_est_cards": [torch.tensor(2.0) for _ in rows],
            "n_join_cols": [1] * 20,
            "n_fanouts": [1] * 20,
            "n_tables": [1] * 20,
            "n_filter_cols": [0] * 20,
        }
        write_cache(cache / split / db, rows, data, {"source": "unit-fixture"})
    data_config = {
        "cache_root": str(cache),
        "data_root": str(root / "unused"),
        "splits": {"train": ["a"], "validation": ["v"], "test": ["t"]},
    }
    config = {
        "model": {
            "width": 8,
            "use_pg": True,
            "arch": "sum",
            "layers": 2,
            "heads": 4,
        },
        "method": "fomaml",
        "seed": 7,
        "steps": 2,
        "meta_batch": 1,
        "accumulation": 1,
        "outer_lr": 0.001,
        "inner_lr": 0.01,
        "inner_steps": 1,
        "adapt_scope": "all",
        "support_sizes": [5],
        "query_size": 4,
        "loss": "smooth_l1",
        "validate_every": 1,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 20,
        "min_delta": 0.0,
        "fixed_budget": True,
        "meta_objective": "joint",
    }
    return config, data_config


class FixedBudgetTrainTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixed_constants(self):
        return {
            "FIXED_STEPS": 2,
            "FIXED_META_BATCH": 1,
            "FIXED_ACCUMULATION": 1,
            "FIXED_QUERY_SIZE": 4,
            "FIXED_SUPPORT_SIZES": (5,),
            "FIXED_LOSS": "smooth_l1",
            "FIXED_INNER_LR": 0.01,
            "FIXED_INNER_STEPS": 1,
            "FIXED_ADAPT_SCOPE": "all",
            "FIXED_OUTER_LR": 0.001,
            "FIXED_SEEDS": (7,),
            "FIXED_MODEL": {
                "width": 8,
                "use_pg": True,
                "arch": "sum",
                "layers": 2,
                "heads": 4,
            },
        }

    def write_plan(self, root, config, data_config):
        plan = build_training_plan(
            data_config,
            seed=config["seed"],
            steps=2,
            meta_batch=1,
            accumulation=1,
            support_sizes=[5],
            query_size=4,
        )
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return plan_path

    def run_fixed(self, root, config, data_config, meta_objective="joint"):
        plan_path = self.write_plan(root, config, data_config)
        config = {
            **config,
            "meta_objective": meta_objective,
            "training_plan": str(plan_path),
        }
        with (
            patch.multiple(train_module, **self.fixed_constants()),
            patch.object(
                train_module,
                "validation_plan",
                side_effect=AssertionError("validation_plan must not be called"),
            ),
            patch.object(
                train_module,
                "validate",
                side_effect=AssertionError("validate must not be called"),
            ),
        ):
            result = train_module.train(config, data_config, root / "out", "cpu")
        return result, config

    def test_joint_fixed_budget_skips_validation_and_logs_pre_post(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            result, _ = self.run_fixed(root, config, data_config)

            self.assertFalse((root / "out" / "validation_manifest.json").exists())
            self.assertTrue((root / "out" / "last.pt").exists())
            self.assertFalse((root / "out" / "best.pt").exists())
            rows = [
                json.loads(line)
                for line in (root / "out" / "loss_audit.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            for step, row in enumerate(rows, start=1):
                self.assertTrue(np.isfinite(row["mean_train_loss"]))
                self.assertTrue(np.isfinite(row["mean_pre_loss"]))
                self.assertTrue(np.isfinite(row["mean_post_loss"]))
                self.assertTrue(np.isfinite(row["grad_norm"]))
                self.assertEqual(row["counts"]["pre_forward_calls"], step)
                self.assertEqual(row["counts"]["joint_pre_grad_calls"], step)

            _, state = load_checkpoint(root / "out" / "last.pt")
            metadata = state["metadata"]
            self.assertEqual(metadata["validation_databases"], [])
            self.assertEqual(metadata["selection_databases"], [])
            self.assertEqual(metadata["validation_monitor_reads"], [])
            self.assertEqual(metadata["selection"], train_module.FIXED_SELECTION)
            self.assertEqual(set(metadata["manifest_hashes"]), {"a"})
            self.assertEqual(metadata["meta_objective"], "joint")
            self.assertEqual(metadata["joint_pre_weight"], 0.5)
            self.assertEqual(len(metadata["initial_state_sha256"]), 64)
            self.assertEqual(metadata["fixed_steps"], 2)
            self.assertIsNone(result["best_validation_macro_log_q95"])
            self.assertEqual(result["meta_objective"], "joint")
            self.assertTrue(result["fixed_budget"])

    def test_post_objective_logs_pre_without_a_joint_gradient_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            result, _ = self.run_fixed(root, config, data_config, meta_objective="post")
            rows = [
                json.loads(line)
                for line in (root / "out" / "loss_audit.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertTrue(np.isfinite(rows[0]["mean_pre_loss"]))
            self.assertEqual(rows[0]["counts"]["joint_pre_grad_calls"], 0)
            self.assertEqual(result["meta_objective"], "post")

    def test_joint_objective_requires_fixed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            config = {**config, "fixed_budget": False}
            with self.assertRaisesRegex(ValueError, "fixed_budget"):
                train_module.train(config, data_config, root / "out", "cpu")

    def test_fixed_budget_rejects_wrong_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            config = {**config, "steps": 3}
            with patch.multiple(
                train_module,
                FIXED_STEPS=2,
                FIXED_META_BATCH=1,
                FIXED_ACCUMULATION=1,
                FIXED_QUERY_SIZE=4,
                FIXED_SUPPORT_SIZES=(5,),
            ):
                with self.assertRaisesRegex(ValueError, "steps=2"):
                    train_module.train(config, data_config, root / "out", "cpu")

    def test_fixed_budget_requires_a_training_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            config = {**config, "training_plan": ""}
            with patch.multiple(
                train_module,
                FIXED_STEPS=2,
                FIXED_META_BATCH=1,
                FIXED_ACCUMULATION=1,
                FIXED_QUERY_SIZE=4,
                FIXED_SUPPORT_SIZES=(5,),
            ):
                with self.assertRaisesRegex(ValueError, "training_plan"):
                    train_module.train(config, data_config, root / "out", "cpu")

    def test_fixed_budget_rejects_wrong_inner_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            plan_path = self.write_plan(root, config, data_config)
            config = {
                **config,
                "training_plan": str(plan_path),
                "inner_steps": 2,
            }
            with patch.multiple(train_module, **self.fixed_constants()):
                with self.assertRaisesRegex(ValueError, "inner_steps=1"):
                    train_module.train(config, data_config, root / "out", "cpu")

    def test_training_plan_requires_fixed_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            plan_path = self.write_plan(root, config, data_config)
            config = {
                **config,
                "fixed_budget": False,
                "meta_objective": "post",
                "training_plan": str(plan_path),
            }
            with self.assertRaisesRegex(ValueError, "training_plan"):
                train_module.train(config, data_config, root / "out", "cpu")

    def test_protocol_constants_are_the_112_budget(self):
        self.assertEqual(train_module.FIXED_STEPS, 4000)
        self.assertEqual(train_module.FIXED_META_BATCH, 4)
        self.assertEqual(train_module.FIXED_ACCUMULATION, 1)
        self.assertEqual(train_module.FIXED_QUERY_SIZE, 64)
        self.assertEqual(train_module.FIXED_SUPPORT_SIZES, (5, 10, 20, 50, 100))
        self.assertEqual(train_module.FIXED_LOSS, "smooth_l1")
        self.assertEqual(train_module.FIXED_INNER_LR, 0.001)
        self.assertEqual(train_module.FIXED_INNER_STEPS, 5)
        self.assertEqual(train_module.FIXED_ADAPT_SCOPE, "all")
        self.assertEqual(train_module.FIXED_OUTER_LR, 0.0003)
        self.assertEqual(train_module.FIXED_SEEDS, (42, 43, 44))
        self.assertEqual(
            train_module.FIXED_MODEL,
            {"width": 64, "use_pg": True, "arch": "sum", "layers": 2, "heads": 4},
        )

    def test_gradient_audit_recomputes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            _, config = self.run_fixed(root, config, data_config)
            self.assertTrue((root / "out" / "initial_state.pt").is_file())
            audit = json.loads((root / "out" / "gradient_audit.json").read_text())
            self.assertEqual(len(audit["gradient_sha256"]), 64)
            self.assertIsNotNone(audit["pre_loss"])
            self.assertTrue(audit["post_loss"] >= 0)
            result = audit_run(root / "out", config["training_plan"])
            self.assertTrue(result["verified"])
            self.assertTrue(all(result["comparisons"].values()))

    def test_fixed_budget_rejects_wrong_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            plan_path = self.write_plan(root, config, data_config)
            config = {
                **config,
                "seed": 8,
                "training_plan": str(plan_path),
            }
            with patch.multiple(train_module, **self.fixed_constants()):
                with self.assertRaisesRegex(ValueError, "seed in"):
                    train_module.train(config, data_config, root / "out", "cpu")

    def test_fixed_budget_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            plan_path = self.write_plan(root, config, data_config)
            config = {**config, "training_plan": str(plan_path)}
            with patch.multiple(train_module, **self.fixed_constants()):
                with self.assertRaisesRegex(ValueError, "does not authorize resume"):
                    train_module.train(
                        config,
                        data_config,
                        root / "out",
                        "cpu",
                        resume=str(root / "missing.pt"),
                    )

    def test_fixed_budget_rejects_complex_query_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            plan_path = self.write_plan(root, config, data_config)
            config = {
                **config,
                "method": "erm",
                "meta_objective": "post",
                "complex_query_weight": 2,
                "training_plan": str(plan_path),
            }
            with patch.multiple(train_module, **self.fixed_constants()):
                with self.assertRaisesRegex(ValueError, "complex_query_weight"):
                    train_module.train(config, data_config, root / "out", "cpu")

    def test_training_plan_store_verification_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, data_config = fixtures(root)
            plan = build_training_plan(
                data_config,
                seed=config["seed"],
                steps=2,
                meta_batch=1,
                accumulation=1,
                support_sizes=[5],
                query_size=4,
            )
            pool = StorePool(str(root / "cache"), max_open=1)
            stores = {"a": pool.get("pretrain", "a")}
            self.assertTrue(verify_plan_against_stores(plan, stores))
            original_hash = plan["data_hashes"]["a"]
            plan["data_hashes"]["a"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "label/feature"):
                verify_plan_against_stores(plan, stores)
            plan["data_hashes"]["a"] = original_hash
            plan["episodes"][0]["support_hashes"][0] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_plan_against_stores(plan, stores)


if __name__ == "__main__":
    unittest.main()
