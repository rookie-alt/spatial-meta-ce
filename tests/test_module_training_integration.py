"""Public training dispatch must preserve the new execution and data boundaries."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_module_model_integration import official_binding

from spatial_meta_ce import module_experiment as contract
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.features import write_cache
from spatial_meta_ce.manifests import file_hash
from spatial_meta_ce.module_experiment import RECIPE, STAGE
from spatial_meta_ce.official_split import build_official_capacity
from spatial_meta_ce.plans import build_training_plan
from spatial_meta_ce.price_capacity import OFFICIAL_TRAIN_DATABASES
from spatial_meta_ce.train import train


class ModuleTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            "cache_root": "/synthetic/cache",
            "data_root": "/synthetic/price",
            "splits": {
                "train": list(OFFICIAL_TRAIN_DATABASES),
                "validation": [],
                "test": ["imdb", "stats", "genome", "ergastf1"],
            },
        }
        self.config = {
            **RECIPE,
            "module_arm": "A1-PG",
            "seed": 42,
            "model": {
                "arch": "price_lite",
                "capacity": official_binding(),
                "pg_head_input": True,
            },
        }

    def test_new_stage_calls_bound_execution_validator_before_io(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist"
            with (
                patch(
                    "spatial_meta_ce.module_experiment.validate_training_config",
                    side_effect=ValueError("unsigned module evidence"),
                ) as validator,
                patch(
                    "spatial_meta_ce.train.StorePool",
                    side_effect=AssertionError("cache read"),
                ),
                patch(
                    "spatial_meta_ce.train.create_model",
                    side_effect=AssertionError("model created"),
                ),
                self.assertRaisesRegex(ValueError, "unsigned module evidence"),
            ):
                train(self.config, self.data, output)
            validator.assert_called_once_with(
                self.config, self.data, None, output_dir=output, device="cpu"
            )
            self.assertFalse(output.exists())

    def test_arbitrary_stage_cannot_use_the_module_dispatch(self):
        for stage in (None, "132-module", "132-module26-extra", "122-A1-PG"):
            with (
                self.subTest(stage=stage),
                patch(
                    "spatial_meta_ce.module_experiment.validate_training_config"
                ) as validator,
                patch(
                    "spatial_meta_ce.train.StorePool",
                    side_effect=AssertionError("cache read"),
                ),
                self.assertRaises(ValueError),
            ):
                train({**self.config, "fixed_stage": stage}, self.data, "unused")
            validator.assert_not_called()

    def test_existing_official_stage_keeps_the_old_fixed_model_guard(self):
        config = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/fomaml_joint_sum_4k.json"
            ).read_text()
        )
        config.update(method="erm", meta_objective="post", fixed_stage="122-A1-PG")
        config["model"] = {**self.config["model"], "group_pooling": "attention"}
        with (
            patch(
                "spatial_meta_ce.module_experiment.validate_training_config"
            ) as validator,
            patch(
                "spatial_meta_ce.train.StorePool",
                side_effect=AssertionError("cache read"),
            ),
            self.assertRaisesRegex(ValueError, "frozen model"),
        ):
            train(config, self.data, "unused")
        validator.assert_not_called()
        self.assertEqual(STAGE, "132-module26")

    def test_five_arms_two_update_synthetic_train_and_checkpoint_contract(self):
        """Patch only the test budget/execution evidence; use the real train loop."""
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = {
                **copy.deepcopy(self.data),
                "cache_root": str(root / "cache"),
                "data_root": str(root / "unused-price"),
            }
            mapping_root = root / "mapping"
            for db in OFFICIAL_TRAIN_DATABASES:
                rows = [
                    {
                        "id": f"{db}/pretrain/{i + 1}",
                        "source_line": i + 1,
                        "query_hash": hashlib.sha256(f"{db}-{i}".encode()).hexdigest(),
                        "exact_hash": hashlib.sha256(
                            f"exact-{db}-{i}".encode()
                        ).hexdigest(),
                        "true": float(i + 20),
                        "pg": 2.0,
                    }
                    for i in range(165)
                ]
                parts = (np.ones(80), np.ones(80), np.ones(8), np.ones(43))
                features = {
                    "data_features": [parts for _ in rows],
                    "true_cards": [r["true"] for r in rows],
                    "pg_est_cards": [2.0 for _ in rows],
                    "n_join_cols": [2] * len(rows),
                    "n_fanouts": [2] * len(rows),
                    "n_tables": [2] * len(rows),
                    "n_filter_cols": [1] * len(rows),
                }
                cache = root / "cache/pretrain" / db
                manifest = write_cache(
                    cache, rows, features, {"purpose": "smoke-not-research"}
                )
                path = mapping_root / "pretrain" / db
                path.mkdir(parents=True)
                records = []
                for row in rows:
                    indices = {
                        "counts": [2, 2, 2, 1],
                        "join_column_tables": [0, 1],
                        "filter_column_tables": [0],
                        "fanout_join_columns": [0, 1],
                    }
                    records.append(
                        {
                            **row,
                            "source_lines": [row["source_line"]],
                            "mapping": indices,
                            "shuffled_mapping": copy.deepcopy(indices),
                        }
                    )
                (path / "mapping.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in records)
                )
                arrays = [
                    *(f"group{i}.npy" for i in range(4)),
                    "cards.npy",
                    "offsets.npy",
                ]
                hashes = {
                    f"cache/{name}": file_hash(cache / name)
                    for name in ["manifest.json", *arrays]
                }
                (path / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": 1,
                            "scope": "official_26_4",
                            "split": "pretrain",
                            "database": db,
                            "cache_content_hash": manifest["content_hash"],
                            "input_sha256": hashes,
                            "array_sha256": {
                                name: hashes[f"cache/{name}"] for name in arrays
                            },
                            "jsonl_sha256": file_hash(path / "mapping.jsonl"),
                        }
                    )
                )
            capacity = build_official_capacity(data)
            data_path = root / "data.json"
            data_path.write_text(json.dumps(data))
            plan = build_training_plan(
                data,
                seed=42,
                steps=2,
                meta_batch=4,
                accumulation=1,
                support_sizes=[5, 10, 20, 50, 100],
                query_size=64,
                allow_empty_validation=True,
            )
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            test_recipe = {**contract.RECIPE, "steps": 2, "validate_every": 2}
            with (
                patch.object(contract, "STEPS", 2),
                patch.object(contract, "RECIPE", test_recipe),
                patch.object(
                    contract,
                    "FIXED_SELECTION",
                    "none (synthetic 2 updates, last.pt only)",
                ),
                patch.object(contract, "validate_training_config", return_value={}),
                patch(
                    "spatial_meta_ce.train.validation_plan",
                    side_effect=AssertionError("validation read"),
                ),
                patch(
                    "spatial_meta_ce.train.validate",
                    side_effect=AssertionError("validation selected"),
                ),
            ):
                for arm in contract.ARMS:
                    config = contract.training_config(
                        arm,
                        42,
                        capacity,
                        data_config_binding={
                            "path": str(data_path),
                            "sha256": file_hash(data_path),
                        },
                        training_plan_binding={
                            "path": str(plan_path),
                            "sha256": file_hash(plan_path),
                            "plan_sha256": plan["plan_sha256"],
                        },
                        mapping_root=str(mapping_root)
                        if contract.mapping_mode(arm)
                        else None,
                    )
                    output = root / arm
                    result = train(config, data, output, "cpu")
                    self.assertEqual(result["counts"]["optimizer_steps"], 2)
                    self.assertFalse((output / "best.pt").exists())
                    self.assertFalse((output / "validation_manifest.json").exists())
                    _, state = load_checkpoint(output / "last.pt")
                    contract.validate_checkpoint(state, arm, 42, expected_config=config)
                    self.assertEqual(
                        state["metadata"]["train_databases"],
                        list(OFFICIAL_TRAIN_DATABASES),
                    )
                    self.assertEqual(state["metadata"]["fixed_steps"], 2)
                    self.assertEqual(state["metadata"]["target_databases_read"], [])
                    self.assertEqual(state["extra"]["sample_plan_episodes"], 8)
                    self.assertEqual(
                        state["extra"]["sample_plan_digest"], plan["plan_sha256"]
                    )


if __name__ == "__main__":
    unittest.main()
