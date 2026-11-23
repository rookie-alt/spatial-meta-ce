"""Synthetic-only tests for the migrated-server unsigned spec preparation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_meta_ce import module_experiment as contract

REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    REPO_ROOT
    / "reports"
    / "official-module-migration-20261123"
    / "prepare_new_server.py"
)


def load_helper():
    spec = importlib.util.spec_from_file_location("prepare_new_server", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_helper()


def budget():
    return {
        "total_seconds": 60.0,
        "phase_seconds": {
            "preparation": 20.0,
            "evaluation": 20.0,
            "certification": 20.0,
        },
        "rss_bytes": 1024,
        "vram_bytes": 1024,
        "output_bytes": 1024,
        "poll_seconds": 0.5,
        "cpu_core_seconds": 40,
        "gpu_allocated_seconds": 40,
        "run_deadline_seconds": 10.0,
        "eval_deadline_seconds": 10.0,
    }


def synthetic_inputs(root):
    """26 tiny caches, three two-update plans and a matching split manifest."""
    import numpy as np
    from spatial_meta_ce.data import GROUP_DIMS, TRANSFORM_VERSION
    from spatial_meta_ce.official_split import build_official_capacity
    from spatial_meta_ce.plans import build_training_plan

    root = root / "inputs"
    root.mkdir()
    cache_root = root / "cache"
    for db in contract.TRAIN_DATABASES:
        path = cache_root / "pretrain" / db
        path.mkdir(parents=True)
        rows = [
            {
                "id": f"{db}/pretrain/{i + 1}",
                "source_line": i + 1,
                "query_hash": f"{i + 1:064x}",
            }
            for i in range(165)
        ]
        payload = {
            "schema": 1,
            "transform": TRANSFORM_VERSION,
            "inspection": {"max_counts": [1, 2, 1, 1]},
            "rows": rows,
        }
        (path / "manifest.json").write_text(
            json.dumps({**payload, "content_hash": contract.canonical_digest(payload)})
        )
        counts = np.asarray([1, 2, 1, 1])
        np.save(path / "offsets.npy", np.arange(166)[:, None] * counts)
        np.save(path / "cards.npy", np.full((165, 2), 2.0))
        for index, dim in enumerate(GROUP_DIMS):
            np.save(
                path / f"group{index}.npy",
                np.ones((165 * counts[index], dim), dtype=np.float32),
            )
    data = {
        "cache_root": str(cache_root),
        "data_root": str(root / "readonly-price"),
        "splits": {
            "train": list(contract.TRAIN_DATABASES),
            "validation": [],
            "test": list(contract.TEST_DATABASES),
        },
    }
    data_path, cap_path = root / "data.json", root / "capacity.json"
    data_path.write_text(json.dumps(data))
    cap = build_official_capacity(data)
    cap_path.write_text(json.dumps(cap))
    plans, entries = {}, {}
    for seed in contract.SEEDS:
        plan = build_training_plan(
            data,
            seed=seed,
            steps=2,
            meta_batch=4,
            accumulation=1,
            support_sizes=[5, 10, 20, 50, 100],
            query_size=64,
            allow_empty_validation=True,
        )
        path = root / f"plan{seed}.json"
        path.write_text(json.dumps(plan, indent=2, sort_keys=True))
        plans[str(seed)] = str(path)
        entries[str(seed)] = {
            "path": str(path),
            "plan_sha256": plan["plan_sha256"],
            "file_sha256": runner_sha256(path),
            "episodes": 8,
        }
    manifest = root / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "kind": "official_split_26_train_4_test",
                "train_databases": list(contract.TRAIN_DATABASES),
                "test_databases": list(contract.TEST_DATABASES),
                "capacity_sha256": cap["capacity_sha256"],
                "max_counts": cap["max_counts"],
                "plans": entries,
            }
        )
    )
    report = {
        "scope": "official_26_4",
        "mapping_root": str(root / "mapping"),
        "cache_root": str(cache_root),
        "synthetic_mocked_certification": True,
    }
    report_path = root / "mapping-report.json"
    report_path.write_text(json.dumps(report))
    budget_path = root / "budget.json"
    budget_path.write_text(json.dumps(budget()))
    return {
        "data_config": str(data_path),
        "capacity": str(cap_path),
        "split_manifest": str(manifest),
        "mapping_report": str(report_path),
        "budget": str(budget_path),
    }


def runner_sha256(path):
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MigrationTrainingPrepTests(unittest.TestCase):
    def prepare(self, root, inputs, **overrides):
        from scripts import run_official_module_experiment as runner

        arguments = {
            "out": root / "prepared",
            "output_root": root / "outputs",
            "code_dir": REPO_ROOT,
            "data_config": inputs["data_config"],
            "capacity": inputs["capacity"],
            "split_manifest": inputs["split_manifest"],
            "mapping_report": inputs["mapping_report"],
            "budget_path": inputs["budget"],
            "cuda_device": 1,
        }
        arguments.update(overrides)
        recipe = {**contract.RECIPE, "steps": 2, "validate_every": 2}
        with (
            patch.object(contract, "STEPS", 2),
            patch.object(contract, "RECIPE", recipe),
            patch.object(runner, "STEPS", 2),
            patch.object(
                runner,
                "_mapping_report",
                return_value=json.loads(Path(inputs["mapping_report"]).read_text()),
            ),
        ):
            return HELPER.prepare(**arguments)

    def test_preparation_writes_fifteen_unsigned_specs_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            result = self.prepare(root, inputs)

            self.assertTrue(result["prepared"])
            self.assertFalse(result["training_run"])
            self.assertFalse(result["model_predictions_run"])
            self.assertFalse(result["authorized"])
            self.assertEqual(result["target_databases_read_for_training"], [])
            self.assertEqual(len(result["runs"]), 15)

            out = root / "prepared"
            readiness = json.loads((out / "readiness.json").read_text())
            self.assertTrue(readiness["prepared"])
            self.assertFalse(readiness["authorized"])
            self.assertEqual(len(readiness["runs"]), 15)

            keys = {record["key"] for record in result["runs"]}
            self.assertEqual(
                keys,
                {
                    f"{arm}-seed{seed}"
                    for arm in contract.ARMS
                    for seed in contract.SEEDS
                },
            )
            for record in result["runs"]:
                folder = out / record["key"]
                spec = json.loads((folder / "training_spec.json").read_text())
                template = json.loads(
                    (folder / "authorization_template.json").read_text()
                )
                self.assertEqual(spec["status"], "unsigned")
                self.assertEqual(spec["action"], "train")
                self.assertEqual(spec["validation_databases"], [])
                self.assertEqual(spec["target_databases_read"], [])
                self.assertEqual(spec["steps"], 2)
                self.assertEqual(spec["execution"]["mode"], "single_run")
                self.assertEqual(
                    spec["execution"]["arm"], record["key"].split("-seed")[0]
                )
                self.assertFalse(template["authorized"])
                self.assertFalse(template["formal_training_authorized"])
                self.assertFalse(template["formal_evaluation_authorized"])
                self.assertIsNone(template["granted_by"])
                self.assertIsNone(template["granted_at"])
                self.assertIn("unsigned", record["unsigned_execute_rejected"])
                self.assertFalse(Path(spec["output_dir"]).exists())

    def test_preparation_reads_only_source_training_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            result = self.prepare(root, inputs)
            for spec_binding in result["common_bindings"].values():
                if not isinstance(spec_binding, dict):
                    continue
                self.assertNotIn("test", spec_binding.get("path", ""))
            cache = root / "inputs" / "cache"
            self.assertTrue((cache / "pretrain").is_dir())
            self.assertFalse((cache / "test").exists())

    def test_preparation_refuses_output_paths_overlapping_readonly_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            data = json.loads(Path(inputs["data_config"]).read_text())
            for label, target in (
                ("data_root", data["data_root"]),
                ("cache_root", data["cache_root"]),
            ):
                with self.subTest(label=label):
                    with self.assertRaises((ValueError, FileExistsError)):
                        self.prepare(root, inputs, out=Path(target) / "prepared")
            with self.assertRaises(ValueError):
                self.prepare(
                    root,
                    inputs,
                    output_root=Path(data["cache_root"]) / "outputs",
                )

    def test_preparation_refuses_existing_output_and_overlapping_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            (root / "prepared").mkdir()
            with self.assertRaises(FileExistsError):
                self.prepare(root, inputs)
            with self.assertRaises(ValueError):
                self.prepare(
                    root,
                    inputs,
                    out=root / "new-prepared" / "inner",
                    output_root=root / "new-prepared",
                )

    def test_preparation_does_not_execute_training_or_construct_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            with (
                patch(
                    "spatial_meta_ce.train.train",
                    side_effect=AssertionError("training must not run"),
                ),
                patch(
                    "spatial_meta_ce.model.create_model",
                    side_effect=AssertionError("model must not be constructed"),
                ),
            ):
                result = self.prepare(root, inputs)
            self.assertFalse(result["training_run"])
            self.assertFalse((root / "outputs").exists())

    def test_existing_run_root_refused_before_preparation_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            (root / "outputs").mkdir()
            with self.assertRaises(FileExistsError):
                self.prepare(root, inputs)
            self.assertFalse((root / "prepared").exists())

    def test_unsafe_paths_refused_before_preparation_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = synthetic_inputs(root)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            for target in (alias / "prepared", root / "real" / ".." / "prepared"):
                with self.subTest(path=str(target)), self.assertRaises(ValueError):
                    self.prepare(root, inputs, out=target)
            self.assertFalse((root / "prepared").exists())
            self.assertFalse((real / "prepared").exists())


if __name__ == "__main__":
    unittest.main()
