"""Synthetic-only tests for the independent official26/4 module contract."""

import copy
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from spatial_meta_ce import module_experiment as contract
from scripts import run_official_module_experiment as runner


def capacity():
    from spatial_meta_ce.data import TRANSFORM_VERSION
    from spatial_meta_ce.manifests import digest
    from spatial_meta_ce.price_capacity import (
        OFFICIAL_CAPACITY_KIND,
        OFFICIAL_TRAIN_DATABASES,
    )

    payload = {
        "schema": 1,
        "kind": OFFICIAL_CAPACITY_KIND,
        "split": "pretrain",
        "transform": TRANSFORM_VERSION,
        "databases": list(OFFICIAL_TRAIN_DATABASES),
        "max_counts": [2, 2, 2, 2],
        "database_max_counts": {db: [2, 2, 2, 2] for db in OFFICIAL_TRAIN_DATABASES},
        "cache_hashes": {db: "a" * 64 for db in OFFICIAL_TRAIN_DATABASES},
        "manifest_sha256": {db: "b" * 64 for db in OFFICIAL_TRAIN_DATABASES},
    }
    return {**payload, "capacity_sha256": digest(payload)}


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


def config(arm="A1-PG", seed=42):
    return contract.training_config(
        arm,
        seed,
        capacity(),
        data_config_binding={"path": "/synthetic/data.json", "sha256": "c" * 64},
        training_plan_binding={
            "path": "/synthetic/plan.json",
            "sha256": "d" * 64,
            "plan_sha256": "e" * 64,
        },
        mapping_root="/synthetic/mappings" if contract.mapping_mode(arm) else None,
    )


def synthetic_preparation(root):
    """26 tiny caches and 3 two-update plans; certification is mocked explicitly."""
    import numpy as np
    from spatial_meta_ce.data import GROUP_DIMS, TRANSFORM_VERSION
    from spatial_meta_ce.official_split import build_official_capacity
    from spatial_meta_ce.plans import build_training_plan

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
    paths, entries = {}, {}
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
        paths[str(seed)] = str(path)
        entries[str(seed)] = {
            "path": str(path),
            "plan_sha256": plan["plan_sha256"],
            "file_sha256": runner.file_sha256(path),
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
    return (
        {
            "data_config": str(data_path),
            "capacity": str(cap_path),
            "split_manifest": str(manifest),
            "training_plans": paths,
            "mapping_report": str(report_path),
            "output_dir": str(root / "output"),
            "budget": budget(),
            "cuda_device": 0,
        },
        data,
        report,
    )


class ModuleContractTests(unittest.TestCase):
    def test_five_configs_share_base_and_keep_modules_independent(self):
        from spatial_meta_ce.price_protocol import exploratory_model_config

        baseline = exploratory_model_config(capacity())
        self.assertEqual(contract.model_config("A1-PG", capacity()), baseline)
        for arm in contract.ARMS:
            value = contract.model_config(arm, capacity())
            self.assertFalse(value["lora"])
            self.assertEqual(value["capacity"], capacity())
            if "Group" in arm:
                self.assertEqual(
                    value["group_pooling"],
                    "mean" if arm.endswith("Mean") else "attention",
                )
                self.assertNotIn("relation_attention", value)
            if "Relation" in arm:
                self.assertTrue(value["relation_attention"])
                self.assertEqual(value["relation_scope"], "official_26_4")
                self.assertNotIn("group_pooling", value)
        self.assertEqual(contract.mapping_mode("A1-PG-RelationTrue"), "mapping")
        self.assertEqual(
            contract.mapping_mode("A1-PG-RelationShuffled"), "shuffled_mapping"
        )

    def test_canonical_config_has_exact_frozen_recipe(self):
        value = config()
        self.assertEqual(value["fixed_stage"], "132-module26")
        self.assertEqual(value["steps"], 24000)
        self.assertEqual(value["support_sizes"], [5, 10, 20, 50, 100])
        self.assertEqual(value["query_size"], 64)
        self.assertEqual(value["method"], "erm")
        self.assertEqual(value["meta_objective"], "post")
        self.assertNotIn("preregistered_step_budget", value)
        self.assertNotIn("source_preflight", value)

    def test_shape_rejects_typed_aliases_unsafe_keys_and_combined_modules(self):
        base = config()
        patches = [
            {"steps": 24000.0},
            {"meta_batch": 4.0},
            {"accumulation": True},
            {"seed": 42.0},
            {"inner_steps": 5.0},
            {"fixed_budget": 1},
            {"preregistered_step_budget": 24000},
            {"resume": None},
            {"initial_checkpoint": "/old/last.pt"},
            {"complex_query_weight": 1},
            {"best_checkpoint": False},
            {"module_arm": "A1"},
            {"model": {**base["model"], "relation_attention": True}},
            {"model": {**base["model"], "pg_head_input": 1}},
            {"support_sizes": (5, 10, 20, 50, 100)},
        ]
        for delta in patches:
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                contract.validate_config_shape({**base, **delta})
        contract.validate_config_shape(base)

    def test_real_training_requires_bound_evidence_without_bypass(self):
        value = config()
        data = {
            "splits": {
                "train": list(contract.TRAIN_DATABASES),
                "validation": [],
                "test": list(contract.TEST_DATABASES),
            }
        }
        with self.assertRaisesRegex(ValueError, "module_execution"):
            contract.validate_training_config(value, data)
        with self.assertRaises(ValueError):
            contract.validate_training_config({**value, "allowtrain": True}, data)
        with self.assertRaisesRegex(ValueError, "resume"):
            contract.validate_training_config(value, data, resume="old.pt")

    def test_checkpoint_checks_are_exact_and_do_not_use_122_validator(self):
        value = config("A1-PG-RelationShuffled")
        state = {
            "schema": 1,
            "transform": "price40-group-logfanout-v1",
            "next_step": 24000,
            "model_type": "price_lite_pg_official_relation_residual",
            "model_config": value["model"],
            "model_config_sha256": contract.canonical_digest(value["model"]),
            "metadata": contract.checkpoint_metadata(value),
            "extra": {
                "counts": {"optimizer_steps": 24000},
                "sample_plan_digest": "e" * 64,
                "sample_plan_episodes": 96000,
            },
        }
        contract.validate_checkpoint(state, value["module_arm"], 42)
        for field, replacement in (
            ("module_arm", "A1-PG-RelationTrue"),
            ("fixed_steps", 4000),
            ("selection", "best"),
            ("target_databases_read", ["imdb"]),
            ("mapping_mode", "mapping"),
            ("adapted", True),
            ("train_databases", list(contract.TRAIN_DATABASES[:-1])),
        ):
            invalid = copy.deepcopy(state)
            invalid["metadata"][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.validate_checkpoint(invalid, value["module_arm"], 42)


class ModuleRunnerGateTests(unittest.TestCase):
    def test_preparation_destinations_reject_all_readonly_roots_before_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = {"data_root": str(root / "price"), "cache_root": str(root / "cache")}
            report = {
                "mapping_root": str(root / "mapping"),
                "source_mapping_root": str(root / "legacy"),
            }
            for parent in ("price", "cache", "mapping", "legacy"):
                destination = root / parent / "must-not-create" / "spec.json"
                with self.subTest(parent=parent), self.assertRaises(ValueError):
                    runner.preparation_path(destination, data, report)
                with self.subTest(parent=parent), self.assertRaises(ValueError):
                    runner._check_input_overlap(destination, data, report)
                self.assertFalse(destination.parent.exists())
            allowed = root / "artifacts" / "spec.json"
            self.assertEqual(runner.preparation_path(allowed, data, report), allowed)
            self.assertFalse(allowed.parent.exists())

    def test_prepare_cli_and_evaluation_api_refuse_input_tree_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs, data, report = synthetic_preparation(root)
            report.update(source_mapping_root=str(root / "legacy"))
            Path(inputs["mapping_report"]).write_text(json.dumps(report))
            budget_path = root / "budget.json"
            budget_path.write_text(json.dumps(budget()))
            for parent in ("readonly-price", "cache", "mapping", "legacy"):
                destination = root / parent / "must-not-create"
                argv = [
                    "prepare",
                    "--data-config",
                    inputs["data_config"],
                    "--mapping-report",
                    inputs["mapping_report"],
                    "--output-dir",
                    inputs["output_dir"],
                    "--out",
                    str(destination),
                    "--budget",
                    str(budget_path),
                    "--cuda-device",
                    "1",
                    "--capacity",
                    inputs["capacity"],
                    "--split-manifest",
                    inputs["split_manifest"],
                ]
                for seed, path in inputs["training_plans"].items():
                    argv.extend(["--training-plan", f"{seed}={path}"])
                with (
                    self.subTest(parent=parent),
                    self.assertRaisesRegex(ValueError, "read-only"),
                ):
                    runner.main(argv)
                self.assertFalse(destination.exists())
                records = root / "records.json"
                if not records.exists():
                    records.write_text("{}")
                with (
                    patch.object(
                        runner, "_frozen_training_records", return_value={"data": data}
                    ),
                    patch.object(runner, "_mapping_report", return_value=report),
                    patch.object(
                        runner,
                        "build_evaluation_plan",
                        side_effect=AssertionError("test read"),
                    ),
                    self.assertRaisesRegex(ValueError, "read-only"),
                ):
                    runner.prepare_evaluation(
                        training_records=[str(records)],
                        data_config=inputs["data_config"],
                        mapping_report=inputs["mapping_report"],
                        plan_path=str(destination / "plan.json"),
                        output_dir=inputs["output_dir"],
                        budget=budget(),
                        cuda_device=1,
                    )
                self.assertFalse(destination.exists())

    def test_module_sampler_uses_owned_rss_and_fresh_worker_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "worker"
            output.mkdir()
            child = object()
            memory = output / "memory.json"
            valid = {
                "allocated_bytes": 20,
                "reserved_bytes": 30,
                "max_allocated_bytes": 40,
                "max_reserved_bytes": 50,
                "interval_seconds": 1.0,
            }
            memory.write_text(json.dumps(valid))
            with (
                patch(
                    "scripts.meta65_joint_supervisor._gpu_bytes",
                    side_effect=AssertionError("global driver query"),
                ),
                patch(
                    "scripts.meta65_joint_supervisor.sample_process",
                    return_value={"rss_bytes": 100, "vram_bytes": 0},
                ) as sample,
            ):
                self.assertEqual(
                    runner.module_resource_sample(child, output, "cuda"),
                    {"rss_bytes": 100, "vram_bytes": 50},
                )
                sample.assert_called_once_with(
                    child, mode="cpu_preflight", output=output
                )
                for invalid in (
                    {"error": "CUDA failed"},
                    {**valid, "allocated_bytes": -1},
                    {**valid, "max_reserved_bytes": True},
                    {"allocated_bytes": 1},
                ):
                    memory.write_text(json.dumps(invalid))
                    with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                        runner.module_resource_sample(child, output, "cuda")
                memory.write_text(json.dumps(valid))
                os.utime(memory, (time.time() - 60, time.time() - 60))
                with self.assertRaises(ValueError):
                    runner.module_resource_sample(child, output, "cuda")
                memory.unlink()
                with self.assertRaises(ValueError):
                    runner.module_resource_sample(child, output, "cuda")
                self.assertEqual(
                    runner.module_resource_sample(
                        child, output, "cuda", telemetry_required=False
                    ),
                    {"rss_bytes": 100, "vram_bytes": 0},
                )
                self.assertEqual(
                    runner.module_resource_sample(child, output, "cpu"),
                    {"rss_bytes": 100, "vram_bytes": 0},
                )
                memory.write_text("not-json")
                with self.assertRaises(ValueError):
                    runner.module_resource_sample(child, output, "cuda")
                memory.unlink()
                (output / "elsewhere.json").write_text(json.dumps(valid))
                memory.symlink_to(output / "elsewhere.json")
                with self.assertRaises(ValueError):
                    runner.module_resource_sample(child, output, "cuda")

    def test_frozen_training_records_bind_parent_budget_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data = {"cache_root": str(root / "cache"), "data_root": str(root / "price")}
            spec = runner.spec_header(
                "train", str(root / "output"), budget(), cuda_device=0
            )
            spec_path = root / "training-spec.json"
            spec_path.write_text(json.dumps(spec))
            auth = {
                **runner.authorization_template(spec, runner.file_sha256(spec_path)),
                "authorized": True,
                "formal_training_authorized": True,
                "granted_by": "synthetic-only",
                "granted_at": "2026-11-18T12:00:00+08:00",
            }
            auth_path = root / "authorization.json"
            auth_path.write_text(json.dumps(auth))
            records, configs = [], {}
            for arm in contract.ARMS:
                for seed in contract.SEEDS:
                    key = contract.run_key(arm, seed)
                    checkpoint = Path(spec["outputs"][key]) / "last.pt"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"synthetic-placeholder")
                    evidence = {
                        "spec": runner.file_binding(spec_path),
                        "authorization": runner.file_binding(auth_path),
                        "output_dir": spec["output_dir"],
                        "run_key": key,
                    }
                    value = config(arm, seed)
                    configs[key] = value
                    records.append(
                        {
                            "key": key,
                            "arm": arm,
                            "seed": seed,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": runner.file_sha256(checkpoint),
                            "optimizer_steps": 24000,
                            "training_plan_sha256": value["training_plan_sha256"],
                            "sample_plan_digest": value["training_plan_sha256"],
                            "sample_plan_episodes": 96000,
                            "module_execution": evidence,
                        }
                    )
            record_path = root / "records.json"
            record_path.write_text(
                json.dumps(
                    {"protocol": contract.PROTOCOL, "completed": True, "runs": records}
                )
            )
            completion = {
                "protocol": contract.PROTOCOL,
                "action": "train",
                "status": "completed",
                "spec_sha256": runner.file_sha256(spec_path),
                "authorization_sha256": runner.file_sha256(auth_path),
                "supervisor": {"status": "completed"},
                "target_databases_read": [],
                "budget_usage": {
                    "sampling_failed": None,
                    "samples": 2,
                    "cpu_core_seconds": 1.0,
                    "gpu_allocated_seconds": 1.0,
                },
            }
            completion_path = root / "output/completion.json"
            completion_path.write_text(json.dumps(completion))
            with patch.object(
                runner,
                "_training_bindings",
                return_value={
                    "configs": configs,
                    "capacity": capacity(),
                    "data": data,
                },
            ):
                frozen = runner._frozen_training_records(
                    [runner.file_binding(record_path)]
                )
                self.assertEqual(len(frozen["records"]), 15)
                self.assertEqual(
                    frozen["completion_bindings"][str(completion_path)],
                    runner.file_binding(completion_path),
                )
                completion["budget_usage"]["gpu_allocated_seconds"] = 0
                completion_path.write_text(json.dumps(completion))
                with self.assertRaisesRegex(ValueError, "budget-certified"):
                    runner._frozen_training_records([runner.file_binding(record_path)])

    def test_worker_injects_execution_record_immediately_before_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            value = config()
            key = "A1-PG-seed42"
            spec = runner.spec_header(
                "train",
                str(root / "output"),
                budget(),
                cuda_device=0,
                execution={"mode": "single_run", "arm": "A1-PG", "seed": 42},
            )
            data = {"synthetic": True}
            spec_binding = {"path": str(root / "spec.json"), "sha256": "a" * 64}
            auth_binding = {"path": str(root / "auth.json"), "sha256": "b" * 64}
            checkpoint_state = {
                "extra": {
                    "sample_plan_digest": value["training_plan_sha256"],
                    "sample_plan_episodes": 96000,
                },
            }

            def train_stub(actual, actual_data, actual_output, device):
                self.assertEqual(
                    actual["module_execution"],
                    {
                        "spec": spec_binding,
                        "authorization": auth_binding,
                        "output_dir": spec["output_dir"],
                        "run_key": key,
                    },
                )
                self.assertIs(actual_data, data)
                self.assertEqual(actual_output, spec["outputs"][key])
                self.assertEqual(device, "cuda")
                path = Path(actual_output)
                path.mkdir(parents=True)
                (path / "last.pt").write_bytes(b"synthetic-placeholder")
                return {
                    "counts": {"optimizer_steps": 24000},
                    "parameters": 7,
                    "total_seconds": 0.01,
                }

            with (
                patch("spatial_meta_ce.train.train", side_effect=train_stub) as mocked,
                patch("torch.load", return_value=checkpoint_state),
                patch.object(runner, "validate_checkpoint") as validator,
            ):
                result = runner._train_all(
                    spec,
                    {"configs": {key: value}, "data": data, "budget": budget()},
                    spec_binding,
                    auth_binding,
                    root,
                    lambda *args: None,
                )
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(validator.call_count, 1)
            self.assertEqual(result["runs"][0]["sample_plan_episodes"], 96000)
            self.assertEqual(result["target_databases_read"], [])

    def test_unsigned_execution_fails_before_verify_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            spec = runner.spec_header(
                "train", str(root / "output"), budget(), cuda_device=0
            )
            path = root / "spec.json"
            path.write_text(json.dumps(spec))
            auth = root / "auth.json"
            auth.write_text(
                json.dumps(
                    runner.authorization_template(spec, runner.file_sha256(path))
                )
            )
            args = SimpleNamespace(
                spec=str(path),
                spec_sha256=runner.file_sha256(path),
                authorization=str(auth),
                authorization_sha256=runner.file_sha256(auth),
                execute=True,
            )
            with patch.object(
                runner, "verify_spec", side_effect=AssertionError("verified inputs")
            ):
                with self.assertRaisesRegex(ValueError, "unsigned"):
                    runner.parent_main(args)
            self.assertFalse((root / "output").exists())

    def test_evaluate_ready_requires_all_runs_before_test_population_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / "records.json"
            path.write_text(
                json.dumps(
                    {"protocol": contract.PROTOCOL, "completed": True, "runs": []}
                )
            )
            with patch.object(
                runner,
                "build_evaluation_plan",
                side_effect=AssertionError("target read"),
            ):
                with self.assertRaisesRegex(ValueError, "fifteen"):
                    runner.prepare_evaluation(
                        training_records=[str(path)],
                        data_config=str(root / "data.json"),
                        mapping_report=str(root / "report.json"),
                        plan_path=str(root / "eval-plan.json"),
                        output_dir=str(root / "output"),
                        budget=budget(),
                        cuda_device=0,
                    )
            self.assertFalse((root / "eval-plan.json").exists())

    def test_prepare_verify_unsigned_refusal_and_bound_training_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs, data, report = synthetic_preparation(root)
            recipe = {**contract.RECIPE, "steps": 2, "validate_every": 2}
            with (
                patch.object(contract, "STEPS", 2),
                patch.object(contract, "RECIPE", recipe),
                patch.object(runner, "STEPS", 2),
                patch.object(runner, "_mapping_report", return_value=report),
            ):
                spec = runner.prepare_training(**inputs)
                self.assertFalse(Path(inputs["output_dir"]).exists())
                self.assertFalse((root / "cache/test").exists())
                spec_path = root / "spec.json"
                spec_path.write_text(json.dumps(spec))
                template = runner.authorization_template(
                    spec, runner.file_sha256(spec_path)
                )
                auth_path = root / "authorization.json"
                auth_path.write_text(json.dumps(template))
                evidence = {
                    "spec": runner.file_binding(spec_path),
                    "authorization": runner.file_binding(auth_path),
                    "output_dir": spec["output_dir"],
                    "run_key": "A1-PG-seed42",
                }
                value = {
                    **runner.verify_spec(spec)["configs"]["A1-PG-seed42"],
                    "module_execution": evidence,
                }
                with self.assertRaisesRegex(ValueError, "unsigned"):
                    contract.validate_training_config(
                        value, data, output_dir=spec["outputs"]["A1-PG-seed42"]
                    )
                signed = {
                    **template,
                    "authorized": True,
                    "formal_training_authorized": True,
                    "granted_by": "synthetic-test-only",
                    "granted_at": "2026-11-18T12:00:00+08:00",
                }
                auth_path.write_text(json.dumps(signed))
                evidence["authorization"] = runner.file_binding(auth_path)
                with (
                    patch("sys.platform", "linux"),
                    patch.dict(
                        os.environ,
                        {
                            "CUDA_VISIBLE_DEVICES": "0",
                            "META65_SUPERVISED_MODE": "formal",
                            "META65_SUPERVISOR_PID": str(os.getppid()),
                        },
                    ),
                ):
                    contract.validate_training_config(
                        value,
                        data,
                        output_dir=spec["outputs"]["A1-PG-seed42"],
                        device="cuda",
                    )
                    for output in (None, str(root / "wrong-run")):
                        with (
                            self.subTest(output=output),
                            self.assertRaisesRegex(ValueError, "output_dir"),
                        ):
                            contract.validate_training_config(
                                value, data, output_dir=output, device="cuda"
                            )
                    with self.assertRaisesRegex(ValueError, "Runtime config/data"):
                        contract.validate_training_config(
                            value,
                            {**data, "cache_root": str(root / "other-cache")},
                            output_dir=spec["outputs"]["A1-PG-seed42"],
                            device="cuda",
                        )
                for actual_device, visibility, platform, mode, parent in (
                    ("cpu", "0", "linux", "formal", str(os.getppid())),
                    ("cuda:7", "0", "linux", "formal", str(os.getppid())),
                    ("cuda", "", "linux", "formal", str(os.getppid())),
                    ("cuda", "0", "darwin", "formal", str(os.getppid())),
                    ("cuda", "0", "linux", "cpu_preflight", str(os.getppid())),
                    ("cuda", "0", "linux", "formal", "99999999"),
                ):
                    with (
                        self.subTest(
                            device=actual_device,
                            visibility=visibility,
                            platform=platform,
                            mode=mode,
                        ),
                        patch("sys.platform", platform),
                        patch.dict(
                            os.environ,
                            {
                                "CUDA_VISIBLE_DEVICES": visibility,
                                "META65_SUPERVISED_MODE": mode,
                                "META65_SUPERVISOR_PID": parent,
                            },
                        ),
                        patch.object(
                            runner,
                            "verify_spec",
                            side_effect=AssertionError(
                                "cache accessed before runtime gate"
                            ),
                        ),
                        self.assertRaisesRegex(
                            ValueError, "device|visibility|Linux|supervisor"
                        ),
                    ):
                        contract.validate_training_config(
                            value,
                            data,
                            output_dir=spec["outputs"]["A1-PG-seed42"],
                            device=actual_device,
                        )
                # Wrong closure is rejected before the cache loader can be called.
                wrong = copy.deepcopy(spec)
                wrong["bindings"]["code_sha256"].pop("spatial_meta_ce/model.py")
                with patch.object(
                    runner, "_cache_stores", side_effect=AssertionError("cache opened")
                ):
                    with self.assertRaisesRegex(ValueError, "closure"):
                        runner.verify_spec(wrong)

    def test_streamed_plan_verifies_canonical_digest_and_live_query_hashes(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(runner, "STEPS", 2),
        ):
            root = Path(directory).resolve()
            rows = [{"query_hash": f"{i + 1:064x}"} for i in range(69)]
            stores = {db: SimpleNamespace(rows=rows) for db in contract.TRAIN_DATABASES}
            data_hashes = {db: "c" * 64 for db in contract.TRAIN_DATABASES}
            episode = {
                "db": contract.TRAIN_DATABASES[0],
                "k": 5,
                "support": list(range(5)),
                "query": list(range(5, 69)),
                "support_hashes": [r["query_hash"] for r in rows[:5]],
                "query_hashes": [r["query_hash"] for r in rows[5:]],
            }
            plan = {
                "schema": 1,
                "kind": "fomaml_joint_source_training_plan",
                "seed": 42,
                "steps": 2,
                "meta_batch": 4,
                "accumulation": 1,
                "support_sizes": [5, 10, 20, 50, 100],
                "query_size": 64,
                "databases": capacity()["cache_hashes"],
                "data_hashes": data_hashes,
                "episodes": [episode] * 8,
            }
            plan["plan_sha256"] = contract.canonical_digest(plan)
            path = root / "plan.json"
            path.write_text(json.dumps(plan, indent=2, sort_keys=True))
            record = runner.file_binding(path, plan=True)
            summary = runner._verify_training_plan(
                record, 42, capacity(), stores, data_hashes
            )
            self.assertNotIn("episodes", summary)
            self.assertEqual(summary["episode_count"], 8)
            with patch.object(
                runner, "_plan_items", side_effect=AssertionError("cached parser")
            ):
                runner._verify_training_plan(
                    record, 42, capacity(), stores, data_hashes
                )
            # The cache is content-bound: changing even whitespace rejects the
            # old binding before any cached verification is reused.
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                runner._verify_training_plan(
                    record, 42, capacity(), stores, data_hashes
                )
            bad = copy.deepcopy(plan)
            bad["episodes"][0]["query_hashes"][0] = "f" * 64
            bad["plan_sha256"] = contract.canonical_digest(
                {k: v for k, v in bad.items() if k != "plan_sha256"}
            )
            path.write_text(json.dumps(bad, indent=2))
            with self.assertRaisesRegex(ValueError, "ID/hash"):
                runner._verify_training_plan(
                    runner.file_binding(path, plan=True),
                    42,
                    capacity(),
                    stores,
                    data_hashes,
                )

    def test_streamed_plan_rejects_duplicate_keys_and_wrong_self_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "plan.json"
            path.write_text('{"schema":1,"schema":1}')
            with self.assertRaisesRegex(ValueError, "repeated"):
                list(runner._plan_items(path))
            path.write_text('{"episodes":[{"k":5,"k":10}]}')
            with self.assertRaisesRegex(ValueError, "Repeated"):
                list(runner._plan_items(path))

    def test_closure_contains_every_model_dispatch_and_mapping_dependency(self):
        for name in (
            "spatial_meta_ce/model.py",
            "spatial_meta_ce/price_model.py",
            "spatial_meta_ce/price_relations.py",
            "spatial_meta_ce/checkpoints.py",
            "spatial_meta_ce/official_mappings.py",
            "spatial_meta_ce/train.py",
            "spatial_meta_ce/data.py",
            "spatial_meta_ce/meta.py",
            "spatial_meta_ce/plans.py",
            "spatial_meta_ce/official_split.py",
            "spatial_meta_ce/module_experiment.py",
            "scripts/prepare_official_module_inputs.py",
            "vendor/price_baseline/encoder.py",
            "vendor/price_baseline/module.py",
        ):
            self.assertIn(name, runner.REQUIRED_CODE)

    def test_code_closure_rejects_missing_extra_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "part.py").write_text("pass\n")
            with patch.object(runner, "REQUIRED_CODE", ("part.py",)):
                hashes = runner.compute_code_sha256(root)
                runner.verify_code_tree(root, hashes, expected_root=root)
                for changed in (
                    {},
                    {**hashes, "extra.py": "a" * 64},
                    {"../part.py": "a" * 64},
                ):
                    with self.assertRaises(ValueError):
                        runner.verify_code_tree(root, changed, expected_root=root)
                with self.assertRaises(ValueError):
                    runner.verify_code_tree(root, hashes, expected_root=root / "other")
                (root / "part.py").unlink()
                (root / "part.py").symlink_to(root / "outside.py")
                with self.assertRaises(ValueError):
                    runner.compute_code_sha256(root)

    def test_unsigned_template_never_grants_training_or_evaluation(self):
        spec = runner.spec_header("train", "/synthetic/output", budget(), cuda_device=0)
        authorization = runner.authorization_template(spec, "a" * 64)
        self.assertFalse(authorization["authorized"])
        self.assertFalse(authorization["formal_training_authorized"])
        self.assertFalse(authorization["formal_evaluation_authorized"])
        with self.assertRaisesRegex(ValueError, "unsigned|authorization"):
            runner.validate_authorization(authorization, spec, "a" * 64)

    def test_authorization_binds_action_budget_output_identity_timezone(self):
        spec = runner.spec_header("train", "/synthetic/output", budget(), cuda_device=0)
        value = {
            **runner.authorization_template(spec, "a" * 64),
            "authorized": True,
            "formal_training_authorized": True,
            "granted_by": "synthetic-test",
            "granted_at": "2026-11-18T10:00:00+08:00",
        }
        runner.validate_authorization(value, spec, "a" * 64)
        for delta in (
            {"protocol": "122-official-26"},
            {"authorized": 1},
            {"formal_evaluation_authorized": True},
            {"output_dir": "/other"},
            {"budget_sha256": "b" * 64},
            {"spec_sha256": "b" * 64},
            {"granted_at": "2026-11-18T10:00:00"},
            {"granted_by": " "},
        ):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                runner.validate_authorization({**value, **delta}, spec, "a" * 64)

    def test_matrix_remains_predeclared_when_running_an_explicit_subset(self):
        spec = runner.spec_header("train", "/synthetic/output", budget(), cuda_device=0)
        self.assertEqual(len(runner.execution_keys(spec)), 15)
        one = {
            **spec,
            "execution": {"mode": "single_run", "arm": "A1-PG-GroupMean", "seed": 43},
        }
        self.assertEqual(runner.execution_keys(one), ["A1-PG-GroupMean-seed43"])
        for changed in (
            {"arms": ["A1-PG"]},
            {"seeds": [42]},
            {"seeds": [42.0, 43, 44]},
            {"execution": {"mode": "single_run", "arm": "A1-PG", "seed": 42.0}},
            {"execution": {"mode": "matrix", "arm": "A1-PG"}},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                runner.execution_keys({**spec, **changed})

    def test_malformed_spec_does_not_import_model_or_write_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "output"
            spec = runner.spec_header("train", str(output), budget(), cuda_device=0)
            spec["steps"] = 1
            with patch.object(
                runner, "compute_code_sha256", side_effect=AssertionError("read code")
            ):
                with self.assertRaises(ValueError):
                    runner.verify_spec(spec)
            self.assertFalse(output.exists())

    def test_cli_help_uses_no_model_import(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/run_official_module_experiment.py",
                "--help",
            ],
            cwd=runner.ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("evaluate-ready", result.stdout)


class IndependentEvaluationTests(unittest.TestCase):
    def test_independent_recompute_preserves_population_and_truth_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            paths = {}
            for arm in ("A1-PG", "A1-PG-GroupMean"):
                path = root / f"{arm}.csv"
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(runner.PREDICTION_COLUMNS)
                    for db in contract.TEST_DATABASES:
                        for population in ("unique-query", "all-rows"):
                            for index, true in enumerate((2.0, 4.0)):
                                writer.writerow(
                                    [
                                        arm,
                                        42,
                                        db,
                                        population,
                                        0,
                                        0,
                                        f"{db}/test/{index}",
                                        f"{index + 1:064x}",
                                        true,
                                        math.log(true),
                                    ]
                                )
                paths[contract.run_key(arm, 42)] = path
            result = runner.recompute_evaluation(paths)
            self.assertEqual(len(result["units"]), 16)
            self.assertEqual(result["paired_seed_deltas"]["A1-PG-GroupMean"]["42"], 0.0)
            self.assertEqual(result["selection_use"], "none")
            self.assertTrue(result["independent_recompute"])
            path = paths["A1-PG-GroupMean-seed42"]
            path.write_text(
                path.read_text().replace(
                    "imdb/test/0,0000000000000000000000000000000000000000000000000000000000000001,2.0",
                    "imdb/test/0,0000000000000000000000000000000000000000000000000000000000000001,3.0",
                )
            )
            with self.assertRaisesRegex(ValueError, "truth|population"):
                runner.recompute_evaluation(paths)

    def test_no_clipping_or_silent_nonfinite_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "pred.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(runner.PREDICTION_COLUMNS)
                writer.writerow(
                    [
                        "A1-PG",
                        42,
                        "imdb",
                        "unique-query",
                        0,
                        0,
                        "imdb/test/0",
                        "a" * 64,
                        1.0,
                        "nan",
                    ]
                )
            with self.assertRaisesRegex(ValueError, "finite"):
                runner.recompute_evaluation({"A1-PG-seed42": path})


if __name__ == "__main__":
    unittest.main()
