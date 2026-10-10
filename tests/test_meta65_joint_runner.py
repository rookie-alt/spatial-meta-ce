import copy
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts import meta65_joint_runner as runner
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import SetEstimator


def identity(n=120):
    return {"query_hash": np.array([f"h{i:03}" for i in range(n)]),
            "query_id": np.array([f"telstra/pretrain/{i+1}" for i in range(n)]),
            "true": np.arange(n, dtype=np.float64) + 16777217,
            "n_tables": np.full(n, 4, dtype=np.int16)}


def plan_for(data):
    indices = list(range(len(data["true"])))
    support = indices[:100]
    query = indices[100:]
    values = {"database": "telstra", "run": 0, "support_max": support,
              "supports": {str(k): support[:k] for k in (5, 10, 20, 50, 100)},
              "query": query,
              "query_digest": runner.digest([(i, str(data["query_hash"][i])) for i in query]),
              "support_digests": {str(k): runner.digest([(i, str(data["query_hash"][i]))
                                                       for i in support[:k]])
                                  for k in (5, 10, 20, 50, 100)}}
    return values


class JointRunnerTests(unittest.TestCase):
    def test_fixed_matrix_and_target_rejection(self):
        self.assertEqual(len(runner.matrix()), 135)
        with self.assertRaises(ValueError):
            runner.source_path("imdb")
        with self.assertRaises(ValueError):
            runner.validate_config({**runner.FROZEN, "k": 5})

    def test_full_plan_preserves_100_exclusion_and_digests(self):
        data = identity()
        plan = plan_for(data)
        result = runner.certify_plan(plan, data, "telstra", 0, expected_n=20)
        self.assertEqual(result["support"], list(range(20)))
        self.assertEqual(result["query"], list(range(100, 120)))
        for mutator in (lambda p: p["query"].append(99),
                        lambda p: p["supports"].update({"20": list(range(1, 21))}),
                        lambda p: p.update(query_digest="bad")):
            bad = copy.deepcopy(plan)
            mutator(bad)
            with self.assertRaises(ValueError):
                runner.certify_plan(bad, data, "telstra", 0, expected_n=20)

    def test_identity_requires_fp64_and_no_truncation_or_duplicate(self):
        data = identity()
        for key, value in (("true", data["true"].astype(np.float32)),
                           ("query_hash", np.array(["a" * 65] * 120)),
                           ("n_tables", np.zeros(120))):
            bad = {**data, key: value}
            with self.assertRaises(ValueError):
                runner.encode_identity(bad)
        result = runner.encode_identity(data)
        np.testing.assert_array_equal(result["true"], data["true"])
        self.assertEqual(result["query_hash"].dtype, np.dtype("S64"))

    def test_metrics_linear_standard_and_empty_bucket_null(self):
        data = {key: value[:4] for key, value in identity().items()}
        data["true"] = np.ones(4)
        pred = np.log([1, 2, 4, 8])
        result = runner.cell_metrics(pred, data)
        self.assertAlmostEqual(result["all"]["q95"], float(np.percentile([1, 2, 4, 8], 95)))
        self.assertIsNone(result["joint"][">=1e6|5+"]["q95"])
        self.assertEqual(sum(r["count"] for r in result["joint"].values()), 4)

    def test_anchor_exact_identity_truth_and_fixed_prediction_tolerance(self):
        data = {key: value[:4] for key, value in identity().items()}
        pred = np.log(data["true"]) + 0.1
        anchor = {**data, "log_pred": pred.copy()}
        runner.verify_anchor(pred, data, anchor)
        bad = {**anchor, "true": anchor["true"] + 0.01}
        with self.assertRaises(ValueError):
            runner.verify_anchor(pred, data, bad)
        with self.assertRaises(ValueError):
            runner.verify_anchor(pred + 0.1, data, anchor)

    def test_none_never_calls_adapt_and_selected_names(self):
        model = SetEstimator(arch="sum", use_pg=True)
        before = runner.state_digest(model)
        with patch("spatial_meta_ce.meta.adapt", side_effect=AssertionError("none called adapt")):
            params, record = runner.operator_params(model, None, "none")
        self.assertEqual(record["selected_parameter_count"], 0)
        self.assertEqual(record["selected_parameter_names"], [])
        self.assertEqual(list(params), list(OrderedDict(model.named_parameters())))
        self.assertEqual(runner.state_digest(model), before)

    def test_head_last_isolate_initialization_and_frozen_parameters(self):
        torch.manual_seed(42)
        model = SetEstimator(arch="sum", use_pg=True)
        support = batch_samples([{"groups": [np.ones((1, dim), dtype=np.float32)
                                            for dim in (40, 40, 4, 43)],
                                  "true": 16777217.25, "pg": 2.0,
                                  "id": "telstra/pretrain/1"}])
        before = runner.state_digest(model)
        for operator, expected in (("head3", 107521), ("last3", 65)):
            first, record = runner.operator_params(model, support, operator)
            second, again = runner.operator_params(model, support, operator)
            self.assertEqual(record["selected_parameter_count"], expected)
            self.assertEqual(record, again)
            self.assertTrue(record["changed_parameter_names"])
            self.assertEqual(runner.state_digest(model), before)
            for name in first:
                self.assertTrue(torch.equal(first[name], second[name]))
                if name not in record["selected_parameter_names"]:
                    self.assertTrue(torch.equal(first[name], dict(model.named_parameters())[name]))

    def test_parent_symlink_is_rejected_and_formal_needs_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "original").mkdir()
            (root / "link").symlink_to(root / "original")
            with self.assertRaises(ValueError):
                runner.checked_path(root / "link" / "new")
            with self.assertRaisesRegex(ValueError, "not authorized"):
                runner.main(["--out", str(root / "output"), "--mode", "formal", "--execute"])

    def test_sha_mutation_and_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "input"
            p.write_text("original")
            binding = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()}
            runner.recheck(binding)
            p.write_text("changed")
            with self.assertRaises(ValueError):
                runner.recheck(binding)
            with self.assertRaises(ValueError):
                runner.safe_output(p, [p])

    def test_json_strict_and_empty_run_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "data.json"
            p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):
                runner.read_json(p)
        rows = [{"all": {"q95": 1}}, {"all": {"q95": None}}]
        self.assertIsNone(runner.mean_metric(rows, "all", "q95"))
        self.assertEqual(json.loads(json.dumps(runner.FROZEN)), runner.FROZEN)

    def test_frozen_root_sibling_is_forbidden_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "frozen"
            frozen.mkdir()
            with patch.object(runner, "FROZEN_ROOT", frozen):
                with self.assertRaisesRegex(ValueError, "frozen|read-only"):
                    runner.safe_output(frozen / "another-run", [])
                self.assertEqual(runner.safe_output(root / "isolated" / "formal", []),
                                 root / "isolated" / "formal")

    def test_launch_binding_rechecks_code_config_protocol_and_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            protocol = root / runner.PROTOCOL
            protocol.parent.mkdir(parents=True)
            protocol.write_text("frozen 56 protocol")
            config = root / "configs" / "meta65_joint_control.json"
            config.parent.mkdir()
            config.write_text(json.dumps(runner.FROZEN))
            code = {}
            for name in runner.CODE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
                code[name] = runner.sha256(path)
            evidence_dir = root / runner.PREFLIGHT_EVIDENCE_DIR
            evidence_dir.mkdir(parents=True)
            source_hashes = {}
            for name in runner.PROVENANCE:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
                source_hashes[name] = runner.sha256(path)
            source_binding = {runner.PROTOCOL: runner.sha256(protocol),
                              "configs/meta65_joint_control.json": runner.sha256(config),
                              **source_hashes}
            remote_source = {str(root / name): value for name, value in {**code, **source_binding}.items()}
            preflight = {
                "status.json": {"status": "preflight_completed", "mode": "cpu_preflight",
                                "cells": 0, "micro_cells": 9, "formal_result": False},
                "verification.json": {"status": "preflight_verified", "mode": "cpu_preflight",
                                      "cells_verified": 9, "all_metrics_verified": True,
                                      "formal_result": False, "summary": None},
                "input-binding.json": {"schema": 1, "mode": "cpu_preflight",
                                       "input_sha256": remote_source, "code_sha256": code,
                                       "estimated_formal_output_bytes": 200_000_000},
                "microcells.json": {"schema": 1, "mode": "cpu_preflight",
                                    "cells": [{} for _ in range(9)],
                                    "none_checks": [{} for _ in range(9)]},
                "environment.json": {"mode": "cpu_preflight", "threads": 1,
                                     "CUDA_VISIBLE_DEVICES": ""},
                "supervisor.json": {"status": "preflight_completed", "mode": "cpu_preflight",
                                    "returncode": 0, "failure": None, "phase": "certification",
                                    "peak_rss_bytes": 100_000_000, "peak_vram_bytes": 0,
                                    "outputbytes": 200_000_000, "elapsed": 30},
            }
            for filename, value in preflight.items():
                (evidence_dir / filename).write_text(json.dumps(value))
            binding = {
                "launch_ready": True, "execution_authorized": False,
                "gpu_formal_executed": False, "code_sha256": code,
                "input_sha256": source_binding,
                "cpu_preflight_evidence_sha256": {
                    str((evidence_dir / filename).relative_to(root)): runner.sha256(evidence_dir / filename)
                    for filename in preflight},
                "resource_budget": runner.FROZEN["limits"],
            }
            binding_path = root / "binding.json"
            binding_path.write_text(json.dumps(binding))
            with patch.object(runner, "PROTOCOL_SHA", runner.sha256(protocol)):
                runner.verify_launch_binding(binding_path, root=root, config_path=config)
                (root / runner.CODE_FILES[0]).write_text("changed")
                with self.assertRaisesRegex(ValueError, "binding|SHA"):
                    runner.verify_launch_binding(binding_path, root=root, config_path=config)
                (root / runner.CODE_FILES[0]).write_text(runner.CODE_FILES[0])
                evidence = evidence_dir / "verification.json"
                evidence.write_text("changed")
                with self.assertRaisesRegex(ValueError, "binding|SHA"):
                    runner.verify_launch_binding(binding_path, root=root, config_path=config)
                evidence.write_text(json.dumps(preflight["verification.json"]))
                preflight_binding = evidence_dir / "input-binding.json"
                old = json.loads(preflight_binding.read_text())
                old["code_sha256"][runner.CODE_FILES[0]] = "0" * 64
                preflight_binding.write_text(json.dumps(old))
                binding["cpu_preflight_evidence_sha256"][str(preflight_binding.relative_to(root))] = runner.sha256(
                    preflight_binding)
                binding_path.write_text(json.dumps(binding))
                with self.assertRaisesRegex(ValueError, "preflight|code|SHA"):
                    runner.verify_launch_binding(binding_path, root=root, config_path=config)
                old["code_sha256"][runner.CODE_FILES[0]] = code[runner.CODE_FILES[0]]
                preflight_binding.write_text(json.dumps(old))
                binding["cpu_preflight_evidence_sha256"][str(preflight_binding.relative_to(root))] = runner.sha256(
                    preflight_binding)
                binding["input_sha256"].pop(runner.PROVENANCE[0])
                binding_path.write_text(json.dumps(binding))
                with self.assertRaisesRegex(ValueError, "binding|SHA"):
                    runner.verify_launch_binding(binding_path, root=root, config_path=config)

    def test_formal_approval_binds_launch_and_is_consumed_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "formal-once"
            config = root / "config.json"
            config.write_text("frozen config")
            binding_path = root / "binding.json"
            binding_path.write_text("frozen binding")
            code = {name: "f" * 64 for name in runner.CODE_FILES}
            approval = {
                "schema": 2, "permission": "one_shot_formal_gpu_matrix",
                "launch_binding_sha256": runner.sha256(binding_path),
                "config_sha256": runner.sha256(config),
                "code_sha256": code, "protocol_sha256": runner.PROTOCOL_SHA,
                "output": str(output), "gpu_uuid": "GPU-test-uuid",
            }
            path = root / "approval.json"
            path.write_text(json.dumps(approval))
            runner.validate_formal_approval(path, output, binding_path, config,
                                            code, gpu_uuid="GPU-test-uuid")
            with self.assertRaisesRegex(ValueError, "binding|SHA"):
                runner.validate_formal_approval(path, output, root / "other-binding", config,
                                                code, gpu_uuid="GPU-test-uuid")
            with self.assertRaisesRegex(ValueError, "GPU"):
                runner.validate_formal_approval(path, output, binding_path, config,
                                                code, gpu_uuid="GPU-other")
            receipt = runner.claim_formal_approval(path, output, approval)
            self.assertTrue(receipt.is_file())
            runner.verify_formal_claim(path, output, parent_pid=os.getpid())
            with self.assertRaisesRegex(ValueError, "consumed|used"):
                runner.claim_formal_approval(path, output, approval)
            with self.assertRaisesRegex(ValueError, "consumed|used"):
                runner.validate_formal_approval(path, output, binding_path, config,
                                                code, gpu_uuid="GPU-test-uuid")

    def test_selected_gpu_must_be_idle_and_inventory_call_is_bounded(self):
        good = subprocess.CompletedProcess([], 0, "GPU-test-uuid, 1, 0\n", "")
        empty = subprocess.CompletedProcess([], 0, "", "")
        with patch("subprocess.run", side_effect=(good, empty)) as query:
            runner.verify_gpu_idle("GPU-test-uuid")
        self.assertEqual(query.call_count, 2)
        self.assertTrue(all(call.kwargs.get("timeout") for call in query.call_args_list))
        for used, utilization in ((101, 0), (1, 1)):
            busy = subprocess.CompletedProcess([], 0, f"GPU-test-uuid, {used}, {utilization}\n", "")
            with patch("subprocess.run", return_value=busy), self.assertRaisesRegex(ValueError, "idle"):
                runner.verify_gpu_idle("GPU-test-uuid")
        with (patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)),
              self.assertRaises((subprocess.TimeoutExpired, ValueError))):
            runner.verify_gpu_idle("GPU-test-uuid")

    def test_parent_consumes_approval_before_spawn_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = root / "config.json"
            config.write_text(json.dumps(runner.FROZEN))
            output = root / "formal-once"
            binding_path = root / "binding.json"
            binding_path.write_text("synthetic frozen binding")
            code = {name: runner.sha256(runner.ROOT / name) for name in runner.CODE_FILES}
            approval = {
                "schema": 2, "permission": "one_shot_formal_gpu_matrix",
                "launch_binding_sha256": runner.sha256(binding_path),
                "config_sha256": runner.sha256(config), "code_sha256": code,
                "protocol_sha256": runner.PROTOCOL_SHA,
                "output": str(output), "gpu_uuid": "GPU-test-uuid",
            }
            receipt = root / "approval.json"
            receipt.write_text(json.dumps(approval))
            with (patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test-uuid"}),
                  patch.object(runner, "verify_launch_binding", return_value={"launch_ready": True}),
                  patch.object(runner, "LAUNCH_BINDING", str(binding_path)),
                  patch.object(runner, "verify_gpu_idle") as idle,
                  patch("scripts.meta65_joint_supervisor.supervise", return_value={"status": "failed"}) as spawn):
                args = ["--config", str(config), "--out", str(output), "--mode", "formal",
                        "--execute", "--approval", str(receipt)]
                with self.assertRaises(SystemExit):
                    runner.main(args)
                idle.assert_called_once_with("GPU-test-uuid")
                spawn.assert_called_once()
                self.assertTrue((root / "approval.json.consumed.json").exists())
                with self.assertRaisesRegex(ValueError, "consumed|used"):
                    runner.main(args)
                spawn.assert_called_once()

    def test_worker_peak_high_water_marks_fail_closed(self):
        limits = runner.FROZEN["limits"]
        valid = {"peak_rss_bytes": limits["rss_bytes"],
                 "peak_vram_bytes": limits["vram_bytes"]}
        runner.check_cell_peaks(valid)
        for key in ("peak_rss_bytes", "peak_vram_bytes"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "limit"):
                runner.check_cell_peaks({**valid, key: limits["rss_bytes" if key == "peak_rss_bytes"
                                                    else "vram_bytes"] + 1})
        with self.assertRaises(ValueError):
            runner.check_cell_peaks({**valid, "peak_vram_bytes": True})
