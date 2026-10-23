"""Fail-closed scope launch gates; synthetic evidence never authorizes real GPUs."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from scripts import meta65_scope_contract as contract
from scripts import meta65_scope_launch as launch
from scripts import meta65_scope_verify as verifier

UUID = "GPU-" + "12345678-1234-1234-1234-123456789abc"


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def certified(mode: str) -> dict:
    return {
        "status": "preflight_verified",
        "mode": mode,
        "formal_result": False,
        "engineering_verified": True,
        "reproduction_verified": False,
        "cells_verified": 12,
        "all_metrics_verified": True,
        "summary": None,
    }


class ScopeLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "code"
        self.root.mkdir()
        self.config = self.root / "configs/meta65_scope.json"
        save(self.config, contract.FROZEN)
        self.data = self.root / "configs/meta65_server_data.json"
        save(
            self.data,
            {
                "cache_root": str(self.base / "cache"),
                "data_root": str(self.base / "PRICE-master"),
            },
        )
        protocol = self.root / contract.PROTOCOL
        protocol.parent.mkdir(parents=True, exist_ok=True)
        protocol.write_text("frozen scope protocol", encoding="utf-8")
        protocol_patch = patch.object(
            contract, "PROTOCOL_SHA", contract.sha256(protocol)
        )
        protocol_patch.start()
        self.addCleanup(protocol_patch.stop)
        self.code = {}
        for name in contract.CODE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            self.code = {**self.code, name: contract.sha256(path)}
        source = self.base / "cache/pretrain/chembl/manifest.json"
        save(source, {"source": True})
        paths = [
            *(self.root / name for name in self.code),
            self.config,
            self.data,
            protocol,
            source,
        ]
        self.inputs = {str(path): contract.sha256(path) for path in paths}
        self.binding = {
            "schema": 1,
            "experiment": "meta65-training-scope-20261011",
            "launch_ready": True,
            "execution_authorized": False,
            "formal_executed": False,
            "code_sha256": self.code,
            "config_sha256": contract.sha256(self.config),
            "protocol_sha256": contract.sha256(protocol),
            "data_config_sha256": contract.sha256(self.data),
            "input_sha256": self.inputs,
            "resource_budget": contract.FROZEN["limits"],
            "gpu_uuid": UUID,
            "preflights": {},
        }
        for mode in ("cpu_preflight", "gpu_preflight"):
            out = self.base / mode
            out.mkdir()
            save(
                out / "status.json",
                {
                    "status": "preflight_completed",
                    "mode": mode,
                    "cells": 12,
                    "formal_result": False,
                    "test_only": False,
                },
            )
            save(out / "verification.json", certified(mode))
            save(
                out / "input-binding.json",
                {
                    "schema": 1,
                    "mode": mode,
                    "code_sha256": self.code,
                    "input_sha256": self.inputs,
                },
            )
            save(
                out / "microcells.json",
                {"schema": 1, "mode": mode, "cells": [{} for _ in range(12)]},
            )
            save(
                out / "environment.json",
                {
                    "mode": mode,
                    "python": sys.version,
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "precision": "FP32 model/FP64 true",
                    "threads": 1,
                    "CUDA_VISIBLE_DEVICES": "" if mode == "cpu_preflight" else UUID,
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "deterministic_algorithms": True,
                    "tf32": False,
                },
            )
            records = []
            for arm in contract.ARMS:
                checkpoint = out / arm / "last.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"synthetic checkpoint")
                records.append(
                    {
                        "arm": arm,
                        "last_step": 2,
                        "optimizer_steps": 2,
                        "config": {"steps": 2},
                        "checkpoint": "last.pt",
                        "checkpoint_sha256": contract.sha256(checkpoint),
                    }
                )
            save(out / "training-records.json", {"schema": 1, "records": records})
            save(out / "reproduction.json", {"status": "not_applicable"})
            sidecar = out.with_name(out.name + ".supervisor.json")
            save(
                sidecar,
                {
                    "status": "preflight_completed",
                    "schema": 1,
                    "mode": mode,
                    "test_only": False,
                    "formal_result": False,
                    "returncode": 0,
                    "failure": None,
                    "cleanup_verified": True,
                    "output": str(out),
                    "supervisor_json": str(sidecar),
                    "gpu_uuid": None if mode == "cpu_preflight" else UUID,
                    "budget": launch.preflight_budget(mode),
                    "elapsed": 1.0,
                    "peak_rss_bytes": 100,
                    "peak_vram_bytes": 0,
                    "outputbytes": 100,
                },
            )
            out.with_name(out.name + ".supervisor.log").write_text(
                "synthetic worker log", encoding="utf-8"
            )
            self.binding["preflights"][mode] = {
                "output": str(out),
                "supervisor_file": str(sidecar),
                "raw_sha256": self.inventory(out),
            }
        self.path = self.root / "reports/launch-binding.json"
        self.persist()
        self.out = self.base / "formal"
        self.approval = self.base / "approval.json"
        self.input_patch = patch.object(
            contract, "bind_inputs", return_value=self.inputs, create=True
        )
        self.input_patch.start()
        self.addCleanup(self.input_patch.stop)
        self.verify_patch = patch.object(
            verifier,
            "verify",
            side_effect=lambda _out, *, formal: certified(Path(_out).name),
        )
        self.mock_verify = self.verify_patch.start()
        self.addCleanup(self.verify_patch.stop)
        self.idle_patch = patch.object(
            launch, "verify_gpu_idle", return_value={"uuid": UUID}
        )
        self.mock_idle = self.idle_patch.start()
        self.addCleanup(self.idle_patch.stop)

    def inventory(self, output: Path) -> dict[str, str]:
        files = [
            *(path for path in output.rglob("*") if path.is_file()),
            output.with_name(output.name + ".supervisor.json"),
            output.with_name(output.name + ".supervisor.log"),
        ]
        return {str(path): contract.sha256(path) for path in files}

    def persist(self) -> None:
        save(self.path, self.binding)

    def refresh(self, mode: str) -> None:
        out = Path(self.binding["preflights"][mode]["output"])
        self.binding = {
            **self.binding,
            "preflights": {
                **self.binding["preflights"],
                mode: {
                    **self.binding["preflights"][mode],
                    "raw_sha256": self.inventory(out),
                },
            },
        }
        self.persist()

    def validate(self) -> dict:
        return launch.verify_launch_binding(
            self.path, root=self.root, config_path=self.config
        )

    def permit(self, **changes: Any) -> None:
        save(
            self.approval,
            {
                "schema": 1,
                "permission": "one_shot_meta65_scope_formal",
                "binding_sha256": contract.sha256(self.path),
                "config_sha256": self.binding["config_sha256"],
                "code_sha256": self.code,
                "protocol_sha256": self.binding["protocol_sha256"],
                "gpu_uuid": UUID,
                "output": str(self.out),
                "resource_budget": self.binding["resource_budget"],
                **changes,
            },
        )

    def authorize(self) -> dict:
        return launch.authorize_formal(
            self.approval,
            self.out,
            self.path,
            self.config,
            root=self.root,
            gpu_uuid=UUID,
        )

    def worker(self, pid: int) -> dict:
        return launch.verify_worker_approval(
            self.approval,
            self.out,
            self.path,
            self.config,
            root=self.root,
            gpu_uuid=UUID,
            parent_pid=pid,
        )

    def test_validates_both_preflights_without_implicit_authorization(self) -> None:
        self.assertEqual(self.validate(), self.binding)
        self.assertEqual(self.mock_verify.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs == {"formal": False}
                for call in self.mock_verify.call_args_list
            )
        )
        self.mock_idle.assert_not_called()
        self.assertFalse(self.approval.exists())
        self.assertFalse(self.out.exists())

    def test_bad_code_config_protocol_data_sha_and_exact_closure(self) -> None:
        original = copy.deepcopy(self.binding)
        for field in ("config_sha256", "protocol_sha256", "data_config_sha256"):
            self.binding = {**original, field: "0" * 64}
            self.persist()
            with self.assertRaises(ValueError):
                self.validate()
        for code in (
            {},
            {**self.code, "extra.py": "0" * 64},
            {**self.code, next(iter(self.code)): "0" * 64},
        ):
            self.binding = {**original, "code_sha256": code}
            self.persist()
            with self.assertRaises(ValueError):
                self.validate()

    def test_empty_and_incomplete_inputs_and_preflight_disagreement(self) -> None:
        original = copy.deepcopy(self.binding)
        for inputs in (
            {},
            {k: v for i, (k, v) in enumerate(self.inputs.items()) if i > 0},
        ):
            self.binding = {**original, "input_sha256": inputs}
            self.persist()
            with self.assertRaises(ValueError):
                self.validate()
        self.binding = original
        out = Path(self.binding["preflights"]["gpu_preflight"]["output"])
        save(
            out / "input-binding.json",
            {
                "schema": 1,
                "mode": "gpu_preflight",
                "code_sha256": self.code,
                "input_sha256": {},
            },
        )
        self.refresh("gpu_preflight")
        with self.assertRaises(ValueError):
            self.validate()

    def test_flags_budget_schema_and_uuid_fail_closed(self) -> None:
        original = copy.deepcopy(self.binding)
        for field, value in (
            ("schema", True),
            ("launch_ready", False),
            ("execution_authorized", True),
            ("formal_executed", True),
            ("experiment", "meta65-joint-control-20261010"),
            (
                "resource_budget",
                {**original["resource_budget"], "total_seconds": 14401},
            ),
            ("gpu_uuid", "GPU-6"),
        ):
            self.binding = {**original, field: value}
            self.persist()
            with self.assertRaises(ValueError):
                self.validate()

    def test_raw_manifest_sha_omission_extra_symlink(self) -> None:
        mode = "cpu_preflight"
        original = copy.deepcopy(self.binding)
        raw = original["preflights"][mode]["raw_sha256"]
        for changed in (
            {},
            {**raw, next(iter(raw)): "0" * 64},
            {k: v for i, (k, v) in enumerate(raw.items()) if i > 0},
            {**raw, str(self.base / "extra.json"): "0" * 64},
        ):
            self.binding = {
                **original,
                "preflights": {
                    **original["preflights"],
                    mode: {**original["preflights"][mode], "raw_sha256": changed},
                },
            }
            self.persist()
            with self.assertRaises(ValueError):
                self.validate()
        self.binding = original
        out = Path(original["preflights"][mode]["output"])
        (out / "injected").symlink_to(self.config)
        self.persist()
        with self.assertRaises(ValueError):
            self.validate()

    def test_partial_preflight_and_untrusted_success_flags(self) -> None:
        out = Path(self.binding["preflights"]["cpu_preflight"]["output"])
        save(
            out / "microcells.json",
            {"schema": 1, "mode": "cpu_preflight", "cells": [{}] * 11},
        )
        self.refresh("cpu_preflight")
        with self.assertRaises(ValueError):
            self.validate()
        save(
            out / "microcells.json",
            {"schema": 1, "mode": "cpu_preflight", "cells": [{}] * 12},
        )
        self.refresh("cpu_preflight")
        self.mock_verify.side_effect = ValueError(
            "independent raw certification failed"
        )
        with self.assertRaisesRegex(ValueError, "independent"):
            self.validate()

    def test_environment_and_supervisor_and_training_rechecked(self) -> None:
        mode = "gpu_preflight"
        out = Path(self.binding["preflights"][mode]["output"])
        cases = (
            (out / "environment.json", "precision", "FP16"),
            (out / "environment.json", "CUBLAS_WORKSPACE_CONFIG", None),
            (out / "environment.json", "torch", "wrong-version"),
            (out / "environment.json", "threads", True),
            (out / "environment.json", "CUDA_VISIBLE_DEVICES", "6"),
            (out.with_name(out.name + ".supervisor.json"), "returncode", 1),
            (out.with_name(out.name + ".supervisor.json"), "cleanup_verified", False),
            (out.with_name(out.name + ".supervisor.json"), "failure", "timeout"),
        )
        for path, field, value in cases:
            old = contract.read_json(path)
            save(path, {**old, field: value})
            self.refresh(mode)
            with self.assertRaises(ValueError):
                self.validate()
            save(path, old)
            self.refresh(mode)
        record = contract.read_json(out / "training-records.json")
        save(
            out / "training-records.json", {**record, "records": record["records"][:1]}
        )
        self.refresh(mode)
        with self.assertRaises(ValueError):
            self.validate()

    def test_supervisor_budget_matches_frozen_mode_limits(self) -> None:
        from scripts import meta65_scope_supervisor as supervisor

        for mode in launch.MODES:
            self.assertEqual(supervisor.limits_for(mode), launch.preflight_budget(mode))
        self.assertEqual(
            supervisor.limits_for("formal")["total_seconds"],
            contract.FROZEN["limits"]["total_seconds"],
        )
        with self.assertRaises(ValueError):
            launch.preflight_budget("formal")

    def test_training_checkpoint_must_be_arm_last_pt_inside_output(self) -> None:
        out = Path(self.binding["preflights"]["cpu_preflight"]["output"])
        record = contract.read_json(out / "training-records.json")
        for name in ("../last.pt", str(self.base / "outside.pt"), "other.pt"):
            save(
                out / "training-records.json",
                {
                    **record,
                    "records": [
                        {**item, "checkpoint": name} for item in record["records"]
                    ],
                },
            )
            self.refresh("cpu_preflight")
            with self.assertRaises(ValueError):
                self.validate()
        save(out / "training-records.json", record)
        self.refresh("cpu_preflight")
        self.assertEqual(self.validate(), self.binding)

    def test_build_binding_round_trips_through_the_gate(self) -> None:
        from scripts import meta65_scope_build_binding as builder

        built = builder.build_binding(
            root=self.root,
            config_path=self.config,
            cpu_output=Path(self.binding["preflights"]["cpu_preflight"]["output"]),
            gpu_output=Path(self.binding["preflights"]["gpu_preflight"]["output"]),
            gpu_uuid=UUID,
        )
        self.assertEqual(built, self.binding)
        builder.write_binding(self.path, built)
        self.assertEqual(self.validate(), built)

    def test_approval_atomic_one_shot_and_worker_parent_binding(self) -> None:
        self.permit()
        receipt = self.authorize()
        self.assertEqual(receipt["status"], "consumed")
        self.assertEqual(receipt["supervisor_pid"], os.getpid())
        self.mock_idle.assert_called_once_with(UUID)
        with self.assertRaisesRegex(ValueError, "consumed|used"):
            self.authorize()
        self.out.mkdir()  # Worker may validate after its output was created.
        pid = receipt["supervisor_pid"]
        env = {
            "META65_SCOPE_SUPERVISOR_PID": str(pid),
            "META65_SCOPE_SUPERVISED_MODE": "formal",
            "CUDA_VISIBLE_DEVICES": UUID,
        }
        with patch.dict(os.environ, env), patch.object(os, "getppid", return_value=pid):
            self.assertEqual(self.worker(pid), receipt)
            with self.assertRaises(ValueError):
                self.worker(pid + 1)
        with (
            patch.dict(
                os.environ, {**env, "META65_SCOPE_SUPERVISOR_PID": str(pid + 1)}
            ),
            patch.object(os, "getppid", return_value=pid),
        ):
            with self.assertRaises(ValueError):
                self.worker(pid)

    def test_worker_does_not_claim_and_rejects_missing_or_tampered_receipt(
        self,
    ) -> None:
        self.permit()
        pid = os.getpid()
        env = {
            "META65_SCOPE_SUPERVISOR_PID": str(pid),
            "META65_SCOPE_SUPERVISED_MODE": "formal",
            "CUDA_VISIBLE_DEVICES": UUID,
        }
        with patch.dict(os.environ, env), patch.object(os, "getppid", return_value=pid):
            with self.assertRaises(ValueError):
                self.worker(pid)
        receipt = self.authorize()
        receipt_path = self.approval.with_name(self.approval.name + ".consumed.json")
        save(receipt_path, {**receipt, "supervisor_pid": pid + 1})
        with patch.dict(os.environ, env), patch.object(os, "getppid", return_value=pid):
            with self.assertRaises(ValueError):
                self.worker(pid)

    def test_busy_gpu_or_changed_preflight_never_consumes_approval(self) -> None:
        self.permit()
        self.mock_idle.side_effect = ValueError("GPU busy")
        with self.assertRaisesRegex(ValueError, "GPU busy"):
            self.authorize()
        self.assertFalse(
            self.approval.with_name(self.approval.name + ".consumed.json").exists()
        )
        self.mock_idle.side_effect = None
        Path(next(iter(self.inputs))).write_text("tampered", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.authorize()
        self.mock_idle.assert_called_once()

    def test_exact_approval_and_fresh_nonoverlapping_paths(self) -> None:
        for fields in (
            {"schema": True},
            {"resource_budget": {}},
            {"output": str(self.out) + "2"},
            {"extra": True},
            {"binding_sha256": "0" * 64},
            {"gpu_uuid": "GPU-6"},
        ):
            self.permit(**fields)
            with self.assertRaises(ValueError):
                self.authorize()
        self.permit()
        for output in (
            self.root,
            self.root / "scripts/new",
            self.base / "cache/new",
            self.base / "PRICE-master/new",
            self.base / "cpu_preflight/new",
        ):
            self.out = output
            self.approval = output.parent / "approval.json"
            self.permit()
            with self.assertRaises(ValueError):
                self.authorize()

    def test_relative_escape_symlink_and_existing_output_rejected(self) -> None:
        self.permit()
        self.out.mkdir()
        with self.assertRaises(ValueError):
            self.authorize()
        self.out.rmdir()
        (self.base / "link").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            launch.verify_launch_binding(
                self.path,
                root=self.root,
                config_path=self.base / "link/configs/meta65_scope.json",
            )
        with self.assertRaises(ValueError):
            launch.verify_launch_binding(
                self.path,
                root=self.root,
                config_path=self.root / "configs/../configs/meta65_scope.json",
            )
        mode = "cpu_preflight"
        self.binding["preflights"][mode]["output"] = "cpu_preflight"
        self.persist()
        with self.assertRaises(ValueError):
            self.validate()

    def test_real_independent_verifier_rejects_flag_only_fixture(self) -> None:
        self.verify_patch.stop()
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == "__main__":
    unittest.main()
