import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import run_s4_anchor as s

LOCAL = Path(__file__).resolve().parents[1]
PARENT = LOCAL


class S4AnchorTests(unittest.TestCase):
    def test_frozen_plan_and_explicit_scales(self):
        with patch.object(s, "ROOT", PARENT):
            plan = s.load_plan(config=LOCAL / "configs/s4_anchor.json")
        self.assertEqual(list(plan["arms"]), list(s.ARMS))
        self.assertEqual(
            [c["model"]["pg_output_scale"] for c in plan["arms"].values()], [1.0, 0.0]
        )
        self.assertTrue(all(c["model"]["use_pg"] for c in plan["arms"].values()))
        self.assertTrue(
            all(
                c["steps"] == 4000 and c["loss"] == "smooth_l1"
                for c in plan["arms"].values()
            )
        )

    def test_canonical_checkpoint_does_not_require_legacy_scale_one(self):
        self.assertEqual(
            s.canonical_model({"width": 64}), {"width": 64, "pg_output_scale": 1.0}
        )
        self.assertNotEqual(
            s.canonical_model({"width": 64}),
            s.canonical_model({"width": 64, "pg_output_scale": 0.0}),
        )

    def test_gpu_initial_idle_and_live_own_pid(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
            with patch.object(
                s.subprocess, "check_output", side_effect=["4, GPU-one, 0, 0\n", ""]
            ):
                self.assertEqual(s.gpu_gate("cuda", "4", initial=True), "GPU-one")
            with patch.object(
                s.subprocess,
                "check_output",
                side_effect=["4, GPU-one, 300, 75\n", f"GPU-one, {os.getpid()}\n"],
            ):
                self.assertEqual(
                    s.gpu_gate("cuda", "4", expected_uuid="GPU-one"), "GPU-one"
                )
            with patch.object(
                s.subprocess,
                "check_output",
                side_effect=["4, GPU-one, 300, 75\n", "GPU-one, 999999\n"],
            ):
                with self.assertRaises(s.GPUBusy):
                    s.gpu_gate("cuda", "4", expected_uuid="GPU-one")

    def test_no_silent_false_verification(self):
        self.assertEqual(s.require_passed({"passed": True}), {"passed": True})
        for value in ({"passed": False}, {}, {"passed": 1}):
            with self.assertRaises(ValueError):
                s.require_passed(value)

    def test_output_fresh_and_symlink_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = {
                "data_root": str(root / "source"),
                "cache_root": str(root / "cache"),
            }
            with patch.object(s, "ROOT", root):
                self.assertEqual(
                    s.validate_output(root / "runs/new", data), root / "runs/new"
                )
                (root / "runs").mkdir()
                (root / "runs/existing").mkdir()
                with self.assertRaises(FileExistsError):
                    s.validate_output(root / "runs/existing", data)
                with self.assertRaises(ValueError):
                    s.validate_output(root / "cache/new", data)
                (root / "runs/link").symlink_to(root / "source")
                with self.assertRaises(ValueError):
                    s.validate_output(root / "runs/link/new", data)

    def fake_verifier(self):
        return Mock(
            verify_rows=Mock(return_value={"passed": True}),
            analyze_rows=Mock(return_value=[]),
            verify_run=Mock(return_value={"passed": True}),
            verify_baseline=Mock(return_value={"passed": True}),
            verify_pair=Mock(return_value={"passed": True}),
            choose=Mock(return_value={"decision": "no_candidate"}),
        )

    def plan(self):
        with patch.object(s, "ROOT", PARENT):
            return s.load_plan(config=LOCAL / "configs/s4_anchor.json")

    def test_plan_changes_fail_closed(self):
        with patch.object(s, "ROOT", PARENT):
            with (
                patch.object(s, "read_json", return_value={}),
                self.assertRaises(ValueError),
            ):
                s.load_plan(config=LOCAL / "configs/s4_anchor.json")
            with (
                patch.object(s, "file_hash", return_value="bad"),
                self.assertRaises(ValueError),
            ):
                s.load_plan(config=LOCAL / "configs/s4_anchor.json")

    def test_data_and_gpu_guards(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = {
                "splits": plan["splits"],
                "data_root": "/gly/weilin/PRICE-master",
                "cache_root": str(root / "artifacts/cache"),
            }
            with patch.object(s, "ROOT", root):
                s.validate_data(data, plan)
                for bad in (
                    {**data, "splits": {}},
                    {**data, "data_root": str(root)},
                    {**data, "cache_root": str(root / "other")},
                ):
                    with self.assertRaises(ValueError):
                        s.validate_data(bad, plan)
        for device, index in [("cpu", "4"), ("cuda", None), ("cuda", "4,5")]:
            with self.assertRaises(ValueError):
                s.gpu_gate(device, index)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}):
            for rows, apps, kind in [
                ("5, GPU-one, 0, 0\n", "", ValueError),
                ("4, GPU-two, 0, 0\n", "", ValueError),
                ("4, GPU-one, 65, 0\n", "", s.GPUBusy),
                ("4, GPU-one, 0, 1\n", "", s.GPUBusy),
                ("4, GPU-one, 0, 0\n", f"GPU-one, {os.getpid()}\n", s.GPUBusy),
            ]:
                with (
                    patch.object(
                        s.subprocess, "check_output", side_effect=[rows, apps]
                    ),
                    self.assertRaises(kind),
                ):
                    s.gpu_gate("cuda", "4", expected_uuid="GPU-one", initial=True)

    def test_frozen_and_runtime_preflight(self):
        sources = {str(i): {} for i in range(26)}
        data = {"splits": {"train": list(sources), "validation": []}}
        with patch.object(s, "ROOT", PARENT):
            self.assertEqual(s.frozen_inputs(), s.BINDINGS)
        with (
            patch.object(s, "file_hash", return_value="bad"),
            self.assertRaises(ValueError),
        ):
            s.frozen_inputs()
        with (
            patch.object(s, "gpu_gate", return_value="GPU-test"),
            patch.object(s.platform, "python_version", return_value="3.11.4"),
            patch.object(s.torch, "__version__", "2.1.0+cu121"),
            patch.object(s.np, "__version__", "1.26.1"),
            patch("sqlglot.__version__", "18.17.0"),
            patch.object(s, "frozen_inputs", return_value={}),
            patch.object(s, "historical_sources", return_value={}),
            patch.object(s, "source_evidence", return_value=sources),
            patch.object(s, "read_json", return_value={"sources": sources}),
            patch.object(s, "file_hash", return_value="sha"),
            patch.object(s, "code_hashes", return_value={}),
        ):
            value, old = s.preflight(data, "4")
            self.assertEqual(value["sources"], sources)
            self.assertEqual(old, {})
            with (
                patch.object(s.np, "__version__", "wrong"),
                self.assertRaises(ValueError),
            ):
                s.preflight(data, "4")
            with (
                patch.object(s, "read_json", return_value={"sources": {}}),
                self.assertRaises(ValueError),
            ):
                s.preflight(data, "4")

    def test_cpu_real_run_config_roundtrip_and_pg_rows(self):
        from tests.test_mapping_ablation import build_store

        importlib.import_module("torch._dynamo")  # before sys.modules mocking
        s.torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for db in ["train", "chembl", "legalacts", "telstra"]:
                build_store(root, db, n_rows=16, n_tables=4)
            data = {
                "data_root": str(root / "source"),
                "cache_root": str(root / "cache"),
                "splits": {
                    "train": ["train"],
                    "validation": list(s.VALIDATION),
                    "test": ["imdb", "stats", "genome", "ergastf1"],
                },
            }
            records = []
            for arm, cfg in s.smoke_configs(self.plan()["arms"]).items():
                with contextlib.redirect_stdout(io.StringIO()):
                    records.append(
                        s.run_one(arm, cfg, data, root / arm, "cpu", None, True)
                    )
            self.assertEqual([r["parameters"] for r in records], [132545, 132545])
            for key in [
                "counts",
                "sample_plan_digest",
                "validation_plan_hash",
                "initialization_hash",
                "pg_identity_digest",
            ]:
                self.assertEqual(records[0][key], records[1][key])
            a_state = s.load_checkpoint(root / s.ARMS[0] / "last.pt")[1]
            self.assertNotIn("pg_output_scale", a_state["model_config"])
            self.assertEqual(
                s.load_checkpoint(root / s.ARMS[1] / "last.pt")[1]["model_config"][
                    "pg_output_scale"
                ],
                0.0,
            )
            for record in records:
                rows = [
                    json.loads(line)
                    for line in (Path(record["output"]) / "endpoint_rows.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertTrue(
                    all(
                        r["total_correction_vs_pg"] == r["log_pred"] - r["pg_input_log"]
                        for r in rows
                    )
                )
                self.assertTrue(all(len(r["token_counts"]) == 4 for r in rows))
            with self.assertRaises(ValueError):
                s.run_one(
                    s.ARMS[0],
                    self.plan()["arms"][s.ARMS[0]],
                    data,
                    root / "formal",
                    "cpu",
                    None,
                )

    def test_execute_stops_b_after_failed_baseline_and_preserves_records(self):
        configs = self.plan()["arms"]
        verifier = self.fake_verifier()
        verifier.verify_baseline.side_effect = ValueError("A not reproduced")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            with (
                patch("scripts.s4_anchor_evidence", verifier),
                patch.dict(sys.modules, {"scripts.s4_anchor_evidence": verifier}),
                patch.object(s, "gpu_gate"),
                patch.object(s, "source_evidence", return_value={}),
                patch.object(s, "read_json", return_value={}),
                patch.object(s, "run_one", return_value={"arm": s.ARMS[0]}) as run,
                self.assertRaises(ValueError),
            ):
                s.execute_arms(
                    configs, {}, output, {"sources": {}, "gpu_uuid": "GPU"}, {}, "4"
                )
            self.assertEqual(run.call_count, 1)
            current = json.loads((output / "status.json").read_text())
            self.assertEqual(current["status"], "reproduction_blocked")
            self.assertEqual(current["arms"][s.ARMS[1]], "not_started")
            self.assertEqual(
                len(json.loads((output / "runs.json").read_text())["runs"]), 1
            )
            self.assertFalse((output / "summary.json").exists())

    def test_execute_smoke_and_formal_success_and_rehash_failure(self):
        configs = self.plan()["arms"]
        verifier = self.fake_verifier()
        for smoke in [True, False]:
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp).resolve()
                with (
                    patch("scripts.s4_anchor_evidence", verifier),
                    patch.dict(sys.modules, {"scripts.s4_anchor_evidence": verifier}),
                    patch.object(s, "gpu_gate"),
                    patch.object(s, "source_evidence", return_value={}),
                    patch.object(s, "read_json", return_value={}),
                    patch.object(
                        s, "run_one", side_effect=[{"arm": a} for a in s.ARMS]
                    ),
                ):
                    result = s.execute_arms(
                        configs,
                        {},
                        output,
                        {"sources": {}, "gpu_uuid": "GPU"},
                        {},
                        "4",
                        smoke,
                    )
                self.assertEqual(len(result["runs"]), 2)
                self.assertEqual("decision" in result, not smoke)
                self.assertEqual(
                    json.loads((output / "verification.json").read_text())["passed"],
                    True,
                )
        for values in [[{"bad": True}], [{}, {"bad": True}]]:
            with tempfile.TemporaryDirectory() as temp:
                with (
                    patch("scripts.s4_anchor_evidence", verifier),
                    patch.dict(sys.modules, {"scripts.s4_anchor_evidence": verifier}),
                    patch.object(s, "gpu_gate"),
                    patch.object(s, "source_evidence", side_effect=values),
                    patch.object(s, "run_one", return_value={}),
                    self.assertRaises(ValueError),
                ):
                    s.execute_arms(
                        configs,
                        {},
                        Path(temp),
                        {"sources": {}, "gpu_uuid": "GPU"},
                        {},
                        "4",
                    )

    def test_supervisor_success_and_output_verification_failure(self):
        for valid in [True, False]:
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp).resolve()
                s.status(output, "outputs_ready", dict.fromkeys(s.ARMS, "verified"))
                child = Mock(pid=1234)
                child.wait.return_value = 0
                with (
                    patch.object(s.subprocess, "Popen", return_value=child),
                    patch.object(
                        s,
                        "verify_outputs",
                        return_value={"file": "sha"},
                        side_effect=None if valid else ValueError("manifest"),
                    ),
                    patch.object(s, "stop_child") as cleanup,
                ):
                    if valid:
                        s.supervise(["fixture"], output, s.time.monotonic() + 60)
                    else:
                        with self.assertRaises(ValueError):
                            s.supervise(["fixture"], output, s.time.monotonic() + 60)
                self.assertEqual(
                    json.loads((output / "status.json").read_text())["status"],
                    "completed" if valid else "failed",
                )
                self.assertEqual(cleanup.call_count, int(not valid))

    def test_supervisor_timeout_cleanup_error_and_no_premature_completion(self):
        for code in [1, 0, "timeout"]:
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp).resolve()
                s.status(
                    output, "running", {s.ARMS[0]: "verified", s.ARMS[1]: "running"}
                )
                child = Mock(pid=1234)
                if code == "timeout":
                    child.wait.side_effect = s.subprocess.TimeoutExpired("fixture", 1)
                else:
                    child.wait.return_value = code
                with (
                    patch.object(s.subprocess, "Popen", return_value=child),
                    patch.object(
                        s, "stop_child", side_effect=RuntimeError("cleanup failed")
                    ),
                    self.assertRaises((RuntimeError, s.subprocess.TimeoutExpired)),
                ):
                    s.supervise(["fixture"], output, s.time.monotonic() + 60)
                current = json.loads((output / "status.json").read_text())
                self.assertEqual(current["status"], "failed")
                self.assertEqual(current["arms"][s.ARMS[0]], "verified")
                self.assertEqual(current["arms"][s.ARMS[1]], "partial")
                self.assertIn("cleanup failed", current["cleanup_error"])

    def test_cleanup_reaps_only_own_group_and_has_bounded_escalation(self):
        child = Mock(pid=1234)
        with patch.object(s.os, "killpg", side_effect=ProcessLookupError):
            s.stop_child(child)
        child.wait.side_effect = [s.subprocess.TimeoutExpired("fixture", 1), 0]
        with patch.object(s.os, "killpg") as kill:
            s.stop_child(child)
        self.assertEqual(kill.call_args_list[0].args, (1234, s.signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (1234, s.signal.SIGKILL))
        child.wait.side_effect = s.subprocess.TimeoutExpired("fixture", 1)
        with patch.object(s.os, "killpg"), self.assertRaises(RuntimeError):
            s.stop_child(child)

    def test_manifest_rejects_missing_hashes_symlinks_and_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            names = [
                "protocol.json",
                "preflight.json",
                "summary.json",
                "runs.json",
                "verification.json",
                "smoke/summary.json",
                "smoke/verification.json",
            ]
            names += [
                f"{prefix}{arm}/{name}"
                for prefix in ("", "smoke/")
                for arm in s.ARMS
                for name in (
                    "last.pt",
                    "result.json",
                    "endpoint_rows.jsonl",
                    "initial_endpoint_rows.jsonl",
                    "validation_manifest.json",
                )
            ]
            names += ["smoke/runs.json"]
            for name in names:
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"passed": True}))
            for prefix in ("", "smoke/"):
                records = []
                for arm in s.ARMS:
                    record = {
                        "arm": arm,
                        **{
                            field: s.file_hash(output / f"{prefix}{arm}/{name}")
                            for field, name in (
                                ("checkpoint_sha256", "last.pt"),
                                ("endpoint_rows_sha256", "endpoint_rows.jsonl"),
                                (
                                    "initial_endpoint_rows_sha256",
                                    "initial_endpoint_rows.jsonl",
                                ),
                                ("validation_plan_hash", "validation_manifest.json"),
                            )
                        },
                    }
                    (output / f"{prefix}{arm}/result.json").write_text(
                        json.dumps(record)
                    )
                    records.append(record)
                (output / f"{prefix}runs.json").write_text(
                    json.dumps({"runs": records})
                )
            hashes = s.output_manifest(output)
            self.assertEqual(s.verify_outputs(output), hashes)
            checkpoint = output / f"{s.ARMS[0]}/last.pt"
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(b"corrupt after verification")
            s.output_manifest(output)
            with self.assertRaisesRegex(ValueError, "record"):
                s.verify_outputs(output)
            checkpoint.write_bytes(original)
            s.output_manifest(output)
            self.assertTrue(s.verify_outputs(output))
            (output / "runs.json").write_text("changed")
            with self.assertRaises(ValueError):
                s.verify_outputs(output)
            (output / "runs.json").unlink()
            with self.assertRaises(ValueError):
                s.verify_outputs(output)

    def test_ordered_pg_identity_matches_independent_verifier(self):
        from scripts.s4_anchor_evidence import verify_rows

        rows = {
            db: [
                {
                    "query_hash": str(i),
                    "true": 10.0,
                    "n_tables": 4,
                    "log_pred": 1.0,
                    "pg_input_log": 1.0,
                    "pg_reference_log": 1.0,
                }
                for i in range(2)
            ]
            for db in s.VALIDATION
        }
        checked = verify_rows(rows, 1.0, None, formal=False)
        self.assertEqual(s.pg_identity(rows), checked["endpoint"]["pg_identity_digest"])
        reversed_rows = {db: list(reversed(group)) for db, group in rows.items()}
        self.assertNotEqual(s.pg_identity(rows), s.pg_identity(reversed_rows))

    def test_worker_original_error_survives_supervisor_failure(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d).resolve()
            s.status(
                output,
                "failed",
                dict.fromkeys(s.ARMS, "not_started"),
                error="FloatingPointError: Nonfinite outer gradient",
            )
            child = Mock(pid=1234)
            child.wait.return_value = 1
            with (
                patch.object(s.subprocess, "Popen", return_value=child),
                patch.object(s, "stop_child"),
                self.assertRaises(RuntimeError),
            ):
                s.supervise(["fixture"], output, s.time.monotonic() + 60)
            status = json.loads((output / "status.json").read_text())
            self.assertIn("Nonfinite outer gradient", status["worker_error"])

    def test_worker_and_deadlines_mocked_full_flow(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            output = root / "runs/new"
            output.mkdir(parents=True)
            s.status(output, "planned", dict.fromkeys(s.ARMS, "not_started"))
            s.write_json(
                output / "supervision.json", {"pid": os.getppid(), "token": "test"}
            )
            env = {
                "S4_PARENT": str(os.getppid()),
                "S4_TOKEN": "test",
                "S4_DEADLINE": str(s.time.monotonic() + 60),
            }
            evidence = {
                "sources": {},
                "gpu_uuid": "GPU",
                "code_sha256": {},
                "input_sha256": {},
            }
            with (
                patch.dict(os.environ, env),
                patch.object(s, "ROOT", root),
                patch.object(s.precision, "bind_parent_lifetime"),
                patch.object(s, "load_plan", return_value=self.plan()),
                patch.object(s, "data_settings", return_value={}),
                patch.object(s, "validate_data"),
                patch.object(s, "preflight", return_value=(evidence, {})),
                patch.object(s, "configure_determinism"),
                patch.object(s, "execute_arms") as execute,
                patch.object(s, "source_evidence", return_value={}),
                patch.object(s, "code_hashes", return_value={}),
                patch.object(s, "frozen_inputs", return_value={}),
            ):
                s.worker(output, "4")
            self.assertEqual(execute.call_count, 2)
            self.assertEqual(execute.call_args_list[0].args[0][s.ARMS[0]]["steps"], 2)
            self.assertEqual(
                execute.call_args_list[1].args[0][s.ARMS[0]]["steps"], 4000
            )
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"],
                "outputs_ready",
            )
            with patch.object(s, "ROOT", root), self.assertRaises(ValueError):
                s.worker(output, "4")
        with self.assertRaises(TimeoutError):
            s.deadline_expired(0, None)

    def test_initial_gpu_query_failure_is_recorded_without_training(self):
        for failure in (
            RuntimeError("nvidia-smi unavailable"),
            TimeoutError("initial deadline"),
        ):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve()
                output = root / "runs/new"
                args = ["s4", "--execute", "--output", str(output), "--gpu-index", "4"]
                data = {
                    "data_root": str(root / "source"),
                    "cache_root": str(root / "cache"),
                }
                with (
                    patch.object(s.sys, "argv", args),
                    patch.object(s.sys, "platform", "linux"),
                    patch.object(s, "ROOT", root),
                    patch.object(s, "load_plan", return_value=self.plan()),
                    patch.object(s, "data_settings", return_value=data),
                    patch.object(s, "validate_data"),
                    patch.object(s, "gpu_gate", side_effect=failure),
                    patch.object(s, "supervise") as supervise,
                ):
                    with self.assertRaises(type(failure)):
                        s.run_main(s.time.monotonic() + 60)
                state = json.loads((output / "status.json").read_text())
                self.assertEqual(state["status"], "failed")
                self.assertIn(str(failure), state["error"])
                supervise.assert_not_called()

    def test_main_dry_run_busy_and_supervision(self):
        for execute, busy in [(False, False), (True, False), (True, True)]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                output = root / "runs/new"
                args = ["s4", "--output", str(output), "--gpu-index", "4"] + (
                    ["--execute"] if execute else []
                )
                data = {
                    "data_root": str(root / "source"),
                    "cache_root": str(root / "cache"),
                }
                with (
                    patch.object(s.sys, "argv", args),
                    patch.object(s.sys, "platform", "linux"),
                    patch.object(s, "ROOT", root),
                    patch.object(s, "load_plan", return_value=self.plan()),
                    patch.object(s, "data_settings", return_value=data),
                    patch.object(s, "validate_data"),
                    patch.object(
                        s, "gpu_gate", side_effect=s.GPUBusy("busy") if busy else None
                    ),
                    patch.object(s, "supervise") as supervisor,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    s.main()
                self.assertEqual(supervisor.call_count, int(execute and not busy))
                self.assertEqual(output.exists(), execute)
                if busy:
                    self.assertEqual(
                        json.loads((output / "status.json").read_text())["status"],
                        "deferred_gpu_busy",
                    )


if __name__ == "__main__":
    unittest.main()
