import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_s5_weight as s


class S5RunnerTests(unittest.TestCase):
    def test_frozen_plan_only_changes_training_weight(self):
        plan = s.load_plan()
        a, b = (plan["arms"][name] for name in s.ARMS)
        self.assertEqual(a["complex_query_weight"], 1)
        self.assertEqual(b["complex_query_weight"], 2)
        self.assertEqual(
            {k: v for k, v in a.items() if k != "complex_query_weight"},
            {k: v for k, v in b.items() if k != "complex_query_weight"},
        )
        self.assertEqual(a["model"]["pg_output_scale"], 1)
        self.assertEqual(a["steps"], 4000)

    def test_matrix_tampering_is_rejected(self):
        value = json.loads(s.CONFIG.read_text())
        for field in ("arms", "model", "timeout_seconds"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                changed = {**value, field: {}}
                path = Path(tmp) / "bad.json"
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    s.load_plan(path)

    def test_protocol_tampering_is_rejected(self):
        with patch.object(s, "file_hash", return_value="0" * 64):
            with self.assertRaises(ValueError):
                s.load_plan()

    def test_source_plan_rejects_target_and_cache_redirection(self):
        plan = s.load_plan()
        data = {
            "splits": plan["splits"],
            "data_root": "/gly/weilin/PRICE-master",
            "cache_root": str(s.ROOT / "artifacts/cache"),
        }
        with patch.object(s, "checked_path", side_effect=lambda root, path: path):
            s.validate_data(data, plan)
            with self.assertRaises(ValueError):
                s.validate_data({**data, "splits": {"train": ["stats"]}}, plan)
            with self.assertRaises(ValueError):
                s.validate_data({**data, "cache_root": "/tmp/cache"}, plan)

    def test_source_evidence_is_checked_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = {"gpu_uuid": "GPU-one", "sources": {"frozen": 1}}
            with (
                patch.object(s.anchor, "gpu_gate"),
                patch.object(s, "source_evidence", return_value={}),
                patch.object(s.anchor, "run_one") as run,
            ):
                with self.assertRaises(ValueError):
                    s.execute_arms(
                        s.load_plan()["arms"], {}, Path(tmp), evidence, {}, "3"
                    )
                run.assert_not_called()
                self.assertEqual(
                    json.loads((Path(tmp) / "status.json").read_text())["status"],
                    "failed",
                )

    def test_baseline_failure_stops_before_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = {"gpu_uuid": "GPU-one", "sources": {}}
            (Path(tmp) / s.ARMS[0]).mkdir()
            with (
                patch.object(s.anchor, "gpu_gate"),
                patch.object(s, "source_evidence", return_value={}),
                patch.object(
                    s.anchor, "run_one", return_value={"arm": s.ARMS[0]}
                ) as run,
                patch.object(s.verifier, "verify_run", return_value={"passed": True}),
                patch.object(
                    s.verifier,
                    "verify_baseline",
                    side_effect=ValueError("reproduction_blocked"),
                ),
            ):
                with self.assertRaises(ValueError):
                    s.execute_arms(
                        s.load_plan()["arms"], {}, Path(tmp), evidence, {}, "3"
                    )
                self.assertEqual(run.call_count, 1)
                current = json.loads((Path(tmp) / "status.json").read_text())
                self.assertEqual(current["status"], "reproduction_blocked")
                self.assertEqual(current["arms"][s.ARMS[1]], "not_started")

    def test_output_verifier_rejects_incomplete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.write_json(root / "output-manifest.json", {})
            with self.assertRaises(ValueError):
                s.verify_outputs(root)

    def test_missing_record_hashes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.write_json(root / "runs.json", {"runs": []})
            with self.assertRaises(ValueError):
                s.verify_record_files(root)

    def test_expired_budget_never_starts_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.status(root, "planned", dict.fromkeys(s.ARMS, "not_started"))
            with patch.object(s.subprocess, "Popen") as popen:
                with self.assertRaises(TimeoutError):
                    s.supervise(["no-command"], root, s.time.monotonic() - 1)
                popen.assert_not_called()
            self.assertNotEqual(
                json.loads((root / "status.json").read_text())["status"], "completed"
            )

    def test_worker_requires_supervisor_before_any_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.write_json(root / "supervision.json", {"pid": 0, "token": "bad"})
            with (
                patch.object(s, "ROOT", root.parent),
                patch.object(s, "checked_path", side_effect=lambda a, b: b),
                patch.object(s, "preflight") as preflight,
            ):
                with self.assertRaises(ValueError):
                    s.worker(root, "3")
                preflight.assert_not_called()


class S5PipelineTests(unittest.TestCase):
    def test_real_smoke_config_is_accepted(self):
        from test_s5_weight_evidence import record

        value = record(formal=False)
        value["config"]["purpose"] = "gpu-smoke-not-research"
        s.verifier.verify_pair(
            [
                value,
                {
                    **record(s.ARMS[1], False),
                    "config": {
                        **record(s.ARMS[1], False)["config"],
                        "purpose": "gpu-smoke-not-research",
                    },
                },
            ],
            False,
        )

    def test_completed_arms_use_real_pair_and_decision(self):
        from test_s5_weight_evidence import record

        with tempfile.TemporaryDirectory() as tmp:
            for arm in s.ARMS:
                (Path(tmp) / arm).mkdir()
            with (
                patch.object(s.anchor, "gpu_gate"),
                patch.object(s, "source_evidence", return_value={}),
                patch.object(
                    s.anchor, "run_one", side_effect=[record(), record(s.ARMS[1])]
                ),
                patch.object(s.verifier, "verify_run", return_value={"passed": True}),
                patch.object(
                    s.verifier, "verify_baseline", return_value={"passed": True}
                ),
            ):
                report = s.execute_arms(
                    s.load_plan()["arms"],
                    {},
                    Path(tmp),
                    {"sources": {}, "gpu_uuid": "GPU-fixture"},
                    {},
                    "3",
                )
            self.assertEqual(report["decision"], "no_candidate")
            self.assertEqual(len(report["runs"]), 2)

    def test_supervisor_success_and_cleanup_error(self):
        from unittest.mock import Mock

        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                s.status(root, "outputs_ready", dict.fromkeys(s.ARMS, "verified"))
                child = Mock(pid=99999)
                child.wait.return_value = 1 if failure else 0
                with (
                    patch.object(s.subprocess, "Popen", return_value=child),
                    patch.object(s, "verify_outputs", return_value={"checked": "sha"}),
                    patch.object(
                        s.anchor,
                        "stop_child",
                        side_effect=RuntimeError("cleanup") if failure else None,
                    ),
                ):
                    if failure:
                        with self.assertRaises(RuntimeError):
                            s.supervise(["synthetic"], root, s.time.monotonic() + 10)
                    else:
                        s.supervise(["synthetic"], root, s.time.monotonic() + 10)
                current = json.loads((root / "status.json").read_text())
                self.assertEqual(
                    current["status"], "failed" if failure else "completed"
                )
                if failure:
                    self.assertIn("cleanup", current["cleanup_error"])

    def test_output_hash_manifest_and_record_crosscheck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for prefix in ("", "smoke/"):
                records = []
                for arm in s.ARMS:
                    directory = root / prefix / arm
                    directory.mkdir(parents=True)
                    fields = {
                        "checkpoint_sha256": "last.pt",
                        "endpoint_rows_sha256": "endpoint_rows.jsonl",
                        "initial_endpoint_rows_sha256": "initial_endpoint_rows.jsonl",
                        "validation_plan_hash": "validation_manifest.json",
                    }
                    row = {"arm": arm}
                    for key, name in fields.items():
                        (directory / name).write_text("synthetic")
                        row[key] = s.file_hash(directory / name)
                    s.write_json(directory / "result.json", row)
                    records.append(row)
                s.write_json(root / prefix / "runs.json", {"runs": records})
                s.write_json(root / prefix / "summary.json", {})
                s.write_json(root / prefix / "verification.json", {"passed": True})
            for name in ("protocol.json", "preflight.json", "weight-audit.json"):
                s.write_json(root / name, {"passed": True})
            s.anchor.output_manifest(root)
            with patch.object(s, "verify_content"):
                s.verify_outputs(root)
            (root / s.ARMS[0] / "last.pt").write_text("changed")
            s.anchor.output_manifest(root)
            with self.assertRaisesRegex(ValueError, "record file hash"):
                s.verify_outputs(root)

    def test_worker_pipeline_and_error_leave_status(self):
        import os
        from contextlib import ExitStack

        for fails in (False, True):
            with (
                self.subTest(fails=fails),
                tempfile.TemporaryDirectory() as tmp,
                ExitStack() as stack,
            ):
                root = Path(tmp).resolve()
                s.write_json(
                    root / "supervision.json", {"pid": 123, "token": "synthetic"}
                )
                s.status(root, "planned", dict.fromkeys(s.ARMS, "not_started"))
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "S5_TOKEN": "synthetic",
                            "S5_PARENT": "123",
                            "S5_DEADLINE": str(s.time.monotonic() + 20),
                        },
                    )
                )
                evidence = {"sources": {}, "code_sha256": {}, "input_sha256": {}}
                patches = [
                    (s.os, "getppid", 123),
                    (s, "checked_path", lambda a, b: b),
                    (s.anchor.precision, "bind_parent_lifetime", None),
                    (s, "data_settings", {}),
                    (s, "validate_data", None),
                    (s, "preflight", (evidence, {})),
                    (s, "audit_training_flow", {"passed": True}),
                    (s, "configure_determinism", None),
                    (s, "source_evidence", {}),
                    (s, "code_hashes", {}),
                    (s, "frozen_inputs", {}),
                    (s.anchor, "output_manifest", {}),
                ]
                for obj, key, value in patches:
                    kw = (
                        {"side_effect": value}
                        if callable(value)
                        else {"return_value": value}
                    )
                    stack.enter_context(patch.object(obj, key, **kw))
                run = stack.enter_context(
                    patch.object(
                        s,
                        "execute_arms",
                        side_effect=ValueError("synthetic error") if fails else None,
                    )
                )
                if fails:
                    with self.assertRaises(ValueError):
                        s.worker(root, "3")
                else:
                    s.worker(root, "3")
                    self.assertEqual(run.call_count, 2)
                current = json.loads((root / "status.json").read_text())
                self.assertEqual(
                    current["status"], "failed" if fails else "outputs_ready"
                )

    def test_cli_dry_busy_execute_and_main_deadline(self):
        import sys
        from contextlib import ExitStack
        import io

        for execute, busy in ((False, False), (True, False), (True, True)):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                output = Path(tmp) / "fresh"
                argv = ["s5", "--output", str(output), "--gpu-index", "3"] + (
                    ["--execute"] if execute else []
                )
                stack.enter_context(patch.object(sys, "argv", argv))
                stack.enter_context(patch.object(sys, "platform", "linux"))
                stack.enter_context(patch.object(s, "data_settings", return_value={}))
                stack.enter_context(patch.object(s, "validate_data"))
                stack.enter_context(
                    patch.object(
                        s.anchor.precision,
                        "validate_output",
                        side_effect=lambda p, *a: p,
                    )
                )
                stack.enter_context(
                    patch.object(
                        s.anchor,
                        "gpu_gate",
                        side_effect=s.anchor.GPUBusy("busy") if busy else None,
                    )
                )
                supervise = stack.enter_context(patch.object(s, "supervise"))
                stack.enter_context(patch.object(sys, "stdout", io.StringIO()))
                s.run_main(s.time.monotonic() + 30)
                self.assertEqual(supervise.call_count, int(execute and not busy))
                if busy:
                    self.assertEqual(
                        json.loads((output / "status.json").read_text())["status"],
                        "deferred_gpu_busy",
                    )
        with (
            patch.object(sys, "argv", ["s5", "--execute"]),
            patch.object(s, "run_main") as run,
            patch.object(s.signal, "setitimer"),
        ):
            s.main()
            run.assert_called_once()
        with self.assertRaises(TimeoutError):
            s.deadline_expired(0, None)


if __name__ == "__main__":
    unittest.main()
