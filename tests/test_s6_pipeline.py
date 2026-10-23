"""S6 integration gates; S6_REVIEW_ROOT optionally selects a live checkout."""

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(os.environ.get("S6_REVIEW_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from scripts import run_s6_tail as s  # noqa: E402
from scripts import s4_anchor_evidence as s4  # noqa: E402
from scripts.s2_source_replay import TRAIN  # noqa: E402
from spatial_meta_ce.data import batch_samples  # noqa: E402
from spatial_meta_ce.train import train  # noqa: E402
from test_s4_anchor_evidence import rows  # noqa: E402
from test_s6_tail_evidence import preflight, record  # noqa: E402


class SyntheticStore:
    def __init__(self, database: str) -> None:
        self.rows = [
            {"id": str(i), "query_hash": f"{database}-查询-{i}"} for i in range(30)
        ]
        self.manifest = {"content_hash": "synthetic-s6"}
        self.cards = np.column_stack(
            (np.arange(30, dtype=np.float64) + 20, np.full(30, 2))
        )
        lengths = np.column_stack(
            (np.ones(30), np.ones(30), 3 + np.arange(30) % 2, np.zeros(30))
        )
        self.offsets = np.vstack((np.zeros(4), np.cumsum(lengths, axis=0))).astype(
            np.int64
        )

    def unique_indices(self) -> np.ndarray:
        return np.arange(30)

    def batch(self, indices: np.ndarray, device: str) -> dict:
        return batch_samples(
            [
                {
                    "groups": [
                        torch.ones(1, 40),
                        torch.ones(1, 40),
                        torch.ones(int(np.diff(self.offsets[:, 2])[i]), 4),
                        torch.ones(0, 43),
                    ],
                    "true": float(self.cards[i, 0]),
                    "pg": float(self.cards[i, 1]),
                    "id": self.rows[i]["id"],
                }
                for i in indices
            ],
            device,
        )


class CPUIntegration(unittest.TestCase):
    def test_actual_training_capture_and_independent_23_source_replay(self) -> None:
        plan = s.load_plan()
        self.assertEqual(plan["splits"]["train"], list(TRAIN))
        stores = {
            db: SyntheticStore(db) for db in TRAIN + tuple(plan["splits"]["validation"])
        }
        pool = Mock()
        pool.get.side_effect = lambda split, db: stores[db]
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        digests = []
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("spatial_meta_ce.train.StorePool", return_value=pool),
            patch.object(s, "StorePool", return_value=pool),
        ):
            data = {"cache_root": "synthetic", "splits": plan["splits"]}
            replay_stores = s.loss_stores(data)
            self.assertEqual(
                [call.args for call in pool.get.call_args_list],
                [("pretrain", db) for db in TRAIN],
            )
            for arm in s.ARMS:
                config = {
                    **s.anchor.smoke_configs(plan["arms"])[arm],
                    "model": {"width": 8, "use_pg": True},
                    "inner_steps": 0,
                }
                out = Path(tmp) / arm
                with redirect_stdout(io.StringIO()):
                    result = train(config, data, out, "cpu")
                verified = s.verify_loss_evidence(out, config, replay_stores)
                self.assertTrue(verified["passed"])
                self.assertEqual(
                    (
                        verified["episodes"],
                        verified["presentations"],
                        verified["snapshot_rows"],
                    ),
                    (8, 104, 8),
                )
                self.assertEqual(result["counts"]["optimizer_steps"], 2)
                self.assertEqual(result["loss_evidence"]["mode"], verified["mode"])
                _, state = s.anchor.load_checkpoint(out / "last.pt", "cpu")
                self.assertEqual(
                    state["extra"]["sample_plan_digest"], verified["sample_plan_digest"]
                )
                self.assertEqual(
                    state["extra"]["loss_evidence"], result["loss_evidence"]
                )
                digests.append(verified["sample_plan_digest"])
                snapshots = [
                    json.loads(line)
                    for line in (out / "loss-snapshots.jsonl").read_text().splitlines()
                ]
                changed = {
                    **snapshots[0],
                    "samples": [{**snapshots[0]["samples"][0], "label": 0.0}]
                    + snapshots[0]["samples"][1:],
                }
                (out / "loss-snapshots.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in [changed] + snapshots[1:])
                    + "\n"
                )
                with self.assertRaisesRegex(ValueError, "Cache label mismatch"):
                    s.verify_loss_evidence(out, config, replay_stores)
        self.assertEqual(digests[0], digests[1])


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(
            self.stack.enter_context(tempfile.TemporaryDirectory())
        ).resolve()
        self.plan = s.load_plan()
        self.evidence = {**preflight(), "input_sha256": {"fixture": "a" * 64}}
        self.data = {"splits": self.plan["splits"]}
        self.checkpoints, self.loss = {}, {}
        self.stack.enter_context(
            patch.object(s4, "COVERAGE", dict.fromkeys(s4.DATABASES, (3, 2)))
        )
        self.stack.enter_context(
            patch.object(
                s.verifier, "SOURCE_HASH", s4._digest(self.evidence["sources"])
            )
        )
        for name, value in (
            ("frozen_inputs", self.evidence["input_sha256"]),
            ("code_hashes", self.evidence["code_sha256"]),
            ("data_settings", self.data),
            ("validate_data", None),
            ("historical_sources", {}),
            ("source_evidence", self.evidence["sources"]),
            ("loss_stores", {}),
        ):
            self.stack.enter_context(patch.object(s, name, return_value=value))
        original_read = s.read_json
        self.stack.enter_context(
            patch.object(
                s.anchor,
                "load_checkpoint",
                side_effect=lambda path, _: (None, original_read(path)),
            )
        )
        self.stack.enter_context(
            patch.object(
                s,
                "verify_loss_evidence",
                side_effect=lambda path, *_: self.loss[str(path)],
            )
        )
        self.stack.enter_context(
            patch.object(
                s,
                "read_json",
                side_effect=lambda path: self.baseline
                if Path(path) == s.ROOT / s.BASELINE
                else original_read(path),
            )
        )
        for name, value in (
            ("protocol.json", self.plan),
            ("preflight.json", self.evidence),
        ):
            self.save(name, value)
        for prefix, formal in (("smoke/", False), ("", True)):
            records = []
            for arm in s.ARMS:
                r = copy.deepcopy(record(arm, formal))
                directory = self.root / prefix / arm
                directory.mkdir(parents=True)
                for initial in (False, True):
                    name = "initial_endpoint" if initial else "endpoint"
                    delta = 0 if initial else (2 if arm == s.ARMS[0] else 1)
                    group = {
                        db: [
                            {**row, "log_pred": row["pg_input_log"] + delta}
                            for row in rs
                        ]
                        for db, rs in rows().items()
                    }
                    checked = s.verifier.verify_rows(group, 1.0, None, formal=formal)
                    r[name] = checked["endpoint"]
                    r[("initial_" if initial else "") + "rows_verification"] = checked
                    r[("initial_" if initial else "") + "pg_identity_digest"] = r[name][
                        "pg_identity_digest"
                    ]
                    path = directory / (name + "_rows.jsonl")
                    path.write_text(
                        "".join(
                            json.dumps({"database": db, **row}) + "\n"
                            for db, rs in group.items()
                            for row in rs
                        )
                    )
                    r[name + "_rows_sha256"] = s.file_hash(path)
                for field, name in (
                    ("checkpoint_sha256", "last.pt"),
                    ("validation_plan_hash", "validation_manifest.json"),
                ):
                    (directory / name).write_text("fixture")
                    r[field] = s.file_hash(directory / name)
                for field, name in (
                    ("audit_sha256", "loss-audit.jsonl"),
                    ("snapshots_sha256", "loss-snapshots.jsonl"),
                ):
                    (directory / name).write_text("fixture")
                    r["loss_verification"][field] = s.file_hash(directory / name)
                self.loss[str(directory)] = copy.deepcopy(r["loss_verification"])
                self.checkpoints[str(directory / "last.pt")] = {
                    "next_step": r["config"]["steps"],
                    "metadata": {
                        "training_signature": {
                            k: v
                            for k, v in r["config"].items()
                            if k not in ("steps", "data_config")
                        }
                    },
                    "extra": {
                        k: r["summary"][k]
                        for k in ("counts", "gradient_stats", "loss_evidence")
                    },
                }
                self.save(
                    prefix + arm + "/last.pt",
                    self.checkpoints[str(directory / "last.pt")],
                )
                r["checkpoint_sha256"] = s.file_hash(directory / "last.pt")
                self.save(prefix + arm + "/result.json", r)
                self.save(
                    prefix + arm + "/loss-verification.json", r["loss_verification"]
                )
                records.append(r)
            if formal:
                self.stack.enter_context(
                    patch.object(
                        s.verifier,
                        "VALIDATION_HASH",
                        records[0]["validation_plan_hash"],
                    )
                )
                self.baseline = {
                    "runs": [{"arm": "A_uniform", "endpoint": records[0]["endpoint"]}]
                }
            self.save(prefix + "runs.json", {"runs": records})
            self.save(
                prefix + "summary.json",
                {"runs": records, **(s.verifier.choose(records) if formal else {})},
            )
            self.save(prefix + "verification.json", {"passed": True})
        s.anchor.output_manifest(self.root)

    def save(self, name: str, value: dict) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        s.write_json(path, value)

    def test_full_content_and_manifest_authenticate_real_decision(self) -> None:
        s.verify_content(self.root)
        hashes = s.verify_outputs(self.root)
        self.assertIn("B_mean_tail10/loss-audit.jsonl", hashes)
        self.assertEqual(
            s.read_json(self.root / "summary.json")["decision"], "promising_single_seed"
        )

    def test_rehashed_decision_endpoint_and_loss_authentication_rejected(self) -> None:
        original = s.read_json(self.root / "summary.json")
        for kind in ("decision", "endpoint", "loss"):
            summary = copy.deepcopy(original)
            candidate = summary["runs"][1]
            if kind == "decision":
                summary = {**summary, "decision": "no_candidate"}
            elif kind == "endpoint":
                candidate = {
                    **candidate,
                    "endpoint": {**candidate["endpoint"], "macro_four_plus_q95": 99},
                }
            else:
                candidate = {
                    **candidate,
                    "loss_verification": {
                        **candidate["loss_verification"],
                        "passed": False,
                    },
                }
            summary = {**summary, "runs": [summary["runs"][0], candidate]}
            self.save("summary.json", summary)
            self.save("runs.json", {"runs": summary["runs"]})
            self.save(s.ARMS[1] + "/result.json", candidate)
            s.anchor.output_manifest(self.root)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                s.verify_outputs(self.root)
        self.save("summary.json", original)

    def test_checkpoint_and_rehashed_file_drift_rejected(self) -> None:
        key = str(self.root / s.ARMS[0] / "last.pt")
        original = self.checkpoints[key]
        for field in ("counts", "gradient_stats", "loss_evidence"):
            self.save(
                s.ARMS[0] + "/last.pt",
                {**original, "extra": {**original["extra"], field: {}}},
            )
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "checkpoint content"),
            ):
                s.verify_content(self.root)
        self.save(s.ARMS[0] + "/last.pt", original)
        (self.root / s.ARMS[0] / "loss-audit.jsonl").write_text("tampered")
        s.anchor.output_manifest(self.root)
        with self.assertRaisesRegex(ValueError, "loss evidence file hash"):
            s.verify_outputs(self.root)

    def test_rehashed_disk_checkpoint_next_step_zero_rejected(self) -> None:
        relative = s.ARMS[0] + "/last.pt"
        checkpoint = s.read_json(self.root / relative)
        self.save(relative, {**checkpoint, "next_step": 0})
        summary = s.read_json(self.root / "summary.json")
        first = {
            **summary["runs"][0],
            "checkpoint_sha256": s.file_hash(self.root / relative),
        }
        records = [first, summary["runs"][1]]
        self.save(s.ARMS[0] + "/result.json", first)
        self.save("runs.json", {"runs": records})
        self.save("summary.json", {**summary, "runs": records})
        s.anchor.output_manifest(self.root)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            s.verify_outputs(self.root)

    def test_execute_success_and_baseline_failure_never_starts_b(self) -> None:
        records = s.read_json(self.root / "runs.json")["runs"]
        with (
            patch.object(s.anchor, "gpu_gate"),
            patch.object(s.anchor, "run_one", side_effect=records) as run,
        ):
            report = s.execute_arms(
                self.plan["arms"], self.data, self.root, self.evidence, {}, "3"
            )
            self.assertEqual(report["decision"], "promising_single_seed")
            self.assertEqual(run.call_count, 2)
        self.baseline = {"runs": []}
        with (
            patch.object(s.anchor, "gpu_gate"),
            patch.object(s.anchor, "run_one", return_value=records[0]) as run,
        ):
            with self.assertRaisesRegex(ValueError, "reproduction_blocked"):
                s.execute_arms(
                    self.plan["arms"], self.data, self.root, self.evidence, {}, "3"
                )
            self.assertEqual(run.call_count, 1)
        current = s.read_json(self.root / "status.json")
        self.assertEqual(current["status"], "reproduction_blocked")
        self.assertEqual(current["arms"][s.ARMS[1]], "not_started")


class LifecycleTests(unittest.TestCase):
    def test_preflight_checks_runtime_and_all_26_source_bindings(self) -> None:
        data = {"splits": s.load_plan()["splits"]}
        sources = dict.fromkeys(
            data["splits"]["train"] + data["splits"]["validation"], {}
        )
        with ExitStack() as stack:
            for obj, name, value in (
                (s.anchor, "gpu_gate", "GPU-fixture"),
                (s.anchor, "own_nvml_pid", 123),
                (s.platform, "python_version", s.anchor.DEPS["python"]),
                (s, "frozen_inputs", {}),
                (s, "code_hashes", {}),
                (s, "historical_sources", {}),
                (s, "source_evidence", sources),
                (s, "read_json", {"sources": sources}),
            ):
                stack.enter_context(patch.object(obj, name, return_value=value))
            stack.enter_context(
                patch.object(s.torch, "__version__", s.anchor.DEPS["torch"])
            )
            stack.enter_context(
                patch.object(s.np, "__version__", s.anchor.DEPS["numpy"])
            )
            stack.enter_context(
                patch.dict(
                    sys.modules, {"sqlglot": Mock(__version__=s.anchor.DEPS["sqlglot"])}
                )
            )
            evidence, historical = s.preflight(data, "3")
            self.assertEqual(len(evidence["sources"]), 26)
            self.assertEqual(evidence["nvml_process_id"], 123)
            self.assertEqual(historical, {})
            with (
                patch.object(s.platform, "python_version", return_value="wrong"),
                self.assertRaisesRegex(ValueError, "dependency"),
            ):
                s.preflight(data, "3")
            with (
                patch.object(s, "source_evidence", return_value={}),
                self.assertRaisesRegex(ValueError, "182-file"),
            ):
                s.preflight(data, "3")

    def test_supervisor_success_timeout_nonzero_and_cleanup(self) -> None:
        for mode in ("success", "timeout", "nonzero", "cleanup", "late", "baseline"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as tmp,
                ExitStack() as stack,
            ):
                root = Path(tmp).resolve()
                state = (
                    "reproduction_blocked" if mode == "baseline" else "outputs_ready"
                )
                s.status(root, state, dict.fromkeys(s.ARMS, "running"))
                child = Mock()
                child.wait.side_effect = (
                    s.subprocess.TimeoutExpired("fixture", 1)
                    if mode == "timeout"
                    else None
                )
                child.wait.return_value = int(
                    mode in ("nonzero", "cleanup", "baseline")
                )
                popen = stack.enter_context(
                    patch.object(s.subprocess, "Popen", return_value=child)
                )
                stop = stack.enter_context(
                    patch.object(
                        s.anchor,
                        "stop_child",
                        side_effect=RuntimeError("cleanup")
                        if mode == "cleanup"
                        else None,
                    )
                )
                stack.enter_context(
                    patch.object(s, "verify_outputs", return_value={"verified": "sha"})
                )
                stack.enter_context(
                    patch.object(
                        s.time,
                        "monotonic",
                        side_effect=[10, 31] if mode == "late" else None,
                        return_value=10,
                    )
                )
                if mode == "success":
                    s.supervise(["fixture"], root, 30)
                    stop.assert_not_called()
                else:
                    with self.assertRaises(
                        (RuntimeError, TimeoutError, s.subprocess.TimeoutExpired)
                    ):
                        s.supervise(["fixture"], root, 30)
                    stop.assert_called_once_with(child)
                current = s.read_json(root / "status.json")
                expected = (
                    "completed"
                    if mode == "success"
                    else "reproduction_blocked"
                    if mode == "baseline"
                    else "partial"
                    if mode in ("timeout", "late")
                    else "failed"
                )
                self.assertEqual(current["status"], expected)
                env = popen.call_args.kwargs["env"]
                self.assertEqual(
                    env["S6_TOKEN"], s.read_json(root / "supervision.json")["token"]
                )
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                if mode == "cleanup":
                    self.assertIn("cleanup", current["cleanup_error"])

    def test_worker_ownership_deadline_success_and_failure(self) -> None:
        for mode in ("owner", "expired", "success", "error"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as tmp,
                ExitStack() as stack,
            ):
                root = Path(tmp).resolve()
                s.write_json(
                    root / "supervision.json", {"pid": 123, "token": "fixture"}
                )
                s.status(root, "planned", dict.fromkeys(s.ARMS, "not_started"))
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "S6_PARENT": "123",
                            "S6_TOKEN": "wrong" if mode == "owner" else "fixture",
                            "S6_DEADLINE": "0"
                            if mode == "expired"
                            else str(s.time.monotonic() + 20),
                        },
                    )
                )
                for obj, name, value in (
                    (s.os, "getppid", 123),
                    (s.anchor.precision, "bind_parent_lifetime", None),
                    (s, "data_settings", {}),
                    (s, "validate_data", None),
                    (s, "configure_determinism", None),
                    (s, "source_evidence", {}),
                    (s, "code_hashes", {}),
                    (s, "frozen_inputs", {}),
                ):
                    stack.enter_context(patch.object(obj, name, return_value=value))
                stack.enter_context(
                    patch.object(s, "checked_path", side_effect=lambda _, path: path)
                )
                evidence = dict(sources={}, code_sha256={}, input_sha256={})
                pre = stack.enter_context(
                    patch.object(s, "preflight", return_value=(evidence, {}))
                )
                run = stack.enter_context(
                    patch.object(
                        s,
                        "execute_arms",
                        side_effect=ValueError("worker failure")
                        if mode == "error"
                        else None,
                    )
                )
                stack.enter_context(patch.object(s.signal, "signal"))
                timer = stack.enter_context(patch.object(s.signal, "setitimer"))
                if mode == "success":
                    s.worker(root, "3")
                    self.assertEqual(run.call_count, 2)
                    self.assertTrue(run.call_args_list[0].args[-1])
                    self.assertEqual(
                        s.read_json(root / "status.json")["status"], "outputs_ready"
                    )
                else:
                    with self.assertRaises((ValueError, TimeoutError)):
                        s.worker(root, "3")
                    if mode in ("owner", "expired"):
                        pre.assert_not_called()
                    else:
                        self.assertEqual(
                            s.read_json(root / "status.json")["status"], "failed"
                        )
                if mode in ("success", "error"):
                    timer.assert_called_with(s.signal.ITIMER_REAL, 0)

    def test_cli_dry_busy_execute_and_absolute_deadline(self) -> None:
        for execute, busy in ((False, False), (True, True), (True, False)):
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                output = Path(tmp) / "fresh"
                argv = ["s6", "--output", str(output), "--gpu-index", "3"] + (
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
                        side_effect=lambda path, *_: path,
                    )
                )
                gate = stack.enter_context(
                    patch.object(
                        s.anchor,
                        "gpu_gate",
                        side_effect=s.anchor.GPUBusy("busy") if busy else None,
                    )
                )
                supervise = stack.enter_context(patch.object(s, "supervise"))
                stream = stack.enter_context(redirect_stdout(io.StringIO()))
                s.run_main(30)
                self.assertEqual(supervise.call_count, int(execute and not busy))
                if not execute:
                    gate.assert_not_called()
                    self.assertFalse(output.exists())
                    self.assertEqual(
                        json.loads(stream.getvalue())["purpose"],
                        "dry-run-not-execution",
                    )
                if busy:
                    self.assertEqual(
                        s.read_json(output / "status.json")["status"],
                        "deferred_gpu_busy",
                    )
        with (
            patch.object(sys, "argv", ["s6", "--execute"]),
            patch.object(s, "run_main") as run,
            patch.object(s.signal, "setitimer") as timer,
        ):
            s.main()
            run.assert_called_once()
            timer.assert_called_with(s.signal.ITIMER_REAL, 0)
        with self.assertRaises(TimeoutError):
            s.deadline_expired(0, None)

    def test_expired_budget_does_not_launch_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            s.status(root, "planned", dict.fromkeys(s.ARMS, "not_started"))
            with (
                patch.object(s.subprocess, "Popen") as popen,
                self.assertRaises(TimeoutError),
            ):
                s.supervise(["fixture"], root, s.time.monotonic() - 1)
            popen.assert_not_called()
            self.assertEqual(s.read_json(root / "status.json")["status"], "partial")


if __name__ == "__main__":
    unittest.main()
