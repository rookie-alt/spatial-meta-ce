import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import run_precision_screen as screen
from tests.test_mapping_ablation import build_store


class PrecisionScreenTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_frozen_four_arm_plan_and_no_adaptation(self):
        plan = screen.load_plan()
        self.assertEqual(list(plan["arms"]), list(screen.ARMS))
        self.assertEqual(
            [c["model"]["width"] for c in plan["arms"].values()], [64, 64, 128, 128]
        )
        self.assertEqual(
            [c["loss"] for c in plan["arms"].values()], ["smooth_l1", "mse"] * 2
        )
        for cfg in plan["arms"].values():
            self.assertEqual(
                (cfg["method"], cfg["seed"], cfg["steps"], cfg["inner_steps"]),
                ("erm", 42, 4000, 0),
            )
            self.assertEqual(cfg["patience"], 41)

    def test_changed_frozen_config_and_protocol_are_rejected(self):
        contract = screen.read_json(screen.CONFIG)
        contract["training"]["steps"] = 3
        with (
            patch.object(screen, "read_json", return_value=contract),
            self.assertRaises(ValueError),
        ):
            screen.load_plan()
        with (
            patch.object(screen, "file_hash", return_value="bad"),
            self.assertRaises(ValueError),
        ):
            screen.load_plan()

    def test_exact_source_splits_required(self):
        plan = screen.load_plan()
        data = screen.data_settings(plan["arms"][screen.ARMS[0]], screen.CONFIG)
        screen.validate_data(data, plan)
        for field, values in [
            ("train", ["stats"]),
            ("validation", ["genome"]),
            ("train", data["splits"]["train"][:-1]),
            ("train", data["splits"]["train"] + ["accidents"]),
        ]:
            bad = {**data, "splits": {**data["splits"], field: values}}
            with self.assertRaises(ValueError):
                screen.validate_data(bad, plan)

    def test_output_rejects_old_run_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "runs").mkdir()
            (root / "source").mkdir()
            data = {
                "data_root": str(root / "source"),
                "cache_root": str(root / "cache"),
            }
            screen.validate_output(root / "runs/new", data, root)
            (root / "runs/old").mkdir()
            with self.assertRaises(FileExistsError):
                screen.validate_output(root / "runs/old", data, root)
            for path in [
                root / "source/new",
                root / "runs-other/new",
                root / "cache/new",
            ]:
                with self.assertRaises(ValueError):
                    screen.validate_output(path, data, root)
            (root / "runs/link").symlink_to(root / "source", target_is_directory=True)
            with self.assertRaises(ValueError):
                screen.validate_output(root / "runs/link/new", data, root)

    def test_cpu_four_arm_smoke_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for db in ["train", "chembl", "legalacts", "telstra"]:
                build_store(root, db, n_rows=10, n_tables=4)
            data = {
                "data_root": str(root / "source"),
                "cache_root": str(root / "cache"),
                "splits": {
                    "train": ["train"],
                    "validation": ["chembl", "legalacts", "telstra"],
                    "test": ["imdb", "stats", "genome", "ergastf1"],
                },
            }
            configs = screen.load_plan()["arms"]
            records = []
            for arm in screen.ARMS:
                cfg = {
                    **configs[arm],
                    "steps": 2,
                    "meta_batch": 1,
                    "support_sizes": [2],
                    "query_size": 3,
                    "validate_every": 1,
                    "validation_runs": 1,
                    "purpose": "synthetic-smoke-not-research",
                }
                records.append(
                    screen.run_one(
                        arm, cfg, data, root / arm, "cpu", expected_coverage=None
                    )
                )
            screen.verify_group(records, steps=2)
            self.assertEqual(
                [r["parameters"] for r in records], [132545, 132545, 288641, 288641]
            )
            for key in ["sample_plan_digest", "counts", "validation_plan_hash"]:
                bad = copy.deepcopy(records)
                bad[1][key] = "different"
                with self.assertRaises(ValueError):
                    screen.verify_group(bad, steps=2)
            bad = copy.deepcopy(records)
            bad[1]["endpoint"]["identity_digest"] = "different"
            with self.assertRaises(ValueError):
                screen.verify_group(bad, steps=2)
            with self.assertRaises(ValueError):
                screen.verify_group(records, steps=4000)

    def fixture_records(self):
        return [
            {
                "arm": a,
                "parameters": 1 if i < 2 else 2,
                "endpoint": {
                    "macro_four_plus_q95": 100.0,
                    "per_database": {
                        db: {
                            "all_tables": {"q95": 100.0},
                            "four_plus_tables": {"q95": 100.0},
                        }
                        for db in ["chembl", "legalacts", "telstra"]
                    },
                },
            }
            for i, a in enumerate(screen.ARMS)
        ]

    def test_candidate_thresholds_and_guard(self):
        records = self.fixture_records()
        self.assertEqual(screen.choose_candidates(records)["decision"], "no_candidate")
        records[1]["endpoint"]["macro_four_plus_q95"] = 90.0
        self.assertEqual(
            screen.choose_candidates(records)["candidates"], [screen.ARMS[1]]
        )
        records[1]["endpoint"]["per_database"]["telstra"]["all_tables"]["q95"] = 110.01
        self.assertEqual(screen.choose_candidates(records)["decision"], "no_candidate")

    def test_nonfinite_and_incomplete_results_rejected(self):
        for value in [float("nan"), float("inf"), -float("inf"), None, "90"]:
            records = self.fixture_records()
            records[1]["endpoint"]["per_database"]["chembl"]["all_tables"]["q95"] = (
                value
            )
            with self.assertRaises((ValueError, TypeError)):
                screen.choose_candidates(records)
        records = self.fixture_records()
        for bad in [records[:2], [records[0]] * 4]:
            with self.assertRaises(ValueError):
                screen.choose_candidates(bad)

    def test_gpu_gate(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "4"}):
            with self.assertRaises(ValueError):
                screen.formal_gpu("cpu", "4")
            with self.assertRaises(ValueError):
                screen.formal_gpu("cuda", "5")
            with (
                patch.object(
                    screen, "available_gpu", side_effect=RuntimeError("occupied")
                ),
                self.assertRaises(RuntimeError),
            ):
                screen.formal_gpu("cuda", "4")

    def test_timeout_terminates_own_group(self):
        with tempfile.TemporaryDirectory() as temp:
            child = unittest.mock.Mock(pid=12345)
            child.wait.side_effect = [screen.subprocess.TimeoutExpired("test", 1), 0]
            with (
                patch.object(screen.subprocess, "Popen", return_value=child),
                patch.object(
                    screen.os, "killpg", side_effect=[None, ProcessLookupError]
                ) as kill,
                self.assertRaises(TimeoutError),
            ):
                screen.supervise(["synthetic-command"], Path(temp), timeout=1)
            self.assertEqual(
                kill.call_args_list,
                [
                    unittest.mock.call(12345, screen.signal.SIGTERM),
                    unittest.mock.call(12345, 0),
                ],
            )
            self.assertEqual(
                json.loads((Path(temp) / "status.json").read_text())["status"],
                "partial",
            )

    def test_historical_source_bundle_is_frozen(self):
        from scripts import precision_screen_evidence as evidence

        plan = screen.load_plan()
        databases = plan["splits"]["train"] + plan["splits"]["validation"]
        self.assertEqual(
            set(evidence.historical_sources(screen.ROOT, databases)), set(databases)
        )
        with (
            patch.object(evidence, "file_hash", return_value="changed"),
            self.assertRaises(ValueError),
        ):
            evidence.historical_sources(screen.ROOT, databases)

    def test_cache_bytes_and_manifest_bind_to_history(self):
        from scripts import precision_screen_evidence as evidence

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            build_store(root, "train", n_tables=4)
            historical = {
                "train": json.loads((root / "sidecars/train/manifest.json").read_text())
            }
            data = {
                "cache_root": str(root / "cache"),
                "splits": {"train": ["train"], "validation": []},
            }
            self.assertEqual(
                evidence.source_evidence(data, historical)["train"]["rows"], 8
            )
            with (root / "cache/pretrain/train/cards.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                evidence.source_evidence(data, historical)
            build_store(root / "alternate", "train", n_rows=9, n_tables=4)
            changed = {**data, "cache_root": str(root / "alternate/cache")}
            with self.assertRaises(ValueError):
                evidence.source_evidence(changed, historical)

    def test_cache_symlink_rejected_before_read(self):
        from scripts import precision_screen_evidence as evidence

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            build_store(root, "train", n_tables=4)
            (root / "cache/pretrain/alias").symlink_to(root / "cache/pretrain/train")
            data = {
                "cache_root": str(root / "cache"),
                "splits": {"train": ["alias"], "validation": []},
            }
            with self.assertRaises(ValueError):
                evidence.source_evidence(data, {"alias": {}})

    def endpoint_rows_fixture(self):
        return {
            db: [
                {
                    "query_hash": "same-cross-db",
                    "true": 100.0,
                    "n_tables": 4,
                    "log_pred": 3.0,
                }
            ]
            for db in screen.VALIDATION
        }

    def test_endpoint_identity_includes_database_truth_and_tables(self):
        from scripts import precision_screen_evidence as evidence

        rows = self.endpoint_rows_fixture()
        baseline = evidence.summarize_endpoint(rows, None)["identity_digest"]
        for key, value in [("true", 101.0), ("n_tables", 5)]:
            changed = copy.deepcopy(rows)
            changed["chembl"][0][key] = value
            self.assertNotEqual(
                evidence.summarize_endpoint(changed, None)["identity_digest"], baseline
            )
        changed = {**rows, "chembl": rows["chembl"] * 2}
        with self.assertRaises(ValueError):
            evidence.summarize_endpoint(changed, None)
        with self.assertRaises(ValueError):
            evidence.summarize_endpoint(rows, screen.COVERAGE)

    def test_raw_prediction_nonfinite_and_overflow_rejected(self):
        from scripts import precision_screen_evidence as evidence

        for value in [float("nan"), float("inf"), -float("inf"), 1000.0]:
            rows = self.endpoint_rows_fixture()
            rows["chembl"][0]["log_pred"] = value
            with self.assertRaises((ValueError, FloatingPointError)):
                evidence.summarize_endpoint(rows, None)

    def test_timeout_escalates_and_preserves_completed_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            screen.status(
                output,
                "running",
                {
                    a: "completed" if i == 0 else "running" if i == 1 else "planned"
                    for i, a in enumerate(screen.ARMS)
                },
            )
            child = unittest.mock.Mock(pid=12345)
            child.wait.side_effect = [
                screen.subprocess.TimeoutExpired("test", 1),
                screen.subprocess.TimeoutExpired("test", 1),
                0,
            ]
            with (
                patch.object(screen.subprocess, "Popen", return_value=child),
                patch.object(screen.os, "killpg") as kill,
                self.assertRaises(TimeoutError),
            ):
                screen.supervise(["test"], output, timeout=1)
            self.assertEqual(
                kill.call_args_list,
                [
                    unittest.mock.call(12345, screen.signal.SIGTERM),
                    unittest.mock.call(12345, screen.signal.SIGKILL),
                    unittest.mock.call(12345, 0),
                    unittest.mock.call(12345, screen.signal.SIGKILL),
                ],
            )
            states = json.loads((output / "status.json").read_text())["arms"]
            self.assertEqual(
                list(states.values()), ["completed", "partial", "planned", "planned"]
            )

    def test_supervisor_nonzero_and_missing_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            for code in [1, 0]:
                screen.status(output, "running", dict.fromkeys(screen.ARMS, "planned"))
                child = unittest.mock.Mock()
                child.wait.return_value = code
                with (
                    patch.object(screen.subprocess, "Popen", return_value=child),
                    self.assertRaises(RuntimeError),
                ):
                    screen.supervise(["test"], output)

    def test_termination_exit_race(self):
        child = unittest.mock.Mock(pid=12345)
        with patch.object(screen.os, "killpg", side_effect=ProcessLookupError):
            screen.terminate_child(child)
        child.wait.assert_called_once()

    def test_worker_requires_timeout_parent(self):
        with (
            patch.dict("os.environ", {"PRECISION_SUPERVISOR_PID": "invalid"}),
            self.assertRaises(RuntimeError),
        ):
            screen.worker(Path("/unused"), "4", False)

    def test_pairing_failure_stops_at_second_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            configs = screen.load_plan()["arms"]
            records = self.fixture_records()
            for record in records:
                record.update(
                    optimizer_steps=4000,
                    counts={"optimizer_steps": 4000},
                    sample_plan_digest="same",
                    sample_plan_episodes=16000,
                    validation_plan_hash="same",
                    manifest_hashes={},
                    initialization_hash="same",
                )
                record["endpoint"]["identity_digest"] = "same"
            records[1]["sample_plan_digest"] = "wrong"
            with (
                patch.object(screen, "run_one", side_effect=records) as run,
                self.assertRaises(ValueError),
            ):
                screen.execute_arms(
                    configs, {}, Path(temp), "cpu", expected_coverage=None
                )
            self.assertEqual(run.call_count, 2)
            state = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual(
                list(state["arms"].values()),
                ["completed", "failed", "planned", "planned"],
            )

    def test_worker_output_rejects_outside_and_non_owned_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "runs").mkdir()
            with patch.object(screen, "ROOT", root):
                with self.assertRaises(ValueError):
                    screen.validate_worker_output(root / "outside")
                output = root / "runs/existing"
                output.mkdir()
                (output / "old.txt").write_text("unrelated")
                with self.assertRaises(ValueError):
                    screen.validate_worker_output(output)

    def test_unresolved_cache_root_symlink_rejected(self):
        plan = screen.load_plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "artifacts").mkdir()
            (root / "outside").mkdir()
            (root / "artifacts/cache").symlink_to(root / "outside")
            data = {
                "splits": plan["splits"],
                "data_root": "/gly/weilin/PRICE-master",
                "cache_root": str(root / "outside"),
            }
            with patch.object(screen, "ROOT", root), self.assertRaises(ValueError):
                screen.validate_data(data, plan)

    def test_real_sigterm_reaps_supervised_child(self):
        import os
        import signal
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as temp:
            code = """
import json, signal, sys
from pathlib import Path
from scripts import run_precision_screen as s
output=Path(sys.argv[1])
s.status(output, 'running', dict.fromkeys(s.ARMS, 'planned'))
original=s.signal.signal
def install(sig, handler):
    result=original(sig, handler)
    if sig == signal.SIGTERM and handler is s.interrupted:
        print('ready', flush=True)
    return result
s.signal.signal=install
try:
    s.supervise([sys.executable, '-c', 'import time; time.sleep(60)'], output, timeout=60)
except TimeoutError:
    print('terminated', flush=True)
"""
            controller = subprocess.Popen(
                [sys.executable, "-B", "-c", code, temp],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=screen.ROOT,
            )
            try:
                self.assertEqual(controller.stdout.readline().strip(), "ready")
                os.kill(controller.pid, signal.SIGTERM)
                stdout, stderr = controller.communicate(timeout=15)
                self.assertEqual(controller.returncode, 0, stderr)
                self.assertIn("terminated", stdout)
                self.assertEqual(
                    json.loads((Path(temp) / "status.json").read_text())["status"],
                    "partial",
                )
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.communicate()

    def test_worker_orchestration_with_mocked_execution(self):
        import os

        for smoke in [False, True]:
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp).resolve()
                plan = screen.load_plan()
                configs = {}

                def execution(arms, *args, configs=configs, **kwargs):
                    configs.update(arms)
                    return {}

                with (
                    patch.dict(
                        os.environ, {"PRECISION_SUPERVISOR_PID": str(os.getppid())}
                    ),
                    patch.object(screen, "validate_worker_output"),
                    patch.object(screen, "bind_parent_lifetime"),
                    patch.object(screen, "validate_data"),
                    patch.object(screen, "formal_gpu", return_value="GPU-test"),
                    patch.object(screen, "configure_determinism"),
                    patch.object(screen, "historical_sources", return_value={}),
                    patch.object(screen, "source_evidence", return_value={}),
                    patch.object(screen, "execute_arms", side_effect=execution),
                ):
                    screen.worker(output, "4", smoke)
                self.assertEqual(configs[screen.ARMS[0]]["steps"], 2 if smoke else 4000)
                self.assertEqual(plan["arms"][screen.ARMS[0]]["steps"], 4000)
                self.assertEqual(
                    json.loads((output / "status.json").read_text())["status"],
                    "completed",
                )

    def test_cli_dry_run_and_supervised_execution(self):
        import contextlib
        import io

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = {
                "data_root": str(root / "source"),
                "cache_root": str(root / "cache"),
            }
            plan = screen.load_plan()
            with (
                patch.object(screen, "ROOT", root),
                patch.object(screen, "load_plan", return_value={"arms": plan["arms"]}),
                patch.object(screen, "data_settings", return_value=data),
                patch.object(screen, "validate_data"),
            ):
                for execute in [False, True]:
                    output = root / "runs" / ("execute" if execute else "dry")
                    args = ["screen", "--output", str(output), "--gpu-index", "4"]
                    if execute:
                        args.append("--execute")
                    with (
                        patch.object(screen.sys, "argv", args),
                        patch.object(screen, "formal_gpu"),
                        patch.object(screen, "supervise") as supervisor,
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        screen.main()
                    self.assertEqual(output.exists(), execute)
                    self.assertEqual(supervisor.call_count, int(execute))

    def test_linux_parent_death_setup_failure_and_success(self):
        import ctypes

        lib = unittest.mock.Mock()
        with (
            patch.object(screen.sys, "platform", "linux"),
            patch.object(ctypes, "CDLL", return_value=lib),
        ):
            lib.prctl.return_value = 0
            with (
                patch.object(screen.os, "getppid", return_value=999),
                self.assertRaises(RuntimeError),
            ):
                screen.bind_parent_lifetime(123)
            screen.bind_parent_lifetime(screen.os.getppid())
            lib.prctl.return_value = -1
            with self.assertRaises(RuntimeError):
                screen.bind_parent_lifetime(screen.os.getppid())
        with (
            patch.object(screen.sys, "platform", "darwin"),
            self.assertRaises(RuntimeError),
        ):
            screen.bind_parent_lifetime(screen.os.getppid())

    def test_supervisor_success(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            screen.status(output, "completed", dict.fromkeys(screen.ARMS, "completed"))
            child = unittest.mock.Mock()
            child.wait.return_value = 0
            with patch.object(screen.subprocess, "Popen", return_value=child):
                screen.supervise(["test"], output)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "completed"
            )

    def test_candidate_tie_uses_capacity_then_name(self):
        records = self.fixture_records()
        for record in records[1:]:
            record["endpoint"]["macro_four_plus_q95"] = 89.0
        self.assertEqual(
            screen.choose_candidates(records)["candidates"], list(screen.ARMS[1:])
        )

    def test_failed_arm_stops_following_arms(self):
        with tempfile.TemporaryDirectory() as temp:
            configs = screen.load_plan()["arms"]
            with (
                patch.object(
                    screen, "run_one", side_effect=FloatingPointError("Nonfinite loss")
                ) as run,
                self.assertRaises(FloatingPointError),
            ):
                screen.execute_arms(
                    configs, {}, Path(temp), "cpu", expected_coverage=None
                )
            self.assertEqual(run.call_count, 1)
            state = json.loads((Path(temp) / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["arms"][screen.ARMS[1]], "planned")


if __name__ == "__main__":
    unittest.main()
