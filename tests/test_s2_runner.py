import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import run_s2_diagnosis as runner


class S2RunnerTests(unittest.TestCase):
    def test_output_must_be_fresh_under_runs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "runs").mkdir()
            self.assertEqual(
                runner.validate_output(root, root / "runs/new"), root / "runs/new"
            )
            for path in (
                root / "runs",
                root / "artifacts/new",
                root / "runs/precision-screen-20261006/new",
            ):
                with self.assertRaises(ValueError):
                    runner.validate_output(root, path)
            (root / "runs/old").mkdir()
            with self.assertRaises(FileExistsError):
                runner.validate_output(root, root / "runs/old")

    def test_csv_utf8_lf_and_json_no_nan(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            runner.csv_output(path, "pool.csv", [{"n": 1, "database": "源库"}])
            self.assertNotIn(b"\r", (path / "pool.csv").read_bytes())
            with self.assertRaises(ValueError):
                runner.write_json(path / "bad.json", {"x": float("nan")})

    def run_child(self, code, timeout=5):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d).resolve()
            runner.update_status(output, "planned")
            try:
                runner.supervise(
                    [sys.executable, "-B", "-c", code, str(output)], output, timeout
                )
            except BaseException:
                self.assertNotEqual(
                    json.loads((output / "status.json").read_text())["status"],
                    "completed",
                )
                raise
            return json.loads((output / "status.json").read_text())

    def test_supervisor_nonzero_timeout_and_bad_manifest(self):
        with self.assertRaises(RuntimeError):
            self.run_child("raise SystemExit(3)")
        with self.assertRaises(TimeoutError):
            self.run_child("import time; time.sleep(10)", timeout=0.02)
        with self.assertRaises(TimeoutError):
            self.run_child("pass", timeout=0)
        with self.assertRaises(ValueError):
            self.run_child(
                'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); (p/"status.json").write_text(json.dumps({"status":"outputs_ready"})); (p/"output-manifest.json").write_text("{}")'
            )

    def test_supervisor_success_requires_all_hashes(self):
        code = """import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
names=("protocol.json","summary.json","pool.csv","exposure.csv","residuals.csv","comparison.csv","verification.json")
for name in names: (p/name).write_text("{}" if name.endswith("json") else "x\\n")
(p/"output-manifest.json").write_text(json.dumps({name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in names}))
(p/"status.json").write_text(json.dumps({"status":"outputs_ready"}))"""
        self.assertEqual(self.run_child(code)["status"], "completed")
        with self.assertRaises(ValueError):
            self.run_child(
                code.replace(
                    "hashlib.sha256((p/name).read_bytes()).hexdigest()", '"bad"'
                )
            )

    def test_worker_rejects_unowned_or_gpu_execution(self):
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}),
        ):
            with self.assertRaises(RuntimeError):
                runner.worker(Path(d), Path(d))

    def test_analysis_pipeline_uses_real_synthetic_statistics_and_verifier(self):
        import numpy as np
        from unittest.mock import patch
        from scripts import s2_source_replay as core

        config = dict(
            seed=42,
            steps=4000,
            meta_batch=1,
            query_size=1,
            support_sizes=[1, 2],
            accumulation=1,
        )
        caches = {
            db: dict(
                true=np.tile(
                    [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 500000.0], 32
                ),
                pg=np.tile(
                    [0.0, 2.0, 5.0, 100.0, 500.0, 20000.0, 100000.0, 1000000.0], 32
                ),
                n_tables=np.tile(np.arange(1, 9), 32),
                raw_rows=512,
                indices=np.arange(0, 512, 2),
                query_hash=[str(i) for i in range(256)],
            )
            for db in ["accidents", "airline", "chembl"]
        }
        contract = dict(
            train=["accidents", "airline"],
            validation=["chembl"],
            config=config,
            sources={db: {} for db in caches},
        )
        for changed in (False, True):
            with (
                tempfile.TemporaryDirectory() as d,
                patch.object(np, "__version__", "1.26.1"),
                patch.object(core, "load_contract", return_value=contract),
                patch.object(
                    core,
                    "snapshot",
                    side_effect=[
                        {"synthetic": "only"},
                        {"changed": "input"} if changed else {"synthetic": "only"},
                    ],
                ),
                patch.object(
                    core, "read_cache", side_effect=lambda root, db, frozen: caches[db]
                ),
                patch.object(core, "certify") as certify,
            ):
                output = Path(d).resolve()
                if changed:
                    with self.assertRaisesRegex(ValueError, "changed"):
                        runner.analyze(runner.ROOT, output)
                    self.assertFalse((output / "summary.json").exists())
                else:
                    result = runner.analyze(runner.ROOT, output)
                    self.assertFalse(result["training"])
                    self.assertTrue(result["replay_verified"])
                    self.assertEqual(
                        set(runner.verify_outputs(output)), set(runner.OUTPUTS)
                    )
                    self.assertEqual(
                        json.loads((output / "verification.json").read_text())[
                            "replay"
                        ]["episodes"],
                        4000,
                    )
                certify.assert_called_once()
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(np, "__version__", "wrong"),
        ):
            with self.assertRaisesRegex(ValueError, "NumPy"):
                runner.analyze(runner.ROOT, Path(d))

    def test_owned_worker_success_failure_and_memory_error_states(self):
        from unittest.mock import patch, Mock
        from scripts.s2_source_replay import ReplayMismatch

        parent = os.getppid()
        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "S2_PARENT": str(parent),
            "S2_TOKEN": "fixture",
            **{name: "1" for name in runner.THREAD_ENV},
        }
        for failure, state in [
            (None, "outputs_ready"),
            (ReplayMismatch("bad digest"), "blocked_replay_unverified"),
            (MemoryError("budget"), "failed"),
            (ValueError("source mismatch"), "failed"),
        ]:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve()
                output = root / "runs/synthetic"
                output.mkdir(parents=True)
                runner.update_status(output, "planned")
                runner.write_json(
                    output / "supervision.json", {"pid": parent, "token": "fixture"}
                )
                libc = Mock()
                libc.prctl.return_value = 0
                with (
                    patch.object(sys, "platform", "linux"),
                    patch.dict(os.environ, env),
                    patch.object(runner.ctypes, "CDLL", return_value=libc),
                    patch.object(runner.resource, "setrlimit"),
                    patch.object(runner, "analyze", side_effect=failure),
                ):
                    if failure:
                        with self.assertRaises(type(failure)):
                            runner.worker(root, output)
                    else:
                        runner.worker(root, output)
                self.assertEqual(
                    json.loads((output / "status.json").read_text())["status"], state
                )

    def test_launcher_and_absolute_deadline(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            with (
                patch.object(runner, "ROOT", root),
                patch.object(sys, "platform", "darwin"),
                patch.object(
                    sys, "argv", ["runner", "--output", str(root / "runs/dry")]
                ),
            ):
                runner.run_main(runner.time.monotonic() + 10)
                self.assertFalse((root / "runs/dry").exists())
            with (
                patch.object(runner, "ROOT", root),
                patch.object(sys, "platform", "linux"),
                patch.object(runner.resource, "setrlimit"),
                patch.object(runner, "supervise") as supervised,
                patch.object(
                    sys,
                    "argv",
                    ["runner", "--execute", "--output", str(root / "runs/new")],
                ),
            ):
                runner.run_main(runner.time.monotonic() + 10)
                supervised.assert_called_once()
            with (
                patch.object(runner, "ROOT", root),
                patch.object(sys, "platform", "linux"),
                patch.object(runner.resource, "setrlimit"),
                patch.object(runner, "worker") as worker,
                patch.object(
                    sys,
                    "argv",
                    ["runner", "--worker", "--output", str(root / "runs/new")],
                ),
            ):
                runner.run_main(runner.time.monotonic() + 10)
                worker.assert_called_once()
        with self.assertRaises(TimeoutError):
            runner.deadline_expired(0, None)
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(sys, "argv", ["runner", "--execute"]),
            patch.object(runner.signal, "signal"),
            patch.object(runner.signal, "setitimer") as timer,
            patch.object(runner, "run_main", side_effect=TimeoutError("fixture")),
        ):
            with self.assertRaises(TimeoutError):
                runner.main()
            self.assertEqual(
                timer.call_args_list[-1].args, (runner.signal.ITIMER_REAL, 0)
            )

    def test_fresh_process_no_torch_or_training(self):
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                'import sys; from scripts import run_s2_diagnosis; assert not any(x == "torch" or x.startswith("spatial_meta_ce") for x in sys.modules)',
            ],
            capture_output=True,
        )
        self.assertEqual(p.returncode, 0, p.stderr.decode())


if __name__ == "__main__":
    unittest.main()
