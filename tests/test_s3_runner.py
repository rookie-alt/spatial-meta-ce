import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts import run_s3_diagnosis as runner
from scripts.s2_source_replay import digest, file_hash


class S3RunnerTests(unittest.TestCase):
    def cache_fixture(self, root: Path, defect: str = "") -> tuple:
        p = root / "artifacts/cache/pretrain/accidents"
        p.mkdir(parents=True)
        rows = [
            {"id": f"accidents/pretrain/{i + 1}", "query_hash": key}
            for i, key in enumerate(["z", "a", "z", "b"])
        ]
        m = {
            "schema": 1,
            "transform": "price40-group-logfanout-v1",
            "unique_queries": 3,
            "provenance": {"excluded": []},
            "rows": rows,
        }
        m = {**m, "content_hash": digest(m)}
        (p / "manifest.json").write_text(json.dumps(m))
        np.save(
            p / "cards.npy",
            np.array([[10, 11], [0, 0], [10, 11], [100, 100]], dtype=float),
        )
        counts = np.array(
            [[1, 2, 2, 0], [0, 0, 3, 1], [1, 2, 2, 0], [3, 4, 4, 2]], dtype=np.int64
        )
        if defect == "counts":
            counts[2, 0] = 2
        offsets = np.vstack([np.zeros(4, dtype=np.int64), np.cumsum(counts, axis=0)])
        np.save(p / "offsets.npy", offsets)
        frozen = {
            "rows": 4,
            "manifest_hash": m["content_hash"],
            "file_sha256": {
                n: file_hash(p / n)
                for n in ("manifest.json", "cards.npy", "offsets.npy")
            },
        }
        return p, frozen

    def test_four_counts_first_occurrence_and_conflict(self):
        for defect in ("", "counts"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve()
                _, frozen = self.cache_fixture(root, defect)
                if defect:
                    with self.assertRaisesRegex(ValueError, "Duplicate"):
                        runner.read_metadata(root, "accidents", frozen)
                else:
                    data = runner.read_metadata(root, "accidents", frozen)
                    self.assertEqual(
                        data["token_counts"].tolist(),
                        [[1, 2, 2, 0], [0, 0, 3, 1], [3, 4, 4, 2]],
                    )
                    self.assertEqual(data["indices"].tolist(), [0, 1, 3])
        with self.assertRaises(ValueError):
            runner.read_metadata(Path("/no-read"), "imdb", {})

    def test_frozen_contract(self):
        c = runner.load_contract(runner.ROOT)
        self.assertEqual(len(c["train"]), 23)
        self.assertEqual(len(c["validation"]), 3)
        self.assertIn(runner.PLAN, c["input_sha256"])

    def test_old_denominators_are_not_silently_changed(self):
        root = runner.ROOT
        s = json.loads((root / runner.S2 / "summary.json").read_text())
        caches = {
            db: {
                "raw_rows": v["raw_rows"],
                "true": np.zeros(v["unique_queries"]),
                "n_tables": np.ones(v["unique_queries"]),
            }
            for db, v in s["source_pool"].items()
        }
        with self.assertRaisesRegex(ValueError, "S1"):
            runner.check_denominators(root, caches)
        caches["sap"] = {**caches["sap"], "raw_rows": 1}
        with self.assertRaisesRegex(ValueError, "S2"):
            runner.check_denominators(root, caches)

    def test_output_paths_no_overwrite_or_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            (root / "runs").mkdir()
            self.assertEqual(
                runner.validate_output(root, root / "runs/new"), root / "runs/new"
            )
            for p in (
                root / "runs",
                root / "artifacts/new",
                root / "runs/precision-screen-20261006/new",
                root / "runs/s2-source-exposure-20261006/new",
            ):
                with self.assertRaises(ValueError):
                    runner.validate_output(root, p)
            (root / "runs/old").mkdir()
            with self.assertRaises(FileExistsError):
                runner.validate_output(root, root / "runs/old")
            (root / "runs/link").symlink_to(root / "runs", target_is_directory=True)
            with self.assertRaises(ValueError):
                runner.validate_output(root, root / "runs/link/new")

    def child(self, code: str, timeout: float = 5) -> dict:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d).resolve()
            runner.update_status(out, "planned")
            try:
                runner.supervise(
                    [sys.executable, "-B", "-c", code, str(out)], out, timeout
                )
            except BaseException:
                self.assertNotEqual(
                    json.loads((out / "status.json").read_text())["status"], "completed"
                )
                raise
            return json.loads((out / "status.json").read_text())

    def test_supervisor_failure_timeout_and_manifest(self):
        with self.assertRaises(RuntimeError):
            self.child("raise SystemExit(4)")
        with self.assertRaises(TimeoutError):
            self.child("import time; time.sleep(5)", 0.02)
        with self.assertRaises(TimeoutError):
            self.child("pass", 0)
        with self.assertRaises(ValueError):
            self.child(
                'import sys,json; from pathlib import Path; p=Path(sys.argv[1]); (p/"status.json").write_text(json.dumps({"status":"outputs_ready"})); (p/"output-manifest.json").write_text("{}")'
            )

    def test_cleanup_timeout_keeps_original_failure_and_failed_state(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d).resolve()
            runner.update_status(output, "running")
            child = Mock()
            child.wait.return_value = 4
            with (
                patch.object(runner.subprocess, "Popen", return_value=child),
                patch.object(
                    runner,
                    "stop_child",
                    side_effect=subprocess.TimeoutExpired("cleanup", 10),
                ),
                self.assertRaisesRegex(RuntimeError, "worker did not complete"),
            ):
                runner.supervise(["synthetic"], output)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("worker did not complete", status["error"])
            self.assertIn("TimeoutExpired", status["cleanup_error"])

    def test_success_supervision_verifies_every_byte(self):
        code = """import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1])
names=("protocol.json","summary.json","conditions.csv","residuals.csv","alignment.csv","verification.json")
for n in names: (p/n).write_text("{}" if n.endswith("json") else "x\\n")
(p/"output-manifest.json").write_text(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names}))
(p/"status.json").write_text(json.dumps({"status":"outputs_ready"}))"""
        self.assertEqual(self.child(code)["status"], "completed")
        with self.assertRaises(ValueError):
            self.child(
                code.replace("hashlib.sha256((p/n).read_bytes()).hexdigest()", '"bad"')
            )

    def test_worker_ownership_and_failure_state(self):
        parent = os.getppid()
        env = {
            "CUDA_VISIBLE_DEVICES": "",
            "S3_PARENT": str(parent),
            "S3_TOKEN": "fixture",
            **{n: "1" for n in runner.THREAD_ENV},
        }
        from scripts.s2_source_replay import ReplayMismatch

        for error, state in [
            (None, "outputs_ready"),
            (MemoryError("budget"), "failed"),
            (ReplayMismatch("digest"), "blocked_replay_unverified"),
        ]:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve()
                out = root / "runs/synthetic"
                out.mkdir(parents=True)
                runner.update_status(out, "planned")
                runner.write_json(
                    out / "supervision.json", {"pid": parent, "token": "fixture"}
                )
                libc = Mock()
                libc.prctl.return_value = 0
                with (
                    patch.object(sys, "platform", "linux"),
                    patch.dict(os.environ, env),
                    patch.object(runner.ctypes, "CDLL", return_value=libc),
                    patch.object(runner.resource, "setrlimit"),
                    patch.object(runner, "analyze", side_effect=error),
                ):
                    if error:
                        with self.assertRaises(type(error)):
                            runner.worker(root, out)
                    else:
                        runner.worker(root, out)
                self.assertEqual(
                    json.loads((out / "status.json").read_text())["status"], state
                )
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4"}),
            self.assertRaises(RuntimeError),
        ):
            runner.worker(Path("/bad"), Path("/bad"))

    def test_synthetic_analysis_pipeline_and_changed_source_stop(self):
        from scripts import s2_source_replay as core

        config = dict(
            seed=42,
            steps=4000,
            meta_batch=1,
            query_size=1,
            support_sizes=[1, 2],
            accumulation=1,
        )
        names = ["accidents", "airline", "chembl", "legalacts", "telstra"]
        caches = {
            db: dict(
                true=np.tile([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0], 60),
                pg=np.tile([0.0, 2.0, 5.0, 100.0, 500.0, 20000.0, 100000.0], 60),
                n_tables=np.tile(np.arange(1, 8), 60),
                token_counts=np.column_stack(
                    [
                        np.ones(420, dtype=int),
                        np.full(420, 2),
                        np.tile(np.arange(1, 8), 60),
                        np.zeros(420, dtype=int),
                    ]
                ),
                raw_rows=840,
                indices=np.arange(0, 840, 2),
                query_hash=[str(i) for i in range(420)],
            )
            for db in names
        }
        contract = dict(
            train=names[:2],
            validation=names[2:],
            config=config,
            sources={db: {} for db in names},
        )
        for changed in (False, True):
            with (
                tempfile.TemporaryDirectory() as d,
                patch.object(np, "__version__", "1.26.1"),
                patch.object(runner, "load_contract", return_value=contract),
                patch.object(
                    core,
                    "snapshot",
                    side_effect=[
                        {"synthetic": "only"},
                        {"changed": "input"} if changed else {"synthetic": "only"},
                    ],
                ),
                patch.object(
                    runner,
                    "read_metadata",
                    side_effect=lambda root, db, frozen: caches[db],
                ),
                patch.object(runner, "check_denominators"),
                patch.object(core, "certify") as certify,
            ):
                out = Path(d).resolve()
                if changed:
                    with self.assertRaisesRegex(ValueError, "changed"):
                        runner.analyze(runner.ROOT, out)
                    self.assertFalse((out / "summary.json").exists())
                else:
                    report = runner.analyze(runner.ROOT, out)
                    self.assertFalse(report["training"])
                    self.assertEqual(report["episodes"], 4000)
                    self.assertEqual(
                        set(runner.verify_outputs(out)), set(runner.OUTPUTS)
                    )
                    self.assertEqual(len(report["alignment"]), 640)
                    self.assertNotIn(b"\r", (out / "conditions.csv").read_bytes())
                certify.assert_called_once()
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(np, "__version__", "wrong"),
        ):
            with self.assertRaisesRegex(ValueError, "NumPy"):
                runner.analyze(runner.ROOT, Path(d))

    def test_positive_old_denominators(self):
        from scripts.s2_exposure_stats import CARDINALITY, TABLES

        names = ["chembl", "legalacts", "telstra"]
        caches = {
            db: dict(
                true=np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]),
                n_tables=np.arange(1, 8),
                raw_rows=7,
            )
            for db in names
        }
        source_pool = {db: dict(raw_rows=7, unique_queries=7) for db in names}
        per_database = {}
        for db in names:
            c = caches[db]
            strata = {}
            card = np.arange(7)
            table = np.minimum(np.maximum(c["n_tables"] - 3, 0), 3)
            for i, label in enumerate(CARDINALITY):
                strata[f"all:true={label}"] = {"n": 1}
            for j, label in enumerate(TABLES):
                strata[f"all:tables={'6plus' if label == '6+' else label}"] = {
                    "n": int((table == j).sum())
                }
            for i, cl in enumerate(CARDINALITY):
                for j, tl in enumerate(TABLES):
                    strata[f"all:true={cl}|tables={'6plus' if tl == '6+' else tl}"] = {
                        "n": int(((card == i) & (table == j)).sum())
                    }
            per_database[db] = {"strata": strata}
        with patch.object(
            runner.json,
            "loads",
            side_effect=[{"source_pool": source_pool}, {"per_database": per_database}],
        ):
            runner.check_denominators(runner.ROOT, caches)

    def test_launcher_and_alarm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
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
        with (
            patch.object(sys, "platform", "linux"),
            patch.object(sys, "argv", ["runner", "--execute"]),
            patch.object(runner.signal, "signal"),
            patch.object(runner.signal, "setitimer") as alarm,
            patch.object(runner, "run_main", side_effect=TimeoutError("fixture")),
        ):
            with self.assertRaises(TimeoutError):
                runner.main()
            self.assertEqual(
                alarm.call_args_list[-1].args, (runner.signal.ITIMER_REAL, 0)
            )

    def test_import_and_dry_run_no_torch_or_source_reads(self):
        code = 'import sys; from scripts import run_s3_diagnosis; assert not any(n=="torch" or n.startswith("spatial_meta_ce") for n in sys.modules)'
        p = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True)
        self.assertEqual(p.returncode, 0, p.stderr.decode())
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            with (
                patch.object(runner, "ROOT", root),
                patch.object(sys, "platform", "darwin"),
                patch.object(
                    sys, "argv", ["runner", "--output", str(root / "runs/dry")]
                ),
                patch.object(
                    runner, "load_contract", side_effect=AssertionError("dry reads")
                ),
            ):
                runner.run_main(runner.time.monotonic() + 10)
            self.assertFalse((root / "runs").exists())
        with self.assertRaises(TimeoutError):
            runner.deadline_expired(0, None)


if __name__ == "__main__":
    unittest.main()
