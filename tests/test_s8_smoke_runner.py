"""Synthetic S8 G1 runner tests; no source training or remote operations."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.run_s8_smoke import (
    SourceStore,
    checkpoint_reload,
    checkpoint_write,
    execute,
    metrics_rows,
    runtime_threadpools,
    smoke_plan,
    tensor_hash,
    timeout_failure,
    train_f0,
)
from spatial_meta_ce.data import GROUP_DIMS, batch_samples
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.s8_contract import HELD_OUT, TRAIN, VALIDATION


class SmokeRunnerTests(unittest.TestCase):
    def bindings(self):
        return {
            "schema": 1,
            "mode": "smoke",
            "purpose": "s8-engineering-smoke-not-research",
            "protocol_sha256": "a" * 64,
            "source_binding_sha256": "b" * 64,
            "panel_identity_digest": "c" * 64,
            "train_databases": list(TRAIN),
            "holdout_databases": list(HELD_OUT),
            "validation_databases": list(VALIDATION),
            "model_config": {
                "width": 64,
                "use_pg": True,
                "arch": "sum",
                "layers": 2,
                "heads": 4,
                "pg_output_scale": 1.0,
            },
            "seed": 42,
            "train_steps": 2,
            "meta_batch": 4,
            "query_size": 64,
            "support_sizes": [5, 10, 20, 50, 100],
            "resourcefixedfields": {
                "batch_size": 128,
                "threads": 4,
                "probe_fit_per_database": 64,
                "probe_adam_steps": 2,
                "timeout_seconds": 1800,
                "address_space_bytes": 12884901888,
            },
        }

    def test_wallclock_signal_is_an_exception_not_process_default_termination(self):
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            timeout_failure(None, None)

    def test_actual_sigalrm_leaves_failed_status(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            script = f"""
import json,signal,time
from pathlib import Path
from unittest.mock import patch
from scripts.run_s8_smoke import execute,timeout_failure
root=Path({str(root)!r})
output=root/'runs/s8-g1-smoke-timeout-test'
signal.signal(signal.SIGALRM,timeout_failure)
signal.alarm(1)
try:
    with patch('scripts.run_s8_smoke.DATA_ROOT',root), patch('scripts.run_s8_smoke.np.__version__','1.26.1'), patch('scripts.run_s8_smoke.torch.__version__','2.1.0+cu121'), patch('scripts.run_s8_smoke.sys.version_info',(3,11,4)), patch('scripts.run_s8_smoke.source.preflight',side_effect=lambda *a:time.sleep(10)):
        execute(root,output)
except TimeoutError:
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='failed' and 'deadline' in status['error']
    print('failed_state_written')
finally:
    signal.alarm(0)
"""
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("failed_state_written", result.stdout)

    def test_frozen_linux_backend_counters_are_read_without_installing_dependency(self):
        import builtins
        from unittest.mock import MagicMock

        original_import = builtins.__import__

        def no_threadpoolctl(name, *args, **kwargs):
            if name == "threadpoolctl":
                raise ImportError("unavailable in frozen venv")
            return original_import(name, *args, **kwargs)

        paths = (
            "/gly/weilin/PRICE-master/venv/lib/numpy.libs/libopenblas64_test.so",
            "/gly/weilin/PRICE-master/venv/lib/torch/lib/libgomp-test.so",
        )
        maps = "\n".join("map " + name for name in paths)
        counter = MagicMock(return_value=4)
        library = MagicMock()
        library.openblas_get_num_threads64_ = counter
        library.omp_get_max_threads = counter
        with (
            patch("builtins.__import__", side_effect=no_threadpoolctl),
            patch("scripts.run_s8_smoke.Path.read_text", return_value=maps),
            patch("scripts.run_s8_smoke.ctypes.CDLL", return_value=library),
        ):
            pools = runtime_threadpools()
        self.assertEqual([row["num_threads"] for row in pools], [4, 4])
        self.assertTrue(
            all(row["measurement"] == "loaded-library-C-counter" for row in pools)
        )
        with (
            patch("builtins.__import__", side_effect=no_threadpoolctl),
            patch(
                "scripts.run_s8_smoke.Path.read_text",
                return_value="map /tmp/libmkl_rt.so",
            ),
            self.assertRaises(ValueError),
        ):
            runtime_threadpools()

    def test_default_plan_does_not_train_or_claim_formal_readiness(self):
        plan = smoke_plan()
        self.assertEqual(plan["status"], "planned")
        self.assertFalse(plan["formal_ready"])
        self.assertEqual(plan["F0_updates"], 2)
        self.assertEqual(plan["probe_fit_unique"], 1216)
        self.assertEqual(plan["probe_adam_updates_each"], 2)

    def test_holdout_guard_rejects_before_any_semantic_file_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            store = SourceStore(root, {}, root, self.bindings())
            with patch.object(store, "_load") as load:
                for db in (*HELD_OUT, "imdb", "../imdb", "unknown"):
                    with self.assertRaises(ValueError):
                        store.get(db)
                load.assert_not_called()

    def test_sparse_metrics_validate_even_without_reporting_percentiles(self):
        row = {
            "database": HELD_OUT[0],
            "cohort": "H",
            "query_hash": "q",
            "true": 0,
            "pg": 1,
            "n_tables": 1,
            "predictions": {
                arm: 0 for arm in ("A_h_adam", "B_h_ls", "C_x_adam", "D_x_ls", "F0")
            },
        }
        records = metrics_rows([row])
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "below_min_cell_n")
        self.assertNotIn("arms", records[0])
        for invalid in (float("nan"), float("inf"), 1000.0, -1e200):
            row["predictions"]["D_x_ls"] = invalid
            with self.assertRaises(ValueError):
                metrics_rows([row])

    def test_fresh_checkpoint_reload_binds_signature_step_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = SetEstimator(arch="sum").eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            # Actual optimizer steps, even though the test loss is synthetic.
            for _ in range(2):
                optimizer.zero_grad()
                sum(p.sum() for p in model.parameters()).backward()
                optimizer.step()
            path = root / "last.pt"
            digest = checkpoint_write(
                path,
                model,
                optimizer,
                self.bindings(),
                {"optimizer_steps": 2, "backward_calls": 8},
                "d" * 64,
            )
            restored, _ = checkpoint_reload(path, digest, self.bindings())
            self.assertEqual(tensor_hash(restored), tensor_hash(model))
            with self.assertRaises(ValueError):
                checkpoint_reload(path, "0" * 64, self.bindings())
            with self.assertRaises(ValueError):
                checkpoint_reload(
                    path, digest, {**self.bindings(), "panel_identity_digest": "e" * 64}
                )
            with self.assertRaises(FileExistsError):
                checkpoint_write(
                    path,
                    model,
                    optimizer,
                    self.bindings(),
                    {"optimizer_steps": 2, "backward_calls": 8},
                    "d" * 64,
                )

    def test_wrong_project_root_is_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "runs" / "s8-g1-smoke-test"
            with self.assertRaisesRegex(ValueError, "root|project"):
                execute(root, output)
            self.assertFalse(output.exists())

    def test_checkpoint_rejects_missing_optimizer_and_non_fp32_weight_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = SetEstimator(arch="sum").eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            for _ in range(2):
                optimizer.zero_grad()
                sum(p.sum() for p in model.parameters()).backward()
                optimizer.step()
            path = root / "last.pt"
            checkpoint_write(
                path,
                model,
                optimizer,
                self.bindings(),
                {"optimizer_steps": 2, "backward_calls": 8},
                "d" * 64,
            )
            original = path.read_bytes()
            for mutation in ("optimizer", "weights"):
                payload = torch.load(path, weights_only=False)
                if mutation == "optimizer":
                    payload["optimizer"]["state"] = {}
                else:
                    payload["model"] = {
                        name: value.double() for name, value in payload["model"].items()
                    }
                bad = root / f"{mutation}.pt"
                torch.save(payload, bad)
                with self.assertRaises(ValueError):
                    checkpoint_reload(
                        bad,
                        hashlib.sha256(bad.read_bytes()).hexdigest(),
                        self.bindings(),
                    )
            self.assertEqual(path.read_bytes(), original)

    def test_real_training_call_uses_only_panel_positions(self):
        class SyntheticStore:
            def __init__(self):
                self.calls = []

            def batch(self, db, positions):
                self.calls.append((db, list(positions)))
                rng = np.random.default_rng(123)
                return batch_samples(
                    [
                        {
                            "id": f"{db}/{i}",
                            "groups": [
                                rng.random((1, d), dtype=np.float32) for d in GROUP_DIMS
                            ],
                            "true": 4.0,
                            "pg": 3.0,
                        }
                        for i in positions
                    ]
                )

        bindings = self.bindings()
        panels = {db: list(range(1024)) for db in TRAIN}
        caches = {
            db: {
                "indices": np.arange(1300),
                "query_hash": [f"{db}/q{i}" for i in range(1300)],
            }
            for db in TRAIN
        }
        store = SyntheticStore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model, optimizer, counts, digest, _ = train_f0(
                store, caches, panels, bindings, root
            )
            self.assertEqual(counts["optimizer_steps"], 2)
            self.assertEqual(counts["backward_calls"], 8)
            self.assertEqual(len(store.calls), 8)
            self.assertTrue(
                all(
                    db in TRAIN and set(indices) <= set(panels[db])
                    for db, indices in store.calls
                )
            )
            self.assertEqual(len(digest), 64)
            episodes = [
                json.loads(line)
                for line in (root / "episodes.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(episodes), 8)
            self.assertEqual(model.arch, "sum")
            self.assertEqual(optimizer.param_groups[0]["lr"], 3e-4)


if __name__ == "__main__":
    unittest.main()
