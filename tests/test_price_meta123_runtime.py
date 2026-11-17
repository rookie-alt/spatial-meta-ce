import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class RuntimeContractTests(unittest.TestCase):
    def runtime(self):
        return importlib.import_module("spatial_meta_ce.price_meta123_runtime")

    def test_authorization_missing_or_unapproved_is_rejected(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = {"root": str(root), "binding_sha256": "a" * 64}
            with self.assertRaisesRegex(ValueError, "authoriz"):
                runtime.validate_authorization({}, binding, "train:E:42")
            auth = {
                "schema": 1,
                "protocol": 123,
                "authorized": False,
                "binding_sha256": "a" * 64,
                "root": str(root),
                "actions": ["train:E:42"],
            }
            with self.assertRaisesRegex(ValueError, "authoriz"):
                runtime.validate_authorization(auth, binding, "train:E:42")

    def test_output_path_cannot_escape_or_reuse(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "outputs").mkdir()
            with self.assertRaises(ValueError):
                runtime.output_path(root, "../other")
            with self.assertRaises(ValueError):
                runtime.output_path(root, "/tmp/other")
            target = runtime.output_path(root, "train-E-42")
            self.assertEqual(target, root / "outputs" / "train-E-42")
            target.mkdir()
            with self.assertRaises(FileExistsError):
                runtime.output_path(root, "train-E-42")
            (root / "outputs" / "linked").symlink_to(root / "elsewhere")
            with self.assertRaises(ValueError):
                runtime.output_path(root, "linked")

    def test_atomic_json_never_overwrites_a_record(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            runtime.write_json(path, {"answer": 1})
            with self.assertRaises(FileExistsError):
                runtime.write_json(path, {"answer": 2})
            self.assertEqual(json.loads(path.read_text()), {"answer": 1})
            with self.assertRaises(ValueError):
                runtime.write_json(path.parent / "nan.json", {"answer": float("nan")})

    def test_worker_rejects_forged_or_stale_supervision(self):
        runtime = self.runtime()
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "supervis"):
                runtime.require_supervision()
        with mock.patch.dict(
            os.environ,
            PRICE_META123_SUPERVISED="1",
            PRICE_META123_SUPERVISOR_PID=str(os.getpid()),
        ):
            with self.assertRaisesRegex(ValueError, "supervis"):
                runtime.require_supervision()

    def test_budget_must_cover_the_entire_planned_matrix(self):
        runtime = self.runtime()
        budget = runtime.project_budget(
            {
                "e_update_seconds": 0.5,
                "m_update_seconds": 1.5,
                "calibration_cell_seconds": 1.0,
                "report_cell_seconds": 3.0,
                "setup_io_seconds": 20.0,
                "verification_seconds": 30.0,
                "bytes_per_prediction": 200.0,
                "peak_rss_bytes": 1000000,
                "peak_vram_bytes": 1000000,
            }
        )
        expected = 1.3 * (20 + 6000 * 0.5 + 6000 * 1.5 + 192 + 276 * 3 + 30)
        self.assertAlmostEqual(budget["projected_seconds"], expected)
        self.assertTrue(budget["feasible"])
        self.assertEqual(budget["max_prediction_rows"], 13558768)

    def test_authorization_cannot_raise_protocol_limits(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binding = {"root": str(root), "binding_sha256": "a" * 64}
            auth = {
                "schema": 1,
                "protocol": 123,
                "authorized": True,
                "binding_sha256": "a" * 64,
                "root": str(root),
                "actions": ["train:E:42"],
                "granted_by": "user",
                "granted_at": "2026-11-16T12:00:00+08:00",
                "budget": {
                    "max_wall_seconds": 7201,
                    "max_rss_bytes": 24 * 1024**3,
                    "max_vram_bytes": 12 * 1024**3,
                    "max_output_bytes": 6 * 1024**3,
                    "poll_seconds": 1,
                },
            }
            with self.assertRaisesRegex(ValueError, "limit|budget"):
                runtime.validate_authorization(auth, binding, "train:E:42")

    def test_revised_budget_covers_d2_with_six_hour_meta_run_cap(self):
        runtime = self.runtime()
        timing = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "reports/price-meta123-d2-20261117/timing.json"
            ).read_text()
        )
        budget = runtime.project_budget(timing)
        self.assertTrue(budget["checks"]["total_time"])
        self.assertTrue(budget["checks"]["m_run_time"])
        self.assertEqual(budget["limits"]["total_seconds"], 24 * 3600)
        self.assertEqual(budget["limits"]["m_run_seconds"], 6 * 3600)

    def test_cost_gate_refuses_zero_external_vram_even_when_worker_nonzero(self):
        runtime = self.runtime()
        timing = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "reports/price-meta123-d2-20261117/timing.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            probe = root / "probe"
            probe.mkdir()
            runtime.write_json(probe / "timing.json", timing)
            record = {"timing_sha256": runtime.file_sha(probe / "timing.json")}
            with (
                mock.patch.object(runtime, "_completed", return_value=(probe, record)),
                mock.patch.object(
                    runtime,
                    "read_json",
                    side_effect=lambda p: timing
                    if Path(p).name == "timing.json"
                    else {
                        "status": "completed",
                        "vram_accounting": "device_total",
                        "peak_vram_bytes": 0,
                    },
                ),
            ):
                with self.assertRaisesRegex(ValueError, "VRAM|显存|external"):
                    runtime._cost_gate({}, "prepare-source")

    def test_csv_unit_calls_independent_verifier_and_keeps_population(self):
        runtime = self.runtime()
        import torch
        import numpy as np

        class Store:
            path = Path("chembl")
            rows = [{"id": "q1", "query_hash": "a" * 64}]
            cards = np.asarray([[2.0, 1.0]])

            def batch(self, indices, device):
                return {"true": torch.tensor([2.0], device=device)}

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

        model = Tiny()
        z = torch.log1p(torch.tensor(2.0)) + 1
        entry = {"report": {"indices": [0]}}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "spatial_meta_ce.price_meta123_model.predict_native",
                return_value=z.reshape(1),
            ):
                unit = runtime._unit(
                    directory,
                    model,
                    Store(),
                    entry,
                    arm="P",
                    seed=0,
                    k=0,
                    run=0,
                    alpha=0.0,
                    region="report",
                )
            self.assertEqual(unit["population"], "report")
            self.assertEqual(unit["n"], 1)
            self.assertAlmostEqual(unit["metrics"]["q95"], 1.0, places=5)

    def test_synthetic_six_run_training_seals_and_then_rejects_tampering(self):
        """Exercise the actual orchestration loops with a tiny deterministic model."""
        runtime = self.runtime()
        import torch
        from spatial_meta_ce import price_meta123_model as native
        from spatial_meta_ce import price_meta123_plans as plans

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.price = torch.nn.Linear(1, 1)

            def forward(self, batch):
                return self.price(batch["x"]).flatten()

        class Store:
            def batch(self, indices, device):
                x = torch.ones(len(indices), 1, device=device)
                return {
                    "x": x,
                    "true": torch.ones(
                        len(indices), dtype=torch.float64, device=device
                    ),
                }

        class Stores:
            opened_databases = ["accidents"]

            def __init__(self, *_args):
                pass

            def get(self, db):
                if db != "accidents":
                    raise AssertionError("opened a validation or target store")
                return Store()

        plan = {
            "databases": {"accidents": {}},
            "plan_sha256": "b" * 64,
            "episodes": [
                {
                    "db": "accidents",
                    "support": {"indices": list(range(20))},
                    "query": {"indices": list(range(20, 84))},
                }
            ]
            * 8,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "outputs").mkdir()
            data = root / "data.json"
            data.write_text("{}")
            binding = {
                "root": str(root),
                "binding_sha256": "a" * 64,
                "data_config": str(data),
                "initial_checkpoint": "unused",
            }
            with (
                mock.patch.object(runtime, "TRAIN_STEPS", 2),
                mock.patch.object(runtime, "_calibration", return_value=1e-6),
                mock.patch.object(runtime, "_plan", return_value=plan),
                mock.patch.object(plans, "SourceStores", Stores),
                mock.patch.object(plans, "verify_plan_stores", return_value=True),
                mock.patch.object(
                    native, "build_model", side_effect=lambda *_a, **_k: Tiny()
                ),
            ):
                for arm in ("E", "M"):
                    for seed in (42, 43, 44):
                        output = root / "outputs" / f"train-{arm}-{seed}"
                        output.mkdir()
                        result = runtime._train(binding, output, arm, seed, "cpu")
                        self.assertEqual(result["updates"], 2)
                        self.assertEqual(result["validation_monitor_reads"], [])
                        runtime.write_json(
                            output.with_name(output.name + ".supervisor.json"),
                            {"status": "completed", "completion": result},
                        )
                sealed = root / "outputs/seal-training"
                sealed.mkdir()
                result = runtime._seal_training(binding, sealed)
                self.assertEqual(len(result["runs"]), 6)
                damaged = root / "outputs/train-E-42/last.pt"
                with damaged.open("ab") as stream:
                    stream.write(b"changed")
                second = root / "outputs/seal-second"
                second.mkdir()
                with self.assertRaisesRegex(ValueError, "matrix"):
                    runtime._seal_training(binding, second)

    def test_completed_rejects_a_changed_completion_copy(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "outputs/check"
            output.mkdir(parents=True)
            record = {
                "status": "completed",
                "binding_sha256": "a" * 64,
                "action": "check",
            }
            runtime.write_json(output / "completion.json", record)
            runtime.write_json(
                output.with_name("check.supervisor.json"),
                {"status": "completed", "completion": {**record, "extra": 1}},
            )
            with self.assertRaisesRegex(ValueError, "sealed"):
                runtime._completed(
                    {"root": str(root), "binding_sha256": "a" * 64}, "check"
                )


if __name__ == "__main__":
    unittest.main()
