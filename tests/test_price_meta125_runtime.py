import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class Official26RuntimeTests(unittest.TestCase):
    def runtime(self):
        return importlib.import_module("spatial_meta_ce.price_meta125_runtime")

    def test_official_weight_and_no_validation_actions(self):
        runtime = self.runtime()
        self.assertEqual(
            runtime.INITIAL_SHA,
            "4f889c4df269d1bca5a50dadd83de7c801a607fe84228b6c21fb3371ae053f92",
        )
        self.assertEqual(runtime.ALPHA, 1e-6)
        for forbidden in ("calibrate-p", "calibrate-e", "prepare-source", "report"):
            self.assertNotIn(forbidden, runtime.ACTIONS)

    def test_authorization_rejects_old_binding_and_forbidden_action(self):
        runtime = self.runtime()
        binding = {"root": "/isolated", "binding_sha256": "a" * 64}
        auth = {
            "schema": 1,
            "protocol": 125,
            "authorized": True,
            **binding,
            "actions": ["train:M:42"],
            "granted_by": "user",
            "granted_at": "2026-11-17",
            "budget": {
                "max_wall_seconds": 21600,
                "max_rss_bytes": 24 * 1024**3,
                "max_vram_bytes": 12 * 1024**3,
                "max_output_bytes": 6 * 1024**3,
                "poll_seconds": 1,
            },
        }
        runtime.validate_authorization(auth, binding, "train:M:42")
        with self.assertRaises(ValueError):
            runtime.validate_authorization(auth, binding, "report")
        with self.assertRaises(ValueError):
            runtime.validate_authorization(
                {**auth, "protocol": 123}, binding, "train:M:42"
            )
        with self.assertRaises(ValueError):
            runtime.validate_authorization(auth, binding, "train:E:42")
        excessive = {**auth, "budget": {**auth["budget"], "max_wall_seconds": 21601}}
        with self.assertRaises(ValueError):
            runtime.validate_authorization(excessive, binding, "train:M:42")

    def test_probe_projection_counts_six_training_runs_without_source_cells(self):
        runtime = self.runtime()
        timing = {
            "e_update_seconds": 0.734,
            "m_update_seconds": 7.59,
            "setup_io_seconds": 400,
            "verification_seconds": 60,
            "checkpoint_pair_bytes": 258814300,
            "peak_rss_bytes": 2559414272,
            "peak_vram_bytes": 938945536,
        }
        value = runtime.project_budget(timing)
        self.assertAlmostEqual(value["projected_seconds"], 1.3 * (460 + 6000 * 8.324))
        self.assertTrue(value["feasible"])
        self.assertEqual(value["projected_output_bytes"], 3 * 258814300 + 512 * 1024**2)
        timing["m_update_seconds"] = 9
        self.assertFalse(runtime.project_budget(timing)["checks"]["m_run_time"])
        timing["m_update_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            runtime.project_budget(timing)

    def test_completed_requires_new_protocol_binding_and_exact_sidecar(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/probe"
            output.mkdir(parents=True)
            binding = {"root": str(root), "binding_sha256": "b" * 64}
            completion = {
                "status": "completed",
                "protocol": 125,
                "action": "probe",
                "binding_sha256": "b" * 64,
                "target_databases_read": [],
            }
            (output / "completion.json").write_text(json.dumps(completion))
            sidecar = output.with_name("probe.supervisor.json")
            sidecar.write_text(
                json.dumps({"status": "completed", "completion": completion})
            )
            runtime.completed(binding, "probe")
            completion["protocol"] = 123
            (output / "completion.json").write_text(json.dumps(completion))
            with self.assertRaises(ValueError):
                runtime.completed(binding, "probe")

    def test_cost_gate_rejects_zero_external_measurement(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "probe"
            output.mkdir()
            (output / "timing.json").write_text("{}")
            output.with_name("probe.supervisor.json").write_text(
                json.dumps(
                    {
                        "vram_accounting": "device_total",
                        "gpu_uuid": "GPU-test",
                        "peak_vram_bytes": 0,
                        "vram_baseline_bytes": 1048576,
                    }
                )
            )
            with mock.patch.object(
                runtime,
                "completed",
                return_value=(
                    output,
                    {"timing_sha256": runtime.file_sha(output / "timing.json")},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "VRAM"):
                    runtime.cost_gate({})

    def test_supervisor_completion_protocol_is_explicit_and_defaults_to_123(self):
        from spatial_meta_ce import price_meta123_supervisor as supervisor

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "completion.json").write_text(
                json.dumps(
                    {"status": "completed", "protocol": 125, "binding_sha256": "a" * 64}
                )
            )
            self.assertIsNone(supervisor._read_completion(path)[0])
            self.assertIsNotNone(supervisor._read_completion(path, 125)[0])

    def test_bind_and_reload_reject_changed_code_and_short_budget_initialization(self):
        runtime = self.runtime()
        config = json.loads(Path("configs/data_price125_official26.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "code").mkdir()
            (root / "inputs").mkdir()
            checkpoint = root / "official.pt"
            checkpoint.write_bytes(b"synthetic-official-checkpoint")
            source_config = root / "config.json"
            source_config.write_text(json.dumps(config))
            closure = {runtime.PROTOCOL_FILE: runtime.PROTOCOL_SHA}
            original_sha = runtime.file_sha

            def sha(path):
                return (
                    runtime.INITIAL_SHA
                    if Path(path) == checkpoint
                    else original_sha(path)
                )

            with (
                mock.patch.object(runtime, "code_closure", return_value=closure),
                mock.patch.object(runtime, "file_sha", side_effect=sha),
            ):
                binding = runtime.bind(root, source_config, checkpoint, root / "code")
                self.assertEqual(
                    runtime.load_binding(root / "inputs/binding.json"), binding
                )
                with mock.patch.object(
                    runtime,
                    "code_closure",
                    return_value={**closure, "changed.py": "a" * 64},
                ):
                    with self.assertRaises(ValueError):
                        runtime.load_binding(root / "inputs/binding.json")
            with mock.patch.object(runtime, "code_closure", return_value=closure):
                with self.assertRaisesRegex(ValueError, "checkpoint"):
                    runtime.bind(root, source_config, checkpoint, root / "code")

    def test_cumulative_time_and_output_reject_exhaustion(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "train-E-42.supervisor.json").write_text(
                json.dumps({"elapsed_seconds": 100})
            )
            (outputs / "probe.supervisor.json").write_text(
                json.dumps({"elapsed_seconds": 400})
            )
            seconds, size = runtime.usage({"root": str(root)})
            self.assertEqual(seconds, 100)
            self.assertGreater(size, 0)
            (outputs / "train-M-42.supervisor.json").write_text(
                json.dumps({"elapsed_seconds": 86300})
            )
            with self.assertRaisesRegex(ValueError, "exhausted"):
                runtime.usage({"root": str(root)})

    def test_gpu_run_is_bound_to_protocol_and_cumulative_remaining_budget(self):
        runtime = self.runtime()
        from spatial_meta_ce import price_meta123_supervisor as supervisor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "inputs").mkdir()
            (root / "outputs").mkdir()
            binding = {
                "root": str(root),
                "code_root": str(root / "code"),
                "binding_sha256": "a" * 64,
            }
            budget = {
                "max_wall_seconds": 21600,
                "max_rss_bytes": 24 * 1024**3,
                "max_vram_bytes": 12 * 1024**3,
                "max_output_bytes": 6 * 1024**3,
                "poll_seconds": 1,
            }
            auth = {
                "schema": 1,
                "protocol": 125,
                "authorized": True,
                "root": str(root),
                "binding_sha256": "a" * 64,
                "actions": ["train:M:42"],
                "granted_by": "user",
                "granted_at": "today",
                "budget": budget,
                "cuda_device": 2,
                "cuda_uuid": "GPU-cf1ca2b2-761c-66cb-7b36-221d730920c7",
            }
            auth_path = root / "inputs/auth.json"
            auth_path.write_text(json.dumps(auth))
            with (
                mock.patch.object(runtime, "load_binding", return_value=binding),
                mock.patch.object(runtime, "usage", return_value=(80000, 1000)),
                mock.patch.object(
                    supervisor, "supervise", return_value={"status": "completed"}
                ) as run,
            ):
                runtime.run_stage(
                    root / "inputs/binding.json", auth_path, "train:M:42", "cuda"
                )
                self.assertEqual(run.call_args.kwargs["completion_protocol"], 125)
                self.assertEqual(
                    run.call_args.kwargs["budget"]["max_wall_seconds"], 6400
                )
                self.assertEqual(
                    run.call_args.kwargs["budget"]["max_output_bytes"],
                    6 * 1024**3 - 1000,
                )
                self.assertEqual(run.call_args.kwargs["gpu_uuid"], auth["cuda_uuid"])

    def test_seal_rejects_missing_or_incorrectly_completed_training_run(self):
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = {"root": str(root), "binding_sha256": "a" * 64}
            with self.assertRaises(FileNotFoundError):
                runtime.seal(binding, root)
            with (
                mock.patch.object(
                    runtime, "completed", return_value=(root, {"updates": 1999})
                ),
                mock.patch.object(
                    runtime,
                    "training_plan",
                    return_value={"databases": {}, "plan_sha256": "b" * 64},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "matrix"):
                    runtime.seal(binding, root)


if __name__ == "__main__":
    unittest.main()
