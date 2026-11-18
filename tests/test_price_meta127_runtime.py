import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class OfficialRuntimeTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta127_runtime")

    def test_recipe_freezes_official_effective_batch_and_epoch_schedule(self):
        m = self.module()
        self.assertEqual(m.RECIPE["effective_batch"], 15000)
        self.assertEqual(m.RECIPE["microbatch"], 512)
        self.assertEqual(m.RECIPE["epochs"], 100)
        self.assertEqual(m.RECIPE["lr"], 1e-5)
        self.assertEqual(m.RECIPE["scheduler_step"], "epoch")
        self.assertEqual(len(m.finetune_stages()), 21)
        self.assertTrue(all("imdb" not in x for x in m.finetune_stages()))

    def test_projection_refuses_an_unaffordable_complete_matrix(self):
        m = self.module()
        affordable = m.project_cost([10, 11, 12, 4], setup_seconds=30)
        self.assertTrue(affordable["feasible"])
        self.assertGreater(affordable["projected_total_seconds"], 12 * 400 * 21)
        self.assertFalse(m.project_cost([1000] * 4, setup_seconds=30)["feasible"])
        for bad in ([1, 2, 3], [1, 2, 3, 0], [1, 2, 3, float("nan")]):
            with self.assertRaises(ValueError):
                m.project_cost(bad, setup_seconds=30)

    def test_authorization_is_bound_to_stage_and_official_recipe(self):
        m = self.module()
        binding = {"root": "/tmp/price127", "binding_sha256": "a" * 64}
        auth = {
            "schema": 1,
            "protocol": 127,
            "authorized": True,
            "root": binding["root"],
            "binding_sha256": binding["binding_sha256"],
            "actions": ["zero", "probe", *m.finetune_stages(), "report"],
            "allowed_targets": list(m.DATABASES),
            "recipe": m.RECIPE,
            "selection_forbidden": True,
            "granted_by": "user: official settings",
            "granted_at": "2026-11-18T12:00:00Z",
        }
        m.validate_authorization(auth, binding, "zero")
        for key, value in (("recipe", {}), ("binding_sha256", "b" * 64)):
            changed = copy.deepcopy(auth)
            changed[key] = value
            with self.assertRaises(ValueError):
                m.validate_authorization(changed, binding, "zero")
        with self.assertRaises(ValueError):
            m.validate_authorization(auth, binding, "finetune-P-0-imdb")

    def test_completed_stage_refuses_mutated_prediction_bytes(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs/zero"
            output.mkdir(parents=True)
            (output / "pred.csv.gz").write_bytes(b"initial")
            binding = {"root": tmp, "binding_sha256": "a" * 64}
            value = m.finish(binding, "zero", output, cells=32)
            (output.with_name("zero.supervisor.json")).write_text(
                json.dumps({"status": "completed", "completion": value})
            )
            m.completed(binding, "zero")
            (output / "pred.csv.gz").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                m.completed(binding, "zero")

    def test_gpu_cost_gate_cannot_use_zero_or_infeasible_probe(self):
        m = self.module()
        binding = {"root": "/tmp/price127", "binding_sha256": "a" * 64}
        for feasible, peak in ((True, 0), (False, 2**30)):
            with (
                mock.patch.object(m, "load_binding", return_value=binding),
                mock.patch.object(m, "read_json", return_value={}),
                mock.patch.object(m, "validate_authorization"),
                mock.patch.object(
                    m,
                    "completed",
                    return_value=(
                        {"cost": {"feasible": feasible}},
                        {
                            "vram_accounting": "device_total",
                            "peak_vram_bytes": peak,
                            "vram_baseline_bytes": 2**20,
                        },
                    ),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "gate failed"):
                    m.run("unused", "unused", m.finetune_stages()[0])

    def test_new_completion_protocol_is_read_without_changing_legacy_default(self):
        from spatial_meta_ce import price_meta123_supervisor as supervisor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            value = {"status": "completed", "protocol": 127, "binding_sha256": "a" * 64}
            (path / "completion.json").write_text(json.dumps(value))
            completion, _, _ = supervisor._read_completion(path, 127)
            self.assertEqual(completion, value)
            completion, failure, _ = supervisor._read_completion(path)
            self.assertIsNone(completion)
            self.assertEqual(failure, "incomplete_completion")

    def test_fixed_sequence_stops_after_first_failed_stage_without_retry(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("inputs", "outputs", "evidence"):
                (root / name).mkdir()
            auth_path = root / "inputs/authorization.json"
            auth_path.write_text("{}")
            binding = {"root": tmp, "binding_sha256": "a" * 64}
            with (
                mock.patch.object(m, "load_binding", return_value=binding),
                mock.patch.object(m, "validate_authorization"),
                mock.patch.object(
                    m, "completed", return_value=({"cost": {"feasible": True}}, {})
                ),
                mock.patch.object(m, "experiment_usage", return_value=(0, 0)),
                mock.patch.object(m, "run", return_value={"status": "failed"}) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "stage failed"):
                    m.sequence(root / "inputs/binding.json", auth_path)
                self.assertEqual(run.call_count, 1)
            result = json.loads((root / "evidence/sequence-failed.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["completed"], [])


if __name__ == "__main__":
    unittest.main()
