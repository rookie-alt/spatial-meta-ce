import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import meta65_holdout_contract as contract
from scripts import run_meta65_holdout as runner


class HoldoutWorkerFilesTests(unittest.TestCase):
    def test_worker_writes_phase_verification_and_terminal_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            data = {
                "data_root": "/tmp/data",
                "cache_root": "/tmp/cache",
                "splits": {
                    "train": list(contract.TRAIN),
                    "validation": list(contract.VALIDATION),
                    "holdout": list(contract.HOLDOUT),
                    "test": list(contract.TARGETS),
                },
            }
            config_path = Path(directory) / "config.json"
            data_path = Path(directory) / "data.json"
            plan_path = Path(directory) / "plan.json"
            for path in (config_path, data_path, plan_path):
                path.write_text("{}", encoding="utf-8")
            with (
                patch.object(runner, "require_supervisor"),
                patch.object(runner, "environment", return_value={
                    "mode": "cpu_preflight", "training": True, "target_read": False,
                }),
                patch.object(runner.contract, "load_config", return_value=contract.FROZEN),
                patch.object(runner.contract, "read_json", return_value=data),
                patch.object(runner.contract, "validate_data", return_value=data),
                patch.object(runner.contract, "safe_output", return_value=output),
                patch.object(runner, "manifest_hashes", return_value={"dummy": "a" * 64}),
                patch.object(runner.contract, "dataset_signature", return_value="b" * 64),
                patch.object(runner, "load_plan", return_value={}),
                patch.object(runner.contract, "input_binding", return_value={"dummy": "c" * 64}),
                patch.object(runner.contract, "sha256", return_value="d" * 64),
                patch.object(runner.contract, "verify_bound_inputs"),
                patch.object(runner.contract, "plan_summary", return_value={}),
                patch.object(runner, "train_holdout", return_value=(object(), {})),
                patch.object(runner, "evaluate_cells", return_value=[{} for _ in range(10)]),
                patch.object(runner, "schedule_verification", return_value={
                    "status": "preflight_verified",
                }),
            ):
                result = runner.worker(
                    output, "cpu_preflight", config_path, data_path, plan_path,
                )
            self.assertEqual(result["status"], "preflight_verified")
            self.assertEqual(
                json.loads((output / "phase.json").read_text())["phase"], "certification"
            )
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "worker_complete"
            )
            self.assertEqual(
                json.loads((output / "verification.json").read_text())["status"],
                "preflight_verified",
            )


if __name__ == "__main__":
    unittest.main()
