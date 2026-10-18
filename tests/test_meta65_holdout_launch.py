import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import build_meta65_holdout_binding as builder
from scripts import meta65_holdout_contract as contract
from scripts import meta65_holdout_launch as launch
from scripts import meta65_holdout_verify as verifier
from tests.test_meta65_holdout_verify import build_output, build_plan

ROOT = Path(__file__).resolve().parents[1]


class HoldoutLaunchTests(unittest.TestCase):
    def test_round_trip_build_and_verify_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in contract.CODE_FILES:
                destination = root / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, destination)
            protocol = root / contract.PROTOCOL
            protocol.parent.mkdir(parents=True)
            shutil.copy2(ROOT / contract.PROTOCOL, protocol)
            config = root / "configs/meta65_source_holdout.json"
            config.parent.mkdir(parents=True)
            config.write_text(json.dumps(contract.FROZEN), encoding="utf-8")
            data_config = root / "configs/meta65_source_holdout_data.json"
            cache = root / "cache"
            data = {
                "data_root": str(root / "data"),
                "cache_root": str(cache),
                "splits": {
                    "train": list(contract.TRAIN),
                    "validation": list(contract.VALIDATION),
                    "holdout": list(contract.HOLDOUT),
                    "test": list(contract.TARGETS),
                },
            }
            data_config.write_text(json.dumps(data), encoding="utf-8")
            for database in contract.ALL_SOURCES:
                folder = cache / "pretrain" / database
                folder.mkdir(parents=True)
                for name in contract.CACHE_FILES:
                    (folder / name).write_bytes(f"{database}:{name}".encode())
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(build_plan()), encoding="utf-8")
            work = root / "work"
            work.mkdir()
            source_preflight = build_output(work, formal=False)
            preflights = {}
            for mode in ("cpu_preflight", "gpu_preflight"):
                output = root / "runs" / mode
                shutil.copytree(source_preflight, output)
                for filename in ("status.json", "environment.json", "input-binding.json"):
                    path = output / filename
                    value = json.loads(path.read_text())
                    value["mode"] = mode
                    path.write_text(json.dumps(value), encoding="utf-8")
                verification = verifier.verify(output, formal=False)
                (output / "verification.json").write_text(
                    json.dumps(verification), encoding="utf-8"
                )
                (output / "status.json").write_text(json.dumps({
                    **json.loads((output / "status.json").read_text()),
                    "status": "preflight_completed",
                }), encoding="utf-8")
                sidecar = output.with_name(output.name + ".supervisor.json")
                sidecar.write_text(json.dumps({
                    "status": "preflight_completed", "mode": mode,
                    "returncode": 0, "failure": None,
                }), encoding="utf-8")
                output.with_name(output.name + ".supervisor.log").write_text("", encoding="utf-8")
                preflights[mode] = output
            binding = builder.build_binding(
                root=root, config_path=config, data_config_path=data_config,
                plan_path=plan_path, cpu_output=preflights["cpu_preflight"],
                gpu_output=preflights["gpu_preflight"],
                gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
            )
            binding_path = root / "binding.json"
            builder.write_binding(binding_path, binding)
            checked = launch.verify_launch_binding(binding_path, root=root, config_path=config)
            self.assertFalse(checked["execution_authorized"])
            self.assertFalse(checked["formal_executed"])
            self.assertEqual(set(checked["preflights"]), {"cpu_preflight", "gpu_preflight"})


if __name__ == "__main__":
    unittest.main()
