"""Build and verify a launch binding from the two holdout preflights."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import meta65_holdout_contract as contract  # noqa: E402
from scripts import meta65_holdout_launch as launch  # noqa: E402
from scripts.meta65_joint_supervisor import artifact_paths  # noqa: E402


def raw_manifest(output: Path) -> dict[str, str]:
    sidecar, log = artifact_paths(output)
    files = [*output.rglob("*"), sidecar, log]
    result = {}
    for path in files:
        if path.is_file():
            key = str(path.relative_to(output)) if path.is_relative_to(output) else path.name
            result[key] = contract.sha256(path)
    return result


def build_binding(*, root: Path, config_path: Path, data_config_path: Path,
                  plan_path: Path, cpu_output: Path, gpu_output: Path,
                  gpu_uuid: str) -> dict:
    root = contract.checked_path(root)
    config_path = contract.checked_path(config_path)
    data_config_path = contract.checked_path(data_config_path)
    plan_path = contract.checked_path(plan_path)
    outputs = {}
    for mode, output in (("cpu_preflight", cpu_output), ("gpu_preflight", gpu_output)):
        output = contract.checked_path(output)
        sidecar, _ = artifact_paths(output)
        outputs[mode] = {
            "output": str(output),
            "supervisor_file": str(sidecar),
            "raw_sha256": raw_manifest(output),
        }
    return {
        "schema": 1,
        "experiment": contract.EXPERIMENT,
        "launch_ready": True,
        "execution_authorized": False,
        "formal_executed": False,
        "code_sha256": {name: contract.sha256(root / name) for name in contract.CODE_FILES},
        "config_sha256": contract.sha256(config_path),
        "data_config_sha256": contract.sha256(data_config_path),
        "protocol_sha256": contract.PROTOCOL_SHA,
        "plan_file": str(plan_path),
        "plan_sha256": contract.sha256(plan_path),
        "input_sha256": contract.input_binding(root, config_path, data_config_path, plan_path),
        "resource_budget": dict(contract.FROZEN["limits"]),
        "gpu_uuid": gpu_uuid,
        "preflights": outputs,
    }


def write_binding(path: Path, binding: dict) -> None:
    path = contract.checked_path(path)
    if path.exists():
        raise FileExistsError("Binding output must be fresh")
    contract.write_json(path, binding)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    binding = build_binding(
        root=args.root, config_path=args.config, data_config_path=args.data_config,
        plan_path=args.plan, cpu_output=args.cpu_preflight, gpu_output=args.gpu_preflight,
        gpu_uuid=args.gpu_uuid,
    )
    write_binding(args.out, binding)
    result = {"binding": str(args.out), "binding_sha256": contract.sha256(args.out),
              "preflights": {mode: len(entry["raw_sha256"])
                             for mode, entry in binding["preflights"].items()}}
    if args.verify:
        launch.verify_launch_binding(args.out, root=args.root, config_path=args.config)
        result["verify_launch_binding"] = "passed"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
