#!/usr/bin/env python3
"""Assemble the Meta65 scope launch binding from certified preflight evidence.

The binding freezes code/config/protocol/input SHA evidence, both preflight raw
manifests, the target GPU UUID and the frozen resource budget. It grants no
permission: launch_ready is engineering readiness only, and formal execution
still requires a separate one-shot approval consumed by the supervisor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import meta65_scope_contract as contract  # noqa: E402
from scripts import meta65_scope_launch as launch  # noqa: E402
from scripts.meta65_joint_supervisor import artifact_paths  # noqa: E402

Json = dict[str, Any]


def raw_manifest(output: Path) -> dict[str, str]:
    """SHA-bind every raw artifact plus the supervisor sidecar and worker log."""
    sidecar, log = artifact_paths(output)
    files = sorted([*(p for p in output.rglob("*") if p.is_file()), sidecar, log], key=str)
    return {str(path): contract.sha256(path) for path in files}


def build_binding(*, root: Path, config_path: Path, cpu_output: Path, gpu_output: Path,
                  gpu_uuid: str) -> Json:
    root, config_path = contract.checked_path(root), contract.checked_path(config_path)
    preflights = {}
    for mode, output in (("cpu_preflight", cpu_output), ("gpu_preflight", gpu_output)):
        output = contract.checked_path(output)
        sidecar, _ = artifact_paths(output)
        preflights[mode] = {"output": str(output), "supervisor_file": str(sidecar),
                            "raw_sha256": raw_manifest(output)}
    return {
        "schema": 1, "experiment": launch.EXPERIMENT,
        "launch_ready": True, "execution_authorized": False, "formal_executed": False,
        "code_sha256": {name: contract.sha256(root / name) for name in contract.CODE_FILES},
        "config_sha256": contract.sha256(config_path),
        "protocol_sha256": contract.PROTOCOL_SHA,
        "data_config_sha256": contract.sha256(root / "configs/meta65_server_data.json"),
        "input_sha256": contract.bind_inputs(root=root, config_path=config_path),
        "resource_budget": dict(contract.FROZEN["limits"]),
        "gpu_uuid": gpu_uuid,
        "preflights": preflights,
    }


def write_binding(binding_path: Path, binding: Json) -> None:
    contract.write_json(contract.checked_path(binding_path), binding)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Isolated code root")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-preflight", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--out", type=Path, required=True, help="Binding path inside root")
    parser.add_argument("--verify", action="store_true",
                        help="Run verify_launch_binding on the written binding")
    args = parser.parse_args(argv)
    binding = build_binding(root=args.root, config_path=args.config,
                            cpu_output=args.cpu_preflight, gpu_output=args.gpu_preflight,
                            gpu_uuid=args.gpu_uuid)
    write_binding(args.out, binding)
    result = {"binding": str(args.out), "binding_sha256": contract.sha256(args.out),
              "input_sha256_files": len(binding["input_sha256"]),
              "preflight_raw_files": {mode: len(entry["raw_sha256"])
                                      for mode, entry in binding["preflights"].items()}}
    if args.verify:
        launch.verify_launch_binding(args.out, root=args.root, config_path=args.config)
        result["verify_launch_binding"] = "passed"
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
