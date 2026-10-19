"""Source-only screening runner for the P1 levers of doc 93.

Arms are frozen in `configs/tail_aligned_screen.json`. Training uses the 23
source training databases; the endpoint is the zero-shot prediction on the
three source validation databases only. The held-out targets in
`data_config['splits']['test']` are never read: this module refuses to run if
one appears among the arms' databases, and `evaluate` is never imported.

Two modes:
  * parent (default): supervise each arm as a bounded child process, then
    apply the frozen 10% / 1.10x gate and write `summary.json`.
  * child (`--child`): train one arm, verify budgets, write its endpoint.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts import tail_aligned_screen_evidence as evidence
from scripts.gate_usage import gate_report
from scripts.mapping_ablation_endpoint import endpoint_rows
from scripts.precision_screen_evidence import summarize_endpoint
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.cli import data_settings, read_json
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.manifests import file_hash
from spatial_meta_ce.train import train

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tail_aligned_screen.json"
SCRIPT = Path(__file__).resolve()
DEFAULT_TIMEOUT = 43200


def load_plan(config_path: Path = CONFIG, data_config: Path | None = None) -> dict:
    plan = read_json(config_path)
    if plan.get("schema") != 1:
        raise ValueError("Unsupported tail-aligned screen schema")
    data_config = data_config or ROOT / plan["data_config"]
    splits = read_json(data_config)["splits"]
    arms = plan["arms"]
    if plan["baseline"] not in arms:
        raise ValueError("Baseline arm missing from the frozen matrix")
    target_hits = set(splits["test"]) & set(splits["train"] + splits["validation"])
    if target_hits:
        raise ValueError(
            f"Target databases leaked into the source split: {sorted(target_hits)}"
        )
    for name, arm in arms.items():
        if "splits" in arm or "test" in arm:
            raise ValueError(f"Arm {name} overrides the data split")
    return {
        **plan,
        "arms": arms,
        "splits": splits,
        "data_config": str(data_config),
        "config_sha256": file_hash(config_path),
        "data_config_sha256": file_hash(data_config),
    }


def arm_config(plan: dict, arm: str) -> dict:
    if arm not in plan["arms"]:
        raise ValueError(f"Unknown arm {arm!r}")
    spec = plan["arms"][arm]
    config = {
        **plan["training"],
        "support_sizes": list(plan["training"]["support_sizes"]),
        "model": {**plan["model"], **spec.get("model", {})},
        "loss": spec["loss"],
        "data_config": plan["data_config"],
        "purpose": "research",
    }
    if "pinball_tau" in spec:
        config["pinball_tau"] = spec["pinball_tau"]
    return config


def validate_output(path: Path, plan: dict) -> Path:
    resolved = path.resolve()
    if path.exists():
        raise FileExistsError("Output already exists")
    if resolved.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Outputs must not live inside the read-only cache")
    return resolved


def run_arm(plan: dict, arm: str, output: Path, device: str) -> dict:
    data = data_settings({"data_config": plan["data_config"]}, plan["data_config"])
    if data["splits"] != plan["splits"]:
        raise ValueError("Source split changed since the plan was frozen")
    config = arm_config(plan, arm)
    summary = train(config, data, output, device=device)
    model, state = load_checkpoint(output / "last.pt", device)
    extra = state["extra"]
    if (
        extra["counts"]["optimizer_steps"] != config["steps"]
        or state["next_step"] != config["steps"]
    ):
        raise ValueError("Frozen optimizer budget incomplete")
    expected_signature = {
        k: v for k, v in config.items() if k not in ("steps", "data_config")
    }
    if state["metadata"]["training_signature"] != expected_signature:
        raise ValueError("Checkpoint does not match the arm configuration")
    exposed = set(state["metadata"]["train_databases"]) | set(
        state["metadata"]["validation_databases"]
    )
    if exposed & set(plan["splits"]["test"]):
        raise ValueError("A target database entered training")

    pool = StorePool(data["cache_root"], max_open=len(evidence.VALIDATION))
    rows_by_db = {
        db: endpoint_rows(model, pool.get("pretrain", db), device)
        for db in evidence.VALIDATION
    }
    endpoint = summarize_endpoint(rows_by_db, expected_coverage=None)
    record = {
        "arm": arm,
        "config": config,
        "output": str(output),
        "checkpoint_sha256": file_hash(output / "last.pt"),
        "manifest_hashes": state["metadata"]["manifest_hashes"],
        "counts": extra["counts"],
        "sample_plan_digest": extra["sample_plan_digest"],
        "summary": summary,
        "endpoint": endpoint,
        "gate_usage": gate_report(model, pool, device, evidence.VALIDATION)
        if getattr(model, "gate_mode", None) is not None
        else None,
    }
    evidence.write_json(output / "result.json", record)
    return record


def run_parent(plan: dict, output: Path, device: str, timeout: int) -> dict:
    evidence.status(output, "running", mode="parent", device=device)
    records = {}
    for arm in plan["arms"]:
        command = evidence.child_command(
            SCRIPT, arm, CONFIG, ROOT / plan["data_config"], output / arm, device
        )
        evidence.supervise(command, output / arm, timeout=timeout)
        records[arm] = read_json(output / arm / "result.json")
        evidence.write_json(output / "runs.json", {"runs": records})
    decision = evidence.choose_candidates(
        {name: record["endpoint"] for name, record in records.items()},
        plan["baseline"],
        **plan["guard"],
    )
    report = {
        "purpose": "research-source-only-single-seed",
        "device": device,
        "config_sha256": plan["config_sha256"],
        "data_config_sha256": plan["data_config_sha256"],
        "target_databases_read": [],
        "runs": records,
        **decision,
    }
    evidence.write_json(output / "summary.json", report)
    evidence.status(output, "completed", mode="parent", decision=decision["decision"])
    print(json.dumps(decision), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 tail-aligned source screen")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--data-config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = load_plan(
        Path(args.config), Path(args.data_config) if args.data_config else None
    )
    output = validate_output(Path(args.output), plan)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "arm": args.arm,
                    "config": arm_config(plan, args.arm),
                },
                indent=2,
            )
        )
        print("Re-run with --execute to train.", flush=True)
        return
    if not args.device.startswith("cuda"):
        raise SystemExit("Formal screening requires --device cuda")
    if args.child:
        run_arm(plan, args.arm, output, args.device)
        return
    run_parent(plan, output, args.device, args.timeout)


if __name__ == "__main__":
    main()
