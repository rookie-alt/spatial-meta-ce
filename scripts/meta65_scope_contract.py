"""Immutable source-only protocol and complete input binding for Meta65 scope."""
from pathlib import Path
from typing import Any

from scripts import meta65_joint_runner as joint
from scripts import run_meta65_screen as screen

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("R_head3", "T_all3")
DATABASES = joint.DATABASES
OPERATORS = ("none", "head3")
TRAIN = screen.TRAIN
TARGETS = screen.TARGETS
COUNTS = joint.COUNTS
RETAINED = joint.RETAINED
FROZEN_ROOT = joint.FROZEN_ROOT
PLAN_SHA = joint.PLAN_SHA
PROTOCOL = "实验文档/65-Meta65元训练范围单因素改进方案.md"
PROTOCOL_SHA = "ce47146f43c643c087f28ccd6df1aa4c89c8c6ac88c360bb774ff2c4e02e0131"
OLD_B_METADATA = RETAINED / "B_head3_head3/metadata.json"
OLD_B_METADATA_SHA = "73570adc80b163c0944a6db5d775c4e4c6d34317a164642144c62f6e80364756"
INITIALIZATION_SHA = "41e594b03ef19c56c3f236237da850929f2ec5c919702b44402dd9b2864d284a"
CUDA_WORKSPACE_CONFIG = ":4096:8"
CACHE_FILES = ("manifest.json", "cards.npy", "offsets.npy", *(f"group{i}.npy" for i in range(4)))
PROVENANCE = (*joint.PROVENANCE, "实验文档/results/meta65-source-screen-20261009/B-metadata.json")
CODE_FILES = tuple(dict.fromkeys((
    "scripts/meta65_scope_contract.py", "scripts/meta65_scope_runner.py",
    "scripts/meta65_scope_verify.py", "scripts/meta65_scope_decision.py",
    "scripts/meta65_scope_supervisor.py", "scripts/meta65_scope_launch.py",
    "scripts/meta65_scope_analysis.py",
    "scripts/run_meta65_screen.py", "scripts/meta65_joint_runner.py",
    "scripts/precision_screen_evidence.py", "scripts/run_s8_preflight.py",
    "scripts/mapping_smoke.py", "scripts/mapping_ablation_endpoint.py",
    "spatial_meta_ce/train.py", "spatial_meta_ce/meta.py", "spatial_meta_ce/data.py",
    "spatial_meta_ce/model.py", "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/manifests.py", "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/metrics.py", "spatial_meta_ce/tail_risk.py",
    "scripts/arch_select.py", "scripts/s2_source_replay.py",
    "spatial_meta_ce/baselines.py", "spatial_meta_ce/cli.py", "spatial_meta_ce/evaluate.py",
    "spatial_meta_ce/features.py", "spatial_meta_ce/s8_contract.py",
    "scripts/meta65_joint_supervisor.py", "scripts/meta65_joint_verify.py",
    "spatial_meta_ce/__init__.py",
)))
FROZEN = {
    "schema": 1, "experiment": "meta65-training-scope-20261011",
    "protocol_file": PROTOCOL, "protocol_sha256": PROTOCOL_SHA,
    "source_binding_sha256": screen.SOURCE_BINDING,
    "training": {**screen.FROZEN["training"]},
    "model": {**screen.FROZEN["model"]},
    "arms": {"R_head3": {"train_scope": "head", "train_steps": 3},
             "T_all3": {"train_scope": "all", "train_steps": 3}},
    "evaluation": {"support_seed": 2718, "support_runs": 5, "k20": 20,
                   "batch_size": 512, "operators": list(OPERATORS), "plan_sha256": PLAN_SHA},
    "limits": {"total_seconds": 14400, "rss_bytes": 12 * 1024**3,
               "vram_bytes": 8 * 1024**3, "output_bytes": 1024**3},
    "cpu_preflight": {"training_steps": 2, "micro_queries": 32, "total_seconds": 900},
    "gpu_preflight": {"training_steps": 2, "micro_queries": 32, "total_seconds": 900},
    "screen": {"min_improvement": 0.1, "max_regression": 0.1, "min_low4_count": 30},
}


def digest(value: Any) -> str:
    return joint.digest(value)


def sha256(path: Path | str) -> str:
    return joint.sha256(checked_path(path))


def read_json(path: Path | str) -> Any:
    return joint.read_json(checked_path(path))


def write_json(path: Path | str, value: Any) -> None:
    joint.write_json(checked_path(path), value)


def checked_path(path: Path | str) -> Path:
    # Permit only macOS's system temporary-directory aliases, not user symlinks.
    return joint.checked_path(path)


def validate_config(config: dict) -> dict:
    if digest(config) != digest(FROZEN):
        raise ValueError("Frozen scope configuration changed or unknown fields supplied")
    return config


def load_config(path: Path | str, root: Path = ROOT) -> dict:
    config = validate_config(read_json(path))
    if sha256(checked_path(root) / PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("Frozen scope protocol SHA changed")
    return config


def arm_config(config: dict, arm: str, mode: str = "formal") -> dict:
    validate_config(config)
    if arm not in ARMS or mode not in ("formal", "cpu_preflight", "gpu_preflight"):
        raise ValueError("Unknown source arm or mode")
    spec = config["arms"][arm]
    steps = config["training"]["steps"] if mode == "formal" else config[mode]["training_steps"]
    return {**config["training"], "steps": steps,
            "model": {**config["model"], "width": 64}, "loss": "smooth_l1",
            "adapt_scope": spec["train_scope"], "inner_steps": spec["train_steps"],
            "purpose": config["experiment"]}


def matrix(mode: str = "formal") -> list[tuple[str, str, str, int]]:
    if mode not in ("formal", "cpu_preflight", "gpu_preflight"):
        raise ValueError("Unknown execution mode")
    return [(arm, operator, db, run) for arm in ARMS for operator in OPERATORS
            for db in DATABASES for run in (range(5) if mode == "formal" else range(1))]


def safe_output(path: Path | str, inputs: list[Path | str], root: Path = ROOT) -> Path:
    output = checked_path(path)
    protected = [*inputs, FROZEN_ROOT, "/gly/weilin/PRICE-master",
                 checked_path(root) / "spatial_meta_ce", checked_path(root) / "scripts",
                 checked_path(root) / "configs", checked_path(root) / "实验文档"]
    if output.exists() or any(output.is_relative_to(checked_path(p)) or
                              checked_path(p).is_relative_to(output) for p in protected):
        raise ValueError("Output exists or overlaps read-only inputs/source/cache/retained")
    if any(p in (*TARGETS, "PRICE-master", "finetune", "test") for p in output.parts):
        raise ValueError("Target/PRICE output forbidden")
    return output


def recheck(binding: dict[str, str]) -> None:
    for name, expected in binding.items():
        if sha256(name) != expected:
            raise ValueError(f"Input SHA changed: {name}")


def check_cell_peaks(record: dict) -> dict:
    """Catch a transient memory peak that the one-second sampler might miss."""
    for field, budget in (("peak_rss_bytes", FROZEN["limits"]["rss_bytes"]),
                          ("peak_vram_bytes", FROZEN["limits"]["vram_bytes"])):
        value = record.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid {field} high-water mark")
        if value > budget:
            raise ValueError(f"{field} exceeded frozen limit")
    return record


def historical_paths(root: Path = ROOT) -> dict[str, Path]:
    return {db: checked_path(root / "实验文档/results" /
                            ("p1-mapping-sidecar-preflight-20261003" if db in DATABASES
                             else "p1-mapping-sidecar-train23-20261004") / db / "manifest.json")
            for db in (*TRAIN, *DATABASES)}


def discover_anchors(diagnosis: dict) -> dict[tuple[str, int], dict]:
    anchors = {}
    for name, expected in diagnosis["input_sha256"].items():
        path = Path(name)
        if (path.suffix != ".npz" or "-k20-run" not in path.name or
                path.parent.parent.name != "B_head3_head3"):
            continue
        db, shot, run = path.stem.split("-")
        if db not in DATABASES or shot != "k20" or run not in {f"run{i}" for i in range(5)}:
            raise ValueError("Unexpected B K20 anchor database/run")
        path = checked_path(path)
        key = (db, int(run[3:]))
        if (key in anchors or path != RETAINED / "B_head3_head3/predictions" / path.name
                or not isinstance(expected, str) or len(expected) != 64
                or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Duplicate, escaped or invalid B anchor")
        anchors[key] = {"database": db, "run": key[1], "path": str(path), "sha256": expected}
    if set(anchors) != {(db, run) for db in DATABASES for run in range(5)}:
        raise ValueError("Exactly 15 B K20 anchors required")
    return anchors


def input_binding(root: Path = ROOT, *, config_path: Path | None = None,
                  data_config_path: Path | None = None) -> tuple[dict, dict, dict, dict, dict, dict]:
    from scripts.precision_screen_evidence import historical_sources, source_evidence

    root = checked_path(root)
    config_path = checked_path(config_path or root / "configs/meta65_scope.json")
    data_config_path = checked_path(data_config_path or root / "configs/meta65_server_data.json")
    load_config(config_path, root)
    data = screen.validate_data(read_json(data_config_path), root)
    paths = [*(root / p for p in (*CODE_FILES, *PROVENANCE, PROTOCOL)),
             config_path, data_config_path, *historical_paths(root).values()]
    binding = {str(checked_path(p)): sha256(p) for p in paths}
    historical = historical_sources(root, list((*TRAIN, *DATABASES)))
    before = source_evidence(data, historical)
    if set(before) != set((*TRAIN, *DATABASES)):
        raise ValueError("Exactly 26 source caches required")
    for db, info in before.items():
        if set(info["file_sha256"]) != set(CACHE_FILES):
            raise ValueError("Exactly seven cache files per source required")
        for name, expected in info["file_sha256"].items():
            binding[str(checked_path(Path(data["cache_root"]) / "pretrain" / db / name))] = expected
    plan_path = checked_path(RETAINED / "endpoint_plan.json")
    completion = read_json(root / PROVENANCE[2])
    if sha256(plan_path) != PLAN_SHA or completion["endpoint_plan_sha256"] != PLAN_SHA:
        raise ValueError("Full retained support plan SHA changed")
    binding[str(plan_path)] = PLAN_SHA
    metadata_path = checked_path(OLD_B_METADATA)
    if (sha256(metadata_path) != OLD_B_METADATA_SHA or
            sha256(root / PROVENANCE[3]) != OLD_B_METADATA_SHA):
        raise ValueError("Old B metadata SHA changed")
    binding[str(metadata_path)] = OLD_B_METADATA_SHA
    anchors = discover_anchors(read_json(root / PROVENANCE[1]))
    binding = {**binding, **{a["path"]: a["sha256"] for a in anchors.values()}}
    recheck(binding)
    plan = read_json(plan_path)
    if set(plan) != set(DATABASES) or any(len(plan[db]) != 5 for db in DATABASES):
        raise ValueError("Full source plan missing")
    return binding, plan, anchors, before, historical, data


def verify_pairing(records: list[dict], *, formal: bool = True) -> None:
    if not records or [r["arm"] for r in records] != list(ARMS[:len(records)]):
        raise ValueError("Invalid partial scope arm matrix")
    for field in ("initialization_digest", "parameters", "sample_plan_digest", "sample_plan_episodes",
                  "support_sample_count", "validation_plan_digest", "endpoint_plan_digest", "optimizer_steps", "counts"):
        if len({digest(r[field]) for r in records}) != 1:
            raise ValueError(f"Cross-arm pairing mismatch: {field}")
    for record in records:
        if record["initialization_digest"] != INITIALIZATION_SHA:
            raise ValueError("Seed42 initialization changed")
        if record["metadata"]["manifest_hashes"] != records[0]["metadata"]["manifest_hashes"]:
            raise ValueError("Cross-arm source manifest mismatch")
        steps = 4000 if formal else 2
        if (record["optimizer_steps"] != steps or record["sample_plan_episodes"] != steps * 4
                or record["counts"]["query_presentations"] != steps * 4 * 64
                or record["counts"]["backward_calls"] != steps * 4 * 4
                or record["counts"]["support_presentations"] != record["support_sample_count"] * 3):
            raise ValueError("Training scope budget mismatch")


def verify_reference_training(record: dict, old: dict, *, formal: bool) -> None:
    fields = ("initialization_digest", "parameters", "sample_plan_digest", "sample_plan_episodes",
              "support_sample_count", "validation_plan_digest", "optimizer_steps", "counts")
    if formal and any(record[field] != old[field] for field in fields):
        raise ValueError("R training does not reproduce old B sampling/count chain")
    if record["metadata"]["manifest_hashes"] != old["metadata"]["manifest_hashes"]:
        raise ValueError("R source manifest differs from old B")


def bind_inputs(root: Path = ROOT, config_path: Path | None = None) -> dict:
    """Launch integration rehashes the same fixed server-data input contract."""
    return input_binding(root, config_path=config_path)[0]
