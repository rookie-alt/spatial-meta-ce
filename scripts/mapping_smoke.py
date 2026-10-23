#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import StorePool
from spatial_meta_ce.manifests import digest, file_hash
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.train import seed_all, train

ARMS = ("sum", "mapping", "shuffled_mapping")
STEPS = 20
META_BATCH = 4
TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}


def smoke_config(base: dict, sidecar_root: str, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown smoke arm")
    config = {
        **base,
        "model": {**base["model"], "arch": "sum" if arm == "sum" else "sum_map"},
        "method": "erm",
        "seed": 42,
        "steps": STEPS,
        "meta_batch": META_BATCH,
        "accumulation": 1,
        "outer_lr": 0.0003,
        "inner_lr": 0.001,
        "inner_steps": 5,
        "adapt_scope": "all",
        "support_sizes": [5],
        "query_size": 64,
        "loss": "smooth_l1",
        "validate_every": 10,
        "validation_runs": 1,
        "validation_seed": 1729,
        "patience": 41,
        "min_delta": 0.0001,
        "purpose": "smoke-only; not research evidence",
    }
    config = {k: v for k, v in config.items() if k not in ("mapping", "data_config")}
    return (
        config
        if arm == "sum"
        else {**config, "mapping": {"root": sidecar_root, "mode": arm}}
    )


def verify_arms(results: dict) -> None:
    if set(results) != set(ARMS):
        raise ValueError("Smoke must contain exactly three arms")
    reference = results["sum"]
    for result in results.values():
        if result["optimizer_steps"] != STEPS:
            raise ValueError("Smoke optimizer budget changed")
        for key in ("sample_plan_digest", "sample_plan_episodes", "counts"):
            if result[key] != reference[key]:
                raise ValueError(f"Three-arm pairing failed: {key}")
    for key in ("parameters", "initialization_hash"):
        if results["mapping"][key] != results["shuffled_mapping"][key]:
            raise ValueError(f"Mapping arms differ: {key}")


def available_gpu(index: str) -> str:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        text=True,
    )
    for row in rows.splitlines():
        number, uuid, memory, utilization = [part.strip() for part in row.split(",")]
        if number == index:
            if uuid in processes or int(memory) > 64 or int(utilization) != 0:
                raise RuntimeError("Selected GPU is occupied; refusing to start")
            return uuid
    raise ValueError("Selected GPU does not exist")


def state_hash(model: SetEstimator) -> str:
    value = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def checkpoint_checks(
    config: dict, data: dict, output: Path, initial_hash: str
) -> dict:
    model, state = load_checkpoint(output / "last.pt", "cuda:0")
    if state_hash(model) == initial_hash:
        raise ValueError("Optimizer did not change model weights")
    if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
        raise FloatingPointError("Nonfinite checkpoint weights")
    mapping = config.get("mapping")
    pool = StorePool(
        data["cache_root"],
        mapping_root=mapping["root"] if mapping else None,
        mapping_mode=mapping["mode"] if mapping else "mapping",
    )
    store = pool.get("pretrain", "legalacts")
    indices = store.unique_indices()[:8]
    with torch.no_grad():
        prediction = model.eval()(store.batch(indices, "cuda:0")).cpu()
        reloaded, _ = load_checkpoint(output / "last.pt", "cuda:0")
        repeated = reloaded.eval()(store.batch(indices, "cuda:0")).cpu()
        cpu, _ = load_checkpoint(output / "last.pt", "cpu")
        cpu_prediction = cpu.eval()(store.batch(indices, "cpu"))
    torch.testing.assert_close(prediction, repeated, rtol=0, atol=0)
    torch.testing.assert_close(prediction, cpu_prediction, **TOLERANCE)
    extra = state["extra"]
    return {
        "device": str(next(model.parameters()).device),
        "optimizer_steps": extra["counts"]["optimizer_steps"],
        "sample_plan_digest": extra["sample_plan_digest"],
        "sample_plan_episodes": extra["sample_plan_episodes"],
        "validation_plan_hash": file_hash(output / "validation_manifest.json"),
        "mapping_provenance": state["metadata"].get("mapping_provenance", {}),
        "checkpoint_roundtrip_exact": True,
        "optimizer_changed_weights": True,
        "cpu_gpu_max_abs_difference": float((prediction - cpu_prediction).abs().max()),
        "cpu_gpu_tolerance": TOLERANCE,
        "checkpoint_sha256": file_hash(output / "last.pt"),
    }


def run_arm(config: dict, data: dict, output: Path) -> dict:
    seed_all(config["seed"])
    initial = SetEstimator(**config["model"])
    initial_hash = state_hash(initial)
    parameters = sum(p.numel() for p in initial.parameters())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    summary = train(config, data, output, device="cuda:0")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        **summary,
        **checkpoint_checks(config, data, output, initial_hash),
        "parameters": parameters,
        "initialization_hash": initial_hash,
        "gpu_seconds_including_data_load": elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def run(cache_root: str, sidecar_root: str, output: str, gpu: str) -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly the requested GPU")
    gpu_uuid = available_gpu(gpu)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Smoke requires one visible CUDA GPU; CPU fallback forbidden"
        )
    destination = Path(output).resolve()
    for protected in (
        Path(cache_root).resolve(),
        Path(sidecar_root).resolve(),
        Path("/gly/weilin/PRICE-master").resolve(),
    ):
        if destination == protected or protected in destination.parents:
            raise ValueError("Smoke outputs cannot modify source/cache/sidecar files")
    destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = {
        "cache_root": cache_root,
        "splits": {
            "train": ["chembl"],
            "validation": ["legalacts", "telstra"],
            "test": ["imdb", "stats", "genome", "ergastf1"],
        },
    }
    base = {"model": {"width": 64, "use_pg": True, "layers": 2, "heads": 4}}
    results = {}
    for arm in ARMS:
        config = smoke_config(base, sidecar_root, arm)
        results[arm] = run_arm(config, data, destination / arm)
        torch.cuda.empty_cache()
    verify_arms(results)
    if len({r["validation_plan_hash"] for r in results.values()}) != 1:
        raise ValueError("Smoke validation plans differ")
    project = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(p.relative_to(project)): file_hash(p)
        for folder in ("spatial_meta_ce", "scripts")
        for p in sorted((project / folder).glob("*.py"))
    }
    report = {
        "status": "passed",
        "purpose": "smoke-only; not research evidence",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_index": gpu,
        "gpu_uuid": gpu_uuid,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "source_data": data,
        "target_databases_opened": [],
        "code_hashes": code_hashes,
        "code_digest": digest(code_hashes),
        "results": results,
    }
    (destination / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    run(args.cache_root, args.sidecar_root, args.output, args.gpu)


if __name__ == "__main__":
    main()
