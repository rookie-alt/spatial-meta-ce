#!/usr/bin/env python3
"""Thin supervised two-arm source training worker; default is a stdlib dry run."""
import argparse
import importlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import meta65_joint_runner as joint  # noqa: E402
from scripts import meta65_scope_contract as contract  # noqa: E402
from scripts import run_meta65_screen as screen  # noqa: E402

# Retain the existing canonical FP64 identity/full-100-exclusion validators.
certify_plan = joint.certify_plan
encode_identity = joint.encode_identity
state_digest = joint.state_digest


def environment(mode: str) -> dict:
    if mode not in ("formal", "cpu_preflight", "gpu_preflight"):
        raise ValueError("Unknown execution mode")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != contract.CUDA_WORKSPACE_CONFIG:
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before torch/CUDA use")
    return joint.environment(mode)


def operator_params(model: Any, support: Any, operator: str) -> tuple[Any, dict]:
    if operator not in contract.OPERATORS:
        raise ValueError("Only none/head3 deployment operators permitted")
    return joint.operator_params(model, support, operator)


def cell_metrics(log_pred: Any, identity: dict) -> dict:
    import numpy as np

    metrics = joint.cell_metrics(log_pred, identity)
    mask = (identity["true"] < 100) & (identity["n_tables"] >= 4)
    count = int(mask.sum())
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(identity["true"], 1))))
    low4 = {"count": count, "status": "empty" if not count else "below_min_n" if count < 30 else "ok",
            **{f"q{p}": float(np.percentile(q[mask], p, method="linear")) if count else None
               for p in (50, 95)}}
    return {**metrics, "low4": low4}


def normalize_cell(cell: dict) -> dict:
    return {"arm": cell["arm"], "operator": cell["operator"], "database": cell["database"],
            "run": cell["run"], "k": cell["k20"],
            "metrics": {bucket: {name: cell["metrics"][bucket][name]
                                 for name in ("count", "q50", "q95")}
                        for bucket in ("all", "4+", "low4")}}


def phase(output: Path, name: str) -> None:
    contract.write_json(output / "phase.json", {"phase": name})


def evaluate_cell(model: Any, store: Any, query: list[int], support: list[int],
                  operator: str, device: str, deadline: Callable[[], None]) -> tuple[Any, Any, dict]:
    import numpy as np
    import torch

    from spatial_meta_ce.meta import forward

    deadline()
    model.eval()
    before = state_digest(model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    batch = None if operator == "none" else joint.certified_batch(store, support, device)
    params, record = operator_params(model, batch, operator)
    if device == "cuda":
        torch.cuda.synchronize()
    adapted = time.monotonic() - started
    started = time.monotonic()
    predictions, labels = [], []
    with torch.no_grad():
        for start in range(0, len(query), contract.FROZEN["evaluation"]["batch_size"]):
            deadline()
            batch = joint.certified_batch(store, query[start:start + 512], device)
            values = forward(model, params, batch).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite raw predictions")
            predictions.append(values)
            labels.append(batch["true"].cpu().numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    after = state_digest(model)
    if after != before:
        raise ValueError("Cell contaminated shared initialization")
    record = {**record, "state_before": before, "state_after": after,
              "adaptation_seconds": adapted, "prediction_seconds": time.monotonic() - started,
              "peak_vram_bytes": int(max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
              if device == "cuda" else 0,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
              * (1 if sys.platform == "darwin" else 1024)}
    joint.check_cell_peaks(record)
    return np.concatenate(predictions), np.concatenate(labels), record


def prepare_identities(output: Path, data: dict, plans: dict, anchors: dict,
                       deadline: Callable[[], None]) -> tuple[dict, dict, dict]:
    import numpy as np

    from spatial_meta_ce.data import FeatureStore

    stores, identities, selected = {}, {}, {}
    (output / "identities").mkdir()
    (output / "predictions").mkdir()
    for db in (*contract.TRAIN, *contract.DATABASES):
        deadline()
        store = FeatureStore(Path(data["cache_root"]) / "pretrain" / db)
        screen.validate_store(store, db)
        if db not in contract.DATABASES:
            continue
        stores[db] = store
        raw = joint.store_identity(store, db)
        for run in range(5):
            indices = certify_plan(plans[db][run], raw, db, run)
            query = np.asarray(indices["query"], dtype=np.int64)
            query = query[np.argsort(raw["query_hash"][query])]
            selected[(db, run)] = {"query": query.tolist(), "support": indices["support"]}
            identity = encode_identity({key: value[query] for key, value in raw.items()})
            path = output / "identities" / f"{db}-run{run}.npz"
            np.savez(path, **identity)
            identities[(db, run)] = (identity, path)
            # Preparation authenticates old B identities only, never micro raw predictions.
            with np.load(anchors[(db, run)]["path"], allow_pickle=False) as archive:
                joint.verify_anchor_identity(identity, {key: archive[key] for key in archive.files})
    return stores, identities, selected


def save_binding(output: Path, mode: str, binding: dict, before: dict, historical: dict,
                 data: dict, anchors: dict, config_path: Path, data_config_path: Path) -> None:
    contract.write_json(output / "input-binding.json", {
        "schema": 1, "mode": mode, "input_sha256": binding,
        "code_sha256": {p: binding[str(ROOT / p)] for p in contract.CODE_FILES},
        "config_file": str(config_path), "config_sha256": contract.sha256(config_path),
        "data_config_file": str(data_config_path), "data_config_sha256": contract.sha256(data_config_path),
        "data": data, "source_before": before, "historical_sources": historical,
        "endpoint_plan_file": str(contract.RETAINED / "endpoint_plan.json"),
        "endpoint_plan_sha256": contract.PLAN_SHA,
        "old_b_metadata_file": str(contract.OLD_B_METADATA),
        "old_b_metadata_sha256": contract.OLD_B_METADATA_SHA, "anchors": list(anchors.values()),
    })
    contract.write_json(output / "source_before.json", before)
    contract.write_json(output / "historical_sources.json", historical)
    contract.write_json(output / "reproduction.json", {
        "schema": 1, "status": "pending" if mode == "formal" else "not_applicable",
        "reproduction_verified": False, "old_b_metadata_file": str(contract.OLD_B_METADATA),
        "old_b_metadata_sha256": contract.OLD_B_METADATA_SHA,
        "anchors": [a for a in anchors.values() if mode == "formal" or a["run"] == 0],
    })


def evaluate_operator(output: Path, model: Any, stores: dict, identities: dict, selected: dict,
                      *, arm: str, operator: str, mode: str, device: str,
                      deadline: Callable[[], None], cells: list[dict]) -> list[dict]:
    import numpy as np

    for db in contract.DATABASES:
        for run in (range(5) if mode == "formal" else range(1)):
            identity, identity_path = identities[(db, run)]
            selection = selected[(db, run)]
            positions = None if mode == "formal" else list(range(32))
            query = selection["query"] if positions is None else selection["query"][:32]
            actual = identity if positions is None else {k: v[positions] for k, v in identity.items()}
            pred, true, record = evaluate_cell(model, stores[db], query, selection["support"],
                                               operator, device, deadline)
            if not np.array_equal(true, actual["true"]):
                raise ValueError("Output FP64 truth changed")
            path = output / "predictions" / f"{arm}-{operator}-{db}-run{run}.npz"
            np.savez(path, log_pred=pred)
            cell = {**record, "arm": arm, "operator": operator, "database": db, "run": run,
                    "k20": 20, "count": len(pred),
                    "identity_file": str(identity_path.relative_to(output)),
                    "identity_sha256": contract.sha256(identity_path),
                    "prediction_file": str(path.relative_to(output)), "prediction_sha256": contract.sha256(path),
                    "metrics": cell_metrics(pred, actual),
                    **({"query_positions": positions} if positions is not None else {})}
            cells = [*cells, cell]
            contract.write_json(output / ("cells.json" if mode == "formal" else "microcells.json"),
                                {"schema": 1, "mode": mode, "cells": cells})
    return cells


def train_and_evaluate(output: Path, config: dict, data: dict, mode: str, plans: dict,
                       anchors: dict, before: dict, deadline: Callable[[], None]) -> list[dict]:
    import torch

    from scripts.meta65_scope_verify import verify_reproduction

    stores, identities, selected = prepare_identities(output, data, plans, anchors, deadline)
    records, cells = [], []
    device = "cpu" if mode == "cpu_preflight" else "cuda"
    formal = mode == "formal"
    old = contract.read_json(contract.OLD_B_METADATA)
    for arm in contract.ARMS:
        phase(output, "training")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model, record = screen.train_arm(contract.arm_config(config, arm, mode), data,
                                         output / arm, device, deadline)
        peaks = {"peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                 * (1 if sys.platform == "darwin" else 1024),
                 "peak_vram_bytes": int(max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
                 if device == "cuda" else 0}
        joint.check_cell_peaks(peaks)
        record = {**record, **peaks, "arm": arm, "endpoint_plan_digest": contract.digest(plans)}
        if record["metadata"]["manifest_hashes"] != {db: info["manifest_hash"] for db, info in before.items()}:
            raise ValueError("Training sources differ from authenticated cache")
        if arm == contract.ARMS[0]:
            contract.verify_reference_training(record, old, formal=formal)
        contract.verify_pairing([*records, record], formal=formal)
        metadata = output / arm / "metadata.json"
        contract.write_json(metadata, record)
        records = [*records, {**record, "metadata_file": str(metadata.relative_to(output)),
                             "metadata_sha256": contract.sha256(metadata)}]
        contract.write_json(output / "training-records.json", {"schema": 1, "records": records})
        phase(output, "evaluation")
        # Gate T before it exists: R head15 -> read-only B raw reproduction -> R none15.
        operators = ("head3", "none") if arm == contract.ARMS[0] else contract.OPERATORS
        for operator in operators:
            cells = evaluate_operator(output, model, stores, identities, selected, arm=arm,
                                      operator=operator, mode=mode, device=device, deadline=deadline, cells=cells)
            if arm == contract.ARMS[0] and operator == "head3":
                reproduced = verify_reproduction(output, cells, formal=formal)
                contract.write_json(output / "reproduction.json", reproduced)
        del model
    return cells


def worker(output: Path, mode: str, config_path: Path, *, root: Path = ROOT,
           data_config_path: Path | None = None, approval: Path | None = None,
           launch_binding: Path | None = None) -> dict:
    require_supervisor(mode)
    config_path = contract.checked_path(config_path)
    data_config_path = contract.checked_path(data_config_path or root / "configs/meta65_server_data.json")
    if mode == "formal":
        formal_gate(approval, output, launch_binding, config_path, worker=True, root=root)
    config = contract.load_config(config_path, root)
    data = screen.validate_data(contract.read_json(data_config_path), root)
    output = contract.safe_output(output, [data["cache_root"], data["data_root"], contract.RETAINED], root)
    runtime = environment(mode)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    contract.write_json(output / "status.json", {"schema": 1, "status": "running", "mode": mode,
                                                "cells": 0, "formal_result": False})
    contract.write_json(output / "environment.json", runtime)
    phase(output, "preparation")
    seconds = config["limits"]["total_seconds"] if mode == "formal" else config[mode]["total_seconds"]

    def deadline() -> None:
        screen.check_deadline(started, seconds)

    binding = None
    try:
        binding, plans, anchors, before, historical, data = contract.input_binding(
            root, config_path=config_path, data_config_path=data_config_path)
        save_binding(output, mode, binding, before, historical, data, anchors, config_path, data_config_path)
        cells = train_and_evaluate(output, config, data, mode, plans, anchors, before, deadline)
        phase(output, "certification")
        verified = certify(output, mode, cells, binding, data, historical, before)
        deadline()
        contract.write_json(output / "verification.json", verified)
        if mode == "formal":
            from scripts.meta65_scope_analysis import analyze
            contract.write_json(output / "analysis.json", analyze(output, formal=True))
        deadline()
        contract.recheck(binding)
        contract.write_json(output / "status.json", {"schema": 1, "status": "worker_complete", "mode": mode,
            "cells": len(cells) if mode == "formal" else 12,
            "micro_cells": len(cells) if mode != "formal" else 0, "formal_result": mode == "formal"})
        return verified
    except BaseException as exc:
        contract.write_json(output / "status.json", {"schema": 1, "status": "failed", "mode": mode,
                           "failure": f"{type(exc).__name__}: {exc}", "formal_result": False})
        if binding is not None:
            from scripts.precision_screen_evidence import source_evidence
            try:
                contract.write_json(output / "source_after.json", source_evidence(data, historical))
                contract.recheck(binding)
            except Exception as audit_error:
                contract.write_json(output / "status.json", {"schema": 1, "status": "failed", "mode": mode,
                    "failure": f"{type(exc).__name__}: {exc}", "audit_failure": str(audit_error),
                    "formal_result": False})
        raise


def certify(output: Path, mode: str, cells: list[dict], binding: dict,
            data: dict, historical: dict, before: dict) -> dict:
    from scripts.meta65_scope_decision import decide
    from scripts.meta65_scope_verify import verify
    from scripts.precision_screen_evidence import historical_sources, source_evidence

    after = source_evidence(data, historical)
    contract.write_json(output / "source_after.json", after)
    if after != before or historical_sources(ROOT, list((*contract.TRAIN, *contract.DATABASES))) != historical:
        raise ValueError("Source/cache/historical manifests changed")
    contract.recheck(binding)
    verified = verify(output, formal=mode == "formal")
    if mode == "formal":
        reproduced = contract.read_json(output / "reproduction.json")
        decision = decide([normalize_cell(c) for c in cells],
                          engineering_verified=verified.get("all_metrics_verified") is True,
                          reproduction_verified=reproduced.get("reproduction_verified") is True)
        summary = verified.get("summary")
        if isinstance(summary, dict) and decision.get("status") != summary.get("status"):
            raise ValueError("Independent classifier disagrees with provisional decision")
        contract.write_json(output / "decision.provisional.json", decision)
    else:
        contract.write_json(output / "decision.provisional.json", {
            "schema": 1, "status": "preflight_not_candidate", "selected_arm": None,
            "formal_result": False, "reproduction_status": "not_applicable"})
    return verified


def require_supervisor(mode: str) -> None:
    if (os.environ.get("META65_SCOPE_SUPERVISED_MODE") != mode or
            os.environ.get("META65_SCOPE_SUPERVISOR_PID") != str(os.getppid())):
        raise ValueError("Worker requires its owning external supervisor")


def formal_gate(approval: Path | None, output: Path, binding: Path | None, config_path: Path,
                *, worker: bool, root: Path = ROOT) -> dict:
    if approval is None or binding is None:
        raise ValueError("Formal execution requires separate approval and launch binding; not authorized")
    try:
        launch = importlib.import_module("scripts.meta65_scope_launch")
    except ImportError as exc:
        raise NotImplementedError("Formal launch authorization integration is not implemented") from exc
    if launch is None or not hasattr(launch, "authorize_formal"):
        raise NotImplementedError("Formal launch authorization integration is not implemented")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if worker:
        require_supervisor("formal")
        return launch.verify_worker_approval(approval, output, binding, config_path,
                                             root=root, gpu_uuid=uuid, parent_pid=os.getppid())
    return launch.authorize_formal(approval, output, binding, config_path, root=root, gpu_uuid=uuid)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/meta65_scope.json")
    parser.add_argument("--data-config", type=Path, default=ROOT / "configs/meta65_server_data.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu_preflight", "gpu_preflight", "formal"), default="cpu_preflight")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--launch-binding", type=Path)
    args = parser.parse_args(argv)
    config_path = contract.checked_path(args.config)
    contract.load_config(config_path)
    output = contract.checked_path(args.out)
    if args.mode == "formal" and (args.approval is None or args.launch_binding is None):
        raise ValueError("Formal execution requires separate approval and launch binding; not authorized")
    if args.mode != "formal" and (args.approval is not None or args.launch_binding is not None):
        raise ValueError("Preflight cannot use formal approval or launch binding")
    if not args.execute:
        print(json.dumps({"status": "prepared_not_executed", "mode": args.mode,
                          "matrix": contract.matrix(args.mode), "formal_result": False}, indent=2))
        return
    data = screen.validate_data(contract.read_json(args.data_config))
    contract.safe_output(output, [data["cache_root"], data["data_root"], contract.RETAINED])
    if args.worker:
        worker(output, args.mode, config_path, data_config_path=args.data_config,
               approval=args.approval, launch_binding=args.launch_binding)
        return
    if args.mode == "formal":
        formal_gate(args.approval, output, args.launch_binding, config_path, worker=False)
    elif args.mode == "gpu_preflight":
        joint.verify_gpu_idle(os.environ.get("CUDA_VISIBLE_DEVICES"))
    from scripts.meta65_scope_supervisor import supervise
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--execute",
               "--config", str(config_path), "--data-config", str(contract.checked_path(args.data_config)),
               "--out", str(output), "--mode", args.mode]
    if args.approval is not None:
        command += ["--approval", str(contract.checked_path(args.approval)),
                    "--launch-binding", str(contract.checked_path(args.launch_binding))]
    record = supervise(command, output, mode=args.mode,
                       gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES") if args.mode != "cpu_preflight" else None,
                       budget_override=None)
    print(json.dumps(record, indent=2))
    if record["status"] not in ("completed", "preflight_completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
