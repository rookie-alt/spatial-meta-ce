"""Bounded CPU-only S7; fixed checkpoints and source arrays, never trains."""

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import s2_source_replay as replay  # noqa: E402
from scripts import s7_source_diagnosis as stats  # noqa: E402

HIST = "实验文档/results/s6-mean-tail-20261007"
CORE = (
    "spatial_meta_ce/model.py",
    "spatial_meta_ce/data.py",
    "spatial_meta_ce/checkpoints.py",
    "spatial_meta_ce/mappings.py",
    "spatial_meta_ce/manifests.py",
)
CODE = (
    *CORE,
    "spatial_meta_ce/__init__.py",
    "scripts/s2_source_replay.py",
    "scripts/s7_source_diagnosis.py",
    "scripts/run_s7_diagnosis.py",
    "scripts/s7_independent_verify.py",
    "configs/s7_diagnosis.json",
    "实验文档/21-S7已有预测误差与源侧泛化诊断协议.md",
)


def checked(path, sha):
    if replay.file_hash(path) != sha:
        raise ValueError(f"Input fingerprint changed: {path}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def endpoint_data(path, caches):
    import numpy as np

    result = {
        db: np.empty(len(c["indices"]), dtype=np.float64) for db, c in caches.items()
    }
    positions = {
        db: {h: i for i, h in enumerate(c["query_hash"])} for db, c in caches.items()
    }
    seen = {db: set() for db in caches}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            db, key = row["database"], row["query_hash"]
            if db not in caches or key not in positions[db] or key in seen[db]:
                raise ValueError("Endpoint identity extra/duplicate/mismatch")
            i = positions[db][key]
            c = caches[db]
            if (
                row["true"] != c["true"][i]
                or row["n_tables"] != c["n_tables"][i]
                or abs(row["pg_reference_log"] - np.log(max(c["pg"][i], 1))) > 1e-12
                or row["pg_input_log"] != float(np.float32(np.log(max(c["pg"][i], 1))))
            ):
                raise ValueError("Endpoint true/PG/tables mismatch")
            pred = row["log_pred"]
            if not np.isfinite(pred):
                raise ValueError("Nonfinite endpoint prediction")
            result[db][i] = pred
            seen[db].add(key)
    if any(len(seen[db]) != len(c["indices"]) for db, c in caches.items()):
        raise ValueError("Incomplete endpoint denominator")
    return result


def cohort_report(data):
    import numpy as np

    strata = []
    for name, mask in [
        ("all", np.ones(len(data["true"]), dtype=bool)),
        ("four_plus", data["n_tables"] >= 4),
    ]:
        strata.append({"bucket": name, **stats.pair(data, mask)})
    tables = {
        "1-3": data["n_tables"] < 4,
        "4": data["n_tables"] == 4,
        "5": data["n_tables"] == 5,
        "6+": data["n_tables"] >= 6,
    }
    for card, cm in stats.cardinality_masks(data["true"]).items():
        for table, tm in tables.items():
            strata.append(
                {"bucket": f"{card}|tables={table}", **stats.pair(data, cm & tm)}
            )
    return strata


def execute(data_root, s6_run, output):
    import numpy as np
    import torch

    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import FeatureStore

    started = time.monotonic()
    cfg = read_json(ROOT / "configs/s7_diagnosis.json")
    if (
        np.__version__ != "1.26.1"
        or torch.__version__ != "2.1.0+cu121"
        or tuple(sys.version_info[:3]) != (3, 11, 4)
    ):
        raise ValueError("Frozen S6 inference/replay environment required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frozen = read_json(ROOT / HIST / "summary.json")
    preflight = read_json(ROOT / HIST / "preflight.json")
    status = read_json(ROOT / HIST / "status.json")
    if status["status"] != "completed" or frozen["decision"] != "no_candidate":
        raise ValueError("S6 completion/decision mismatch")
    inputs = {}
    code_hash = {name: replay.file_hash(ROOT / name) for name in CODE}
    for name in CORE:
        checked(ROOT / name, preflight["code_sha256"][name])
    for name in ("summary.json", "preflight.json", "status.json"):
        local = ROOT / HIST / name
        remote = replay.safe(data_root, s6_run / name)
        sha = replay.file_hash(local)
        checked(remote, sha)
        inputs[str(remote)] = sha
    for db in (*replay.TRAIN, *replay.VALIDATION):
        for name, sha in preflight["sources"][db]["file_sha256"].items():
            path = replay.safe(
                data_root, data_root / "artifacts/cache/pretrain" / db / name
            )
            checked(path, sha)
            inputs[str(path)] = sha
    for record in frozen["runs"]:
        for name, key in [
            ("last.pt", "checkpoint_sha256"),
            ("endpoint_rows.jsonl", "endpoint_rows_sha256"),
        ]:
            path = replay.safe(data_root, s6_run / record["arm"] / name)
            checked(path, record[key])
            inputs[str(path)] = record[key]
    stats.write_json(
        output / "protocol.json",
        {
            "config": cfg,
            "protocol_sha256": replay.file_hash(
                ROOT / "实验文档/21-S7已有预测误差与源侧泛化诊断协议.md"
            ),
            "code_sha256": code_hash,
            "input_sha256": inputs,
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "s6_decision": frozen["decision"],
            "training": False,
            "adaptation": False,
            "target_data_read": False,
            "device": "cpu",
        },
    )
    print("S7 input authentication complete", flush=True)
    caches = {
        db: replay.read_cache(data_root, db, preflight["sources"][db])
        for db in (*replay.TRAIN, *replay.VALIDATION)
    }
    sampled = replay.replay(
        caches, list(replay.TRAIN), frozen["runs"][0]["config"], prefixes=(4000,)
    )
    replay.certify(sampled)
    states = sampled["prefixes"][4000]["databases"]
    held = {db: caches[db] for db in replay.VALIDATION}
    predictions = {
        arm: endpoint_data(s6_run / record["arm"] / "endpoint_rows.jsonl", held)
        for arm, record in zip(stats.ARMS, frozen["runs"])
    }
    validation = {
        db: {**held[db], **{arm: predictions[arm][db] for arm in stats.ARMS}}
        for db in replay.VALIDATION
    }
    for arm, record in zip(stats.ARMS, frozen["runs"]):
        for db, data in validation.items():
            for bucket, mask in [
                ("all_tables", np.ones(len(data["true"]), dtype=bool)),
                ("four_plus_tables", data["n_tables"] >= 4),
            ]:
                actual = stats.metrics(data[arm][mask], data["true"][mask])
                for q in ("q50", "q95"):
                    if not np.isclose(
                        actual[q],
                        record["endpoint"]["per_database"][db][bucket][q],
                        rtol=1e-5,
                        atol=1e-8,
                    ):
                        raise ValueError("Historical S6 endpoint failed reproduction")
    stats.write_json(
        output / "legalacts.json", stats.legalacts_report(validation["legalacts"])
    )
    print("S7 historical endpoint/legalacts complete", flush=True)
    selections, manifest = {}, []
    with (output / "sample-manifest.jsonl").open("x") as stream:
        for db in replay.TRAIN:
            seen, unseen = stats.memberships(states[db])
            c = caches[db]
            for cohort, candidates in [("seen", seen), ("unseen", unseen)]:
                selected = stats.deterministic_sample(
                    db,
                    cohort,
                    c["query_hash"],
                    candidates,
                    cfg["sample_limit_per_database_cohort"],
                )
                selections[(db, cohort)] = selected
                member = np.array(selected, dtype=np.int64)
                row = {
                    "database": db,
                    "cohort": cohort,
                    "pool_unique": len(candidates),
                    "sample_unique": len(selected),
                    "manifest_hash": preflight["sources"][db]["manifest_hash"],
                    "selected_identity_digest": replay.digest(
                        [c["query_hash"][i] for i in selected]
                    ),
                    "sample_presentations": int(
                        sum((states[db]["support"] + states[db]["query"])[member])
                    ),
                    "pool_cardinality_counts": {
                        k: int(m[candidates].sum())
                        for k, m in stats.cardinality_masks(c["true"]).items()
                    },
                    "sample_cardinality_counts": {
                        k: int(m[member].sum())
                        for k, m in stats.cardinality_masks(c["true"]).items()
                    },
                    "pool_four_plus": int((c["n_tables"][candidates] >= 4).sum()),
                    "sample_four_plus": int((c["n_tables"][member] >= 4).sum()),
                }
                manifest.append(row)
                for i in selected:
                    stream.write(
                        json.dumps(
                            {
                                "database": db,
                                "cohort": cohort,
                                "unique_position": i,
                                "raw_index": int(c["indices"][i]),
                                "query_hash": c["query_hash"][i],
                                "true": float(c["true"][i]),
                                "pg": float(c["pg"][i]),
                                "n_tables": int(c["n_tables"][i]),
                                "presentations": int(
                                    states[db]["support"][i] + states[db]["query"][i]
                                ),
                            }
                        )
                        + "\n"
                    )
    stats.write_json(
        output / "sample-plan.json",
        {
            "sample_digest": sampled["digest"],
            "episodes": 16000,
            "presentations": 1621915,
            "cohorts": manifest,
            "selected_rows_sha256": replay.file_hash(output / "sample-manifest.jsonl"),
        },
    )
    models = {}
    for arm, record in zip(stats.ARMS, frozen["runs"]):
        model, checkpoint = load_checkpoint(s6_run / record["arm"] / "last.pt", "cpu")
        meta = checkpoint["metadata"]
        if (
            checkpoint["next_step"] != 4000
            or meta["train_databases"] != list(replay.TRAIN)
            or meta["validation_databases"] != list(replay.VALIDATION)
            or meta["manifest_hashes"] != record["manifest_hashes"]
            or meta["training_signature"]
            != {
                k: v
                for k, v in record["config"].items()
                if k not in ("steps", "data_config")
            }
            or checkpoint["extra"]["sample_plan_digest"] != sampled["digest"]
        ):
            raise ValueError("Fixed checkpoint content differs from S6")
        model.eval()
        models[arm] = model
    bridge = {}
    with torch.inference_mode():
        for db in replay.VALIDATION:
            c = caches[db]
            chosen = sorted(
                range(len(c["indices"])),
                key=lambda i: hashlib.sha256(
                    f"s7-bridge-v1|{db}|{c['query_hash'][i]}".encode()
                ).digest(),
            )[: cfg["bridge_queries_per_validation_database"]]
            store = FeatureStore(data_root / "artifacts/cache/pretrain" / db)
            batch = store.batch(c["indices"][chosen], "cpu")
            bridge[db] = {}
            for arm in stats.ARMS:
                values = models[arm](batch).cpu().numpy().astype(np.float64)
                diff = float(np.abs(values - validation[db][arm][chosen]).max())
                if not np.isfinite(values).all() or diff > cfg["bridge_raw_log_atol"]:
                    raise ValueError("CPU/GPU raw prediction bridge failed")
                bridge[db][arm] = {
                    "queries": len(chosen),
                    "max_abs_raw_log_difference": diff,
                }
            del store, batch
    results = []
    with (
        (output / "source-predictions.jsonl").open("x") as stream,
        torch.inference_mode(),
    ):
        for db in replay.TRAIN:
            c = caches[db]
            store = FeatureStore(data_root / "artifacts/cache/pretrain" / db)
            for cohort in ("seen", "unseen"):
                chosen = np.asarray(selections[(db, cohort)], dtype=np.int64)
                output_values = {arm: [] for arm in stats.ARMS}
                for offset in range(0, len(chosen), cfg["batch_size"]):
                    positions = chosen[offset : offset + cfg["batch_size"]]
                    batch = store.batch(c["indices"][positions], "cpu")
                    values = {
                        arm: models[arm](batch).cpu().numpy().astype(np.float64)
                        for arm in stats.ARMS
                    }
                    for arm in stats.ARMS:
                        output_values[arm].extend(values[arm].tolist())
                    for j, i in enumerate(positions):
                        row = {
                            "database": db,
                            "cohort": cohort,
                            "query_hash": c["query_hash"][i],
                            "true": float(c["true"][i]),
                            "pg": float(c["pg"][i]),
                            "n_tables": int(c["n_tables"][i]),
                            **{arm: float(values[arm][j]) for arm in stats.ARMS},
                        }
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                data = {k: c[k][chosen] for k in ("true", "pg", "n_tables")}
                data = {
                    **data,
                    "query_hash": [c["query_hash"][i] for i in chosen],
                    **{arm: np.asarray(output_values[arm]) for arm in stats.ARMS},
                }
                if len(chosen):
                    stats.validate(data)
                    report = cohort_report(data)
                else:
                    report = [{"bucket": "all", "n": 0, "status": "below_min_cell_n"}]
                results.append({"database": db, "cohort": cohort, "statistics": report})
            del store
            print(f"S7 inference complete: {db}", flush=True)
    for db, data in validation.items():
        results.append(
            {
                "database": db,
                "cohort": "source_validation_full",
                "statistics": cohort_report(data),
            }
        )
    stats.write_json(
        output / "generalization.json",
        {
            "purpose": "sampled-fit-versus-developed-source-holdout",
            "rows": results,
            "cpu_bridge": bridge,
            "causal_explanation_verified": False,
            "warning": "Seen/unseen composition and exposure differ. Validation uses historical full GPU predictions.",
        },
    )
    from scripts.s7_independent_verify import verify

    proof = verify(output, data_root, s6_run, caches, states)
    stats.write_json(output / "verification.json", proof)
    if not proof["passed"]:
        raise ValueError("Independent S7 verification failed")
    for path, sha in inputs.items():
        checked(Path(path), sha)
    for name, sha in code_hash.items():
        checked(ROOT / name, sha)
    stats.write_json(
        output / "summary.json",
        {
            "status": "completed",
            "training": False,
            "inference": True,
            "device": "cpu",
            "adaptation": False,
            "target_data_read": False,
            "s6_decision": "no_candidate",
            "sample_digest": sampled["digest"],
            "inference_queries": sum(r["sample_unique"] for r in manifest),
            "source_databases": 26,
            "per_training_database_cohort_limit": 2048,
            "seconds": time.monotonic() - started,
            "inputs_unchanged": True,
            "code_unchanged": True,
            "independent_verification_passed": True,
            "causal_explanation_verified": False,
            "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--s6-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    s6_run = replay.safe(data_root, args.s6_run)
    output = replay.safe(data_root, args.output)
    if (
        s6_run != data_root / "runs/s6-mean-tail-20261007"
        or output != data_root / "runs/s7-fit-generalization-20261007"
    ):
        raise ValueError("S7 requires fixed history and a new fixed output directory")
    if output.exists():
        raise FileExistsError("S7 requires fresh output, no retries")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cfg = read_json(ROOT / "configs/s7_diagnosis.json")
    resource.setrlimit(
        resource.RLIMIT_AS, (cfg["address_space_bytes"], cfg["address_space_bytes"])
    )
    output.mkdir()

    def timed_out(signum, frame):
        raise TimeoutError("S7 hard wall-clock deadline reached")

    signal.signal(signal.SIGALRM, timed_out)
    signal.signal(signal.SIGTERM, timed_out)
    signal.signal(signal.SIGINT, timed_out)
    signal.alarm(cfg["timeout_seconds"])
    stats.write_json(
        output / "status.json",
        {"status": "running", "training": False, "device": "cpu"},
    )
    try:
        execute(data_root, s6_run, output)
        hashes = {
            p.name: replay.file_hash(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "status.json"
        }
        stats.write_json(
            output / "status.json",
            {
                "status": "completed",
                "output_sha256": hashes,
                "training": False,
                "target_data_read": False,
            },
        )
    except BaseException as error:
        stats.write_json(
            output / "status.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "training": False,
                "target_data_read": False,
            },
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
