#!/usr/bin/env python3
"""Model-forward latency only (plan L1, doc 110).

What this measures
------------------
One already-encoded sample -> one forward pass -> one log-cardinality.  This is
the narrowest latency number the project can report, and the one that maps onto
PRICE's published ``INFERENCE TIME (MS)`` column.  It deliberately **excludes**
SQL parsing, feature extraction, label acquisition, plan generation and
database execution; ``protocol.json`` records that so the number can never be
quoted as an end-to-end speedup.

Why a population pass instead of "one query x N repeats"
--------------------------------------------------------
``spatial_meta_ce/evaluate.py``'s ``latency()`` times a single fixed row 30
times.  That cannot show how latency depends on query shape (join columns,
fanouts, tables and filters all change the tensor sizes).  This script times
every row of a database's test cache once, after discarding a warm-up prefix,
and reports percentiles over that population.  A repeated-single-row number is
produced as well, so the historical ``batch1_p50_ms`` values stay comparable.

Two devices are never merged: every result file names exactly one device and
one thread count, and the protocol records both.  Records-model metrics cannot
substitute for this: parameter count is not a latency measurement.

Read-only.  No training, no adaptation, no checkpoint selection, no labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.baselines import PriceAdapter  # noqa: E402
from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.data import StorePool  # noqa: E402
from spatial_meta_ce.meta import forward  # noqa: E402

PERCENTILES = (50, 90, 95, 99)
PRICE_MAXIMA_MODES = ("cache", "pooled_test")
THROUGHPUT_BATCH = 128
THROUGHPUT_REPEATS = 3
STABILITY_TOLERANCE = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict:
    """Percentiles plus moments of a non-empty millisecond sample."""
    if not values:
        raise ValueError("Cannot summarize an empty latency sample")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Latency samples must be finite")
    if (array < 0).any():
        raise ValueError("Latency samples must be non-negative")
    result = {f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES}
    result.update(
        {
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
            "n": int(array.size),
        }
    )
    return result


def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def timed_ms(call, device: str) -> float:
    """Wall time of one no-grad call, synchronised on both sides."""
    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        call()
    sync(device)
    return (time.perf_counter() - started) * 1000.0


def environment(device: str, threads: int) -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "requested_threads": threads,
        "env_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "device": device,
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }
    if str(device).startswith("cuda"):
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "gpu_capability": list(torch.cuda.get_device_capability(device)),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return info


def price_maxima(caches: dict, database: str, mode: str) -> list[int]:
    """PRICE padding capacity.

    ``cache``      per-database test maxima -- PRICE's own per-dataset padding.
    ``pooled_test`` elementwise max over the four measured test caches, which is
        the capacity the project's own PRICE numbers were produced under for the
        four-target comparison.

    ``spatial_meta_ce/evaluate.py:255-261`` instead maxes over **test and
    finetune**, and that rule cannot be reproduced from this cache root because
    ``finetune/imdb`` is absent (protocol.md §IMDB 当前障碍) while ``imdb`` is one
    of the measured databases.  The gap is recorded in ``protocol.json`` rather
    than approximated silently: a capacity produced under a different rule is not
    the same capacity, and PRICE timings computed under different capacities must
    never be quoted together.
    """
    if mode == "cache":
        return [int(value) for value in caches[database]["test"]]
    if mode == "pooled_test":
        counts = np.ones(4, dtype=np.int64)
        for splits in caches.values():
            counts = np.maximum(counts, np.asarray(splits["test"], dtype=np.int64))
        return counts.tolist()
    raise ValueError(f"Unknown maxima mode {mode!r}")


def single_batches(store, device: str) -> list[dict]:
    """One batch per sample, built up front so the timed region is forward only."""
    return [store.batch([index], device) for index in range(len(store))]


def population(run, batches: list[dict], warmup: int, device: str) -> list[float]:
    for batch in batches[:warmup]:
        timed_ms(lambda batch=batch: run(batch), device)
    return [
        timed_ms(lambda batch=batch: run(batch), device) for batch in batches[warmup:]
    ]


def repeated(batch: dict, run, repeat_times: int, device: str) -> list[float]:
    for _ in range(3):
        timed_ms(lambda: run(batch), device)
    return [timed_ms(lambda: run(batch), device) for _ in range(repeat_times)]


def throughput(store, run, device: str) -> dict:
    total = len(store)
    sizes = [
        list(range(start, min(start + THROUGHPUT_BATCH, total)))
        for start in range(0, total, THROUGHPUT_BATCH)
    ]
    best = None
    for _ in range(THROUGHPUT_REPEATS):
        batches = [store.batch(chunk, device) for chunk in sizes]
        rows = sum(len(batch["pg_log"]) for batch in batches)
        seconds = (
            sum(timed_ms(lambda batch=batch: run(batch), device) for batch in batches)
            / 1000.0
        )
        candidate = rows / seconds if seconds else float("inf")
        best = candidate if best is None else max(best, candidate)
    return {
        "batch_size": THROUGHPUT_BATCH,
        "batches": len(sizes),
        "rows": total,
        "repeats": THROUGHPUT_REPEATS,
        "rows_per_second": best,
        "aggregation": "best of repeats (least-squares-free: a throughput ceiling)",
    }


def measure_ours(model, store, device: str, warmup: int, repeat_times: int) -> dict:
    params = dict(model.named_parameters())
    run = lambda batch: forward(model, params, batch)  # noqa: E731
    batches = single_batches(store, device)
    first = population(run, batches, warmup, device)
    second = population(run, batches, warmup, device)
    median_first = float(np.median(first))
    stability = abs(float(np.median(second)) - median_first) / max(median_first, 1e-9)
    return {
        "population": summarize(first),
        "population_repeat": summarize(second),
        "stability_relative_delta": stability,
        "stability_ok": stability <= STABILITY_TOLERANCE,
        "single_row_repeated": summarize(
            repeated(batches[0], run, repeat_times, device)
        ),
        "throughput": throughput(store, run, device),
    }


def measure_price(
    path: Path, store, database, caches, device, warmup, repeat_times
) -> dict:
    run_of = {}
    for mode in PRICE_MAXIMA_MODES:
        maxima = price_maxima(caches, database, mode)
        adapter = PriceAdapter(str(path), maxima, device)
        adapter.eval()
        run = lambda batch, adapter=adapter: adapter(batch)  # noqa: E731
        batches = single_batches(store, device)
        latencies = population(run, batches, warmup, device)
        run_of[mode] = {
            "maxima": maxima,
            "population": summarize(latencies),
            "single_row_repeated": summarize(
                repeated(batches[0], run, repeat_times, device)
            ),
        }
    return run_of


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Model-forward latency (read-only)")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument(
        "--allowed-targets",
        nargs="+",
        required=True,
        help="databases this invocation is authorised to read; every --databases entry must appear here",
    )
    parser.add_argument("--price-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat-times", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    requested = list(dict.fromkeys(args.databases))
    unapproved = set(requested) - set(args.allowed_targets)
    if unapproved:
        raise SystemExit(
            "Refusing to read databases outside --allowed-targets: "
            + ", ".join(sorted(unapproved))
        )
    if args.threads < 1:
        raise SystemExit("--threads must be >= 1")
    if args.warmup < 1:
        raise SystemExit("--warmup must be >= 1")
    if args.repeat_times < 1:
        raise SystemExit("--repeat-times must be >= 1")

    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")

    torch.set_num_threads(args.threads)
    model, state = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    metadata = state["metadata"]
    exposed = set(metadata.get("train_databases") or ()) | set(
        metadata.get("validation_databases") or ()
    )
    if exposed & set(requested):
        raise SystemExit("Checkpoint was trained or selected on a requested database")
    if metadata.get("adapted_database"):
        raise SystemExit("An adapted checkpoint is not an independent starting point")

    pool = StorePool(args.cache_root)
    caches = {}
    stores = {}
    finetune_available = {}
    for database in requested:
        store = pool.get("test", database)
        if len(store) <= args.warmup:
            raise SystemExit(
                f"{database}: {len(store)} test rows cannot absorb warm-up {args.warmup}"
            )
        stores[database] = store
        finetune_available[database] = (
            Path(args.cache_root) / "finetune" / database / "manifest.json"
        ).is_file()
        caches[database] = {
            "content_hash": store.manifest.get("content_hash"),
            "samples": len(store),
            "test": store.manifest["inspection"]["max_counts"],
        }

    result = {
        "schema": 1,
        "purpose": "inference-latency-model-forward-only",
        "unit": "milliseconds per forward pass",
        "includes_feature_extraction": False,
        "includes_label_acquisition": False,
        "includes_plan_generation": False,
        "includes_database_execution": False,
        "claim_boundary": (
            "One encoded sample, one forward pass. Excludes SQL parsing, feature "
            "extraction, labels, plan generation and execution. Not a database "
            "speedup measurement; inference time is not end-to-end acceleration."
        ),
        "databases": {},
    }
    protocol = {
        "schema": 1,
        "script": Path(__file__).name,
        "script_sha256": sha256_file(Path(__file__)),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "checkpoint_metadata": {
            "method": metadata.get("method"),
            "train_databases": metadata.get("train_databases"),
            "validation_databases": metadata.get("validation_databases"),
            "model_config": state["model_config"],
        },
        "databases_read": requested,
        "allowed_targets": list(args.allowed_targets),
        "caches": caches,
        "sampling": {
            "population": "every test-cache row once, first --warmup rows untimed",
            "single_row_repeated": f"row 0 repeated {args.repeat_times} times",
            "throughput": f"batch {THROUGHPUT_BATCH} over all rows, {THROUGHPUT_REPEATS} repeats",
            "warmup": args.warmup,
        },
        "stability_check": {
            "rule": "two independent population passes in one process",
            "tolerance_relative_delta": STABILITY_TOLERANCE,
            "action": "recorded per database; a breach is reported, never silently averaged",
        },
        "threads": args.threads,
        "device": args.device,
        "environment": environment(args.device, args.threads),
        "price": None,
    }
    if args.price_checkpoint:
        price_path = Path(args.price_checkpoint)
        if not price_path.exists():
            raise SystemExit(f"PRICE checkpoint not found: {price_path}")
        protocol["price"] = {
            "checkpoint": str(price_path.resolve()),
            "checkpoint_sha256": sha256_file(price_path),
            "maxima_modes": {
                "cache": "per-database test inspection.max_counts (PRICE's own per-dataset padding)",
                "pooled_test": (
                    "elementwise max over the four measured test caches; the capacity "
                    "the four-target PRICE comparison was produced under"
                ),
            },
            "evaluate_py_rule": {
                "definition": "elementwise max over test AND finetune for every test database",
                "source": "spatial_meta_ce/evaluate.py:255-261",
                "reproducible": False,
                "reason": (
                    "finetune/imdb is absent from this cache root (protocol.md §IMDB "
                    "当前障碍), so the rule cannot be evaluated for a database that is "
                    "being measured; no approximation is substituted"
                ),
                "finetune_cache_present": finetune_available,
            },
        }

    breaches = []
    for database in requested:
        entry = {
            "samples": caches[database]["samples"],
            "models": {
                "ours": measure_ours(
                    model, stores[database], args.device, args.warmup, args.repeat_times
                )
            },
        }
        if protocol["price"]:
            entry["models"]["price"] = measure_price(
                Path(args.price_checkpoint),
                stores[database],
                database,
                caches,
                args.device,
                args.warmup,
                args.repeat_times,
            )
        result["databases"][database] = entry
        ours = entry["models"]["ours"]
        if not ours["stability_ok"]:
            breaches.append(
                {
                    "database": database,
                    "relative_delta": ours["stability_relative_delta"],
                    "tolerance": STABILITY_TOLERANCE,
                }
            )
            print(
                f"WARNING: {database} population passes differ by "
                f"{ours['stability_relative_delta']:.4f} > {STABILITY_TOLERANCE}",
                file=sys.stderr,
                flush=True,
            )
        print(
            json.dumps(
                {
                    "database": database,
                    "ours_p50": ours["population"]["p50"],
                    "ours_p99": ours["population"]["p99"],
                    "stability": round(ours["stability_relative_delta"], 4),
                    "stability_ok": ours["stability_ok"],
                }
            ),
            flush=True,
        )

    result["stability_breaches"] = breaches
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    out.with_suffix(".protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
