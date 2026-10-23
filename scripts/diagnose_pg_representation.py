import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

DATABASES = ("chembl", "legalacts", "telstra")
ARMS = ("sum", "true", "shuffled")
SEEDS = (42, 43, 44)
CARDINALITY_BINS = (
    (10, 100),
    (100, 1000),
    (1000, 10000),
    (10000, 100000),
    (100000, np.inf),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_head(
    h_a: np.ndarray,
    h_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    b_a: float,
    b_b: float,
) -> tuple[np.ndarray, ...]:
    bias = np.full(len(h_a), b_b - b_a)
    readout = ((h_a + h_b) / 2) @ (w_b - w_a)
    representation = (h_b - h_a) @ ((w_a + w_b) / 2)
    return bias, readout, representation


def groups(truth: np.ndarray, tables: np.ndarray) -> dict[str, np.ndarray]:
    primary = tables >= 4
    return {
        "4plus": primary,
        **{f"tables={n}": primary & (tables == n) for n in np.unique(tables[primary])},
        **{
            f"true=[{lo},{hi})": primary & (truth >= lo) & (truth < hi)
            for lo, hi in CARDINALITY_BINS
        },
    }


def q95(log_pred: np.ndarray, truth: np.ndarray) -> float:
    error = np.abs(np.maximum(log_pred, 0) - np.log(np.maximum(truth, 1)))
    return float(np.quantile(np.exp(error), 0.95))


def validate_rows(rows: list[dict], original: dict) -> None:
    if len(rows) != len(original["log_pred"]):
        raise ValueError("Unique query count differs from frozen predictions")
    for key in ("query_hash", "n_tables", "true"):
        if not np.array_equal(np.asarray([row[key] for row in rows]), original[key]):
            raise ValueError(f"Frozen prediction pairing mismatch: {key}")


def capture(
    model: torch.nn.Module, store: object, device: str
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    from scripts.mapping_ablation_endpoint import endpoint_rows

    pg_chunks: tuple[np.ndarray, ...] = ()
    hidden_chunks: tuple[np.ndarray, ...] = ()

    def pg_hook(_module: torch.nn.Module, inputs: tuple) -> None:
        nonlocal pg_chunks
        pg_chunks = (*pg_chunks, inputs[0][:, -2].detach().cpu().numpy())

    def hidden_hook(_module: torch.nn.Module, inputs: tuple) -> None:
        nonlocal hidden_chunks
        hidden_chunks = (*hidden_chunks, inputs[0].detach().cpu().numpy())

    pg_handle = model.head[0].register_forward_pre_hook(pg_hook)
    hidden_handle = model.head[-1].register_forward_pre_hook(hidden_hook)
    try:
        rows = endpoint_rows(model, store, device)
    finally:
        pg_handle.remove()
        hidden_handle.remove()
    return (
        rows,
        np.concatenate(pg_chunks).astype(np.float64),
        np.concatenate(hidden_chunks).astype(np.float64),
    )


def endpoint(records: dict, mask: np.ndarray) -> dict:
    pg, pred, truth = (records[key][mask] for key in ("pg", "pred", "true"))
    pg_error = pg - np.log(np.maximum(truth, 1))
    correction = pred - pg
    model_error = pred - np.log(np.maximum(truth, 1))
    np.testing.assert_allclose(model_error, pg_error + correction, rtol=0, atol=2e-12)
    return {
        "n": int(mask.sum()),
        "pg_q95": q95(pg, truth),
        "model_q95": q95(pred, truth),
        "pg_signed_error_mean": float(pg_error.mean()),
        "head_correction_mean": float(correction.mean()),
        "model_signed_error_mean": float(model_error.mean()),
        "head_correction_p10_p50_p90": np.quantile(
            correction, [0.1, 0.5, 0.9]
        ).tolist(),
        "head_improves_absolute_log_error_fraction": float(
            np.mean(np.abs(model_error) < np.abs(pg_error))
        ),
        "pg_under_fraction": float(np.mean(pg_error < 0)),
        "model_under_fraction_raw_log": float(np.mean(model_error < 0)),
        "pg_error_head_correction_covariance": float(
            np.mean((pg_error - pg_error.mean()) * (correction - correction.mean()))
        ),
    }


def pair(a: dict, b: dict, mask: np.ndarray) -> dict:
    np.testing.assert_array_equal(a["pg"], b["pg"])
    np.testing.assert_array_equal(a["query_hash"], b["query_hash"])
    np.testing.assert_array_equal(a["true"], b["true"])
    np.testing.assert_array_equal(a["n_tables"], b["n_tables"])
    bias, readout, representation = split_head(
        a["hidden"][mask],
        b["hidden"][mask],
        a["weight"],
        b["weight"],
        a["bias"],
        b["bias"],
    )
    delta = b["pred"][mask] - a["pred"][mask]
    np.testing.assert_allclose(
        bias + readout + representation, delta, rtol=2e-5, atol=5e-5
    )
    return {
        "n": int(mask.sum()),
        "delta_prediction_mean": float(delta.mean()),
        "pg_delta_max_abs": float(np.max(np.abs(b["pg"][mask] - a["pg"][mask]))),
        "delta_bias": float(bias[0]),
        "readout_term_mean": float(readout.mean()),
        "representation_term_mean": float(representation.mean()),
        "readout_term_centered_rms": float(np.std(readout)),
        "representation_term_centered_rms": float(np.std(representation)),
        "reconstruction_max_abs": float(
            np.max(np.abs(delta - bias - readout - representation))
        ),
    }


def self_check() -> None:
    pg = np.array([2.0, 3.0])
    truth = np.array([1.0, 5.0])
    residual = np.array([-0.25, 0.5])
    np.testing.assert_allclose((pg + residual) - truth, (pg - truth) + residual)
    h_a = np.array([[1.0, 2.0], [0.0, 3.0]])
    h_b = np.array([[2.0, 1.0], [4.0, 2.0]])
    w_a, w_b = np.array([2.0, -1.0]), np.array([1.0, 3.0])
    bias, readout, representation = split_head(h_a, h_b, w_a, w_b, -0.5, 0.75)
    np.testing.assert_allclose(
        bias + readout + representation,
        h_b @ w_b + 0.75 - h_a @ w_a + 0.5,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(bias, [1.25, 1.25])
    rows = [
        {"query_hash": "a", "n_tables": 4, "true": 10.0},
        {"query_hash": "b", "n_tables": 5, "true": 20.0},
    ]
    original = {
        "query_hash": np.array(["a", "b"]),
        "n_tables": np.array([4, 5]),
        "true": np.array([10.0, 20.0]),
        "log_pred": np.array([2.0, 3.0]),
    }
    validate_rows(rows, original)
    try:
        validate_rows(rows[::-1], original)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched query alignment accepted")
    print("self-check passed")


def analyze(project: Path, run_root: Path, device: str) -> dict:
    sys.path.insert(0, str(project))
    from scripts.arch_select import checkpoint_pool
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.cli import data_settings, read_json

    data = data_settings(
        read_json(project / "configs/data.json"), project / "configs/data.json"
    )
    if tuple(data["splits"]["validation"]) != DATABASES or set(
        data["splits"]["train"] + data["splits"]["validation"]
    ) & {"imdb", "stats", "genome", "ergastf1"}:
        raise ValueError("Source split changed or includes target databases")
    diagnosis_root = run_root / "diagnosis-20261004"
    old = read_json(diagnosis_root / "diagnosis.json")
    overlap = read_json(diagnosis_root / "tail_overlap.json")
    if sha256(diagnosis_root / "diagnosis.json") != overlap["diagnosis_sha256"]:
        raise ValueError("Prior diagnosis provenance changed")
    matrix = read_json(run_root / "ablation_runs.json")
    if {(row["arm"], row["seed"]) for row in matrix["runs"]} != {
        (arm, seed) for arm in ARMS for seed in SEEDS
    } or len(matrix["runs"]) != 9:
        raise ValueError("Frozen run matrix differs from nine planned checkpoints")
    report = {
        "purpose": "source-only posthoc PG and representation diagnostic; not selection or calibration",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(Path(__file__)),
        "training": False,
        "adaptation": False,
        "target_data_read": False,
        "databases": list(DATABASES),
        "device": device,
        "input_sha256": {
            "diagnosis.json": sha256(diagnosis_root / "diagnosis.json"),
            "tail_overlap.json": sha256(diagnosis_root / "tail_overlap.json"),
            "ablation_runs.json": sha256(run_root / "ablation_runs.json"),
        },
        "definitions": {
            "pg_error": "pg_log-log(max(true,1)), not q-error",
            "head_correction": "raw_log_pred-pg_log=w·h+b",
            "model_error": "pg_error+head_correction",
            "head_difference": "delta_bias+(wB-wA)·(hA+hB)/2+(hB-hA)·(wA+wB)/2",
            "interpretation": "Exact coordinate-dependent accounting, not causal attribution or a portable calibration",
        },
        "per_database": {},
    }
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for database in DATABASES:
        runs = {}
        for arm in ARMS:
            for seed in SEEDS:
                name = f"{arm}-seed{seed}"
                path = run_root / name / "last.pt"
                checkpoint_sha = sha256(path)
                if checkpoint_sha != old["runs"][name]["checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
                prediction_path = diagnosis_root / f"{name}-{database}.npz"
                if (
                    sha256(prediction_path)
                    != overlap["npz_sha256"][prediction_path.name]
                ):
                    raise ValueError(f"Prediction SHA256 mismatch: {name}-{database}")
                model, state = load_checkpoint(path, device)
                if state["next_step"] != 4000 or not model.use_pg:
                    raise ValueError("Not the frozen 4000-step PG-residual checkpoint")
                pool = checkpoint_pool(state, data)
                with np.load(prediction_path, allow_pickle=False) as npz:
                    original = {
                        key: npz[key]
                        for key in ("log_pred", "true", "n_tables", "query_hash")
                    }
                rows, pg, hidden = capture(
                    model, pool.get("pretrain", database), device
                )
                validate_rows(rows, original)
                pred = np.array([row["log_pred"] for row in rows])
                np.testing.assert_allclose(
                    pred, original["log_pred"], rtol=1e-6, atol=2e-5
                )
                weight = (
                    model.head[-1]
                    .weight.detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .astype(np.float64)
                )
                bias = float(model.head[-1].bias.detach().cpu().item())
                correction = hidden @ weight + bias
                np.testing.assert_allclose(pred - pg, correction, rtol=2e-5, atol=5e-5)
                runs = {
                    **runs,
                    name: {
                        "pg": pg,
                        "pred": pred,
                        "true": original["true"],
                        "n_tables": original["n_tables"],
                        "query_hash": original["query_hash"],
                        "hidden": hidden,
                        "weight": weight,
                        "bias": bias,
                    },
                }
                del model, pool
                print(f"verified {name}-{database}", flush=True)
        reference = runs["sum-seed42"]
        masks = groups(reference["true"], reference["n_tables"])
        if not masks["4plus"].any():
            raise ValueError("Primary source bucket empty")
        for name, run in runs.items():
            np.testing.assert_array_equal(reference["pg"], run["pg"])
            np.testing.assert_array_equal(reference["true"], run["true"])
            np.testing.assert_array_equal(reference["n_tables"], run["n_tables"])
            np.testing.assert_array_equal(reference["query_hash"], run["query_hash"])
        baseline = {
            key: endpoint(reference, mask)
            for key, mask in masks.items()
            if mask.sum() > 0
        }
        endpoints = {
            name: {
                key: endpoint(run, mask)
                for key, mask in masks.items()
                if mask.sum() > 0
            }
            for name, run in runs.items()
        }
        for name, stats in endpoints.items():
            if not np.isclose(
                stats["4plus"]["model_q95"],
                old["runs"][name]["per_database"][database]["primary"]["q95"],
                rtol=1e-7,
            ):
                raise ValueError(f"Frozen Q95 mismatch: {name}-{database}")
        pairs = {}
        for arm in ARMS:
            for a, b in ((42, 43), (42, 44), (43, 44)):
                key = f"{arm}-seed{a}--seed{b}"
                pairs = {
                    **pairs,
                    key: {
                        label: pair(
                            runs[f"{arm}-seed{a}"], runs[f"{arm}-seed{b}"], mask
                        )
                        for label, mask in masks.items()
                        if mask.sum() > 0
                    },
                }
        for seed in SEEDS:
            key = f"shuffled--true-seed{seed}"
            pairs = {
                **pairs,
                key: {
                    label: pair(
                        runs[f"shuffled-seed{seed}"], runs[f"true-seed{seed}"], mask
                    )
                    for label, mask in masks.items()
                    if mask.sum() > 0
                },
            }
        report["per_database"] = {
            **report["per_database"],
            database: {
                "baseline": baseline,
                "endpoints": endpoints,
                "pairs": pairs,
                "queries": len(reference["pg"]),
            },
        }
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["checks"] = {
        "checkpoint_hashes_verified": 9,
        "prior_prediction_hashes_verified": 27,
        "frozen_endpoint_q95_verified": 27,
        "pg_pairing_across_all_models": True,
        "head_reconstruction_and_delta_verified": True,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if args.project is None or args.run_root is None or args.output is None:
            parser.error("--project, --run-root and --output are required")
        report = analyze(args.project, args.run_root, args.device)
        payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(payload + "\n")
        print(f"wrote {args.output}")
