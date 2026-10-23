import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATABASES = ("chembl", "legalacts", "telstra")
VARIANTS = ("sum", "true", "shuffled")
SEEDS = (42, 43, 44)
CARDINALITY_BINS = (
    (10, 100),
    (100, 1000),
    (1000, 10000),
    (10000, 100000),
    (100000, np.inf),
)


def decompose(x: np.ndarray, y: np.ndarray) -> dict:
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("Need matching vectors with at least two elements")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite predictions")
    x_centered = x - x.mean()
    variance = np.mean(x_centered**2)
    delta = y - x
    shift = float(delta.mean())
    energy = float(np.mean(delta**2))
    slope = float(np.mean(x_centered * (y - y.mean())) / variance) if variance else None
    intercept = float(y.mean() - slope * x.mean()) if slope is not None else None
    residual_energy = (
        float(np.mean((y - intercept - slope * x) ** 2)) if slope is not None else None
    )
    y_variance = float(np.var(y))
    return {
        "n": len(x),
        "mean_shift": shift,
        "median_shift": float(np.median(delta)),
        "shift_p10_p90": np.quantile(delta, [0.1, 0.9]).tolist(),
        "increase_fraction": float(np.mean(delta > 0)),
        "rms": float(np.sqrt(energy)),
        "centered_rms": float(np.sqrt(np.mean((delta - shift) ** 2))),
        "shift_energy_fraction": shift**2 / energy if energy else None,
        "slope": slope,
        "intercept": intercept,
        "affine_rms": float(np.sqrt(residual_energy))
        if residual_energy is not None
        else None,
        "affine_delta_energy_fraction": 1 - residual_energy / energy
        if energy and residual_energy is not None
        else None,
        "prediction_r_squared": 1 - residual_energy / y_variance
        if y_variance and residual_energy is not None
        else None,
    }


def ranks(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def error_stats(prediction: np.ndarray, true: np.ndarray) -> dict:
    signed = np.maximum(prediction, 0) - np.log(np.maximum(true, 1))
    log_q = np.abs(signed)
    with np.errstate(over="raise", invalid="raise"):
        q = np.exp(log_q)
    return {
        "n": len(true),
        "q50": float(np.quantile(q, 0.5)),
        "q95": float(np.quantile(q, 0.95)),
        "q99": float(np.quantile(q, 0.99)),
        "mean_log_q": float(log_q.mean()),
        "under_fraction": float(np.mean(signed < 0)),
        "signed_mean": float(signed.mean()),
        "signed_median": float(np.median(signed)),
        "signed_p10_p90": np.quantile(signed, [0.1, 0.9]).tolist(),
    }


def groups(true: np.ndarray, tables: np.ndarray) -> dict:
    primary = tables >= 4
    return {
        "4plus": primary,
        **{f"tables={n}": primary & (tables == n) for n in np.unique(tables[primary])},
        **{
            f"true=[{low},{high})": primary & (true >= low) & (true < high)
            for low, high in CARDINALITY_BINS
        },
    }


def load_predictions(root: Path, expected_hashes: dict) -> tuple:
    paths = {
        f"{variant}-seed{seed}-{database}": root
        / f"{variant}-seed{seed}-{database}.npz"
        for database, variant, seed in itertools.product(DATABASES, VARIANTS, SEEDS)
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    for name, value in hashes.items():
        if value != expected_hashes[name + ".npz"]:
            raise ValueError(f"Prediction hash mismatch: {name}")
    arrays = {}
    for name, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            row = {
                key: data[key] for key in ("log_pred", "true", "n_tables", "query_hash")
            }
        shape = row["log_pred"].shape
        if (
            len(shape) != 1
            or not shape[0]
            or any(a.shape != shape for a in row.values())
        ):
            raise ValueError(f"Invalid array shape: {name}")
        if any(not np.isfinite(row[k]).all() for k in ("log_pred", "true", "n_tables")):
            raise ValueError(f"Nonfinite data: {name}")
        if (row["true"] < 0).any() or (row["n_tables"] < 1).any():
            raise ValueError(f"Invalid label or table count: {name}")
        if len(np.unique(row["query_hash"])) != shape[0]:
            raise ValueError(f"Duplicate query hashes: {name}")
        arrays = {**arrays, name: row}
    for database in DATABASES:
        reference = arrays[f"sum-seed42-{database}"]
        for variant, seed in itertools.product(VARIANTS, SEEDS):
            row = arrays[f"{variant}-seed{seed}-{database}"]
            for key in ("query_hash", "true", "n_tables"):
                if not np.array_equal(row[key], reference[key]):
                    raise ValueError(
                        f"Pairing mismatch: {database}/{variant}/{seed}/{key}"
                    )
    return arrays, hashes


def endpoint_stats(row: dict) -> dict:
    return {
        group: {
            **error_stats(row["log_pred"][mask], row["true"][mask]),
            "prediction_vs_truth": decompose(
                np.log(np.maximum(row["true"][mask], 1)), row["log_pred"][mask]
            ),
        }
        for group, mask in groups(row["true"], row["n_tables"]).items()
        if int(mask.sum()) >= 2
    }


def pair_stats(left: dict, right: dict) -> dict:
    masks = groups(left["true"], left["n_tables"])
    primary = masks["4plus"]
    global_shift = float((right["log_pred"] - left["log_pred"])[primary].mean())
    return {
        name: {
            **decompose(left["log_pred"][mask], right["log_pred"][mask]),
            "rank_spearman": float(
                np.corrcoef(
                    ranks(left["log_pred"][mask]), ranks(right["log_pred"][mask])
                )[0, 1]
            )
            if np.ptp(left["log_pred"][mask]) > 0
            and np.ptp(right["log_pred"][mask]) > 0
            else None,
            "mean_shift_minus_4plus_mean": float(
                (right["log_pred"] - left["log_pred"])[mask].mean() - global_shift
            ),
        }
        for name, mask in masks.items()
        if mask.sum() >= 2
    }


def analyze(root: Path) -> dict:
    overlap_path = root / "tail_overlap.json"
    diagnosis_path = root / "diagnosis.json"
    overlap = json.loads(overlap_path.read_text())
    if (
        hashlib.sha256(diagnosis_path.read_bytes()).hexdigest()
        != overlap["diagnosis_sha256"]
    ):
        raise ValueError("Diagnosis evidence hash mismatch")
    diagnosis = json.loads(diagnosis_path.read_text())
    arrays, hashes = load_predictions(root, overlap["npz_sha256"])
    endpoints = {name: endpoint_stats(row) for name, row in arrays.items()}
    for name, endpoint in endpoints.items():
        run, database = name.rsplit("-", 1)
        original = diagnosis["runs"][run]["per_database"][database]["primary"]
        for metric in ("q50", "q95", "q99", "mean_log_q", "under_fraction"):
            if not np.isclose(
                endpoint["4plus"][metric], original[metric], rtol=1e-7, atol=1e-10
            ):
                raise ValueError(f"Endpoint reproduction mismatch: {name}/{metric}")
    seed_pairs = {
        f"{variant}-seed{a}--seed{b}-{database}": pair_stats(
            arrays[f"{variant}-seed{a}-{database}"],
            arrays[f"{variant}-seed{b}-{database}"],
        )
        for database, variant in itertools.product(DATABASES, VARIANTS)
        for a, b in itertools.combinations(SEEDS, 2)
    }
    arm_pairs = {
        f"{left}--{right}-seed{seed}-{database}": pair_stats(
            arrays[f"{left}-seed{seed}-{database}"],
            arrays[f"{right}-seed{seed}-{database}"],
        )
        for database, seed in itertools.product(DATABASES, SEEDS)
        for left, right in (("shuffled", "true"), ("sum", "true"))
    }
    return {
        "purpose": "posthoc source-only raw log-prediction shift decomposition; not calibration",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root.resolve()),
        "databases": list(DATABASES),
        "training": False,
        "inference": False,
        "target_data_read": False,
        "checks": {
            "npz_hashes_verified": 27,
            "unique_query_pairing_verified": True,
            "reproduced_endpoints": 27,
            "self_check": True,
        },
        "analysis_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "diagnosis_sha256": overlap["diagnosis_sha256"],
        "tail_overlap_sha256": hashlib.sha256(overlap_path.read_bytes()).hexdigest(),
        "npz_sha256": hashes,
        "definitions": {
            "delta": "right.log_pred-left.log_pred, no transform to production predictions",
            "shift_energy_fraction": "mean(delta)^2/mean(delta^2), undefined at zero drift",
            "affine_delta_energy_fraction": "1-MSE(right-a-b*left)/mean(delta^2)",
            "prediction_vs_truth": "OLS raw log_pred=a+b*log(max(true,1)), descriptive on same sample",
            "q_error": "exp(abs(max(log_pred,0)-log(max(true,1))))",
            "interpretation": "No causal attribution, adjusted metrics, significance, or target extrapolation",
        },
        "endpoints": endpoints,
        "seed_pairs": seed_pairs,
        "arm_pairs": arm_pairs,
    }


def self_check() -> None:
    x = np.arange(3, dtype=np.float64)
    cases = (
        (x + 2, (2, 1, 0, 2, 1, 1)),
        (x - 2, (-2, 1, 0, -2, 1, 1)),
        (2 * x, (1, 0.6, np.sqrt(2 / 3), 0, 2, 1)),
        (np.array([0.0, 2.0, 1.0]), (0, 0, np.sqrt(2 / 3), 0.5, 0.5, 0.25)),
        (x, (0, None, 0, 0, 1, 1)),
    )
    for y, expected in cases:
        result = decompose(x, y)
        keys = (
            "mean_shift",
            "shift_energy_fraction",
            "centered_rms",
            "intercept",
            "slope",
            "prediction_r_squared",
        )
        for key, value in zip(keys, expected, strict=True):
            assert (
                result[key] is None if value is None else np.isclose(result[key], value)
            )
    assert decompose(x, np.ones(3))["prediction_r_squared"] is None
    constant = {
        "log_pred": np.ones(3),
        "true": np.ones(3) * 10,
        "n_tables": np.ones(3, dtype=int) * 4,
    }
    assert endpoint_stats(constant)["4plus"]["n"] == 3
    assert endpoint_stats(constant)["4plus"]["prediction_vs_truth"]["slope"] is None
    assert pair_stats(constant, constant)["4plus"]["rank_spearman"] is None
    np.testing.assert_allclose(ranks(np.array([2.0, 1.0, 2.0])), [1.5, 0, 1.5])
    stats = error_stats(np.array([-2.0, 0.0, 2.0]), np.array([0.0, np.e, 1.0]))
    assert np.isclose(stats["mean_log_q"], 1)
    for a, b in (
        (np.array([]), np.array([])),
        (x, x[:2]),
        (x, x * np.nan),
    ):
        try:
            decompose(a, b)
        except ValueError:
            continue
        raise AssertionError("Invalid decomposition input accepted")
    print("self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if args.input is None or args.output is None:
            parser.error("--input and --output are required unless --self-check")
        report = analyze(args.input)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized + "\n")
        print(f"Verified 27 endpoints; wrote {args.output}")
