import json
from pathlib import Path

import numpy as np

from spatial_meta_ce.manifests import digest, file_hash
from spatial_meta_ce.metrics import q_values

ARMS = ("A_w64_smooth", "B_w64_mse", "C_w128_smooth", "D_w128_mse")
DATABASES = ("chembl", "legalacts", "telstra")
PAIRS = {
    "B-A": (ARMS[0], ARMS[1]),
    "D-C": (ARMS[2], ARMS[3]),
    "C-A": (ARMS[0], ARMS[2]),
    "D-B": (ARMS[1], ARMS[3]),
}
MIN_CELL_N = 200
MIN_Q99_N = 1000
FORMAL_NAME = "precision-screen-20261006"


def signed_errors(prediction: np.ndarray, truth: np.ndarray) -> tuple:
    pred = np.asarray(prediction, dtype=np.float64)
    true = np.asarray(truth, dtype=np.float64)
    q, _, log_q = q_values(pred, true)
    if not np.isfinite(q).all():
        raise FloatingPointError(
            "Q-error overflow; refusing clipped diagnostic results"
        )
    y = np.log(np.maximum(true, 1))
    return pred - y, np.maximum(pred, 0) - y, q, log_q


def metrics(
    prediction: np.ndarray, truth: np.ndarray, include_q99: bool = True
) -> dict:
    raw, official, q, _ = signed_errors(prediction, truth)
    absolute = np.abs(raw)
    result = {f"q{p}": float(np.percentile(q, p)) for p in (50, 90, 95)}
    if include_q99:
        result["q99"] = float(np.percentile(q, 99))
    return {
        **result,
        "n": len(q),
        "raw_log_mse": float(np.mean(raw**2)),
        "raw_log_mae": float(np.mean(absolute)),
        "raw_log_smooth_l1": float(
            np.mean(np.where(absolute < 1, 0.5 * raw**2, absolute - 0.5))
        ),
        "official_log_mae": float(np.mean(np.abs(official))),
        "raw_signed_log_error_mean": float(np.mean(raw)),
        "raw_signed_log_error_median": float(np.median(raw)),
        "official_over_fraction": float(np.mean(official > 0)),
        "official_under_fraction": float(np.mean(official < 0)),
        "official_exact_fraction": float(np.mean(official == 0)),
    }


def paired(
    reference: np.ndarray,
    candidate: np.ndarray,
    truth: np.ndarray,
    include_q99: bool = True,
) -> dict:
    ref_raw, ref_official, _, _ = signed_errors(reference, truth)
    cand_raw, cand_official, _, _ = signed_errors(candidate, truth)
    delta = np.abs(cand_official) - np.abs(ref_official)
    shift = np.asarray(candidate) - np.asarray(reference)
    quantiles = (0.1, 0.5, 0.9, 0.95)
    return {
        "n": len(delta),
        "reference": metrics(reference, truth, include_q99),
        "candidate": metrics(candidate, truth, include_q99),
        "improved_vs_reference_fraction": float(np.mean(delta < 0)),
        "worsened_vs_reference_fraction": float(np.mean(delta > 0)),
        "tied_vs_reference_fraction": float(np.mean(delta == 0)),
        "official_absolute_log_error_delta_mean": float(np.mean(delta)),
        "official_absolute_log_error_delta_p10_p50_p90_p95": np.quantile(
            delta, quantiles
        ).tolist(),
        "raw_squared_log_error_delta_mean": float(np.mean(cand_raw**2 - ref_raw**2)),
        "prediction_log_delta_mean": float(np.mean(shift)),
        "prediction_log_delta_p10_p50_p90_p95": np.quantile(shift, quantiles).tolist(),
        "prediction_log_delta_centered_sd": float(np.std(shift)),
    }


def cardinality_masks(truth: np.ndarray) -> dict:
    bounds = ((10, 100), (100, 1000), (1000, 10000), (10000, 100000), (100000, np.inf))
    return {
        "true=0": truth == 0,
        "true=(0,10)": (truth > 0) & (truth < 10),
        **{f"true=[{lo},{hi})": (truth >= lo) & (truth < hi) for lo, hi in bounds},
    }


def table_masks(tables: np.ndarray) -> dict:
    return {
        "tables=1-3": (tables >= 1) & (tables <= 3),
        "tables=4": tables == 4,
        "tables=5": tables == 5,
        "tables=6plus": tables >= 6,
    }


def describe_cell(data: dict, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    counts = {"n": n, "fraction_of_database": n / len(mask)}
    if n < MIN_CELL_N:
        return {**counts, "status": "below_min_cell_n"}
    preds = data["predictions"]
    arms = {
        arm: metrics(pred[mask], data["true"][mask], include_q99=n >= MIN_Q99_N)
        for arm, pred in preds.items()
    }
    comparisons = {}
    for arm in ARMS[1:]:
        comparison = paired(preds[ARMS[0]][mask], preds[arm][mask], data["true"][mask])
        if n < MIN_Q99_N:
            for key in ("reference", "candidate"):
                comparison[key] = {
                    k: v for k, v in comparison[key].items() if k != "q99"
                }
        comparisons[arm] = comparison
    return {
        **counts,
        "status": "descriptive",
        "arms": arms,
        "comparisons_vs_A": comparisons,
    }


def tail_migration(
    reference: np.ndarray, candidate: np.ndarray, truth: np.ndarray
) -> dict:
    _, _, ref_q, _ = signed_errors(reference, truth)
    _, _, cand_q, _ = signed_errors(candidate, truth)
    ta, tb = float(np.percentile(ref_q, 95)), float(np.percentile(cand_q, 95))
    a, b = ref_q >= ta, cand_q >= tb
    overlap, entered, left = a & b, ~a & b, a & ~b
    return {
        "n": len(truth),
        "reference_threshold_q95": ta,
        "candidate_threshold_q95": tb,
        "reference_tail_n": int(a.sum()),
        "candidate_tail_n": int(b.sum()),
        "reference_tail_fraction": float(a.mean()),
        "candidate_tail_fraction": float(b.mean()),
        "overlap_n": int(overlap.sum()),
        "entered_tail_n": int(entered.sum()),
        "left_reference_tail_n": int(left.sum()),
        "retained_reference_tail_fraction": float(overlap.sum() / a.sum()),
        "tail_jaccard": float(overlap.sum() / (a | b).sum()),
        "fixed_reference_tail_comparison": (
            {"n": int(a.sum()), "status": "below_min_cell_n"}
            if a.sum() < MIN_CELL_N
            else {
                "status": "descriptive",
                **paired(
                    reference[a],
                    candidate[a],
                    truth[a],
                    include_q99=a.sum() >= MIN_Q99_N,
                ),
            }
        ),
    }


def composition(data: dict, mask: np.ndarray, prediction: np.ndarray) -> dict:
    y = np.log(np.maximum(data["true"], 1))
    pred = np.maximum(prediction, 0)
    return {
        "n": int(mask.sum()),
        "cardinality_counts": {
            k: int((mask & v).sum()) for k, v in cardinality_masks(data["true"]).items()
        },
        "table_counts": {
            k: int((mask & v).sum()) for k, v in table_masks(data["n_tables"]).items()
        },
        "over_n": int(((pred > y) & mask).sum()),
        "under_n": int(((pred < y) & mask).sum()),
    }


def pg_residual(data: dict, mask: np.ndarray, include_q99: bool = True) -> dict:
    y = np.log(np.maximum(data["true"][mask], 1))
    pg_input, pg_ref = data["pg_input_log"][mask], data["pg_reference_log"][mask]
    if not np.isfinite(pg_input).all() or not np.isfinite(pg_ref).all():
        raise ValueError("Nonfinite PG input")
    pg_error = pg_input - y
    result = {
        "n": len(y),
        "pg_reference": metrics(pg_ref, data["true"][mask], include_q99),
        "pg_input_fp32": metrics(pg_input, data["true"][mask], include_q99),
        "pg_input_minus_reference_max_abs": float(np.max(np.abs(pg_input - pg_ref))),
        "arms": {},
    }
    for arm in ARMS:
        pred = data["predictions"][arm][mask]
        residual, raw_error = pred - pg_input, pred - y
        delta = np.abs(raw_error) - np.abs(pg_error)
        reconstruction = pg_error + residual - raw_error
        if np.max(np.abs(reconstruction)) > 2e-12:
            raise ValueError("PG input residual accounting mismatch")
        result["arms"][arm] = {
            "raw_signed_pg_error_mean": float(pg_error.mean()),
            "raw_signed_model_error_mean": float(raw_error.mean()),
            "residual_mean": float(residual.mean()),
            "residual_p10_p50_p90": np.quantile(residual, [0.1, 0.5, 0.9]).tolist(),
            "residual_opposes_pg_error_fraction": float(
                np.mean(residual * pg_error < 0)
            ),
            "residual_reinforces_pg_error_fraction": float(
                np.mean(residual * pg_error > 0)
            ),
            "overshoot_fraction": float(
                np.mean(
                    (pg_error * raw_error < 0) & (np.abs(raw_error) > np.abs(pg_error))
                )
            ),
            "improved_vs_pg_input_fraction": float(np.mean(delta < 0)),
            "worsened_vs_pg_input_fraction": float(np.mean(delta > 0)),
            "tied_vs_pg_input_fraction": float(np.mean(delta == 0)),
            "official_comparison_vs_pg_reference": paired(
                pg_ref, pred, data["true"][mask], include_q99
            ),
            "reconstruction_max_abs": float(np.max(np.abs(reconstruction))),
        }
    return result


def analyze_database(data: dict) -> dict:
    true, tables, preds = data["true"], data["n_tables"], data["predictions"]
    if set(preds) != set(ARMS):
        raise ValueError("Exact four-arm coverage required")
    if not len(true) or not np.isfinite(true).all() or (true < 0).any():
        raise ValueError("Invalid truth")
    if tables.shape != true.shape or (tables < 1).any():
        raise ValueError("Invalid table counts")
    masks = {
        "all_tables": np.ones(len(true), dtype=bool),
        "four_plus_tables": tables >= 4,
    }
    if not masks["four_plus_tables"].any():
        raise ValueError("Empty frozen primary bucket")
    global_metrics, pairs, tails, residuals = {}, {}, {}, {}
    for name, mask in masks.items():
        global_metrics[name] = {
            a: metrics(p[mask], true[mask]) for a, p in preds.items()
        }
        pairs[name] = {
            key: paired(preds[left][mask], preds[right][mask], true[mask])
            for key, (left, right) in PAIRS.items()
        }
        tails[name] = {}
        for arm in ARMS[1:]:
            a, b, t = preds[ARMS[0]][mask], preds[arm][mask], true[mask]
            report = tail_migration(a, b, t)
            aq, bq = signed_errors(a, t)[2], signed_errors(b, t)[2]
            old, new = (
                aq >= report["reference_threshold_q95"],
                bq >= report["candidate_threshold_q95"],
            )
            entered, left = (
                np.zeros(len(true), dtype=bool),
                np.zeros(len(true), dtype=bool),
            )
            entered[mask], left[mask] = ~old & new, old & ~new
            tails[name][f"{arm[0]}-A"] = {
                **report,
                "entered_composition": composition(data, entered, preds[arm]),
                "left_composition": composition(data, left, preds[arm]),
            }
        residuals[name] = pg_residual(data, mask)
    strata = {}
    cards, table_groups = cardinality_masks(true), table_masks(tables)
    for name, mask in {**cards, **table_groups}.items():
        strata[f"all:{name}"] = describe_cell(data, mask)
    for card, cm in cards.items():
        for table, tm in table_groups.items():
            strata[f"all:{card}|{table}"] = describe_cell(data, cm & tm)
    pg_conditioned = {}
    pg_error = data["pg_reference_log"] - np.log(np.maximum(true, 1))
    for bucket, base_mask in masks.items():
        for label, group_mask in {
            "pg_under": pg_error < 0,
            "pg_over": pg_error > 0,
            "pg_exact": pg_error == 0,
        }.items():
            mask = base_mask & group_mask
            n = int(mask.sum())
            pg_conditioned[f"{bucket}:{label}"] = (
                {"n": n, "status": "below_min_cell_n"}
                if n < MIN_CELL_N
                else {
                    "status": "descriptive",
                    **pg_residual(data, mask, include_q99=n >= MIN_Q99_N),
                }
            )
    return {
        "global_metrics": global_metrics,
        "factor_pairs": {
            key: {name: pairs[name][key] for name in masks} for key in PAIRS
        },
        "tail_migration": tails,
        "strata": strata,
        "pg_residual": residuals,
        "pg_conditioned": pg_conditioned,
    }


def safe_path(root: Path, path: Path) -> Path:
    absolute = path.absolute()
    if any(p.is_symlink() for p in (absolute, *absolute.parents)):
        raise ValueError("Symlink paths are forbidden")
    if ".." in path.parts or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes frozen input root")
    return path


def read_cache(path: Path, frozen: dict) -> dict:
    safe_path(path.parent, path)
    names = ("manifest.json", "cards.npy", "offsets.npy")
    for name in names:
        source = safe_path(path, path / name)
        if file_hash(source) != frozen["file_sha256"][name]:
            raise ValueError(f"Frozen cache bytes changed: {path.name}/{name}")
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        digest({k: v for k, v in manifest.items() if k != "content_hash"})
        != frozen["manifest_hash"]
    ):
        raise ValueError("Frozen cache manifest identity changed")
    if manifest["content_hash"] != frozen["manifest_hash"] or manifest.get(
        "provenance", {}
    ).get("excluded"):
        raise ValueError("Cache manifest or exclusion scope changed")
    cards = np.load(path / "cards.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(path / "offsets.npy", mmap_mode="r", allow_pickle=False)
    rows = manifest["rows"]
    if cards.shape != (len(rows), 2) or offsets.shape != (len(rows) + 1, 4):
        raise ValueError("Cache shape mismatch")
    if (
        not np.isfinite(cards).all()
        or (cards < 0).any()
        or offsets.dtype.kind not in "iu"
    ):
        raise ValueError("Invalid cache values")
    counts = np.diff(offsets, axis=0)
    if (counts < 0).any() or (counts[:, 2] < 1).any():
        raise ValueError("Invalid cache offsets or table count")
    first, indices = {}, []
    for i, row in enumerate(rows):
        key = row["query_hash"]
        if key not in first:
            first[key] = i
            indices.append(i)
        elif (
            not np.array_equal(cards[i], cards[first[key]])
            or counts[i, 2] != counts[first[key], 2]
        ):
            raise ValueError("Duplicate query has inconsistent truth, PG or tables")
    selected = cards[indices]
    pg_reference = np.log(np.maximum(selected[:, 1], 1))
    return {
        "query_hash": [rows[i]["query_hash"] for i in indices],
        "true": np.asarray(selected[:, 0], dtype=np.float64),
        "n_tables": np.asarray(counts[indices, 2], dtype=np.int64),
        "pg_reference_log": pg_reference,
        "pg_input_log": pg_reference.astype(np.float32).astype(np.float64),
    }


def read_predictions(path: Path, caches: dict, expected_hash: str) -> dict:
    if file_hash(path) != expected_hash:
        raise ValueError("Frozen prediction file changed")
    indexes = {
        db: {h: i for i, h in enumerate(data["query_hash"])}
        for db, data in caches.items()
    }
    predictions = {
        db: np.empty(len(data["true"]), dtype=np.float64) for db, data in caches.items()
    }
    seen = {db: np.zeros(len(data["true"]), dtype=bool) for db, data in caches.items()}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            db = row["database"]
            if db not in DATABASES or db not in caches:
                raise ValueError("Prediction contains a target or unknown database")
            key = row["query_hash"]
            if key not in indexes[db]:
                raise ValueError("Prediction query is missing from frozen cache")
            i = indexes[db][key]
            if seen[db][i]:
                raise ValueError("Duplicate prediction identity")
            if (
                row["true"] != caches[db]["true"][i]
                or row["n_tables"] != caches[db]["n_tables"][i]
            ):
                raise ValueError("Prediction truth or table count changed")
            value = row["log_pred"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
            ):
                raise ValueError("Nonfinite or nonnumeric prediction")
            predictions[db][i], seen[db][i] = value, True
    if not all(mask.all() for mask in seen.values()):
        raise ValueError("Incomplete prediction coverage")
    if file_hash(path) != expected_hash:
        raise ValueError("Prediction changed during read")
    return predictions


def validate_output(path: Path, root: Path) -> Path:
    safe_path(root / "runs", path)
    resolved = path.resolve()
    runs, formal = (root / "runs").resolve(), (root / "runs" / FORMAL_NAME).resolve()
    if (
        resolved == runs
        or resolved.is_relative_to(formal)
        or formal.is_relative_to(resolved)
    ):
        raise ValueError("Output overlaps existing input directory")
    if path.exists():
        raise FileExistsError("Use a new S1 output directory")
    return resolved
