"""Source-only descriptive S7 statistics. No model/training imports or selection."""

import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from scripts.s2_source_replay import TRAIN

MIN_N = 200
ARMS = ("A", "B")
CARD_EDGES = (0, 1, 10, 100, 1000, 10000, 100000, np.inf)


def write_json(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def errors(pred, true):
    pred, true = np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64)
    if (
        pred.ndim != 1
        or pred.shape != true.shape
        or not np.isfinite(pred).all()
        or not np.isfinite(true).all()
        or (true < 0).any()
    ):
        raise ValueError("Invalid prediction/true arrays")
    label = np.log(np.maximum(true, 1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.exp(np.abs(np.maximum(pred, 0.0) - label))
        raw = pred - label
    if not np.isfinite(q).all() or not np.isfinite(raw * raw).all():
        raise ValueError("Nonfinite standard Q or raw log error")
    return q, raw, np.maximum(pred, 0.0) - label


def metrics(pred, true):
    q, raw, signed = errors(pred, true)
    if not len(q):
        return {"n": 0, "status": "empty"}
    return {
        "n": len(q),
        "q50": float(np.percentile(q, 50)),
        "q95": float(np.percentile(q, 95)),
        "raw_log_mae": float(np.abs(raw).mean()),
        "raw_log_mse": float((raw * raw).mean()),
        "raw_signed_log_error": float(raw.mean()),
        "under_fraction": float((signed < 0).mean()),
        "over_fraction": float((signed > 0).mean()),
        "exact_fraction": float((signed == 0).mean()),
    }


def validate(data):
    n = len(data["query_hash"])
    if (
        not n
        or len(set(data["query_hash"])) != n
        or any(not isinstance(h, str) or not h for h in data["query_hash"])
    ):
        raise ValueError("Invalid or duplicate query identity")
    for key in ("true", "pg", "n_tables", *ARMS):
        a = np.asarray(data[key])
        if a.shape != (n,) or not np.isfinite(a).all():
            raise ValueError(f"Invalid array: {key}")
    if (
        (data["pg"] < 0).any()
        or (data["n_tables"] < 1).any()
        or not np.equal(data["n_tables"], np.floor(data["n_tables"])).all()
    ):
        raise ValueError("Invalid PG or table count")
    for arm in ARMS:
        errors(data[arm], data["true"])


def pair(data, mask, min_n=MIN_N):
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != data["true"].shape:
        raise ValueError("Invalid cohort mask")
    n = int(mask.sum())
    if n < min_n:
        return {"n": n, "status": "below_min_cell_n"}
    truth = data["true"][mask]
    a, b = data["A"][mask], data["B"][mask]
    qa, _, _ = errors(a, truth)
    qb, _, _ = errors(b, truth)
    pg = np.log(np.maximum(data["pg"][mask], 1))
    qpg, _, _ = errors(pg, truth)
    return {
        "n": n,
        "status": "descriptive",
        "A": metrics(a, truth),
        "B": metrics(b, truth),
        "PG": metrics(pg, truth),
        "q95_relative_change_b_vs_a": float(
            np.percentile(qb, 95) / np.percentile(qa, 95) - 1
        ),
        "improved_fraction_b_vs_a": float((qb < qa).mean()),
        "worse_fraction_b_vs_a": float((qb > qa).mean()),
        "unchanged_fraction_b_vs_a": float((qb == qa).mean()),
        "improved_vs_pg": {
            arm: float((q < qpg).mean()) for arm, q in zip(ARMS, (qa, qb))
        },
        "mean_raw_prediction_shift_b_minus_a": float((b - a).mean()),
        "mean_correction_vs_pg": {
            "A": float((a - pg).mean()),
            "B": float((b - pg).mean()),
        },
    }


def cardinality_masks(true):
    return {
        f"[{low},{high})": (true >= low) & (true < high)
        for low, high in pairwise(CARD_EDGES)
    }


def direction_masks(pred, true):
    _, _, signed = errors(pred, true)
    return {"under": signed < 0, "over": signed > 0, "exact": signed == 0}


def tail_masks(data, base):
    qa, _, _ = errors(data["A"], data["true"])
    qb, _, _ = errors(data["B"], data["true"])
    if not base.any():
        return {
            k: np.zeros_like(base) for k in ("persistent", "entered", "left", "neither")
        }
    a = base & (qa >= np.percentile(qa[base], 95))
    b = base & (qb >= np.percentile(qb[base], 95))
    return {
        "persistent": a & b,
        "entered": ~a & b,
        "left": a & ~b,
        "neither": base & ~a & ~b,
    }


def composition(data, mask):
    return {
        "n": int(mask.sum()),
        "cardinality_counts": {
            k: int((mask & m).sum()) for k, m in cardinality_masks(data["true"]).items()
        },
        "b_error_direction_counts": {
            k: int((mask & m).sum())
            for k, m in direction_masks(data["B"], data["true"]).items()
        },
        "a_error_direction_counts": {
            k: int((mask & m).sum())
            for k, m in direction_masks(data["A"], data["true"]).items()
        },
        "pg_error_direction_counts": {
            k: int((mask & m).sum())
            for k, m in direction_masks(
                np.log(np.maximum(data["pg"], 1)), data["true"]
            ).items()
        },
        "stats": pair(data, mask),
    }


def legalacts_report(data):
    validate(data)
    base = data["n_tables"] >= 4
    qa, _, _ = errors(data["A"], data["true"])
    qb, _, _ = errors(data["B"], data["true"])
    strata = []
    axes = {
        "true": cardinality_masks(data["true"]),
        "pg_direction": direction_masks(
            np.log(np.maximum(data["pg"], 1)), data["true"]
        ),
        "a_direction": direction_masks(data["A"], data["true"]),
    }
    for axis, masks in axes.items():
        for name, mask in masks.items():
            strata.append({"axis": axis, "bucket": name, **pair(data, base & mask)})
    for card, cm in axes["true"].items():
        for pg, pm in axes["pg_direction"].items():
            strata.append(
                {
                    "axis": "true_x_pg_direction",
                    "bucket": f"{card}|{pg}",
                    **pair(data, base & cm & pm),
                }
            )
    tails = tail_masks(data, base)
    near = np.zeros_like(base)
    if base.any():
        near = (
            base
            & (qb >= np.percentile(qb[base], 94))
            & (qb <= np.percentile(qb[base], 96))
        )
    return {
        "purpose": "posthoc-legalacts-not-selection",
        "all": pair(data, np.ones(len(base), dtype=bool)),
        "four_plus": pair(data, base),
        "table_counts": {
            str(k): int((base & (data["n_tables"] == k)).sum())
            for k in np.unique(data["n_tables"][base])
        },
        "tail_groups": [
            {"group": name, **composition(data, mask)} for name, mask in tails.items()
        ],
        "old_tail_stats": pair(data, tails["persistent"] | tails["left"]),
        "thresholds": {
            "A_q95": float(np.percentile(qa[base], 95)),
            "B_q95": float(np.percentile(qb[base], 95)),
        }
        if base.any()
        else {},
        "q95_neighborhood": composition(data, near),
        "strata": strata,
        "causal_explanation_verified": False,
        "warning": "Quantiles are not additive; strata overlap across axes; small cells only count.",
    }


def deterministic_sample(db, cohort, hashes, candidates, limit):
    if (
        db not in TRAIN
        or cohort not in ("seen", "unseen")
        or type(limit) is not int
        or limit < 1
    ):
        raise ValueError("Invalid source database, cohort or sample limit")
    candidates = [int(i) for i in candidates]
    if len(set(candidates)) != len(candidates) or any(
        i < 0 or i >= len(hashes) for i in candidates
    ):
        raise ValueError("Invalid sample candidates")
    return sorted(
        candidates,
        key=lambda i: hashlib.sha256(
            f"s7-fixed-v1|{db}|{cohort}|{hashes[i]}".encode()
        ).digest(),
    )[:limit]


def memberships(roles):
    a, b = np.asarray(roles["support"]), np.asarray(roles["query"])
    if (
        a.ndim != 1
        or a.shape != b.shape
        or any(v.dtype.kind not in "iu" or (v < 0).any() for v in (a, b))
    ):
        raise ValueError("Invalid exposure counts")
    return np.flatnonzero((a + b) > 0), np.flatnonzero((a + b) == 0)
