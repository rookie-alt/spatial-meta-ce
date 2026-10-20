"""Pure helpers for the P2-2 direction-signal audit (numpy only).

Kept separate from ``scripts/direction_signal_audit.py`` so the cache/feature
plumbing can be unit-tested without scikit-learn, and so the audit script stays
within the project's file-size ceiling.

Nothing here trains a model; these functions only read a validated derived
cache and build labels/features/matrices.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
GROUP_DIMS = (40, 40, 4, 43)
CACHE_TRANSFORM = "price40-group-logfanout-v1"
TIE_TOLERANCE = 1e-9
TABLE_BUCKETS = ("1", "2-3", "4-5", "6+")
FEATURE_GROUPS = ("counts", "pg", "table", "filter", "join_hist", "fanout")


def assert_source_database(db: str) -> None:
    """Refuse a held-out target or a name that is not a bare database id."""
    if db in TARGETS:
        raise ValueError(f"Refusing to audit held-out target database {db!r}")
    if not db or db in {".", ".."} or "/" in db or "\\" in db or Path(db).name != db:
        raise ValueError(f"Invalid database name {db!r}")


def table_bucket(n_tables: int) -> str:
    if n_tables < 2:
        return "1"
    if n_tables < 4:
        return "2-3"
    if n_tables < 6:
        return "4-5"
    return "6+"


def direction_label(true: float, pred0: float) -> int | None:
    """1 = raise the prediction, 0 = lower it, None = exact tie."""
    if not math.isfinite(true) or true < 0:
        raise ValueError("Truth must be finite and nonnegative")
    if not math.isfinite(pred0):
        raise ValueError("K=0 prediction must be finite")
    residual = math.log(max(true, 1.0)) - pred0
    if residual > TIE_TOLERANCE:
        return 1
    if residual < -TIE_TOLERANCE:
        return 0
    return None


def log_error(pred: float, true: float) -> float:
    return abs(pred - math.log(max(true, 1.0)))


def auc_score(labels, scores) -> float:
    """Tie-aware ROC AUC; raises when one class is absent."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional and aligned")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Average rank (1-based) across ties.
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[y == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def bootstrap_auc_ci(labels, scores, reps: int = 1000, seed: int = 0) -> list:
    """Percentile 95% CI over query-level resamples; null when a class is lost."""
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        index = rng.integers(0, len(y), len(y))
        sample = y[index]
        if sample.min() == sample.max():
            continue
        draws.append(auc_score(sample, s[index]))
    if not draws:
        return [None, None]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_cache(cache_dir: Path) -> dict:
    """Validate one derived cache and return its arrays and row identities."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("transform") != CACHE_TRANSFORM:
        raise ValueError(f"Unsupported cache schema/transform in {cache_dir}")
    payload = {k: v for k, v in manifest.items() if k != "content_hash"}
    if manifest_digest(payload) != manifest.get("content_hash"):
        raise ValueError(f"Cache manifest content hash mismatch in {cache_dir}")
    offsets = np.load(cache_dir / "offsets.npy")
    cards = np.load(cache_dir / "cards.npy")
    groups = [np.load(cache_dir / f"group{i}.npy", mmap_mode="r") for i in range(4)]
    rows = manifest["rows"]
    if cards.shape != (len(rows), 2) or offsets.shape != (len(rows) + 1, 4):
        raise ValueError(f"Cache arrays disagree with manifest in {cache_dir}")
    for index, dim in enumerate(GROUP_DIMS):
        if groups[index].shape != (int(offsets[-1, index]), dim):
            raise ValueError(f"Invalid cache group dimensions in {cache_dir}")
    return {
        "manifest": manifest,
        "offsets": offsets,
        "cards": cards,
        "groups": groups,
        "rows": rows,
    }


def unique_rows(cache: dict) -> np.ndarray:
    """Indices of the first occurrence of each query_hash, in file order."""
    seen = set()
    result = []
    for index, row in enumerate(cache["rows"]):
        if row["query_hash"] not in seen:
            seen.add(row["query_hash"])
            result.append(index)
    return np.asarray(result, dtype=np.int64)


def block(cache: dict, row: int, group: int) -> np.ndarray:
    offsets = cache["offsets"]
    return np.asarray(
        cache["groups"][group][offsets[row, group] : offsets[row + 1, group]]
    )


def feature_row(cache: dict, row: int) -> tuple:
    """Summarise one query's four feature groups into a named flat vector."""
    offsets = cache["offsets"]
    names = ["n_join_cols", "n_fanouts", "n_tables", "n_filter_cols"]
    values = [
        float(offsets[row + 1, 0] - offsets[row, 0]),
        float(offsets[row + 1, 1] - offsets[row, 1]),
        float(offsets[row + 1, 2] - offsets[row, 2]),
        float(offsets[row + 1, 3] - offsets[row, 3]),
    ]
    names.append("pg_log")
    values.append(math.log(max(float(cache["cards"][row, 1]), 1.0)))

    table = block(cache, row, 2)
    for stat, column in (("logsize", 0), ("avi", 1), ("minsel", 2), ("ebo", 3)):
        if len(table):
            for reducer, label in (
                (np.mean, "mean"),
                (np.min, "min"),
                (np.max, "max"),
                (np.std, "std"),
            ):
                names.append(f"table_{stat}_{label}")
                values.append(float(reducer(table[:, column])))
            names.append(f"table_{stat}_sumlog")
            values.append(float(np.log(np.maximum(table[:, column], 1e-12)).sum()))
        else:
            for label in ("mean", "min", "max", "std", "sumlog"):
                names.append(f"table_{stat}_{label}")
                values.append(float("nan"))

    filters = block(cache, row, 3)
    filter_names = (
        "sel_mean",
        "sel_min",
        "sel_max",
        "sel_std",
        "width_mean",
        "width_min",
        "width_max",
        "hist_nonzero",
        "hist_max",
        "hist_logmean",
    )
    if len(filters):
        selectivity = filters[:, 42]
        width = filters[:, 41] - filters[:, 40]
        values += [
            float(selectivity.mean()),
            float(selectivity.min()),
            float(selectivity.max()),
            float(selectivity.std()),
            float(width.mean()),
            float(width.min()),
            float(width.max()),
            float((filters[:, :40] > 0).sum(axis=1).mean()),
            float(filters[:, :40].max(axis=1).mean()),
            float(np.log1p(filters[:, :40]).mean()),
        ]
    else:
        values += [float("nan")] * len(filter_names)
    names += [f"filter_{label}" for label in filter_names]

    joins = block(cache, row, 0)
    for column in range(40):
        names.append(f"join_hist_bin{column}")
        values.append(float(joins[:, column].mean()) if len(joins) else float("nan"))
    if len(joins):
        mass = joins / np.maximum(joins.sum(axis=1, keepdims=True), 1e-12)
        entropy = -(mass * np.log(mass + 1e-12)).sum(axis=1)
        values += [
            float(entropy.mean()),
            float(joins.max(axis=1).mean()),
            float((joins > 0).sum(axis=1).mean()),
        ]
    else:
        values += [float("nan")] * 3
    names += ["join_hist_entropy", "join_hist_max", "join_hist_nonzero"]

    fanout = block(cache, row, 1)
    fanout_names = ("mean", "max", "std", "frac_one", "logmean")
    if len(fanout):
        values += [
            float(fanout.mean()),
            float(fanout.max()),
            float(fanout.std()),
            float((fanout == 1).mean()),
            float(np.log1p(fanout).mean()),
        ]
    else:
        values += [float("nan")] * len(fanout_names)
    names += [f"fanout_{label}" for label in fanout_names]
    return names, values


def feature_group(name: str) -> str:
    if name.startswith("table_"):
        return "table"
    if name.startswith("filter_"):
        return "filter"
    if name.startswith("join_hist_"):
        return "join_hist"
    if name.startswith("fanout_"):
        return "fanout"
    if name == "pg_log":
        return "pg"
    return "counts"


def build_matrix(cache: dict, indices: np.ndarray) -> tuple:
    rows = [feature_row(cache, int(index)) for index in indices]
    names = rows[0][0]
    matrix = np.asarray([row[1] for row in rows], dtype=np.float32)
    if matrix.shape != (len(indices), len(names)):
        raise ValueError("Feature matrix shape mismatch")
    return matrix, names


def structural_signature(cache: dict, row: int) -> str:
    """Coarse near-duplicate identity: table stats plus filter selectivity.

    Two queries with the same tables, the same per-table selectivity summary
    and the same filter selectivity are effectively the same query template
    with different literals; grouping CV by query_hash alone would leak them
    across folds.

    Table rows are sorted before flattening so that join-order variants of the
    same template hash equally, and selectivity is rounded to two decimals
    because the cached values are derived estimates, not literals.
    """
    table = block(cache, row, 2)
    filters = block(cache, row, 3)
    offsets = cache["offsets"]
    table_key = ()
    if len(table):
        normalized = np.round(table[:, 1:], 2)
        table_key = tuple(normalized[np.lexsort(normalized.T[::-1])].ravel().tolist())
    filter_key = ()
    if len(filters):
        filter_key = tuple(sorted(np.round(filters[:, 42], 2).tolist()))
    key = (
        int(offsets[row + 1, 2] - offsets[row, 2]),
        int(offsets[row + 1, 0] - offsets[row, 0]),
        int(offsets[row + 1, 3] - offsets[row, 3]),
        table_key,
        filter_key,
    )
    return hashlib.sha256(repr(key).encode()).hexdigest()
