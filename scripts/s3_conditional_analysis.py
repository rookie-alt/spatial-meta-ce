"""Streaming source-only S3 conditional residual diagnostics (NumPy + stdlib).

Inputs are verified, first-query-hash unique caches and the certified S2 replay.
No IO, model imports, training, or input mutation occurs here.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

import numpy as np

CARDINALITY = (
    "0",
    "(0,10)",
    "[10,100)",
    "[100,1000)",
    "[1000,10000)",
    "[10000,100000)",
    "[100000,inf)",
)
TABLES = ("1-3", "4", "5", "6+")
MIN_UNIQUE = 200
UNDEFINED = "undefined"
STAT_FIELDS = tuple(
    f"{kind}_{stat}"
    for kind in ("r_ref", "r_fp32")
    for stat in ("mean", "median", "p10", "p90")
) + ("positive_share", "negative_share", "zero_share")


def _ratio(numerator: float, denominator: float) -> float | str:
    return float(numerator / denominator) if denominator else UNDEFINED


def _bins(values: np.ndarray) -> np.ndarray:
    return np.where(
        values == 0,
        0,
        np.searchsorted([10, 100, 1000, 10000, 100000], values, side="right") + 1,
    )


def _integer(values: np.ndarray, minimum: int = 0) -> bool:
    return np.issubdtype(values.dtype, np.integer) and bool(np.all(values >= minimum))


def _prepare(cache: dict) -> dict:
    true = np.asarray(cache["true"], dtype=np.float64)
    pg = np.asarray(cache["pg"], dtype=np.float64)
    tables = np.asarray(cache["n_tables"])
    tokens = np.asarray(cache["token_counts"])
    indices = np.asarray(cache["indices"])
    n = len(true) if true.ndim == 1 else 0
    if (
        not n
        or pg.shape != (n,)
        or tables.shape != (n,)
        or tokens.shape != (n, 4)
        or indices.shape != (n,)
        or not _integer(tables, 1)
        or not _integer(tokens)
        or not _integer(indices)
        or len(cache["query_hash"]) != n
        or not isinstance(cache["raw_rows"], (int, np.integer))
        or cache["raw_rows"] < n
        or np.any(indices >= cache["raw_rows"])
        or np.any(~np.isfinite(true))
        or np.any(~np.isfinite(pg))
        or np.any(true < 0)
        or np.any(pg < 0)
    ):
        raise ValueError("Invalid unique cache labels, indices, or token counts")
    lt, lp = np.log(np.maximum(true, 1)), np.log(np.maximum(pg, 1))
    card, table = _bins(true), np.minimum(np.maximum(tables - 3, 0), 3)
    return {
        "ref": lt - lp,
        "fp32": lt.astype(np.float32).astype(np.float64)
        - lp.astype(np.float32).astype(np.float64),
        "cardinality": card,
        "n_tables": table,
        "cross": card * 4 + table,
        "keys": np.column_stack((_bins(pg), tokens)),
        "n": n,
    }


def _replay_weights(state: dict, n: int) -> dict:
    arrays = {
        key: np.asarray(state[key], dtype=np.float64)
        for key in ("support", "query", "mass")
    }
    for key, value in arrays.items():
        if (
            value.shape != (n,)
            or np.any(~np.isfinite(value))
            or np.any(value < 0)
            or (key != "mass" and np.any(value != np.floor(value)))
        ):
            raise ValueError("Invalid replay weights or role counts")
    combined = arrays["support"] + arrays["query"]
    if np.any(~np.isfinite(combined)):
        raise ValueError("Nonfinite combined presentation counts")
    return {"combined_presentations": combined, "nominal_mass": arrays["mass"]}


def _layers() -> list[tuple[str, str, int]]:
    return [("all", "all", 0)] + [
        (axis, label, i)
        for axis, labels in (
            ("cardinality", CARDINALITY),
            ("n_tables", TABLES),
            ("cross", tuple(f"{c}|{t}" for c in CARDINALITY for t in TABLES)),
        )
        for i, label in enumerate(labels)
    ]


def _select(data: dict, names: Sequence[str], axis: str, index: int) -> tuple:
    selected = {
        db: np.arange(data[db]["n"])
        if axis == "all"
        else np.flatnonzero(data[db][axis] == index)
        for db in names
    }
    keys = np.unique(
        np.concatenate([data[db]["keys"][selected[db]] for db in names]), axis=0
    )
    # One vectorized unique operation maps all layer rows, never scan per cell.
    lengths = [len(selected[db]) for db in names]
    if len(keys):
        _, inverse = np.unique(
            np.concatenate([data[db]["keys"][selected[db]] for db in names]),
            axis=0,
            return_inverse=True,
        )
    else:
        inverse = np.empty(0, dtype=np.int64)
    ids = dict(zip(names, np.split(inverse, np.cumsum(lengths)[:-1])))
    return selected, keys, ids


def _quantiles(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.quantile(values, [0.1, 0.5, 0.9], method="linear")
    positive = weights > 0
    order = np.argsort(values[positive], kind="stable")
    sorted_values, sorted_weights = values[positive][order], weights[positive][order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    locations = np.searchsorted(
        cumulative, np.array([0.1, 0.5, 0.9]) * cumulative[-1], side="left"
    )
    return sorted_values[np.minimum(locations, len(sorted_values) - 1)]


def _stats(
    ref: np.ndarray,
    fp32: np.ndarray,
    weights: np.ndarray,
    unweighted: bool,
    sufficient: bool,
) -> dict:
    if not sufficient:
        return dict.fromkeys(STAT_FIELDS)
    mass = float(weights.sum())
    statistics = {}
    for values, label in ((ref, "r_ref"), (fp32, "r_fp32")):
        p10, median, p90 = _quantiles(values, None if unweighted else weights)
        statistics.update(
            {
                f"{label}_mean": float(np.dot(values, weights) / mass),
                f"{label}_median": float(median),
                f"{label}_p10": float(p10),
                f"{label}_p90": float(p90),
            }
        )
    return {
        **statistics,
        **{
            f"{sign}_share": float(weights[mask].sum() / mass)
            for sign, mask in (
                ("positive", ref > 0),
                ("negative", ref < 0),
                ("zero", ref == 0),
            )
        },
    }


def _group(
    data: dict,
    selected: dict,
    ids: dict,
    names: Sequence[str],
    population: str,
    replay: dict,
    nkeys: int,
) -> dict:
    cid = np.concatenate([ids[db] for db in names])
    ref = np.concatenate([data[db]["ref"][selected[db]] for db in names])
    fp32 = np.concatenate([data[db]["fp32"][selected[db]] for db in names])
    weights = np.concatenate(
        [
            np.ones(len(selected[db]))
            if population == "pool_unique"
            else np.full(len(selected[db]), 1 / (len(names) * data[db]["n"]))
            if population == "db_equal_pool"
            else replay[db][population][selected[db]]
            for db in names
        ]
    )
    observed = (
        np.ones(len(cid), dtype=bool)
        if population in ("pool_unique", "db_equal_pool")
        else np.concatenate(
            [replay[db]["combined_presentations"][selected[db]] > 0 for db in names]
        )
    )
    counts = np.bincount(cid, minlength=nkeys).astype(np.int64)
    mass = np.zeros(nkeys, dtype=np.float64)
    sums = np.zeros(nkeys, dtype=np.float64)
    if len(cid):
        order = np.argsort(cid, kind="stable")
        present = np.flatnonzero(counts)
        starts = np.concatenate(([0], np.cumsum(counts)))[:-1][present]
        # Segmented FP64 reduction avoids bincount's serial weighted accumulation.
        mass[present] = np.add.reduceat(weights[order], starts, dtype=np.float64)
        sums[present] = np.add.reduceat(
            (weights * ref)[order], starts, dtype=np.float64
        )
    db_counts = {
        db: np.bincount(ids[db], minlength=nkeys).astype(np.int64) for db in names
    }
    return {
        "ids": cid,
        "ref": ref,
        "fp32": fp32,
        "weights": weights,
        "counts": counts,
        "observed": np.bincount(cid[observed], minlength=nkeys).astype(np.int64),
        "mass": mass,
        "sums": sums,
        "db_counts": db_counts,
        "unweighted": population == "pool_unique",
    }


def _rows(
    group: dict, name: str, population: str, axis: str, bucket: str, keys: np.ndarray
) -> tuple[list[dict], list[dict]]:
    conditions, residuals = [], []
    order = np.argsort(group["ids"], kind="stable")
    starts = np.concatenate(([0], np.cumsum(group["counts"])))
    coverage = sum((counts > 0).astype(int) for counts in group["db_counts"].values())
    group_pool_unique = int(group["counts"].sum())
    group_weight_mass = float(group["mass"].sum())
    for i, key in enumerate(keys):
        pool, observed = int(group["counts"][i]), int(group["observed"][i])
        mass = float(group["mass"][i])
        common = {
            "database": name,
            "population": population,
            "axis": axis,
            "bucket": bucket,
            "condition": ",".join(str(int(x)) for x in key),
            "pg_bucket": CARDINALITY[int(key[0])],
            **dict(
                zip(
                    (
                        "n_join_column_tokens",
                        "n_fanout_tokens",
                        "n_table_tokens",
                        "n_filter_tokens",
                    ),
                    map(int, key[1:]),
                )
            ),
            "pool_unique": pool,
            "observed_unique": observed,
            "group_pool_unique": group_pool_unique,
            "weight_mass": mass,
            "group_weight_mass": group_weight_mass,
            "share": _ratio(mass, group_weight_mass),
            "source_db_coverage": int(coverage[i]),
        }
        sufficient = pool >= MIN_UNIQUE and observed >= MIN_UNIQUE and mass > 0
        pos = order[starts[i] : starts[i + 1]]
        conditions.append(common)
        residuals.append(
            {
                **common,
                "status": "ok" if sufficient else "sparse",
                "quantile_method": "linear" if group["unweighted"] else "inverse_ecdf",
                **_stats(
                    group["ref"][pos],
                    group["fp32"][pos],
                    group["weights"][pos],
                    group["unweighted"],
                    sufficient,
                ),
            }
        )
    return conditions, residuals


def _mean(
    group: dict, mask: np.ndarray, require_observed: bool = False
) -> float | None:
    mass = float(group["mass"][mask].sum())
    if (
        group["counts"][mask].sum() < MIN_UNIQUE
        or mass <= 0
        or (require_observed and group["observed"][mask].sum() < MIN_UNIQUE)
    ):
        return None
    return float(group["sums"][mask].sum() / mass)


def _alignment(
    source: dict, target: dict, population: str, database: str, axis: str, bucket: str
) -> dict:
    common = (
        (source["counts"] >= MIN_UNIQUE)
        & (target["counts"] >= MIN_UNIQUE)
        & (source["observed"] >= MIN_UNIQUE)
        & (source["mass"] > 0)
        & (target["mass"] > 0)
    )
    full = np.ones(len(common), dtype=bool)
    coverage = {
        db: int(counts[common].sum()) for db, counts in target["db_counts"].items()
    }
    status = "ok" if common.any() else "no_common_support"
    if database == "validation_db_equal_pool" and common.any():
        adequate = all(
            coverage[db] >= MIN_UNIQUE and bool(np.any(common & (counts >= MIN_UNIQUE)))
            for db, counts in target["db_counts"].items()
        )
        if not adequate:
            status = "incomplete_validation_support"
    full_train, full_validation = _mean(source, full, True), _mean(target, full)
    source_unique, target_unique = (
        int(source["counts"].sum()),
        int(target["counts"].sum()),
    )
    source_mass, target_mass = float(source["mass"].sum()), float(target["mass"].sum())
    sc, tc = int(source["counts"][common].sum()), int(target["counts"][common].sum())
    sm, tm = float(source["mass"][common].sum()), float(target["mass"][common].sum())
    result = {
        "axis": axis,
        "bucket": bucket,
        "source_population": population,
        "validation_database": database,
        "status": status,
        "common_conditions": int(common.sum()),
        "source_pool_unique": source_unique,
        "validation_pool_unique": target_unique,
        "source_observed_unique": int(source["observed"].sum()),
        "source_weight_mass": source_mass,
        "validation_weight_mass": target_mass,
        "source_common_unique": sc,
        "validation_common_unique": tc,
        "source_common_observed_unique": int(source["observed"][common].sum()),
        "source_common_weight_mass": sm,
        "validation_common_weight_mass": tm,
        "source_unique_coverage": _ratio(sc, source_unique),
        "validation_unique_coverage": _ratio(tc, target_unique),
        "source_mass_coverage": _ratio(sm, source_mass),
        "validation_mass_coverage": _ratio(tm, target_mass),
        "source_db_coverage": sum(
            bool(counts[common].sum()) for counts in source["db_counts"].values()
        ),
        "validation_db_coverage": sum(n > 0 for n in coverage.values()),
        "validation_db_coverage_json": json.dumps(
            coverage, ensure_ascii=False, separators=(",", ":")
        ),
        "full_train_mean": full_train,
        "full_validation_mean": full_validation,
        "full_mean_difference": None
        if full_train is None or full_validation is None
        else full_validation - full_train,
        **dict.fromkeys(
            (
                "common_train_mean",
                "common_validation_mean",
                "common_mean_difference",
                "d_signed",
                "d_absolute",
            )
        ),
    }
    if status != "ok":
        return result
    train_mean, val_mean = _mean(source, common, True), _mean(target, common)
    difference = target["sums"][common] / target["mass"][common] - (
        source["sums"][common] / source["mass"][common]
    )
    q = target["mass"][common] / tm
    return {
        **result,
        "common_train_mean": train_mean,
        "common_validation_mean": val_mean,
        "common_mean_difference": val_mean - train_mean,
        "d_signed": float(np.dot(q, difference)),
        "d_absolute": float(np.dot(q, np.abs(difference))),
    }


def iter_analysis(
    caches: dict, replay: dict, train: Sequence[str], validation: Sequence[str]
) -> Iterator[dict]:
    """Yield one g chunk at a time; all groups retain that g's complete C union.

    Database sequences define deterministic row/JSON order. The caller must consume
    chunks without retaining all 40 tables if bounded output memory is required.
    """
    names = tuple(train) + tuple(validation)
    if (
        not train
        or not validation
        or len(set(names)) != len(names)
        or set(caches) != set(names)
    ):
        raise ValueError("Require nonempty, disjoint train/validation covering caches")
    data = {db: _prepare(caches[db]) for db in names}
    states = replay["prefixes"][4000]["databases"]
    weights = {db: _replay_weights(states[db], data[db]["n"]) for db in train}
    specs = (
        [(db, (db,), "pool_unique") for db in names]
        + [
            ("train_pooled", tuple(train), "pool_unique"),
            ("train_db_equal_pool", tuple(train), "db_equal_pool"),
            ("validation_db_equal_pool", tuple(validation), "db_equal_pool"),
        ]
        + [
            (db, (db,), pop)
            for db in (*train, "train_pooled")
            for pop in ("combined_presentations", "nominal_mass")
        ]
    )
    sources = (
        ("train_pooled", "pool_unique", "train_pooled_pool"),
        ("train_db_equal_pool", "db_equal_pool", "train_db_equal_pool"),
        ("train_pooled", "combined_presentations", "combined_presentations"),
        ("train_pooled", "nominal_mass", "nominal_mass"),
    )
    for axis, bucket, index in _layers():
        selected, keys, ids = _select(data, names, axis, index)
        chunk = {
            "axis": axis,
            "bucket": bucket,
            "conditions": [],
            "residuals": [],
            "alignment": [],
        }
        summaries = {}
        for name, dbs, population in specs:
            members = tuple(train) if name == "train_pooled" else dbs
            group = _group(data, selected, ids, members, population, weights, len(keys))
            conditions, residuals = _rows(group, name, population, axis, bucket, keys)
            chunk["conditions"].extend(conditions)
            chunk["residuals"].extend(residuals)
            summaries[name, population] = {
                key: value
                for key, value in group.items()
                if key in ("counts", "observed", "mass", "sums", "db_counts")
            }
        for name, pop, source_label in sources:
            for db in (*validation, "validation_db_equal_pool"):
                target_pop = (
                    "db_equal_pool"
                    if db == "validation_db_equal_pool"
                    else "pool_unique"
                )
                chunk["alignment"].append(
                    _alignment(
                        summaries[name, pop],
                        summaries[db, target_pop],
                        source_label,
                        db,
                        axis,
                        bucket,
                    )
                )
        # Do not retain the last group's raw sorted samples across yielded chunks.
        del group, conditions, residuals, summaries, selected, keys, ids
        yield chunk
        del chunk
