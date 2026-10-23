"""Pure NumPy S2 tables. Inputs are already source-verified, per-database unique rows."""

from __future__ import annotations

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


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: list[float]
) -> np.ndarray:
    """Inverse weighted empirical CDF, excluding zero-weight observations."""
    if (
        values.shape != weights.shape
        or np.any(~np.isfinite(weights))
        or np.any(weights < 0)
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("Invalid weighted quantile input")
    positive = weights > 0
    if not np.any(positive):
        raise ValueError("Weighted quantiles require positive mass")
    order = np.argsort(values[positive], kind="stable")
    sorted_values = values[positive][order]
    cumulative = np.cumsum(weights[positive][order], dtype=np.float64)
    q = np.asarray(probabilities, dtype=np.float64)
    if np.any(~np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    indices = np.searchsorted(cumulative, q * cumulative[-1], side="left")
    return sorted_values[np.minimum(indices, len(sorted_values) - 1)]


def _ratio(numerator: float, denominator: float) -> float | str:
    return float(numerator / denominator) if denominator else UNDEFINED


def _prepare(cache: dict) -> dict:
    true, pg = np.asarray(cache["true"]), np.asarray(cache["pg"])
    n_tables = np.asarray(cache["n_tables"])
    if (
        true.ndim != 1
        or not len(true)
        or pg.shape != true.shape
        or n_tables.shape != true.shape
        or np.any(~np.isfinite(true))
        or np.any(~np.isfinite(pg))
        or np.any(true < 0)
        or np.any(pg < 0)
        or np.any(~np.isfinite(n_tables))
        or np.any(n_tables < 1)
        or np.any(n_tables != np.floor(n_tables))
    ):
        raise ValueError("Invalid unique cache labels or table counts")
    log_true = np.log(np.maximum(true.astype(np.float64), 1))
    log_pg = np.log(np.maximum(pg.astype(np.float64), 1))
    card = np.where(
        true == 0,
        0,
        np.searchsorted([10, 100, 1000, 10000, 100000], true, side="right") + 1,
    )
    table = np.clip(n_tables - 3, 0, 3).astype(int)
    return {
        "ref": log_true - log_pg,
        "fp32": log_true.astype(np.float32).astype(np.float64)
        - log_pg.astype(np.float32).astype(np.float64),
        "cardinality": card,
        "n_tables": table,
        "cross": card * 4 + table,
        "raw_rows": int(cache["raw_rows"]),
    }


def _buckets() -> list[tuple[str, str, int]]:
    return [("all", "all", 0)] + [
        (axis, label, i)
        for axis, labels in [
            ("cardinality", CARDINALITY),
            ("n_tables", TABLES),
            (
                "cross",
                tuple(f"{card}|{table}" for card in CARDINALITY for table in TABLES),
            ),
        ]
        for i, label in enumerate(labels)
    ]


def _mask(data: dict, axis: str, index: int) -> np.ndarray:
    return (
        np.ones(len(data["ref"]), dtype=bool) if axis == "all" else data[axis] == index
    )


def _combine(data: dict, names: list[str]) -> dict:
    return {
        **{
            key: np.concatenate([data[db][key] for db in names])
            for key in ["ref", "fp32", "cardinality", "n_tables", "cross"]
        },
        "raw_rows": sum(data[db]["raw_rows"] for db in names),
    }


def _coverage(data: dict, names: list[str], axis: str, index: int) -> int:
    return sum(bool(np.any(_mask(data[db], axis, index))) for db in names)


def _pool_rows(data: dict, groups: list[tuple[str, list[str], bool]]) -> list[dict]:
    rows = []
    for name, names, equal in groups:
        combined = _combine(data, names)
        weights = np.concatenate(
            [
                np.full(
                    len(data[db]["ref"]), 1 / len(data[db]["ref"]) if equal else 1.0
                )
                for db in names
            ]
        )
        for axis, bucket, index in _buckets():
            mask = _mask(combined, axis, index)
            rows.append(
                {
                    "database": name,
                    "population": "db_equal_pool" if equal else "pool_unique",
                    "axis": axis,
                    "bucket": bucket,
                    "raw_rows": combined["raw_rows"],
                    "duplicate_rows": combined["raw_rows"] - len(mask),
                    "pool_unique": int(mask.sum()),
                    "pool_total": len(mask),
                    "weight_mass": float(weights[mask].sum()),
                    "share": _ratio(weights[mask].sum(), weights.sum()),
                    "source_db_coverage": _coverage(data, names, axis, index),
                }
            )
    return rows


def _residual_row(data: dict, mask: np.ndarray, weights: np.ndarray | None) -> dict:
    pool_unique = int(mask.sum())
    selected_weights = np.ones(pool_unique) if weights is None else weights[mask]
    observed = int(np.count_nonzero(selected_weights))
    mass = float(selected_weights.sum())
    sufficient = pool_unique >= MIN_UNIQUE and observed >= MIN_UNIQUE
    result = {
        "pool_unique": pool_unique,
        "observed_unique": observed,
        "weight_mass": mass,
        "status": "ok" if sufficient else "sparse",
        "quantile_method": "linear" if weights is None else "inverse_ecdf",
        **{
            f"{kind}_{stat}": None
            for kind in ["r_ref", "r_fp32"]
            for stat in ["mean", "median", "p10", "p90"]
        },
        **{f"{sign}_share": None for sign in ["positive", "negative", "zero"]},
    }
    if not sufficient:
        return result
    statistics = {}
    for source, label in [("ref", "r_ref"), ("fp32", "r_fp32")]:
        values = data[source][mask]
        quantiles = (
            np.quantile(values, [0.1, 0.5, 0.9], method="linear")
            if weights is None
            else weighted_quantile(values, selected_weights, [0.1, 0.5, 0.9])
        )
        statistics.update(
            {
                f"{label}_mean": float(np.average(values, weights=selected_weights)),
                f"{label}_p10": float(quantiles[0]),
                f"{label}_median": float(quantiles[1]),
                f"{label}_p90": float(quantiles[2]),
            }
        )
    reference = data["ref"][mask]
    return {
        **result,
        **statistics,
        **{
            f"{sign}_share": float(selected_weights[condition].sum() / mass)
            for sign, condition in [
                ("positive", reference > 0),
                ("negative", reference < 0),
                ("zero", reference == 0),
            ]
        },
    }


def _residuals(
    data: dict,
    name: str,
    names: list[str],
    population: str,
    step: int,
    weights: np.ndarray | None,
) -> list[dict]:
    combined = _combine(data, names)
    return [
        {
            "database": name,
            "population": population,
            "step": step,
            "axis": axis,
            "bucket": bucket,
            "source_db_coverage": _coverage(data, names, axis, index),
            **_residual_row(combined, _mask(combined, axis, index), weights),
        }
        for axis, bucket, index in _buckets()
    ]


def _state_arrays(prefix: dict, names: list[str], data: dict) -> dict:
    arrays = {
        key: np.concatenate([np.asarray(prefix["databases"][db][key]) for db in names])
        for key in ["support", "query", "mass"]
    }
    size = sum(len(data[db]["ref"]) for db in names)
    for key, values in arrays.items():
        if (
            values.shape != (size,)
            or np.any(~np.isfinite(values))
            or np.any(values < 0)
            or (key != "mass" and np.any(values != np.floor(values)))
        ):
            raise ValueError("Invalid replay weights or role counts")
    return {**arrays, "combined": arrays["support"] + arrays["query"]}


def _exposure_rows(
    data: dict, prefix: dict, names: list[str], name: str, step: int
) -> list[dict]:
    arrays, combined = _state_arrays(prefix, names, data), _combine(data, names)
    global_totals = {
        key: sum(
            float(np.asarray(state[key]).sum())
            for state in prefix["databases"].values()
        )
        for key in ["support", "query", "mass"]
    }
    global_totals = {
        **global_totals,
        "combined": global_totals["support"] + global_totals["query"],
    }
    rows = []
    for axis, bucket, index in _buckets():
        mask = _mask(combined, axis, index)
        unique = int(np.count_nonzero(arrays["combined"][mask]))
        values = {
            output: float(arrays[key][mask].sum())
            if key == "mass"
            else int(arrays[key][mask].sum())
            for key, output in [
                ("support", "support"),
                ("query", "query"),
                ("combined", "combined_presentations"),
                ("mass", "nominal_mass"),
            ]
        }
        shares = {
            f"{output}_share": _ratio(values[output], arrays[key].sum())
            for key, output in [
                ("support", "support"),
                ("query", "query"),
                ("combined", "combined_presentations"),
                ("mass", "nominal_mass"),
            ]
        }
        global_shares = {
            f"{output}_global_share": _ratio(values[output], global_totals[key])
            for key, output in [
                ("support", "support"),
                ("query", "query"),
                ("combined", "combined_presentations"),
                ("mass", "nominal_mass"),
            ]
        }
        rows.append(
            {
                "database": name,
                "step": step,
                "axis": axis,
                "bucket": bucket,
                "pool_unique": int(mask.sum()),
                "observed_unique": unique,
                "coverage": _ratio(unique, mask.sum()),
                "source_db_coverage": _coverage(data, names, axis, index),
                **values,
                **shares,
                **global_shares,
            }
        )
    return rows


def _repeat_summary(data: dict, prefix: dict, db: str) -> dict:
    state = prefix["databases"][db]
    repeats = _state_arrays(prefix, [db], data)["combined"]
    conditions = [
        ("0", repeats == 0),
        ("1", repeats == 1),
        ("2", repeats == 2),
        ("3-5", (repeats >= 3) & (repeats <= 5)),
        ("6-10", (repeats >= 6) & (repeats <= 10)),
        ("11+", repeats >= 11),
    ]
    return {
        "episodes": int(state["episodes"]),
        "k_counts": dict(state["k_counts"]),
        "repeat_counts": {label: int(mask.sum()) for label, mask in conditions},
        "repeat_shares": {label: float(mask.mean()) for label, mask in conditions},
        "repeat_quantile_method": "linear",
        "repeat_quantiles": dict(
            zip(
                ["min", "p10", "median", "p90", "max"],
                map(float, np.quantile(repeats, [0, 0.1, 0.5, 0.9, 1])),
            )
        ),
    }


def _distribution(
    data: dict,
    names: list[str],
    axis: str,
    weights: np.ndarray | None = None,
    equal: bool = False,
) -> np.ndarray:
    size = {"cardinality": 7, "n_tables": 4, "cross": 28}[axis]
    values = np.concatenate([data[db][axis] for db in names])
    if weights is None:
        weights = np.concatenate(
            [
                np.full(
                    len(data[db]["ref"]), 1 / len(data[db]["ref"]) if equal else 1.0
                )
                for db in names
            ]
        )
    total = float(weights.sum())
    if total == 0:
        return np.full(size, np.nan)
    return np.bincount(values, weights=weights, minlength=size) / total


def _comparisons(
    data: dict, replay: dict, train: list[str], validation: list[str]
) -> list[dict]:
    rows = []
    for step, prefix in sorted(replay["prefixes"].items()):
        arrays = _state_arrays(prefix, train, data)
        sources = [
            ("train_db_equal_pool", None, True),
            ("train_pooled_pool", None, False),
            ("combined_presentations", arrays["combined"], False),
            ("nominal_mass", arrays["mass"], False),
        ]
        targets = [(db, [db], False) for db in validation] + [
            ("validation_db_equal_pool", validation, True)
        ]
        for axis in ["cardinality", "n_tables", "cross"]:
            for source, weights, equal in sources:
                p = _distribution(data, train, axis, weights, equal)
                for target, names, target_equal in targets:
                    q = _distribution(data, names, axis, equal=target_equal)
                    tv = (
                        float(0.5 * np.abs(p - q).sum())
                        if np.all(np.isfinite(p))
                        else UNDEFINED
                    )
                    for _, bucket, index in [
                        entry for entry in _buckets() if entry[0] == axis
                    ]:
                        valid = bool(np.isfinite(p[index]))
                        rows.append(
                            {
                                "step": step,
                                "axis": axis,
                                "bucket": bucket,
                                "source_population": source,
                                "validation_database": target,
                                "source_share": float(p[index]) if valid else UNDEFINED,
                                "validation_share": float(q[index]),
                                "difference": float(p[index] - q[index])
                                if valid
                                else UNDEFINED,
                                "ratio": _ratio(p[index], q[index])
                                if valid
                                else UNDEFINED,
                                "source_db_coverage": _coverage(
                                    data, train, axis, index
                                ),
                                "axis_tv": tv,
                            }
                        )
    return rows


def build_tables(
    caches: dict, replay: dict, train: list[str], validation: list[str]
) -> dict:
    """Return CSV-ready complete-denominator rows and repetition metadata.

    Pool residual rows have step=0; replay residuals and comparisons use each prefix.
    Sparse details are None (empty CSV fields), never fake zero residual statistics.
    Database-equal weights are 1/N per database; their normalization is explicit.
    """
    names = train + validation
    if (
        not train
        or not validation
        or len(set(names)) != len(names)
        or set(caches) != set(names)
    ):
        raise ValueError(
            "Caches must match disjoint, nonempty train and validation lists"
        )
    if set(replay["prefixes"]) != {1000, 2000, 4000}:
        raise ValueError("Replay requires the frozen 1000/2000/4000 prefixes")
    data = {db: _prepare(caches[db]) for db in names}
    groups = [(db, [db], False) for db in names] + [
        ("train_pooled", train, False),
        ("train_db_equal_pool", train, True),
        ("validation_db_equal_pool", validation, True),
    ]
    residuals, exposure, distributions = [], [], {}
    for name, members, equal in groups:
        weights = (
            np.concatenate(
                [
                    np.full(len(data[db]["ref"]), 1 / len(data[db]["ref"]))
                    for db in members
                ]
            )
            if equal
            else None
        )
        residuals.extend(
            _residuals(
                data,
                name,
                members,
                "db_equal_pool" if equal else "pool_unique",
                0,
                weights,
            )
        )
    for step, prefix in sorted(replay["prefixes"].items()):
        for name, members in [(db, [db]) for db in train] + [("train_pooled", train)]:
            arrays = _state_arrays(prefix, members, data)
            exposure.extend(_exposure_rows(data, prefix, members, name, step))
            for population, key in [
                ("combined_presentations", "combined"),
                ("nominal_mass", "mass"),
            ]:
                residuals.extend(
                    _residuals(data, name, members, population, step, arrays[key])
                )
        distributions[str(step)] = {
            db: _repeat_summary(data, prefix, db) for db in train
        }
    return {
        "pool": _pool_rows(data, groups),
        "exposure": exposure,
        "residuals": residuals,
        "comparison": _comparisons(data, replay, train, validation),
        "distribution_summary": {
            "prefixes": distributions,
            "minimum_unique": MIN_UNIQUE,
            "weighted_quantiles": "inverse_ecdf",
            "unweighted_quantiles": "linear",
        },
    }
