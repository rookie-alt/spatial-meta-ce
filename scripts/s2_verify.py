"""Independent, NumPy-only S2 verification; never imports the producer modules.

Scratch arrays are owned locally. No episode table or per-query Python statistics
are materialized. All CSV cells and row identities are checked, not just totals.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

ROW_ATOL = 1e-10
TOTAL_ATOL = 1e-8
PREFIXES = (1000, 2000, 4000)
CARD = (
    "0",
    "(0,10)",
    "[10,100)",
    "[100,1000)",
    "[1000,10000)",
    "[10000,100000)",
    "[100000,inf)",
)
TABLE = ("1-3", "4", "5", "6+")
AXES = {
    "all": ("all",),
    "cardinality": CARD,
    "n_tables": TABLE,
    "cross": tuple(f"{c}|{t}" for c in CARD for t in TABLE),
}
KEYS = {
    "pool": ("database", "population", "axis", "bucket"),
    "exposure": ("database", "step", "axis", "bucket"),
    "residuals": ("database", "population", "step", "axis", "bucket"),
    "comparison": (
        "step",
        "axis",
        "bucket",
        "source_population",
        "validation_database",
    ),
}


def _same(actual: object, expected: object, where: str, atol: float = ROW_ATOL) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{where}: dictionary fields differ")
        for key, value in expected.items():
            _same(actual[key], value, f"{where}.{key}", atol)
    elif isinstance(expected, (np.ndarray, float, int, np.number)):
        try:
            left, right = np.asarray(actual), np.asarray(expected)
            valid = (
                left.dtype.kind in "fiu"
                and left.shape == right.shape
                and np.all(np.isfinite(left))
                and np.allclose(left, right, rtol=0, atol=atol)
            )
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f"{where}: numeric mismatch")
    elif actual != expected:
        raise ValueError(f"{where}: value mismatch")


def _ratio(n: float, d: float) -> float | str:
    return float(n / d) if d else "undefined"


def _inputs(caches: dict, train: list, validation: list, config: dict) -> dict:
    names = train + validation
    if (
        not train
        or not validation
        or len(set(names)) != len(names)
        or set(names) != set(caches)
        or config["steps"] != 4000
        or config["accumulation"] != 1
        or config["meta_batch"] < 1
        or config["query_size"] < 1
        or not config["support_sizes"]
        or len(set(config["support_sizes"])) != len(config["support_sizes"])
        or any(k < 1 or int(k) != k for k in config["support_sizes"])
    ):
        raise ValueError("Invalid S2 inputs or frozen prefixes")
    data = {}
    for db in names:
        cache = caches[db]
        true, pg, tables = (np.asarray(cache[k]) for k in ("true", "pg", "n_tables"))
        n = len(true)
        indices = np.asarray(cache["indices"])
        if (
            not n
            or any(
                a.shape != (n,) or not np.all(np.isfinite(a))
                for a in (true, pg, tables, indices)
            )
            or np.any(true < 0)
            or np.any(pg < 0)
            or np.any(tables < 1)
            or np.any(tables != np.floor(tables))
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(np.diff(indices) <= 0)
            or cache["raw_rows"] < n
            or indices[-1] >= cache["raw_rows"]
            or len(cache["query_hash"]) != n
            or len(set(cache["query_hash"])) != n
            or any(not isinstance(h, str) or not h for h in cache["query_hash"])
        ):
            raise ValueError(f"{db}: invalid per-database unique cache")
        if db in train and n < max(config["support_sizes"]) + config["query_size"]:
            raise ValueError(f"{db}: insufficient sampling pool")
        logs = [np.log(np.maximum(a.astype(np.float64), 1)) for a in (true, pg)]
        card = np.select(
            [
                true == 0,
                true < 10,
                true < 100,
                true < 1000,
                true < 10000,
                true < 100000,
            ],
            [0, 1, 2, 3, 4, 5],
            default=6,
        )
        table = np.select([tables <= 3, tables == 4, tables == 5], [0, 1, 2], default=3)
        data[db] = {
            "r_ref": logs[0] - logs[1],
            "r_fp32": logs[0].astype(np.float32).astype(float)
            - logs[1].astype(np.float32).astype(float),
            "all": np.zeros(n, dtype=np.int8),
            "cardinality": card,
            "n_tables": table,
            "cross": card * 4 + table,
        }
    return data


def _replay(caches: dict, replay: dict, train: list, config: dict) -> dict:
    if set(replay["prefixes"]) != set(PREFIXES):
        raise ValueError("Missing or unexpected replay prefix")
    sizes, batch, query = (
        config["support_sizes"],
        config["meta_batch"],
        config["query_size"],
    )
    # Integer counts by K provide a different summation algorithm from the producer.
    scratch = {
        db: {
            "support": np.zeros(len(caches[db]["indices"]), dtype=np.int64),
            "query": np.zeros(len(caches[db]["indices"]), dtype=np.int64),
            "by_k": np.zeros((len(sizes), len(caches[db]["indices"])), dtype=np.int32),
            "episodes": 0,
            "k_counts": {k: 0 for k in sizes},
        }
        for db in train
    }
    rng, chain = (
        np.random.default_rng(config["seed"] + 1),
        hashlib.sha256(b"").hexdigest(),
    )
    presentations = 0
    for number in range(1, 4000 * batch + 1):
        db, k = str(rng.choice(train)), int(rng.choice(sizes))
        cache, state = caches[db], scratch[db]
        raw = rng.choice(cache["indices"], size=k + query, replace=False)
        positions = np.searchsorted(cache["indices"], raw)
        hashes = cache["query_hash"]
        event = {
            "db": db,
            "support": [hashes[i] for i in positions[:k]],
            "query": [hashes[i] for i in positions[k:]],
        }
        chain = hashlib.sha256(
            json.dumps(
                {"previous": chain, "episode": event},
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        state["support"][positions[:k]] += 1
        state["query"][positions[k:]] += 1
        state["by_k"][sizes.index(k), positions] += 1
        state["episodes"] += 1
        state["k_counts"][k] += 1
        presentations += k + query
        if number % batch == 0 and number // batch in PREFIXES:
            _prefix(
                replay["prefixes"][number // batch],
                scratch,
                sizes,
                batch,
                query,
                chain,
                number,
                presentations,
            )
    _same(replay["digest"], chain, "final digest")
    return {"episodes": 4000 * batch, "presentations": presentations, "digest": chain}


def _prefix(
    prefix: dict,
    states: dict,
    sizes: list,
    batch: int,
    query: int,
    chain: str,
    number: int,
    presentations: int,
) -> None:
    _same(prefix["digest"], chain, "prefix digest")
    _same(prefix["episodes"], number, "prefix episodes", 0)
    _same(prefix["presentations"], presentations, "prefix presentations", 0)
    if set(prefix["databases"]) != set(states):
        raise ValueError("Prefix database set differs")
    totals = []
    for db, state in states.items():
        mass = sum(
            (state["by_k"][i] / (batch * (k + query)) for i, k in enumerate(sizes))
        )
        expected = {
            key: state[key] for key in ("support", "query", "episodes", "k_counts")
        }
        for key, value in expected.items():
            _same(prefix["databases"][db][key], value, f"{db}.{key}", 0)
        _same(prefix["databases"][db]["mass"], mass, f"{db}.mass")
        totals.append(float(mass.sum()))
    _same(math.fsum(totals), number / batch, "nominal conservation", TOTAL_ATOL)
    _same(
        sum(int(s["support"].sum() + s["query"].sum()) for s in states.values()),
        presentations,
        "role conservation",
        0,
    )


class _Rows:
    def __init__(self, tables: dict) -> None:
        self.tables = {}
        self.counts = {}
        for name, fields in KEYS.items():
            rows = tables[name]
            indexed = {tuple(row[k] for k in fields): row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError(f"{name}: duplicate row")
            self.tables[name], self.counts[name] = indexed, len(rows)

    def check(self, name: str, row: dict) -> None:
        key = tuple(row[k] for k in KEYS[name])
        actual = self.tables[name].pop(key, None)
        if actual is None:
            raise ValueError(f"{name}: missing row {key}")
        _same(actual, row, f"{name}:{key}")

    def finish(self) -> dict:
        if any(self.tables.values()):
            raise ValueError("Unexpected extra CSV rows")
        return self.counts


def _join(data: dict, members: list) -> dict:
    return {
        key: np.concatenate([data[d][key] for d in members]) for key in data[members[0]]
    }


def _weights(caches: dict, members: list, equal: bool) -> np.ndarray:
    return np.concatenate(
        [
            np.full(
                len(caches[d]["true"]), 1 / len(caches[d]["true"]) if equal else 1.0
            )
            for d in members
        ]
    )


def _coverage(data: dict, members: list, axis: str, i: int) -> int:
    return sum(int(np.count_nonzero(data[d][axis] == i) > 0) for d in members)


def _quantiles(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    probabilities = np.array([0.1, 0.5, 0.9])
    if weights is None:
        ranks = probabilities * (len(values) - 1)
        lo, hi = np.floor(ranks).astype(int), np.ceil(ranks).astype(int)
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (
            ranks - lo
        )
    ordered_weights = weights[order]
    positive = ordered_weights > 0
    cumulative = np.cumsum(ordered_weights[positive], dtype=np.float64)
    indices = np.searchsorted(cumulative, probabilities * cumulative[-1], side="left")
    return sorted_values[positive][indices]


def _residual(joined: dict, mask: np.ndarray, weights: np.ndarray | None) -> dict:
    n = int(np.count_nonzero(mask))
    w = np.ones(n) if weights is None else weights[mask]
    observed, mass = int(np.count_nonzero(w)), float(w.sum())
    sufficient = n >= 200 and observed >= 200
    row = {
        "pool_unique": n,
        "observed_unique": observed,
        "weight_mass": mass,
        "status": "ok" if sufficient else "sparse",
        "quantile_method": "linear" if weights is None else "inverse_ecdf",
    }
    for kind in ("r_ref", "r_fp32"):
        values = joined[kind][mask]
        stats = {"mean": None, "median": None, "p10": None, "p90": None}
        if sufficient:
            p10, median, p90 = _quantiles(values, None if weights is None else w)
            stats = {
                "mean": float(np.dot(values, w) / mass),
                "median": float(median),
                "p10": float(p10),
                "p90": float(p90),
            }
        row.update({f"{kind}_{key}": value for key, value in stats.items()})
    ref = joined["r_ref"][mask]
    for sign, selected in [
        ("positive", ref > 0),
        ("negative", ref < 0),
        ("zero", ref == 0),
    ]:
        row[sign + "_share"] = float(w[selected].sum() / mass) if sufficient else None
    return row


def _pools(
    rows: _Rows, caches: dict, data: dict, train: list, validation: list
) -> None:
    groups = [(d, [d], False) for d in train + validation] + [
        ("train_pooled", train, False),
        ("train_db_equal_pool", train, True),
        ("validation_db_equal_pool", validation, True),
    ]
    for name, members, equal in groups:
        joined, weights = _join(data, members), _weights(caches, members, equal)
        total, raw = len(weights), sum(caches[d]["raw_rows"] for d in members)
        population = "db_equal_pool" if equal else "pool_unique"
        for axis, labels in AXES.items():
            for i, label in enumerate(labels):
                mask = joined[axis] == i
                base = {
                    "database": name,
                    "population": population,
                    "axis": axis,
                    "bucket": label,
                    "source_db_coverage": _coverage(data, members, axis, i),
                }
                mass = float(weights[mask].sum())
                rows.check(
                    "pool",
                    {
                        **base,
                        "raw_rows": raw,
                        "duplicate_rows": raw - total,
                        "pool_unique": int(mask.sum()),
                        "pool_total": total,
                        "weight_mass": mass,
                        "share": mass / float(weights.sum()),
                    },
                )
                rows.check(
                    "residuals",
                    {
                        **base,
                        "step": 0,
                        **_residual(joined, mask, weights if equal else None),
                    },
                )


def _exposures(rows: _Rows, data: dict, replay: dict, train: list) -> dict:
    summaries = {}
    for step in PREFIXES:
        states = replay["prefixes"][step]["databases"]
        global_sums = {
            key: math.fsum(float(states[d][key].sum()) for d in train)
            for key in ("support", "query", "mass")
        }
        global_sums["combined"] = global_sums["support"] + global_sums["query"]
        summaries[str(step)] = {}
        for name, members in [(d, [d]) for d in train] + [("train_pooled", train)]:
            joined = _join(data, members)
            arrays = {
                key: np.concatenate([states[d][key] for d in members])
                for key in ("support", "query", "mass")
            }
            arrays["combined"] = arrays["support"] + arrays["query"]
            for axis, labels in AXES.items():
                for i, label in enumerate(labels):
                    mask = joined[axis] == i
                    n, observed = (
                        int(mask.sum()),
                        int(np.count_nonzero(arrays["combined"][mask])),
                    )
                    base = {
                        "database": name,
                        "step": step,
                        "axis": axis,
                        "bucket": label,
                        "source_db_coverage": _coverage(data, members, axis, i),
                    }
                    row = {
                        **base,
                        "pool_unique": n,
                        "observed_unique": observed,
                        "coverage": _ratio(observed, n),
                    }
                    for key, column in [
                        ("support", "support"),
                        ("query", "query"),
                        ("combined", "combined_presentations"),
                        ("mass", "nominal_mass"),
                    ]:
                        value = float(arrays[key][mask].sum())
                        row.update(
                            {
                                column: value,
                                column + "_share": _ratio(value, arrays[key].sum()),
                                column + "_global_share": _ratio(
                                    value, global_sums[key]
                                ),
                            }
                        )
                    rows.check("exposure", row)
                    for population, key in [
                        ("combined_presentations", "combined"),
                        ("nominal_mass", "mass"),
                    ]:
                        rows.check(
                            "residuals",
                            {
                                **base,
                                "population": population,
                                **_residual(joined, mask, arrays[key]),
                            },
                        )
            if name in train:
                summaries[str(step)][name] = _repetition(
                    states[name], arrays["combined"]
                )
    return {
        "prefixes": summaries,
        "minimum_unique": 200,
        "weighted_quantiles": "inverse_ecdf",
        "unweighted_quantiles": "linear",
    }


def _repetition(state: dict, repeat: np.ndarray) -> dict:
    bins = np.searchsorted([1, 2, 3, 6, 11], repeat, side="right")
    counts = dict(
        zip(
            ("0", "1", "2", "3-5", "6-10", "11+"),
            map(int, np.bincount(bins, minlength=6)),
        )
    )
    return {
        "episodes": state["episodes"],
        "k_counts": state["k_counts"],
        "repeat_counts": counts,
        "repeat_shares": {key: value / len(repeat) for key, value in counts.items()},
        "repeat_quantile_method": "linear",
        "repeat_quantiles": dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                map(float, np.quantile(repeat, [0, 0.1, 0.5, 0.9, 1])),
            )
        ),
    }


def _distribution(data: dict, members: list, axis: str, weights: dict) -> np.ndarray:
    counts = sum(
        (
            np.bincount(data[d][axis], weights=weights[d], minlength=len(AXES[axis]))
            for d in members
        )
    )
    return counts / counts.sum()


def _comparisons(
    rows: _Rows, caches: dict, data: dict, replay: dict, train: list, validation: list
) -> None:
    unit = {d: np.ones(len(caches[d]["true"])) for d in train + validation}
    equal = {d: unit[d] / len(unit[d]) for d in unit}
    for step in PREFIXES:
        states = replay["prefixes"][step]["databases"]
        sources = {
            "train_db_equal_pool": equal,
            "train_pooled_pool": unit,
            "combined_presentations": {
                d: states[d]["support"] + states[d]["query"] for d in train
            },
            "nominal_mass": {d: states[d]["mass"] for d in train},
        }
        for axis, labels in AXES.items():
            if axis == "all":
                continue
            for source, weights in sources.items():
                p = _distribution(data, train, axis, weights)
                for target, members in [(d, [d]) for d in validation] + [
                    ("validation_db_equal_pool", validation)
                ]:
                    q = _distribution(data, members, axis, equal)
                    for i, label in enumerate(labels):
                        rows.check(
                            "comparison",
                            {
                                "step": step,
                                "axis": axis,
                                "bucket": label,
                                "source_population": source,
                                "validation_database": target,
                                "source_share": float(p[i]),
                                "validation_share": float(q[i]),
                                "difference": float(p[i] - q[i]),
                                "ratio": _ratio(p[i], q[i]),
                                "source_db_coverage": _coverage(data, train, axis, i),
                                "axis_tv": float(np.abs(p - q).sum() / 2),
                            },
                        )


def verify(
    caches: dict,
    replay: dict,
    tables: dict,
    train: list,
    validation: list,
    config: dict,
) -> dict:
    """Raise ValueError on any mismatch; inputs are never modified.

    Source file authentication belongs to the caller. This independently verifies
    the in-memory unique pools, full replay and every exported statistical cell.
    """
    try:
        data = _inputs(caches, train, validation, config)
        replay_counts = _replay(caches, replay, train, config)
        rows = _Rows(tables)
        _pools(rows, caches, data, train, validation)
        summary = _exposures(rows, data, replay, train)
        _same(tables["distribution_summary"], summary, "distribution_summary")
        _comparisons(rows, caches, data, replay, train, validation)
        counts = rows.finish()
    except (KeyError, TypeError, IndexError, OverflowError) as exc:
        raise ValueError(
            f"Incomplete or malformed S2 verification input: {exc}"
        ) from exc
    return {
        "checks": [
            "independent_rng_hash_chain",
            "every_prefix_per_query_roles_mass",
            "episodes_k_counts_conservation",
            "complete_unique_csv_rows",
            "all_40_buckets_pool_exposure_comparison",
            "residuals_all_statistics",
            "repeat_distributions",
        ],
        "prefixes": list(PREFIXES),
        "row_counts": counts,
        "replay": replay_counts,
        "databases": len(caches),
        "unique_queries": sum(len(c["true"]) for c in caches.values()),
        "tolerances": {"rtol": 0, "row_atol": ROW_ATOL, "total_atol": TOTAL_ATOL},
    }
