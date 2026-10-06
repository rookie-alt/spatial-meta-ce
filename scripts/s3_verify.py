"""Streaming independent S3 verification. Never imports production statistics."""

from __future__ import annotations

import json
import math

import numpy as np

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
MINIMUM = 200
ROW_ATOL = 1e-10
TOTAL_ATOL = 1e-8


def _compare(actual: object, expected: object, path: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path}: fields differ")
        for key in expected:
            if key == "validation_db_coverage_json":
                try:
                    parsed = json.loads(actual[key])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}.{key}: invalid JSON") from exc
                _compare(parsed, json.loads(expected[key]), f"{path}.{key}")
            else:
                _compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, int):
        if (
            isinstance(actual, (bool, np.bool_))
            or not isinstance(actual, (int, np.integer))
            or actual != expected
        ):
            raise ValueError(f"{path}: integer mismatch")
    elif isinstance(expected, float):
        if (
            isinstance(actual, (bool, np.bool_))
            or not isinstance(actual, (int, float, np.number))
            or not math.isfinite(actual)
            or abs(actual - expected)
            > (
                TOTAL_ATOL
                if path.rsplit(".", 1)[-1].endswith("weight_mass")
                else ROW_ATOL
            )
        ):
            raise ValueError(f"{path}: numeric mismatch")
    elif actual != expected:
        raise ValueError(f"{path}: value mismatch")


def _ratio(n: float, d: float) -> float | str:
    return float(n / d) if d else "undefined"


def _classification(values: np.ndarray) -> np.ndarray:
    return np.select(
        [
            values == 0,
            values < 10,
            values < 100,
            values < 1000,
            values < 10000,
            values < 100000,
        ],
        range(6),
        default=6,
    )


def _statistics(values: np.ndarray, weights: np.ndarray, linear: bool) -> dict:
    positive = weights > 0
    x, w = values[positive], weights[positive]
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    if linear:
        positions = np.array([0.5, 0.1, 0.9]) * (len(x) - 1)
        lo, hi = np.floor(positions).astype(int), np.ceil(positions).astype(int)
        quantiles = x[lo] + (x[hi] - x[lo]) * (positions - lo)
    else:
        cumulative = np.cumsum(w, dtype=np.float64)
        indices = np.searchsorted(
            cumulative, np.array([0.5, 0.1, 0.9]) * cumulative[-1], side="left"
        )
        quantiles = x[np.minimum(indices, len(x) - 1)]
    return {
        "mean": float(np.dot(x, w) / w.sum()),
        "median": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "p90": float(quantiles[2]),
    }


def _cell(parts: list, linear: bool) -> dict:
    if not parts:
        return {
            "unique": 0,
            "observed": 0,
            "mass": 0.0,
            "coverage": 0,
            "sum": 0.0,
            "stats": None,
            "directions": None,
        }
    ref = np.concatenate([p[0] for p in parts])
    fp = np.concatenate([p[1] for p in parts])
    weights = np.concatenate([p[2] for p in parts])
    observed = int(np.count_nonzero(weights))
    mass = float(weights.sum())
    enough = len(ref) >= MINIMUM and observed >= MINIMUM
    stats = (
        {
            f"r_{name}_{key}": value
            for name, arr in (("ref", ref), ("fp32", fp))
            for key, value in _statistics(arr, weights, linear).items()
        }
        if enough
        else None
    )
    directions = (
        {
            name: float(weights[mask].sum() / mass)
            for name, mask in (
                ("positive_share", ref > 0),
                ("negative_share", ref < 0),
                ("zero_share", ref == 0),
            )
        }
        if enough
        else None
    )
    return {
        "unique": len(ref),
        "observed": observed,
        "mass": mass,
        "coverage": len(parts),
        "sum": float(np.dot(ref, weights)),
        "stats": stats,
        "directions": directions,
    }


def _aggregate(cells: dict, keys: set | None = None) -> dict:
    selected = [v for k, v in cells.items() if keys is None or k in keys]
    return {
        name: sum((c[name] for c in selected), 0.0 if name in ("mass", "sum") else 0)
        for name in ("unique", "observed", "mass", "sum")
    }


def _mean(total: dict) -> float | None:
    if total["unique"] < MINIMUM or total["observed"] < MINIMUM or not total["mass"]:
        return None
    return float(total["sum"] / total["mass"])


class Verification:
    """Own only scratch arrays and one layer's expected cells at a time."""

    def __init__(
        self, caches: dict, replay: dict, train: list, validation: list
    ) -> None:
        self.train, self.validation = tuple(train), tuple(validation)
        names = self.train + self.validation
        if (
            not train
            or not validation
            or len(set(names)) != len(names)
            or set(names) != set(caches)
        ):
            raise ValueError("Invalid database partition")
        if set(names) & {"imdb", "stats", "genome", "ergastf1"}:
            raise ValueError("Forbidden target database")
        self.data = {db: self._prepare(db, caches[db]) for db in names}
        try:
            prefix = replay["prefixes"][4000]
            states = prefix["databases"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Missing final 4000 replay prefix") from exc
        if set(states) != set(train):
            raise ValueError("Replay database set differs")
        for db in train:
            state, n = states[db], len(self.data[db]["ref"])
            arrays = [np.asarray(state[k]) for k in ("support", "query", "mass")]
            if any(
                a.shape != (n,)
                or a.dtype.kind not in "iuf"
                or not np.all(np.isfinite(a))
                or np.any(a < 0)
                for a in arrays
            ):
                raise ValueError(f"{db}: invalid replay arrays")
            if any(a.dtype.kind not in "iu" for a in arrays[:2]):
                raise ValueError(f"{db}: replay counts must be integers")
            counts = arrays[0].astype(np.int64) + arrays[1].astype(np.int64)
            if np.any(counts < 0) or np.any((counts > 0) != (arrays[2] > 0)):
                raise ValueError(f"{db}: replay count/mass support differs")
            self.data[db] = {
                **self.data[db],
                "presentations": counts,
                "mass": arrays[2].astype(np.float64, copy=True),
            }
        total_count = sum(int(self.data[db]["presentations"].sum()) for db in train)
        total_mass = math.fsum(float(self.data[db]["mass"].sum()) for db in train)
        if (
            total_count != prefix["presentations"]
            or abs(total_mass - 4000) > TOTAL_ATOL
        ):
            raise ValueError("Replay total conservation differs")
        self.seen: set = set()
        self.rows = dict.fromkeys(("conditions", "residuals", "alignment"), 0)

    @staticmethod
    def _prepare(db: str, cache: dict) -> dict:
        true, pg, tables, indices, counts = [
            np.asarray(cache[k])
            for k in ("true", "pg", "n_tables", "indices", "token_counts")
        ]
        n = len(true)
        if not n or any(
            a.shape != (n,) or a.dtype.kind not in "iuf" or not np.all(np.isfinite(a))
            for a in (true, pg, tables, indices)
        ):
            raise ValueError(f"{db}: invalid cache arrays")
        if (
            np.any(true < 0)
            or np.any(pg < 0)
            or np.any(tables < 1)
            or np.any(tables != np.floor(tables))
        ):
            raise ValueError(f"{db}: invalid labels")
        if (
            counts.shape != (n, 4)
            or counts.dtype.kind not in "iu"
            or np.any(counts < 0)
        ):
            raise ValueError(f"{db}: token_counts require nonnegative integers")
        if (
            indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices[1:] <= indices[:-1])
            or not isinstance(cache["raw_rows"], (int, np.integer))
            or cache["raw_rows"] < n
            or indices[-1] >= cache["raw_rows"]
        ):
            raise ValueError(f"{db}: invalid first indices")
        hashes = cache["query_hash"]
        if (
            len(hashes) != n
            or len(set(hashes)) != n
            or any(not isinstance(h, str) or not h for h in hashes)
        ):
            raise ValueError(f"{db}: invalid local query hashes")
        keys = np.column_stack((_classification(pg), counts))
        unique, codes = np.unique(keys, axis=0, return_inverse=True)
        labels = tuple(tuple(int(x) for x in row) for row in unique)
        log_true, log_pg = [
            np.log(np.maximum(a.astype(np.float64), 1)) for a in (true, pg)
        ]
        card = _classification(true)
        table = np.select([tables <= 3, tables == 4, tables == 5], [0, 1, 2], default=3)
        return {
            "ref": log_true - log_pg,
            "fp": log_true.astype(np.float32).astype(float)
            - log_pg.astype(np.float32).astype(float),
            "codes": codes,
            "keys": labels,
            "cardinality": card,
            "n_tables": table,
            "cross": card * 4 + table,
        }

    def _groups(self, axis: str, bucket: str) -> dict:
        number = AXES[axis].index(bucket)
        groups = {}
        for db, data in self.data.items():
            positions = (
                np.arange(len(data["ref"]))
                if axis == "all"
                else np.flatnonzero(data[axis] == number)
            )
            # One grouping sort per database/layer; never scan N for every C.
            order = np.argsort(data["codes"][positions], kind="stable")
            positions = positions[order]
            codes = data["codes"][positions]
            starts = np.r_[0, np.flatnonzero(codes[1:] != codes[:-1]) + 1, len(codes)]
            groups[db] = {
                data["keys"][int(codes[start])]: positions[start:end]
                for start, end in zip(starts[:-1], starts[1:])
                if end > start
            }
        return groups

    def _views(self, groups: dict, keys: set) -> dict:
        specifications = [(db, "pool_unique", (db,), False, True) for db in self.data]
        specifications += [
            ("train_pooled", "pool_unique", self.train, False, True),
            ("train_db_equal_pool", "db_equal_pool", self.train, True, False),
            ("validation_db_equal_pool", "db_equal_pool", self.validation, True, False),
        ]
        specifications += [
            (db, pop, (db,) if db != "train_pooled" else self.train, False, False)
            for pop in ("combined_presentations", "nominal_mass")
            for db in (*self.train, "train_pooled")
        ]
        views = {}
        for database, population, members, equal, linear in specifications:
            cells = {}
            for key in keys:
                parts = []
                for db in members:
                    pos = groups[db].get(key)
                    if pos is None:
                        continue
                    data = self.data[db]
                    if population == "combined_presentations":
                        weight = data["presentations"][pos].astype(float)
                    elif population == "nominal_mass":
                        weight = data["mass"][pos]
                    else:
                        weight = np.full(
                            len(pos),
                            1 / (len(members) * len(data["ref"])) if equal else 1.0,
                        )
                    parts.append((data["ref"][pos], data["fp"][pos], weight))
                cells[key] = _cell(parts, linear)
            views[(database, population)] = (cells, linear)
        return views

    @staticmethod
    def _row(
        database: str,
        population: str,
        axis: str,
        bucket: str,
        key: tuple,
        cell: dict,
        total: dict,
        linear: bool,
    ) -> tuple[dict, dict]:
        common = dict(
            database=database,
            population=population,
            axis=axis,
            bucket=bucket,
            condition=",".join(map(str, key)),
            pg_bucket=CARD[key[0]],
            **dict(
                zip(
                    (
                        "n_join_column_tokens",
                        "n_fanout_tokens",
                        "n_table_tokens",
                        "n_filter_tokens",
                    ),
                    key[1:],
                )
            ),
            pool_unique=cell["unique"],
            observed_unique=cell["observed"],
            group_pool_unique=total["unique"],
            weight_mass=cell["mass"],
            group_weight_mass=total["mass"],
            share=_ratio(cell["mass"], total["mass"]),
            source_db_coverage=cell["coverage"],
        )
        details = {
            f"r_{prefix}_{stat}": None
            for prefix in ("ref", "fp32")
            for stat in ("mean", "median", "p10", "p90")
        }
        details.update(
            dict.fromkeys(("positive_share", "negative_share", "zero_share"))
        )
        if cell["stats"] is not None:
            details.update(cell["stats"])
            details.update(cell["directions"])
        return common, {
            **common,
            "status": "ok" if cell["stats"] is not None else "sparse",
            "quantile_method": "linear" if linear else "inverse_ecdf",
            **details,
        }

    def _alignment(
        self,
        axis: str,
        bucket: str,
        source_name: str,
        target: str,
        source: dict,
        validation: dict,
        views: dict,
        groups: dict,
    ) -> dict:
        keys = {
            k
            for k in source
            if source[k]["unique"] >= MINIMUM
            and validation[k]["unique"] >= MINIMUM
            and source[k]["observed"] >= MINIMUM
        }
        st, vt = _aggregate(source), _aggregate(validation)
        sc, vc = _aggregate(source, keys), _aggregate(validation, keys)
        coverage = {
            db: sum(views[(db, "pool_unique")][0][k]["unique"] for k in keys)
            for db in ((target,) if target in self.validation else self.validation)
        }
        status = "ok" if keys else "no_common_support"
        if (
            keys
            and target == "validation_db_equal_pool"
            and any(
                coverage[db] < MINIMUM
                or not any(
                    source[k]["unique"] >= MINIMUM
                    and source[k]["observed"] >= MINIMUM
                    and views[(db, "pool_unique")][0][k]["unique"] >= MINIMUM
                    for k in keys
                )
                for db in self.validation
            )
        ):
            status = "incomplete_validation_support"
        sm, vm = _mean(st), _mean(vt)
        cm, cv = (_mean(sc), _mean(vc)) if status == "ok" else (None, None)
        signed = absolute = None
        if status == "ok":
            differences = [
                (
                    validation[k]["mass"] / vc["mass"],
                    validation[k]["sum"] / validation[k]["mass"]
                    - source[k]["sum"] / source[k]["mass"],
                )
                for k in keys
            ]
            signed = math.fsum(q * d for q, d in differences)
            absolute = math.fsum(q * abs(d) for q, d in differences)
        return dict(
            axis=axis,
            bucket=bucket,
            source_population=source_name,
            validation_database=target,
            status=status,
            common_conditions=len(keys),
            source_pool_unique=st["unique"],
            validation_pool_unique=vt["unique"],
            source_observed_unique=st["observed"],
            source_weight_mass=st["mass"],
            validation_weight_mass=vt["mass"],
            source_common_unique=sc["unique"],
            validation_common_unique=vc["unique"],
            source_common_observed_unique=sc["observed"],
            source_common_weight_mass=sc["mass"],
            validation_common_weight_mass=vc["mass"],
            source_unique_coverage=_ratio(sc["unique"], st["unique"]),
            validation_unique_coverage=_ratio(vc["unique"], vt["unique"]),
            source_mass_coverage=_ratio(sc["mass"], st["mass"]),
            validation_mass_coverage=_ratio(vc["mass"], vt["mass"]),
            source_db_coverage=sum(
                any(k in groups[db] for k in keys) for db in self.train
            ),
            validation_db_coverage=sum(n > 0 for n in coverage.values()),
            validation_db_coverage_json=json.dumps(coverage, sort_keys=True),
            full_train_mean=sm,
            full_validation_mean=vm,
            full_mean_difference=vm - sm if sm is not None and vm is not None else None,
            common_train_mean=cm,
            common_validation_mean=cv,
            common_mean_difference=cv - cm
            if cm is not None and cv is not None
            else None,
            d_signed=signed,
            d_absolute=absolute,
        )

    def check(self, chunk: dict) -> None:
        if set(chunk) != {"axis", "bucket", "conditions", "residuals", "alignment"}:
            raise ValueError("chunk: fields differ")
        axis, bucket = chunk["axis"], chunk["bucket"]
        if axis not in AXES or bucket not in AXES[axis] or (axis, bucket) in self.seen:
            raise ValueError("Invalid or duplicate layer")
        groups = self._groups(axis, bucket)
        keys = set().union(*(set(v) for v in groups.values()))
        views = self._views(groups, keys)
        expected = {name: {} for name in self.rows}
        for (db, pop), (cells, linear) in views.items():
            total = _aggregate(cells)
            for key, cell in cells.items():
                condition, residual = self._row(
                    db, pop, axis, bucket, key, cell, total, linear
                )
                identity = (db, pop, condition["condition"])
                expected["conditions"][identity] = condition
                expected["residuals"][identity] = residual
        sources = {
            "train_pooled_pool": ("train_pooled", "pool_unique"),
            "train_db_equal_pool": ("train_db_equal_pool", "db_equal_pool"),
            "combined_presentations": ("train_pooled", "combined_presentations"),
            "nominal_mass": ("train_pooled", "nominal_mass"),
        }
        for name, view in sources.items():
            for target in (*self.validation, "validation_db_equal_pool"):
                target_view = (
                    target,
                    "pool_unique" if target in self.validation else "db_equal_pool",
                )
                expected["alignment"][(name, target)] = self._alignment(
                    axis,
                    bucket,
                    name,
                    target,
                    views[view][0],
                    views[target_view][0],
                    views,
                    groups,
                )
        for table, wanted in expected.items():
            identities = set()
            if not isinstance(chunk[table], list):
                raise ValueError(f"{table}: rows must be a list")
            for row in chunk[table]:
                fields = (
                    ("source_population", "validation_database")
                    if table == "alignment"
                    else ("database", "population", "condition")
                )
                try:
                    identity = tuple(row[k] for k in fields)
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"{table}: missing identity") from exc
                if identity in identities or identity not in wanted:
                    raise ValueError(f"{table}: duplicate or unexpected row {identity}")
                identities.add(identity)
                _compare(row, wanted[identity], f"{table}:{identity}")
            if identities != set(wanted):
                raise ValueError(f"{table}: missing row")
        self.seen.add((axis, bucket))
        self.rows = {table: n + len(chunk[table]) for table, n in self.rows.items()}

    def finish(self) -> dict:
        required = {
            (axis, bucket) for axis, buckets in AXES.items() for bucket in buckets
        }
        if self.seen != required:
            raise ValueError("Incomplete 40 layers")
        return {
            "groups": len(self.seen),
            "rows": dict(self.rows),
            "checks": {
                "complete_layers": True,
                "unique_layers": True,
                "all_fields_and_values": True,
                "replay_conservation": True,
            },
        }
