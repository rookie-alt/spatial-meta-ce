"""Independent, fail-closed S4 record and FP64 query evidence verification.

No model, checkpoint loader, metric producer, target data, or selection producer
is imported. Callers own file-byte authentication; these gates verify contents.
"""

import hashlib
import json
import math

import numpy as np

ARMS = ("A_anchor1", "B_anchor0")
DATABASES = ("chembl", "legalacts", "telstra")
BUCKETS = ("all_tables", "four_plus_tables")
SOURCES = (
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "employee",
    "financial",
    "fnhk",
    "grants",
    "hepatitis",
    "hockey",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
    *DATABASES,
)
COVERAGE = {
    "chembl": (50000, 33947),
    "legalacts": (49504, 9178),
    "telstra": (50000, 20311),
}
PROTOCOL_HASH = "d63c4f503f55eb87b59896e54f0379768e8829c8d48f2bad0a3f6224b53e0be3"
INITIAL_HASH = "41e594b03ef19c56c3f236237da850929f2ec5c919702b44402dd9b2864d284a"
SAMPLE_DIGEST = "ad433d964fe562efacd19c8a03005bfe7df530d96bed0ca3535684854f6f4de8"
VALIDATION_HASH = "0a110551da2db90ff619d8461cd06a1a4b62eb82341f6c4219eb7f6436e3ddb2"
SOURCE_HASH = "0e1c45f411a47bc4d4e504cc21319cddc08e11dc086411d2c321b79fe91d2cac"
FROZEN_MODEL = {"arch": "sum", "use_pg": True, "layers": 2, "heads": 4, "width": 64}
FROZEN_CONFIG = {
    "method": "erm",
    "seed": 42,
    "steps": 4000,
    "meta_batch": 4,
    "accumulation": 1,
    "query_size": 64,
    "support_sizes": [5, 10, 20, 50, 100],
    "outer_lr": 3e-4,
    "inner_lr": 1e-3,
    "inner_steps": 0,
    "adapt_scope": "all",
    "validate_every": 100,
    "validation_runs": 5,
    "validation_seed": 1729,
    "patience": 41,
    "min_delta": 1e-4,
    "loss": "smooth_l1",
    "data_config": "configs/data.json",
    "purpose": "research",
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: object, minimum: float | None = None) -> float:
    _require(type(value) in (int, float), "Expected finite numeric value, not bool")
    _require(math.isfinite(value), "Nonfinite numeric value")
    _require(minimum is None or value >= minimum, "Numeric value below minimum")
    return float(value)


def _integer(value: object, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, "Invalid integer count")
    return value


def _hash(value: object) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "Invalid SHA256",
    )
    return value


def _endpoint(endpoint: dict, formal: bool) -> dict:
    try:
        per_db = endpoint["per_database"]
        _require(
            set(per_db) == set(DATABASES), "Exact source endpoint coverage required"
        )
        total = 0
        for db in DATABASES:
            _require(set(per_db[db]) == set(BUCKETS), "Invalid endpoint buckets")
            counts = []
            for bucket in BUCKETS:
                cell = per_db[db][bucket]
                counts.append(_integer(cell["queries"], 1))
                q50, q95 = (_number(cell[k], 1) for k in ("q50", "q95"))
                _require(q50 <= q95, "Q percentiles are not monotonic")
            _require(counts[1] <= counts[0], "Invalid subset coverage")
            _require(
                not formal or tuple(counts) == COVERAGE[db],
                "Formal endpoint denominator changed",
            )
            total += counts[0]
        _require(
            _integer(endpoint["unique_queries"], 1) == total,
            "Unique denominator mismatch",
        )
        _hash(endpoint["identity_digest"])
        if "pg_identity_digest" in endpoint:
            _hash(endpoint["pg_identity_digest"])
        for bucket, name in zip(
            BUCKETS, ("macro_all_table_q95", "macro_four_plus_q95")
        ):
            actual = _number(endpoint[name], 1)
            expected = float(np.mean([per_db[db][bucket]["q95"] for db in DATABASES]))
            _require(
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                "Macro is not equal-weight arithmetic Q95",
            )
        return endpoint
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete endpoint") from error


def _run(record: dict, formal: bool) -> None:
    try:
        _require(type(formal) is bool, "formal must be bool")
        arm = record["arm"]
        _require(arm in ARMS, "Invalid S4 arm")
        config = record["config"]
        expected = {
            **FROZEN_CONFIG,
            **(
                {
                    "steps": 2,
                    "support_sizes": [5],
                    "query_size": 8,
                    "validate_every": 1,
                    "validation_runs": 1,
                }
                if not formal
                else {}
            ),
        }
        _require(
            set(config) == set(expected) | {"model"},
            "Unexpected or missing config field",
        )
        for key, value in expected.items():
            if key == "purpose" and not formal:
                _require(
                    config[key] in ("research", "gpu-smoke-not-research"),
                    "Invalid smoke purpose",
                )
            else:
                _require(
                    _digest(config[key]) == _digest(value),
                    f"Frozen config changed: {key}",
                )
        model = config["model"]
        scale = _number(model["pg_output_scale"])
        _require(scale == (1 if arm == ARMS[0] else 0), "Arm scale mismatch")
        _require(
            {k: v for k, v in model.items() if k != "pg_output_scale"} == FROZEN_MODEL,
            "Frozen model changed",
        )
        _require(
            _digest({k: v for k, v in model.items() if k != "pg_output_scale"})
            == _digest(FROZEN_MODEL),
            "Model types changed",
        )
        steps = expected["steps"]
        counts = record["counts"]
        _require(
            set(counts)
            == {
                "optimizer_steps",
                "backward_calls",
                "support_presentations",
                "query_presentations",
            },
            "Invalid counter fields",
        )
        for key, value in (
            ("optimizer_steps", steps),
            ("backward_calls", steps * 4),
            ("support_presentations", 0),
        ):
            _require(_integer(counts[key]) == value, f"Incomplete budget: {key}")
        _require(
            _integer(record["optimizer_steps"]) == steps, "Optimizer budget changed"
        )
        _require(
            _integer(record["sample_plan_episodes"]) == steps * 4,
            "Episode budget changed",
        )
        queries = _integer(counts["query_presentations"], 1)
        _require(
            queries == (1621915 if formal else 104),
            "Query presentation budget changed",
        )
        _require(
            _hash(record["initialization_hash"]) == INITIAL_HASH,
            "Initialization changed",
        )
        _require(_integer(record["parameters"]) == 132545, "Parameter count changed")
        for key, frozen in (
            ("sample_plan_digest", SAMPLE_DIGEST),
            ("validation_plan_hash", VALIDATION_HASH),
        ):
            value = _hash(record[key])
            _require(not formal or value == frozen, f"Frozen {key} changed")
        _require(
            set(record["manifest_hashes"]) == set(SOURCES),
            "26-source manifest coverage changed",
        )
        for value in record["manifest_hashes"].values():
            _hash(value)
        for key in ("checkpoint_sha256", "endpoint_rows_sha256"):
            _hash(record[key])
        checks = record["checkpoint_checks"]
        _require(
            checks["roundtrip_exact"] is True
            and checks["optimizer_changed_weights"] is True,
            "Checkpoint checks failed",
        )
        _require(
            _integer(checks["checkpoint_steps"]) == steps, "Checkpoint step mismatch"
        )
        _require(
            _number(checks["cpu_device_max_abs_difference"], 0) <= 1e-5,
            "Checkpoint device disagreement",
        )
        _endpoint(record["endpoint"], formal)
        for key in (
            "pg_identity_digest",
            "initial_pg_identity_digest",
            "initial_endpoint_rows_sha256",
        ):
            if key in record:
                _hash(record[key])
        if (
            "pg_identity_digest" in record
            and "pg_identity_digest" in record["endpoint"]
        ):
            _require(
                record["pg_identity_digest"]
                == record["endpoint"]["pg_identity_digest"],
                "Record/endpoint PG identity mismatch",
            )
        if "summary" in record:
            summary = record["summary"]
            _require(_integer(summary["last_step"]) == steps, "Summary step mismatch")
            _require(
                _digest(summary["counts"]) == _digest(counts), "Summary counts mismatch"
            )
            _require(
                _integer(summary["parameters"]) == 132545, "Summary parameter mismatch"
            )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete S4 record") from error


def verify_run(record: dict, preflight: dict, formal: bool) -> dict:
    _run(record, formal)
    try:
        _require(preflight["protocol_sha256"] == PROTOCOL_HASH, "Protocol SHA changed")
        sources = preflight["sources"]
        _require(set(sources) == set(SOURCES), "Preflight source coverage changed")
        _require(
            _digest(sources) == SOURCE_HASH, "Frozen 182-file source evidence changed"
        )
        _require(
            record["manifest_hashes"]
            == {db: sources[db]["manifest_hash"] for db in SOURCES},
            "Record/source manifest mismatch",
        )
        for key, version in (
            ("torch", "2.1.0+cu121"),
            ("numpy", "1.26.1"),
            ("sqlglot", "18.17.0"),
        ):
            _require(preflight[key] == version, "Frozen dependency changed")
        _require(preflight["python"].startswith("3.11.4 "), "Frozen Python changed")
        _require(bool(preflight["gpu_uuid"]), "Missing GPU UUID")
        if "gpu_uuid" in record:
            _require(
                record["gpu_uuid"] == preflight["gpu_uuid"], "GPU binding mismatch"
            )
        _hash(preflight["config_sha256"])
        _require(bool(preflight["code_sha256"]), "Missing source hash binding")
        for value in preflight["code_sha256"].values():
            _hash(value)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete preflight") from error
    return {"passed": True, "formal": formal, "arm": record["arm"]}


def verify_pair(records: list, formal: bool) -> dict:
    _require(
        isinstance(records, list) and len(records) == 2, "Exactly two S4 arms required"
    )
    for record in records:
        _run(record, formal)
    _require({r["arm"] for r in records} == set(ARMS), "Exact A/B pair required")
    a, b = ({r["arm"]: r for r in records}[arm] for arm in ARMS)
    configs = [
        {
            **r["config"],
            "model": {
                k: v for k, v in r["config"]["model"].items() if k != "pg_output_scale"
            },
        }
        for r in (a, b)
    ]
    _require(_digest(configs[0]) == _digest(configs[1]), "Paired train config changed")
    for key in (
        "counts",
        "sample_plan_digest",
        "sample_plan_episodes",
        "validation_plan_hash",
        "manifest_hashes",
        "initialization_hash",
        "parameters",
    ):
        _require(_digest(a[key]) == _digest(b[key]), f"Pair mismatch: {key}")
    for key in (
        "gpu_uuid",
        "pg_identity_digest",
        "initial_pg_identity_digest",
        "sources",
        "dependencies",
    ):
        _require((key in a) == (key in b), f"Incomplete pair field: {key}")
        if key in a:
            _require(_digest(a[key]) == _digest(b[key]), f"Pair mismatch: {key}")
    for key in ("identity_digest", "pg_identity_digest"):
        _require(
            a["endpoint"].get(key) == b["endpoint"].get(key),
            f"Endpoint pair mismatch: {key}",
        )
    _require(
        "pg_identity_digest" in a or "pg_identity_digest" in a["endpoint"],
        "Missing paired PG identity",
    )
    if "initial_endpoint" in a or "initial_endpoint" in b:
        _require(
            "initial_endpoint" in a and "initial_endpoint" in b,
            "Missing initial endpoint",
        )
        for r in (a, b):
            _endpoint(r["initial_endpoint"], formal)
        _require(
            a["initial_endpoint"]["identity_digest"]
            == b["initial_endpoint"]["identity_digest"]
            == a["endpoint"]["identity_digest"],
            "Initial/final identity mismatch",
        )
    return {"passed": True, "formal": formal, "arms": list(ARMS)}


def choose(records: list) -> dict:
    verify_pair(records, True)
    a, b = ({r["arm"]: r["endpoint"] for r in records}[arm] for arm in ARMS)
    improvement = 1 - b["macro_four_plus_q95"] / a["macro_four_plus_q95"]
    primary = b["macro_four_plus_q95"] <= 0.9 * a["macro_four_plus_q95"]
    guardrail = all(
        b["per_database"][db][bucket]["q95"]
        <= 1.1 * a["per_database"][db][bucket]["q95"]
        for db in DATABASES
        for bucket in BUCKETS
    )
    return {
        "decision": "promising_single_seed"
        if primary and guardrail
        else "no_candidate",
        "baseline": ARMS[0],
        "candidates": [ARMS[1]] if primary and guardrail else [],
        "relative_improvement": improvement,
        "primary_improvement": {ARMS[0]: 0.0, ARMS[1]: improvement},
    }


def verify_baseline(record: dict, historical_summary: dict) -> dict:
    _run(record, True)
    _require(
        record["arm"] == ARMS[0], "reproduction_blocked: baseline must be current A"
    )
    try:
        old = historical_summary["runs"][0]
        _require(
            old["arm"] == "A_w64_smooth", "reproduction_blocked: historical A missing"
        )
        historical = _endpoint(old["endpoint"], True)
        actual = record["endpoint"]
        pairs = [
            (actual[name], historical[name])
            for name in ("macro_four_plus_q95", "macro_all_table_q95")
        ]
        pairs += [
            (
                actual["per_database"][db][bucket][q],
                historical["per_database"][db][bucket][q],
            )
            for db in DATABASES
            for bucket in BUCKETS
            for q in ("q50", "q95")
        ]
        _require(
            all(math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-8) for a, b in pairs),
            "reproduction_blocked: A metrics exceed frozen tolerance",
        )
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "reproduction_blocked: incomplete historical summary"
        ) from error
    return {"passed": True, "status": "reproduced", "rtol": 1e-5, "atol": 1e-8}


def _arrays(rows_by_db: dict) -> dict:
    _require(
        isinstance(rows_by_db, dict) and tuple(rows_by_db) == DATABASES,
        "Exact ordered source databases required",
    )
    result = {}
    for db, rows in rows_by_db.items():
        _require(isinstance(rows, list) and bool(rows), "Empty query rows")
        seen = set()
        for row in rows:
            try:
                key = row["query_hash"]
                _require(
                    isinstance(key, str) and bool(key) and key not in seen,
                    "Duplicate or invalid query identity",
                )
                seen.add(key)
                _require(row.get("database", db) == db, "Query database mismatch")
                _integer(row["n_tables"], 1)
                _number(row["true"], 0)
                for field in ("log_pred", "pg_input_log", "pg_reference_log"):
                    _number(row[field])
            except (KeyError, TypeError) as error:
                raise ValueError("Incomplete query row") from error
        data = {
            k: np.asarray([r[k] for r in rows], dtype=np.float64)
            for k in (
                "true",
                "n_tables",
                "log_pred",
                "pg_input_log",
                "pg_reference_log",
            )
        }
        _require(bool((data["n_tables"] >= 4).any()), "Empty primary bucket")
        result[db] = data
    return result


def _stats(data: dict, mask: np.ndarray, scale: float) -> dict:
    truth = np.log(np.maximum(data["true"][mask], 1.0))
    pred, pg, reference = (
        data[k][mask] for k in ("log_pred", "pg_input_log", "pg_reference_log")
    )
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.exp(np.abs(np.maximum(pred, 0) - truth))
        raw = pred - truth
        mse = float(np.mean(raw**2))
    _require(
        bool(np.isfinite(q).all()) and math.isfinite(mse), "Q or raw error overflow"
    )
    anchor = scale * pg
    head = pred - anchor
    correction = head + (scale - 1) * pg
    error = np.maximum(np.abs(anchor + head - pred), np.abs(correction + pg - pred))
    _require(
        bool(np.isfinite(error).all()) and float(error.max()) <= 2e-12,
        "PG accounting identity mismatch",
    )
    output = {
        "q50": float(np.percentile(q, 50)),
        "q95": float(np.percentile(q, 95)),
        "raw_log_mae": float(np.mean(np.abs(raw))),
        "raw_log_mse": mse,
        "raw_signed_log_error_mean": float(np.mean(raw)),
        "anchor_mean": float(anchor.mean()),
        "head_output_mean": float(head.mean()),
        "head_output_mode": "pg_residual_correction"
        if scale == 1
        else "direct_log_prediction",
        "total_correction_vs_pg_mean": float(correction.mean()),
        "identity_max_abs_error": float(error.max()),
        "pg_input_minus_reference_max_abs": float(np.max(np.abs(pg - reference))),
    }
    for name, values in (("pg_reference", reference), ("pg_input", pg)):
        signed = values - truth
        output = {
            **output,
            **{
                f"{name}_{label}_fraction": float(condition.mean())
                for label, condition in (
                    ("under", signed < 0),
                    ("over", signed > 0),
                    ("exact", signed == 0),
                )
            },
        }
    return output


def analyze_rows(rows_by_db: dict, scale: float) -> list[dict]:
    scale = _number(scale)
    _require(scale in (0, 1), "Scale must be zero or one")
    output = []
    for db, data in _arrays(rows_by_db).items():
        true, tables = data["true"], data["n_tables"]
        cards = {"true=0": true == 0, "true=(0,10)": (true > 0) & (true < 10)}
        cards = {
            **cards,
            **{
                f"true=[{lo},{hi})": (true >= lo) & (true < hi)
                for lo, hi in (
                    (10, 100),
                    (100, 1000),
                    (1000, 10000),
                    (10000, 100000),
                    (100000, np.inf),
                )
            },
        }
        table_masks = {
            "tables=1-3": tables <= 3,
            "tables=4": tables == 4,
            "tables=5": tables == 5,
            "tables=6plus": tables >= 6,
        }
        bases = {
            "all_tables": np.ones(len(true), dtype=bool),
            "four_plus_tables": tables >= 4,
        }
        masks = {
            **bases,
            **cards,
            **table_masks,
            **{
                f"{c}|{t}": cm & tm
                for c, cm in cards.items()
                for t, tm in table_masks.items()
            },
        }
        signed = data["pg_reference_log"] - np.log(np.maximum(true, 1))
        masks = {
            **masks,
            **{
                f"{b}:pg_{label}": bm & direction
                for b, bm in bases.items()
                for label, direction in (
                    ("under", signed < 0),
                    ("over", signed > 0),
                    ("exact", signed == 0),
                )
            },
        }
        # Validate every row via the all-row bucket, including small diagnostic cells.
        for bucket, mask in masks.items():
            n = int(mask.sum())
            detailed = bucket in BUCKETS or n >= 200
            output.append(
                {
                    "database": db,
                    "bucket": bucket,
                    "n": n,
                    "status": "descriptive" if detailed else "below_min_cell_n",
                    **(_stats(data, mask, scale) if detailed else {}),
                }
            )
    return output


def verify_rows(
    rows_by_db: dict, scale: float, recorded_endpoint: dict | None, formal: bool = True
) -> dict:
    cells = analyze_rows(rows_by_db, scale)
    identity, pg_identity, per_db = [], [], {}
    for db, rows in rows_by_db.items():
        per_db[db] = {}
        for bucket in BUCKETS:
            cell = next(
                c for c in cells if c["database"] == db and c["bucket"] == bucket
            )
            per_db[db][bucket] = {
                "queries": cell["n"],
                "q50": cell["q50"],
                "q95": cell["q95"],
            }
        for r in rows:
            key = (db, r["query_hash"], r["true"], r["n_tables"])
            identity.append(key)
            pg_identity.append((*key, r["pg_input_log"], r["pg_reference_log"]))
    endpoint = {
        "per_database": per_db,
        "identity_digest": _digest(sorted(identity)),
        "pg_identity_digest": _digest(pg_identity),
        "unique_queries": len(identity),
        "macro_four_plus_q95": float(
            np.mean([per_db[db][BUCKETS[1]]["q95"] for db in DATABASES])
        ),
        "macro_all_table_q95": float(
            np.mean([per_db[db][BUCKETS[0]]["q95"] for db in DATABASES])
        ),
    }
    _endpoint(endpoint, formal)
    if recorded_endpoint is not None:
        _endpoint(recorded_endpoint, formal)
        for key in ("identity_digest", "unique_queries"):
            _require(
                recorded_endpoint[key] == endpoint[key],
                "Recorded query identity mismatch",
            )
        if "pg_identity_digest" in recorded_endpoint:
            _require(
                recorded_endpoint["pg_identity_digest"]
                == endpoint["pg_identity_digest"],
                "Recorded ordered PG identity mismatch",
            )
        for db in DATABASES:
            for bucket in BUCKETS:
                for key in ("queries", "q50", "q95"):
                    a, b = (
                        recorded_endpoint["per_database"][db][bucket][key],
                        endpoint["per_database"][db][bucket][key],
                    )
                    _require(
                        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12),
                        "Recorded endpoint metric mismatch",
                    )
    return {"passed": True, "formal": formal, "endpoint": endpoint, "analysis": cells}
