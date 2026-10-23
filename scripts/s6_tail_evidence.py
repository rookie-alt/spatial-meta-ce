"""Independent S6 gates; no training, model or target-data imports.
Caller authenticates record, row, loss-audit and snapshot file bytes before
these content gates; passing flags alone do not authenticate execution.
"""

import math

from scripts.s4_anchor_evidence import (
    FROZEN_CONFIG,
    FROZEN_MODEL,
    INITIAL_HASH,
    SAMPLE_DIGEST,
    VALIDATION_HASH,
    SOURCE_HASH,
    SOURCES,
    DATABASES,
    BUCKETS,
    COVERAGE as COVERAGE,
    _digest,
    _endpoint,
    _hash,
    _integer,
    _number,
    _require,
    verify_rows as verify_rows,
    analyze_rows as analyze_rows,
)

ARMS = ("A_mean", "B_mean_tail10")
PROTOCOL_HASH = "9420a8e98664eb037ea95fa27d1e198241c99e624eeeb854253bd4d387b6c96e"


def _config(record: dict, formal: bool) -> int:
    _require(type(formal) is bool, "formal must be bool")
    arm = record["arm"]
    _require(arm in ARMS, "Invalid S6 arm")
    expected = {
        **FROZEN_CONFIG,
        **(
            {
                "steps": 2,
                "support_sizes": [5],
                "query_size": 8,
                "validate_every": 1,
                "validation_runs": 1,
                "purpose": "gpu-smoke-not-research",
            }
            if not formal
            else {}
        ),
    }
    config = record["config"]
    _require(
        set(config) == set(expected) | {"model", "tail_risk_mode"},
        "Unexpected or missing config field",
    )
    for key, value in expected.items():
        _require(
            _digest(config[key]) == _digest(value), f"Frozen config changed: {key}"
        )
    _require(
        config["tail_risk_mode"] == ("mean" if arm == ARMS[0] else "mean_tail10"),
        "Arm tail risk mode mismatch",
    )
    model = config["model"]
    _require(
        set(model) == set(FROZEN_MODEL) | {"pg_output_scale"}, "Model fields changed"
    )
    _require(_number(model["pg_output_scale"]) == 1, "S6 scale must be one")
    _require(
        _digest({k: v for k, v in model.items() if k != "pg_output_scale"})
        == _digest(FROZEN_MODEL),
        "Frozen model changed",
    )
    return expected["steps"]


def _budget(record: dict, steps: int, formal: bool) -> None:
    expected = {
        "optimizer_steps": steps,
        "backward_calls": steps * 4,
        "support_presentations": 0,
        "query_presentations": 1621915 if formal else 104,
    }
    counts = record["counts"]
    _require(set(counts) == set(expected), "Invalid counter fields")
    for key, value in expected.items():
        _require(_integer(counts[key]) == value, f"Incomplete budget: {key}")
    _require(
        _integer(record["optimizer_steps"], 1) == steps, "Optimizer budget changed"
    )
    _require(
        _integer(record["sample_plan_episodes"], 1) == steps * 4,
        "Episode budget changed",
    )
    _require(
        _hash(record["initialization_hash"]) == INITIAL_HASH, "Initialization changed"
    )
    _require(_integer(record["parameters"], 1) == 132545, "Parameter count changed")
    for key, frozen in (
        ("sample_plan_digest", SAMPLE_DIGEST),
        ("validation_plan_hash", VALIDATION_HASH),
    ):
        value = _hash(record[key])
        _require(not formal or value == frozen, f"Frozen {key} changed")
    _require(set(record["manifest_hashes"]) == set(SOURCES), "Source coverage changed")
    for value in record["manifest_hashes"].values():
        _hash(value)


def _summary(record: dict, steps: int, formal: bool) -> None:
    summary = record["summary"]
    _require(_integer(summary["last_step"], 1) == steps, "Summary step mismatch")
    _require(
        _digest(summary["counts"]) == _digest(record["counts"]),
        "Summary counts mismatch",
    )
    _require(
        _integer(summary["parameters"], 1) == 132545, "Summary parameters mismatch"
    )
    stats = summary["gradient_stats"]
    _require(
        set(stats) == {"count", "total_norm", "max_norm", "clipped_count"},
        "Invalid gradient fields",
    )
    count = _integer(stats["count"], 1)
    clipped = _integer(stats["clipped_count"])
    total, maximum = (_number(stats[k], 0) for k in ("total_norm", "max_norm"))
    _require(count == steps and clipped <= count, "Gradient count mismatch")
    _require(
        maximum <= total <= count * maximum + 1e-8, "Gradient norm accounting mismatch"
    )

    loss = summary["loss_evidence"]
    _require(
        set(loss) == {"audit_rows", "snapshot_rows", "mode"},
        "Invalid summary loss fields",
    )
    _require(_integer(loss["audit_rows"]) == steps * 4, "Loss audit budget mismatch")
    _require(
        _integer(loss["snapshot_rows"]) == (16 if formal else 8),
        "Loss snapshot budget mismatch",
    )
    _require(
        loss["mode"] == record["config"]["tail_risk_mode"], "Summary loss mode mismatch"
    )


def _loss(record: dict, steps: int, formal: bool) -> None:
    check = record["loss_verification"]
    _require(check["passed"] is True, "Independent loss verification failed")
    _require(
        check["mode"] == record["config"]["tail_risk_mode"],
        "Verified loss mode mismatch",
    )
    _require(_integer(check["episodes"]) == steps * 4, "Verified episode mismatch")
    _require(
        _integer(check["presentations"]) == (1621915 if formal else 104),
        "Verified presentation mismatch",
    )
    _require(
        _hash(check["sample_plan_digest"]) == record["sample_plan_digest"],
        "Verified sample plan mismatch",
    )
    _require(
        _integer(check["snapshot_rows"]) == (16 if formal else 8),
        "Verified snapshot mismatch",
    )
    for key in ("audit_sha256", "snapshots_sha256"):
        _hash(check[key])


def _same_endpoint(raw: dict, verified: dict) -> None:
    for key in ("identity_digest", "pg_identity_digest", "unique_queries"):
        _require(raw[key] == verified[key], "Verified/raw identity mismatch")
    pairs = [
        (raw[key], verified[key])
        for key in ("macro_all_table_q95", "macro_four_plus_q95")
    ]
    for db in DATABASES:
        for bucket in BUCKETS:
            a, b = (e["per_database"][db][bucket] for e in (raw, verified))
            _require(a["queries"] == b["queries"], "Verified/raw denominator mismatch")
            pairs += [(a[q], b[q]) for q in ("q50", "q95")]
    _require(
        all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in pairs),
        "Verified/raw endpoint metric mismatch",
    )


def _rows(record: dict, formal: bool) -> None:
    for prefix in ("", "initial_"):
        endpoint = _endpoint(record[prefix + "endpoint"], formal)
        check = record[prefix + "rows_verification"]
        _require(
            check["passed"] is True and check["formal"] is formal,
            "Independent rows verification failed",
        )
        verified = _endpoint(check["endpoint"], formal)
        _same_endpoint(endpoint, verified)
        _hash(record[prefix + "endpoint_rows_sha256"])
        identity = _hash(record[prefix + "pg_identity_digest"])
        _require(
            _hash(endpoint["pg_identity_digest"]) == identity,
            "Raw PG identity mismatch",
        )
    _require(
        record["initial_pg_identity_digest"] == record["pg_identity_digest"],
        "Initial/final PG identity mismatch",
    )
    _require(
        record["initial_endpoint"]["identity_digest"]
        == record["endpoint"]["identity_digest"],
        "Initial/final query identity mismatch",
    )


def _record(record: dict, formal: bool) -> None:
    try:
        steps = _config(record, formal)
        _budget(record, steps, formal)
        _summary(record, steps, formal)
        _loss(record, steps, formal)
        _hash(record["checkpoint_sha256"])
        checks = record["checkpoint_checks"]
        _require(
            checks["roundtrip_exact"] is True
            and checks["optimizer_changed_weights"] is True,
            "Checkpoint checks failed",
        )
        _require(
            _integer(checks["checkpoint_steps"], 1) == steps, "Checkpoint step mismatch"
        )
        _require(
            _number(checks["cpu_device_max_abs_difference"], 0) <= 1e-5,
            "Checkpoint device disagreement",
        )
        _require(
            isinstance(record["gpu_uuid"], str)
            and record["gpu_uuid"].startswith("GPU-")
            and len(record["gpu_uuid"]) > 4,
            "Missing GPU UUID",
        )
        _rows(record, formal)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete S6 record") from error


def verify_run(record: dict, preflight: dict, formal: bool) -> dict:
    """Validate S6 content; caller separately authenticates file SHA256s."""
    _record(record, formal)
    try:
        _require(preflight["protocol_sha256"] == PROTOCOL_HASH, "Protocol SHA changed")
        sources = preflight["sources"]
        _require(
            set(sources) == set(SOURCES) and _digest(sources) == SOURCE_HASH,
            "Frozen 182-file source evidence changed",
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
        _require(record["gpu_uuid"] == preflight["gpu_uuid"], "GPU binding mismatch")
        _hash(preflight["config_sha256"])
        _require(bool(preflight["code_sha256"]), "Missing source hash binding")
        for value in preflight["code_sha256"].values():
            _hash(value)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete preflight") from error
    return {"passed": True, "formal": formal, "arm": record["arm"]}


def verify_pair(records: list[dict], formal: bool) -> dict:
    """Require a complete paired S6 run, including independent loss evidence."""
    _require(
        isinstance(records, list) and len(records) == 2, "Exactly two S6 arms required"
    )
    for record in records:
        _record(record, formal)
    _require({r["arm"] for r in records} == set(ARMS), "Exact A/B pair required")
    a, b = ({r["arm"]: r for r in records}[arm] for arm in ARMS)
    configs = [
        {k: v for k, v in r["config"].items() if k != "tail_risk_mode"} for r in (a, b)
    ]
    _require(_digest(configs[0]) == _digest(configs[1]), "Paired config changed")
    for key in (
        "counts",
        "sample_plan_digest",
        "sample_plan_episodes",
        "validation_plan_hash",
        "manifest_hashes",
        "initialization_hash",
        "parameters",
        "gpu_uuid",
        "pg_identity_digest",
        "initial_pg_identity_digest",
        "initial_endpoint",
    ):
        _require(_digest(a[key]) == _digest(b[key]), f"Pair mismatch: {key}")
    for key in ("sources", "dependencies"):
        _require((key in a) == (key in b), f"Incomplete pair field: {key}")
        if key in a:
            _require(_digest(a[key]) == _digest(b[key]), f"Pair mismatch: {key}")
    _require(
        a["endpoint"]["identity_digest"] == b["endpoint"]["identity_digest"],
        "Endpoint pair identity mismatch",
    )
    return {"passed": True, "formal": formal, "arms": list(ARMS)}


def choose(records: list[dict]) -> dict:
    """Apply frozen thresholds after complete formal pair verification."""
    verify_pair(records, True)
    a, b = ({r["arm"]: r["endpoint"] for r in records}[arm] for arm in ARMS)
    improvement = 1 - b["macro_four_plus_q95"] / a["macro_four_plus_q95"]
    primary = b["macro_four_plus_q95"] <= 0.9 * a["macro_four_plus_q95"]
    guards = {
        q: all(
            b["per_database"][db][bucket][q] <= 1.1 * a["per_database"][db][bucket][q]
            for db in DATABASES
            for bucket in BUCKETS
        )
        for q in ("q50", "q95")
    }
    passed = primary and all(guards.values())
    return {
        "decision": "promising_single_seed" if passed else "no_candidate",
        "baseline": ARMS[0],
        "candidates": [ARMS[1]] if passed else [],
        "relative_improvement": improvement,
        "primary_improvement": {ARMS[0]: 0.0, ARMS[1]: improvement},
        "q95_guardrail": guards["q95"],
        "q50_guardrail": guards["q50"],
    }


def verify_baseline(record: dict, historical_summary: dict) -> dict:
    """Reproduce all fourteen historical S5 A metrics."""
    try:
        _record(record, True)
        _require(record["arm"] == ARMS[0], "baseline must be current A")
        old = historical_summary["runs"][0]
        _require(old["arm"] == "A_uniform", "historical S5 A missing")
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
            "A metrics exceed frozen tolerance",
        )
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError(f"reproduction_blocked: {error}") from error
    return {"passed": True, "status": "reproduced", "rtol": 1e-5, "atol": 1e-8}
