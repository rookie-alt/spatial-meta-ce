"""Independent S5 gates; no training, model or target-data imports.
Caller authenticates record and row-file bytes before these content gates.
"""

from collections import Counter
import hashlib
import math

import numpy as np

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
from scripts.s2_source_replay import TRAIN

ARMS = ("A_uniform", "B_complex2")
PROTOCOL_HASH = "4ac019d51b384aba7fc616bbdadc536439a33e633596f011f4d78b48011a1d56"
MASS_KEYS = ("a_low_mass", "a_high_mass", "b_low_mass", "b_high_mass")


def _config(record, formal):
    _require(type(formal) is bool, "formal must be bool")
    arm = record["arm"]
    _require(arm in ARMS, "Invalid S5 arm")
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
        set(config) == set(expected) | {"model", "complex_query_weight"},
        "Unexpected or missing config field",
    )
    for key, value in expected.items():
        _require(
            _digest(config[key]) == _digest(value), f"Frozen config changed: {key}"
        )
    _require(
        _number(config["complex_query_weight"]) == (1 if arm == ARMS[0] else 2),
        "Arm weight mismatch",
    )
    model = config["model"]
    _require(
        set(model) == set(FROZEN_MODEL) | {"pg_output_scale"}, "Model fields changed"
    )
    _require(_number(model["pg_output_scale"]) == 1, "S5 scale must be one")
    _require(
        _digest({k: v for k, v in model.items() if k != "pg_output_scale"})
        == _digest(FROZEN_MODEL),
        "Frozen model changed",
    )
    return expected["steps"]


def _budget(record, steps, formal):
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


def _summary(record, steps):
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


def _same_endpoint(raw, verified):
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


def _rows(record, formal):
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


def _record(record, formal):
    try:
        steps = _config(record, formal)
        _budget(record, steps, formal)
        _summary(record, steps)
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
        raise ValueError("Incomplete S5 record") from error


def verify_run(record, preflight, formal):
    """Validate S5 content; caller separately authenticates file SHA256s."""
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


def verify_pair(records, formal):
    _require(
        isinstance(records, list) and len(records) == 2, "Exactly two S5 arms required"
    )
    for record in records:
        _record(record, formal)
    _require({r["arm"] for r in records} == set(ARMS), "Exact A/B pair required")
    a, b = ({r["arm"]: r for r in records}[arm] for arm in ARMS)
    configs = [
        {k: v for k, v in r["config"].items() if k != "complex_query_weight"}
        for r in (a, b)
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


def choose(records):
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


def verify_baseline(record, historical_summary):
    try:
        _record(record, True)
        _require(record["arm"] == ARMS[0], "baseline must be current A")
        old = historical_summary["runs"][0]
        _require(old["arm"] == "A_anchor1", "historical S4 A missing")
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


def _flow_inputs(stores, config):
    _require(set(stores) == set(TRAIN), "Exact source training databases required")
    for key in ("steps", "meta_batch", "query_size", "accumulation"):
        _integer(config[key], 1)
    _integer(config["seed"])
    _require(
        config["meta_batch"] == 4 and config["accumulation"] == 1,
        "S5 coefficient budget requires meta_batch=4 and accumulation=1",
    )
    sizes = config["support_sizes"]
    _require(isinstance(sizes, list) and bool(sizes), "Missing support sizes")
    for k in sizes:
        _integer(k, 1)
    _require(len(set(sizes)) == len(sizes), "Duplicate support sizes")
    for db in TRAIN:
        store = stores[db]
        hashes, unique, tables = (
            store[k] for k in ("query_hashes", "unique", "tables")
        )
        _require(
            isinstance(hashes, list)
            and bool(hashes)
            and all(isinstance(h, str) and bool(h) for h in hashes),
            "Invalid query hashes",
        )
        _require(
            isinstance(unique, np.ndarray)
            and unique.ndim == 1
            and unique.dtype.kind in "iu",
            "Invalid unique indices",
        )
        _require(
            isinstance(tables, np.ndarray)
            and tables.shape == (len(hashes),)
            and tables.dtype.kind in "iu"
            and bool((tables >= 1).all()),
            "Invalid table counts",
        )
        _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
        _require(
            np.array_equal(tables, tables[first[inverse]]),
            "Conflicting duplicate table counts",
        )
        expected = np.sort(first)
        _require(np.array_equal(unique, expected), "First-occurrence indices changed")
        _require(
            len(unique) >= max(sizes) + config["query_size"], "Insufficient unique pool"
        )


def _counter_verification(counter, states, steps):
    independent = {}
    for db in TRAIN:
        groups = [
            (n, h, count) for (name, n, h), count in counter.items() if name == db
        ]
        masses = (
            math.fsum(count * (n - h) / (4 * n) for n, h, count in groups),
            math.fsum(count * h / (4 * n) for n, h, count in groups),
            math.fsum(count * (n - h) / (4 * (n + h)) for n, h, count in groups),
            math.fsum(count * 2 * h / (4 * (n + h)) for n, h, count in groups),
        )
        for key, value in zip(MASS_KEYS, masses):
            _require(
                math.isclose(states[db][key], value, rel_tol=0, abs_tol=1e-8),
                "Independent coefficient mass mismatch",
            )
        _require(
            states[db]["low_presented"] == sum(c * (n - h) for n, h, c in groups)
            and states[db]["high_presented"] == sum(c * h for n, h, c in groups)
            and states[db]["mixed_episodes"]
            == sum(c for n, h, c in groups if 0 < h < n),
            "Independent presentation count mismatch",
        )
        independent = {**independent, db: dict(zip(MASS_KEYS, masses))}
    totals = [
        math.fsum(s[k] for s in independent.values() for k in pair)
        for pair in (MASS_KEYS[:2], MASS_KEYS[2:])
    ]
    _require(
        all(math.isclose(v, steps, rel_tol=0, abs_tol=1e-8) for v in totals),
        "Nominal coefficient conservation failed",
    )
    return {
        "passed": True,
        "algorithm": "integer_db_N_H_counter",
        "atol": 1e-8,
        "per_database": independent,
    }


def audit_flow(stores, config):
    """Replay raw first-occurrence sampling and both nominal loss coefficients.

    N=K+Q for ERM; only table counts determine weights. No labels or forward pass.
    Full-budget runs must match the exact frozen digest, never an approximation.
    """
    try:
        _flow_inputs(stores, config)
        formal = config["steps"] == FROZEN_CONFIG["steps"]
        if formal:
            for key, value in FROZEN_CONFIG.items():
                _require(
                    _digest(config[key]) == _digest(value),
                    f"Frozen flow config changed: {key}",
                )
        rng = np.random.default_rng(config["seed"] + 1)
        previous = hashlib.sha256(b"").hexdigest()
        states = {
            db: {
                "mixed_episodes": 0,
                "low_presented": 0,
                "high_presented": 0,
                **dict.fromkeys(MASS_KEYS, 0.0),
            }
            for db in TRAIN
        }
        counter = Counter()
        episodes = config["steps"] * config["meta_batch"]
        for _ in range(episodes):
            db = str(rng.choice(TRAIN))
            k = int(rng.choice(config["support_sizes"]))
            n = k + config["query_size"]
            raw = rng.choice(stores[db]["unique"], size=n, replace=False)
            hashes = stores[db]["query_hashes"]
            episode = {
                "db": db,
                "support": [hashes[int(i)] for i in raw[:k]],
                "query": [hashes[int(i)] for i in raw[k:]],
            }
            previous = _digest({"previous": previous, "episode": episode})
            high = int(np.count_nonzero(stores[db]["tables"][raw] >= 4))
            low = n - high
            masses = (
                low / (4 * n),
                high / (4 * n),
                low / (4 * (n + high)),
                2 * high / (4 * (n + high)),
            )
            old = states[db]
            states = {
                **states,
                db: {
                    "mixed_episodes": old["mixed_episodes"] + int(0 < high < n),
                    "low_presented": old["low_presented"] + low,
                    "high_presented": old["high_presented"] + high,
                    **{key: old[key] + value for key, value in zip(MASS_KEYS, masses)},
                },
            }
            counter = counter + Counter({(db, n, high): 1})
        mixed = sum(s["mixed_episodes"] for s in states.values())
        _require(mixed > 0, "no_effective_intervention")
        presentations = sum(
            s["low_presented"] + s["high_presented"] for s in states.values()
        )
        _require(
            not formal
            or (
                previous == SAMPLE_DIGEST
                and episodes == 16000
                and presentations == 1621915
            ),
            "Frozen formal sample plan mismatch",
        )
        verification = _counter_verification(counter, states, config["steps"])
        totals = [
            math.fsum(s[k] for s in states.values() for k in pair)
            for pair in (MASS_KEYS[:2], MASS_KEYS[2:])
        ]
        _require(
            all(
                math.isclose(v, config["steps"], rel_tol=0, abs_tol=1e-8)
                for v in totals
            ),
            "Replay coefficient conservation failed",
        )
    except (KeyError, TypeError, AttributeError, IndexError) as error:
        raise ValueError("Incomplete flow inputs") from error
    return {
        "schema": 1,
        "passed": True,
        "sample_plan_digest": previous,
        "episodes": episodes,
        "presentations": presentations,
        "nominal_mass_A": totals[0],
        "nominal_mass_B": totals[1],
        "mixed_episodes": mixed,
        "per_database": states,
        "verification": verification,
    }
