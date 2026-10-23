"""Content-only S6 loss verification; no production loss or training helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.s2_source_replay import EXPECTED_DIGEST, TRAIN

AUDIT_FIELDS = frozenset(
    "step episode_index database k n m tau h e mode mean_loss tail_mean mixed_loss coefficient_sum coefficient_square_sum low_mass high_mass under_mass over_mass exact_mass sample_plan_digest".split()
)
SAMPLE_FIELDS = frozenset(
    "query_hash role log_pred label loss coefficient n_tables".split()
)
MASS_FIELDS = (
    "coefficient_sum",
    "low_mass",
    "high_mass",
    "under_mass",
    "over_mass",
    "exact_mass",
)
RTOL, ATOL = 1e-6, 1e-6
FORMAL_STEPS = 4000
FORMAL_PRESENTATIONS = 1621915


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid integer: {name}")
    return value


def _number(value, name, nonnegative=True):
    _require(
        type(value) in (int, float) and np.isfinite(value),
        f"Invalid finite number: {name}",
    )
    _require(not nonnegative or value >= 0, f"Negative number: {name}")
    return float(value)


def _fp32(value, name, nonnegative=False):
    number = _number(value, name, nonnegative)
    _require(abs(number) <= np.finfo(np.float32).max, f"FP32 overflow: {name}")
    _require(float(np.float32(number)) == number, f"Not lossless FP32: {name}")
    return number


def _close(actual, expected, name, atol=ATOL):
    _require(
        np.allclose(actual, expected, rtol=RTOL, atol=atol), f"Numeric mismatch: {name}"
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _object(pairs):
    result = dict(pairs)
    _require(len(result) == len(pairs), "Duplicate JSON field")
    return result


def _read(path):
    # Hash exactly the bytes parsed, not a second potentially changed read.
    _require(not path.is_symlink(), "Symlink evidence is forbidden")
    raw = path.read_bytes()
    try:
        rows = [
            json.loads(line, object_pairs_hook=_object)
            for line in raw.decode("utf-8").splitlines()
        ]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSONL: {path.name}") from error
    _require(all(type(row) is dict for row in rows), "JSONL rows must be objects")
    return rows, hashlib.sha256(raw).hexdigest()


def _config(config):
    _require(type(config) is dict, "Configuration must be an object")
    for field in ("steps", "meta_batch", "accumulation", "seed", "query_size"):
        _integer(config.get(field), field, 1)
    _require(
        config["meta_batch"] == 4
        and config["accumulation"] == 1
        and config["seed"] == 42,
        "Frozen ERM sampling configuration changed",
    )
    _require(
        config.get("tail_risk_mode") in ("mean", "mean_tail10"),
        "Invalid tail risk mode",
    )
    sizes = config.get("support_sizes")
    _require(type(sizes) is list and bool(sizes), "Missing support sizes")
    for size in sizes:
        _integer(size, "support size", 1)
    _require(len(set(sizes)) == len(sizes), "Duplicate support sizes")
    if config["steps"] == FORMAL_STEPS:
        _require(
            config["query_size"] == 64 and sizes == [5, 10, 20, 50, 100],
            "Frozen formal budget changed",
        )


def _stores(stores, config):
    _require(set(stores) == set(TRAIN), "Exactly the 23 TRAIN sources are required")
    for db in TRAIN:
        store = stores[db]
        _require(
            type(store) is dict
            and set(store) == {"query_hashes", "unique", "tables", "labels"},
            f"Invalid store schema: {db}",
        )
        keys, unique, tables, labels = (
            store[k] for k in ("query_hashes", "unique", "tables", "labels")
        )
        _require(
            type(keys) is list
            and bool(keys)
            and all(type(k) is str and k for k in keys),
            f"Invalid cache identities: {db}",
        )
        _require(
            isinstance(unique, np.ndarray)
            and unique.ndim == 1
            and unique.dtype.kind in "iu",
            f"Invalid unique indices: {db}",
        )
        _require(
            isinstance(tables, np.ndarray)
            and tables.shape == (len(keys),)
            and tables.dtype.kind in "iu"
            and np.all(tables >= 1),
            f"Invalid tables: {db}",
        )
        _require(
            isinstance(labels, np.ndarray)
            and labels.shape == (len(keys),)
            and labels.dtype == np.float64
            and np.isfinite(labels).all()
            and np.all(labels >= 0),
            f"Invalid FP64-promoted labels: {db}",
        )
        _require(
            np.all(labels <= np.finfo(np.float32).max)
            and np.array_equal(labels.astype(np.float32).astype(np.float64), labels),
            f"Labels are not original FP32: {db}",
        )
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        _require(
            np.array_equal(unique, np.sort(first)),
            f"First-occurrence indices changed: {db}",
        )
        _require(
            np.array_equal(labels, labels[first[inverse]])
            and np.array_equal(tables, tables[first[inverse]]),
            f"Conflicting duplicate cache identity: {db}",
        )
        _require(
            len(unique) >= max(config["support_sizes"]) + config["query_size"],
            f"Insufficient source pool: {db}",
        )


def _coefficients(n, m, h, e, mode):
    # Reproduce FP32 scalar coefficients before FP64 accumulation.
    tail = np.array([1 / m, (m - h) / (m * e), 0], dtype=np.float32)
    combined = (
        np.full(3, 1 / n, dtype=np.float32)
        if mode == "mean"
        else np.float32(0.5 / n) + np.float32(0.5) * tail
    )
    return tail, combined


def _audit(row, identity, mode):
    _require(
        set(row) == AUDIT_FIELDS,
        "Invalid audit schema (including untrusted passed flags)",
    )
    for field in ("step", "episode_index", "k", "n", "m", "h", "e"):
        _integer(row[field], field)
    _require(
        all(row[field] == value for field, value in identity.items()),
        "Audit sampling identity/order mismatch",
    )
    _require(row["mode"] == mode, "Audit mode mismatch")
    n, m, h, e = (row[field] for field in ("n", "m", "h", "e"))
    _require(
        m == (n + 9) // 10 and 0 <= h < m and m - h <= e <= n - h, "Invalid tail counts"
    )
    for field in (
        "tau",
        "mean_loss",
        "tail_mean",
        "mixed_loss",
        "coefficient_square_sum",
        *MASS_FIELDS,
    ):
        _number(row[field], field)
    for field in ("tau", "mean_loss", "tail_mean", "mixed_loss"):
        _fp32(row[field], field, True)
    _require(
        row["tail_mean"] + ATOL + RTOL * row["mean_loss"] >= row["mean_loss"],
        "Tail mean below mean",
    )
    expected = (
        row["mean_loss"]
        if mode == "mean"
        else 0.5 * row["mean_loss"] + 0.5 * row["tail_mean"]
    )
    _close(row["mixed_loss"], expected, "mixed loss")
    for total in (
        row["coefficient_sum"],
        row["low_mass"] + row["high_mass"],
        row["under_mass"] + row["over_mass"] + row["exact_mass"],
    ):
        _require(abs(total - 1) <= ATOL, "Episode mass is not one")
    _, coeff = _coefficients(n, m, h, e, mode)
    counts = np.array([h, e, n - h - e], dtype=np.int64)
    _close(
        row["coefficient_square_sum"],
        np.sum(counts * coeff.astype(np.float64) ** 2),
        "Coefficient square sum",
    )


def _snapshot(snapshot, row, store, chosen):
    _require(
        set(snapshot) == {"audit", "samples"}
        and _digest(snapshot["audit"]) == _digest(row),
        "Snapshot audit mismatch/order",
    )
    samples, n, k = snapshot["samples"], row["n"], row["k"]
    _require(
        type(samples) is list and len(samples) == n,
        "Snapshot must contain complete support+query",
    )
    for position, (sample, index) in enumerate(zip(samples, chosen)):
        _require(
            type(sample) is dict and set(sample) == SAMPLE_FIELDS,
            "Invalid sample schema",
        )
        _require(
            sample["query_hash"] == store["query_hashes"][index]
            and sample["role"] == ("support" if position < k else "query"),
            "Sample identity/role/order mismatch",
        )
        _integer(sample["n_tables"], "n_tables", 1)
        _require(sample["n_tables"] == store["tables"][index], "Cache table mismatch")
        for field in ("log_pred", "label", "loss", "coefficient"):
            _fp32(sample[field], field, field != "log_pred")
        _require(sample["label"] == store["labels"][index], "Cache label mismatch")
    pred, label, loss, coeff = (
        np.array([s[field] for s in samples], dtype=np.float64)
        for field in ("log_pred", "label", "loss", "coefficient")
    )
    absolute = np.abs(pred - label)
    # FP64 independent arithmetic; stored FP32 loss alone determines ties.
    independent = np.where(absolute < 1, 0.5 * absolute**2, absolute - 0.5)
    _close(loss, independent, "Independent SmoothL1", atol=1e-7)
    tau = np.sort(loss)[-row["m"]]
    high, equal = loss > tau, loss == tau
    h, e = int(high.sum()), int(equal.sum())
    _require(
        row["tau"] == tau and row["h"] == h and row["e"] == e,
        "Actual FP32 tail threshold/ties mismatch",
    )
    tail_values, combined = _coefficients(n, row["m"], h, e, row["mode"])
    groups = np.where(high, 0, np.where(equal, 1, 2))
    expected_coeff = combined[groups].astype(np.float64)
    _close(coeff, expected_coeff, "Sample coefficients")
    _require(abs(coeff.sum() - 1) <= ATOL, "Snapshot coefficient mass is not one")
    _close(row["mean_loss"], loss.mean(), "Snapshot mean")
    _close(
        row["tail_mean"],
        np.sum(tail_values[groups].astype(np.float64) * loss),
        "Snapshot tail mean",
    )
    _close(row["mixed_loss"], np.sum(coeff * loss), "FP64 independent mixed reduction")
    tables = store["tables"][chosen]
    mass = dict(
        coefficient_sum=coeff.sum(),
        low_mass=coeff[tables < 4].sum(),
        high_mass=coeff[tables >= 4].sum(),
        under_mass=coeff[pred < label].sum(),
        over_mass=coeff[pred > label].sum(),
        exact_mass=coeff[pred == label].sum(),
        coefficient_square_sum=np.sum(coeff**2),
    )
    for field, value in mass.items():
        _close(row[field], value, f"Snapshot {field}")


def verify_loss_evidence(directory, config, stores) -> dict:
    """Return content checks, never execution certification.

    Per-database masses include the fixed outer 1/4 episode factor. Full audit
    replay binds all support and query identities; prescribed snapshots alone
    permit independent prediction/label/loss checks.
    """
    _config(config)
    _stores(stores, config)
    directory = Path(directory)
    audit, audit_sha = _read(directory / "loss-audit.jsonl")
    snapshots, snapshots_sha = _read(directory / "loss-snapshots.jsonl")
    episodes = config["steps"] * 4
    _require(len(audit) == episodes, "Missing/extra audit episodes")
    snapshot_steps = (
        {1, 1000, 2000, 4000}
        if config["steps"] == FORMAL_STEPS
        else set(range(1, config["steps"] + 1))
    )
    _require(len(snapshots) == len(snapshot_steps) * 4, "Missing/extra fixed snapshots")
    rng = np.random.default_rng(config["seed"] + 1)
    previous = hashlib.sha256(b"").hexdigest()
    totals = {
        db: {"episodes": 0, "presentations": 0, **dict.fromkeys(MASS_FIELDS, 0.0)}
        for db in TRAIN
    }
    presentations, snapshot_index = 0, 0
    for episode, row in enumerate(audit, 1):
        db = str(rng.choice(TRAIN))
        k = int(rng.choice(config["support_sizes"]))
        n, step = k + config["query_size"], (episode - 1) // 4 + 1
        store = stores[db]
        chosen = rng.choice(store["unique"], n, replace=False)
        hashes = [store["query_hashes"][i] for i in chosen]
        previous = _digest(
            {
                "previous": previous,
                "episode": {"db": db, "support": hashes[:k], "query": hashes[k:]},
            }
        )
        identity = dict(
            step=step,
            episode_index=episode,
            database=db,
            k=k,
            n=n,
            sample_plan_digest=previous,
        )
        _audit(row, identity, config["tail_risk_mode"])
        if step in snapshot_steps:
            _snapshot(snapshots[snapshot_index], row, store, chosen)
            snapshot_index += 1
        current = totals[db]
        totals = {
            **totals,
            db: {
                "episodes": current["episodes"] + 1,
                "presentations": current["presentations"] + n,
                **{field: current[field] + row[field] / 4 for field in MASS_FIELDS},
            },
        }
        presentations += n
    if config["steps"] == FORMAL_STEPS:
        _require(
            episodes == 16000
            and presentations == FORMAL_PRESENTATIONS
            and previous == EXPECTED_DIGEST,
            "Historical formal sampling budget/digest mismatch",
        )
    return dict(
        mode=config["tail_risk_mode"],
        episodes=episodes,
        presentations=presentations,
        sample_plan_digest=previous,
        snapshot_rows=snapshot_index,
        passed=True,
        per_database=totals,
        audit_sha256=audit_sha,
        snapshots_sha256=snapshots_sha,
        verification_scope="Content verification only: full audit sampling replay and nominal mass checks; independent SmoothL1 and coefficients only for prescribed complete snapshots. Fixed snapshots cannot authenticate all 16000 forward passes; no execution, gradient or checkpoint certification.",
    )
