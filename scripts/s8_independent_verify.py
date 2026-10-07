"""Read-only, independent authentication of S8 G1 smoke artifacts.

No production contract/probe/metric/train/state imports. Predictions are *raw
natural-log cardinalities*, not cardinalities or residuals. This API authenticates
supplied unique-source arrays, not their on-disk provenance or token semantics.
It does not issue research nominations or verify a linear solver's optimality.

bindings: schema=1 preserves the original contract; schema=2 is smoke-only
and checks numerical record consistency, not an independent mathematical proof.
No summary.json is required (the runner writes it after this verification).
The database lists are
``train_databases``, ``holdout_databases``, ``validation_databases``.
``resourcefixedfields`` binds batch_size=128, threads=4,
probe_fit_per_database=64, probe_adam_steps=2. panel_identity_digest is SHA256
of JSON
(sort_keys=True, ensure_ascii=False, default separators) of {db: panel hashes}.
metrics.json: {"rows": [{database, cohort, bucket, n, status, arms: {arm:
{q50, q95, raw_log_mae, raw_log_mse}}}]}; below-200 cells omit ``arms`` entirely.
events.jsonl must have exactly one nomination_sealed with file_sha256,
followed by at least one H_semantic_load with database/hash metadata (or the
runner's nomination_sha256, which must match the sealed file).
Only recorded order is verified, not the content provenance of H event hashes;
caller-supplied caches may already have been read by the caller.
Episode steps are 1..2; global episode counters are 1..8. The SHA panel
order defines identity/fit; F0 samples sorted raw cache indices, not SHA order.
All malformed evidence raises; only completely verified smoke returns
passed=True. No files are written.
"""

# Malformed evidence consistently raises ValueError as the caller contract.
# ruff: noqa: TRY004

import hashlib
import json
import math
from pathlib import Path

import numpy as np

TRAIN = (
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "financial",
    "fnhk",
    "hepatitis",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
)
HOLDOUT = ("accidents", "grants", "employee", "hockey")
VALIDATION = ("chembl", "legalacts", "telstra")
ARMS = ("A_h_adam", "B_h_ls", "C_x_adam", "D_x_ls", "F0")
PANEL_SIZE = 1024
MIN_T_POOL = 1224
SETTINGS = {
    "schema": 1,
    "mode": "smoke",
    "purpose": "s8-engineering-smoke-not-research",
    "seed": 42,
    "train_steps": 2,
    "meta_batch": 4,
    "query_size": 64,
    "support_sizes": [5, 10, 20, 50, 100],
}
METRIC_NAMES = ("q50", "q95", "raw_log_mae", "raw_log_mse")
NUMERICAL_POLICY = "s8-g1-resource-engineering-v2"
V2_BINDINGS = ("numerical_acceptance_policy", "acceptance_protocol_sha256")
ENGINEERING_FLAGS = (
    "metadata_pass",
    "normal_residual_pass",
    "objective_pass",
    "reference_objective_pass",
    "reference_prediction_pass",
    "reference_spectrum_pass",
    "svd_backend_pass",
    "retained_subspace_pass",
    "reference_retained_subspace_pass",
    "observed_rank_agreement",
)
STRICT_CERTIFICATE = {
    "status": "unknown",
    "scopes": ["true_rank", "rigorous_retained_subspace", "strict_linear_optimality"],
    "reason": "no_verified_arithmetic_certificate",
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _read(path, lines=False):
    text = path.read_text(encoding="utf-8")
    try:
        if lines:
            # Blank rows are malformed rather than silently dropped.
            return [
                json.loads(line, object_pairs_hook=_object)
                for line in text.splitlines()
            ]
        return json.loads(text, object_pairs_hook=_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"malformed artifact: {path.name}") from exc


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _number(value, context):
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{context}: expected finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{context}: numeric overflow") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}: nonfinite number")
    return value


def _bindings(bindings):
    if not isinstance(bindings, dict):
        raise ValueError("smoke bindings must be a dict")
    schema = bindings.get("schema")
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("smoke binding mismatch: schema")
    if schema == 1:
        if any(key in bindings for key in V2_BINDINGS):
            raise ValueError("schema1 cannot contain v2 numerical bindings")
    else:
        policy = bindings.get("numerical_acceptance_policy")
        if type(policy) is not str or policy != NUMERICAL_POLICY:
            raise ValueError("smoke numerical policy binding mismatch")
        sha = bindings.get("acceptance_protocol_sha256")
        if (
            type(sha) is not str
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("smoke acceptance protocol SHA256 binding invalid")
    for key, expected in SETTINGS.items():
        if key == "schema":
            expected = schema
        actual = bindings.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"smoke binding mismatch: {key}")
        if key == "support_sizes" and any(type(v) is not int for v in actual):
            raise ValueError("smoke binding support_sizes must be integers")
    for key, expected in (
        ("train_databases", TRAIN),
        ("holdout_databases", HOLDOUT),
        ("validation_databases", VALIDATION),
    ):
        if bindings.get(key) != list(expected):
            raise ValueError(f"smoke split binding mismatch: {key}")
    resources = bindings.get("resourcefixedfields")
    fixed = {
        "batch_size": 128,
        "threads": 4,
        "probe_fit_per_database": 64,
        "probe_adam_steps": 2,
    }
    if not isinstance(resources, dict):
        raise ValueError("smoke resource binding required")
    for key, expected in fixed.items():
        if type(resources.get(key)) is not int or resources[key] != expected:
            raise ValueError(f"smoke resource binding mismatch: {key}")


def _record_field(record, key, expected, context):
    actual = record.get(key)
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{context}: invalid or missing {key}")


def _record_seconds(record, key, context):
    seconds = _number(record.get(key), f"{context}/{key}")
    if seconds < 0:
        raise ValueError(f"{context}/{key}: negative seconds")
    return seconds


def _certificate_diagnostics(cert, rows, columns, context):
    """Authenticate existing diagnostics against their flags, not the matrix."""
    for key, expected in (
        ("engineering_scope", "same_matrix_fixed_cutoff_numeric_consistency"),
        ("spectrum_specification", "s8-fp64-spectrum-v1.1"),
        ("spectrum_dtype", "float64"),
        ("spectrum_numpy_version", "1.26.1"),
    ):
        _record_field(cert, key, expected, context)
    shape = cert.get("spectrum_matrix_shape")
    if (
        type(shape) is not list
        or len(shape) != 2
        or any(not _integer(n) for n in shape)
        or shape != [rows, columns]
    ):
        raise ValueError(f"{context}: certificate matrix shape mismatch")
    # These are frozen metadata definitions, not newly chosen solver tolerances.
    eps = 2.0**-52
    d, k = max(rows, columns), min(rows, columns)
    gamma = d * eps / (1 - d * eps)
    orth_gamma = (d + k) * eps / (1 - (d + k) * eps)
    for key, expected in (
        ("spectrum_epsilon", eps),
        ("spectrum_gamma", gamma),
        ("svd_normalized_orthogonality_bound", orth_gamma),
    ):
        _record_field(cert, key, expected, context)
    values = {}
    for key in (
        "spectrum_error_bound",
        "spectrum_comparison_bound",
        "svd_scaled_reconstruction_residual",
        "svd_u_orthogonality_residual",
        "svd_v_orthogonality_residual",
        "svd_u_normalized_orthogonality_residual",
        "svd_v_normalized_orthogonality_residual",
        "objective_ls",
        "objective_adam",
        "reference_objective",
        "normalized_normal_residual",
    ):
        value = _number(cert.get(key), f"{context}/{key}")
        if value < 0:
            raise ValueError(f"{context}/{key}: negative diagnostic")
        values[key] = value
    eta = values["spectrum_error_bound"]
    if not math.isfinite(2 * eta) or values["spectrum_comparison_bound"] != 2 * eta:
        raise ValueError(f"{context}: inconsistent spectrum comparison bound")
    if values["svd_scaled_reconstruction_residual"] > gamma:
        raise ValueError(f"{context}: reconstruction flag contradicts residual")
    for side in ("u", "v"):
        raw = values[f"svd_{side}_orthogonality_residual"]
        normalized = values[f"svd_{side}_normalized_orthogonality_residual"]
        # Reuse the relative metadata precision; this is an identity check,
        # never a new acceptance threshold for the underlying solver.
        if (
            not math.isclose(normalized, raw / math.sqrt(k), rel_tol=1e-12, abs_tol=0)
            or normalized > orth_gamma
        ):
            raise ValueError(f"{context}: orthogonality diagnostic/flag mismatch")
    jl, ja, reference = (
        values[key] for key in ("objective_ls", "objective_adam", "reference_objective")
    )
    limit = ja + 1e-8 * max(1.0, ja)
    if not math.isfinite(limit) or jl > limit:
        raise ValueError(f"{context}: objective flag contradicts recorded objectives")
    if abs(jl - reference) > 1e-12 + 1e-12 * abs(reference):
        raise ValueError(f"{context}: reference objective flag contradicts record")
    if values["normalized_normal_residual"] > 1e-8:
        raise ValueError(f"{context}: normal flag contradicts recorded residual")
    intervals = cert.get("rank_cutoff_intervals")
    if type(intervals) is not dict or set(intervals) != {
        "svd",
        "reported_ls",
        "reference_ls",
    }:
        raise ValueError(f"{context}: complete cutoff intervals required")
    for bounds in [cert.get("rank_cutoff_interval"), *intervals.values()]:
        if type(bounds) is not list or len(bounds) != 2:
            raise ValueError(f"{context}: invalid cutoff interval shape")
        lo, hi = [_number(x, f"{context}/cutoff") for x in bounds]
        if not 0 <= lo <= hi:
            raise ValueError(f"{context}: invalid cutoff interval order/sign")
    if _digest(cert["rank_cutoff_interval"]) != _digest(intervals["svd"]):
        raise ValueError(f"{context}: cutoff interval alias mismatch")
    # Without spectra, only authenticate stored interval structure/margin.
    # Do not infer strict rank or replay the spectrum from these records.
    margin = _number(cert.get("rank_interval_margin"), f"{context}/margin")
    rank, stability = cert["observed_ranks"]["svd"], cert["rank_stability"]
    if (
        stability == "unknown"
        and margin > 0
        or stability == "stable"
        and margin <= 0
        and not (rank == 0 and eta == 0 and margin == 0)
    ):
        raise ValueError(f"{context}: rank stability contradicts interval margin")


def _certificate_record(cert, rows, columns, context):
    """Check reported booleans/ranks only, never re-solve or prove optimality."""
    if not isinstance(cert, dict):
        raise ValueError(f"{context}: certificate must be a dict")
    for key in (
        *ENGINEERING_FLAGS,
        "engineering_verified",
        "spectrum_pass",
        "reference_pass",
    ):
        _record_field(cert, key, True, context)
    _record_field(cert, "certification_linalg_calls", 2, context)
    ranks = cert.get("observed_ranks")
    if (
        not isinstance(ranks, dict)
        or set(ranks) != {"reported_ls", "svd", "reference_ls"}
        or any(not _integer(r) or not 0 <= r <= columns for r in ranks.values())
        or len(set(ranks.values())) != 1
    ):
        raise ValueError(f"{context}: observed ranks must be equal bounded integers")
    rank = ranks["reported_ls"]
    stability = cert.get("rank_stability")
    if not isinstance(stability, str) or stability not in ("stable", "unknown"):
        raise ValueError(f"{context}: invalid rank stability")
    passed = stability == "stable"  # All engineering conjuncts were required True.
    for key, expected in (
        ("pass", passed),
        ("solver_verified", passed),
        ("rank_limited", rank < columns),
        ("numerical_rank_limited", rank < columns),
        ("fullrank_numeric_solver", passed and rank == columns),
        ("full_linear_optimality_verified", False),
        ("full_linear_optimality", False),
    ):
        _record_field(cert, key, expected, context)
    if cert.get("strict_certificate") != STRICT_CERTIFICATE:
        raise ValueError(
            f"{context}: strict certificate must retain exact unknown scope"
        )
    _certificate_diagnostics(cert, rows, columns, context)
    return rank, stability


def _numeric_records(run_dir, bindings):
    path = run_dir / "probe-smoke.json"
    if bindings["schema"] == 1:
        if path.exists():
            record = _read(path)
            if isinstance(record, dict) and (
                record.get("schema") == 2 or any(key in record for key in V2_BINDINGS)
            ):
                raise ValueError("schema1 cannot authenticate v2 numerical records")
        return {}
    record = _read(path)
    if not isinstance(record, dict):
        raise ValueError("probe-smoke record must be a dict")
    for key, expected in (
        ("schema", 2),
        ("numerical_acceptance_policy", bindings["numerical_acceptance_policy"]),
        ("acceptance_protocol_sha256", bindings["acceptance_protocol_sha256"]),
        ("purpose", "smoke-not-research"),
        ("probe_fit_rows", 1216),
    ):
        _record_field(record, key, expected, "probe-smoke")
    for stage, group in (("probe", "profiles"), ("resource", "resource_profile")):
        profiles = record.get(group)
        if not isinstance(profiles, dict) or set(profiles) != {"h", "x"}:
            raise ValueError(f"{group}: exactly h/x profiles required")
        for feature, columns in (("h", 65), ("x", 775)):
            context = f"{feature}-{stage}"
            profile = profiles[feature]
            if not isinstance(profile, dict):
                raise ValueError(f"{context}: profile must be a dict")
            cert = profile.get("certificate")
            rank, stability = _certificate_record(
                cert, 1216 if stage == "probe" else 19456, columns, context
            )
            stage_cert = _read(run_dir / f"{context}-certificate.json")
            # Canonical JSON equality also distinguishes bool/int/float records.
            if _digest(stage_cert) != _digest(cert):
                raise ValueError(f"{context}: stage/combined certificate mismatch")
            if stage == "probe":
                for key, expected in (
                    ("fit_rows", 1216),
                    ("parameters", columns),
                    ("adam_updates", 2),
                ):
                    _record_field(profile, key, expected, context)
                _record_seconds(profile, "seconds", context)
                if cert["pass"] is not True:
                    raise ValueError(f"{context}: small probe requires legacy pass")
            else:
                for key, expected in (
                    ("resource_only", True),
                    ("not_used_for_predictions_or_selection", True),
                    ("rows", 19456),
                    ("columns", columns),
                    ("matrix_bytes", 8 * 19456 * columns),
                    ("linalg_calls", 3),
                    ("rank", rank),
                    ("full_linear_optimality_verified", False),
                    (
                        "stage_status",
                        "resource_profiled_with_rank_stability_unknown"
                        if stability == "unknown"
                        else "resource_profiled",
                    ),
                ):
                    _record_field(profile, key, expected, context)
                normalization = _record_seconds(
                    profile, "standardization_seconds", context
                )
                solver = _record_seconds(
                    profile, "solver_certification_seconds", context
                )
                total = _record_seconds(profile, "seconds", context)
                if not math.isclose(
                    total, normalization + solver, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"{context}: inconsistent seconds sum")
    return {
        "engineering_verified": True,
        "strict_certificate_status": "unknown",
        "numerical_record_checks": {
            "passed": True,
            "scope": "record_consistency_not_independent_mathematical_proof",
            "certificate_files": 4,
            "probe_profiles": 2,
            "resource_profiles": 2,
        },
    }


def _events(run_dir):
    events = _read(run_dir / "events.jsonl", lines=True)
    nomination_sha = hashlib.sha256(
        (run_dir / "nomination.json").read_bytes()
    ).hexdigest()
    seal = None
    loads = 0
    for i, row in enumerate(events):
        if not isinstance(row, dict) or not isinstance(row.get("event"), str):
            raise ValueError("invalid event row")
        if row["event"] == "nomination_sealed":
            if seal is not None:
                raise ValueError("duplicate nomination_sealed event")
            if row.get("file_sha256") != nomination_sha:
                raise ValueError("nomination event SHA mismatch")
            seal = i
        elif row["event"] == "H_semantic_load":
            if seal is None:
                raise ValueError("H semantic event before nomination sealed")
            if row.get("database") not in HOLDOUT:
                raise ValueError("H semantic event database mismatch")
            if "nomination_sha256" in row:
                if row["nomination_sha256"] != nomination_sha:
                    raise ValueError("H semantic event nomination SHA mismatch")
            elif "hash" not in row:
                raise ValueError("H semantic event requires database/hash metadata")
            if "hash" in row and (not isinstance(row["hash"], str) or not row["hash"]):
                raise ValueError("invalid H semantic event hash metadata")
            loads += 1
    if seal is None or not loads:
        raise ValueError("missing nomination/H event")
    return nomination_sha, loads


def _caches(caches):
    sources = {*TRAIN, *VALIDATION, *HOLDOUT}
    if not isinstance(caches, dict) or set(caches) != sources:
        raise ValueError("source cache whitelist must equal the 26 source databases")
    for db in (*TRAIN, *VALIDATION, *HOLDOUT):
        cache = caches[db]
        required = {"indices", "query_hash", "true", "pg", "n_tables"}
        if not isinstance(cache, dict) or not required <= cache.keys():
            raise ValueError(f"cache fields missing: {db}")
        hashes = cache["query_hash"]
        if (
            not isinstance(hashes, list)
            or not hashes
            or any(not isinstance(h, str) or not h for h in hashes)
            or len(set(hashes)) != len(hashes)
        ):
            raise ValueError(f"cache unique identity mismatch: {db}")
        n = len(hashes)
        if db in TRAIN and n < MIN_T_POOL:
            raise ValueError(f"cache training pool requires at least 1224 unique: {db}")
        if db in VALIDATION and n < 256 or db in HOLDOUT and n < 32:
            raise ValueError(f"cache smoke pool too small: {db}")
        for key in ("indices", "true", "pg", "n_tables"):
            array = cache[key]
            if (
                not isinstance(array, np.ndarray)
                or array.shape != (n,)
                or array.dtype.kind
                not in ("iu" if key in ("indices", "n_tables") else "fiu")
                or not np.isfinite(array).all()
                or (array < 0).any()
            ):
                raise ValueError(f"invalid cache array: {db}/{key}")
        if len(set(map(int, cache["indices"]))) != n:
            raise ValueError(f"cache raw indices duplicate: {db}")
        if (cache["n_tables"] < 1).any():
            raise ValueError(f"cache invalid n_tables: {db}")


def _select(db, hashes, prefix, size):
    return sorted(
        range(len(hashes)),
        key=lambda i: (
            hashlib.sha256(f"{prefix}|{db}|{hashes[i]}".encode()).hexdigest(),
            hashes[i],
        ),
    )[:size]


def _panels(bindings, panels, caches):
    if not isinstance(panels, dict) or set(panels) != set(TRAIN):
        raise ValueError("panel database whitelist mismatch")
    for db in TRAIN:
        positions = panels[db]
        if (
            not isinstance(positions, list)
            or any(not _integer(i) for i in positions)
            or positions
            != _select(db, caches[db]["query_hash"], "s8-train-panel-v1", PANEL_SIZE)
        ):
            raise ValueError(f"panel SHA identity/order mismatch: {db}")
    identities = {db: [caches[db]["query_hash"][i] for i in panels[db]] for db in TRAIN}
    sha = _digest(identities)
    if bindings.get("panel_identity_digest") != sha:
        raise ValueError("panel identity digest mismatch")
    return sha


def _episodes(rows, panels, caches):
    if len(rows) != 8:
        raise ValueError("episode count must be 2 x 4 = 8")
    rng = np.random.default_rng(43)
    observed = set()
    index = 0
    for step in range(1, 3):
        for _episode in range(4):
            db = str(rng.choice(TRAIN))
            k = int(rng.choice([5, 10, 20, 50, 100]))
            cache = caches[db]
            raw = rng.choice(
                np.sort(cache["indices"][panels[db]]), size=k + 64, replace=False
            )
            raw_mapping = {int(r): i for i, r in enumerate(cache["indices"])}
            positions = [raw_mapping[int(r)] for r in raw]
            expected = {
                "step": step,
                "episode": index + 1,
                "database": db,
                "support_positions": positions[:k],
                "query_positions": positions[k:],
                "support_hashes": [cache["query_hash"][i] for i in positions[:k]],
                "query_hashes": [cache["query_hash"][i] for i in positions[k:]],
            }
            row = rows[index]
            if not isinstance(row, dict) or row != expected:
                raise ValueError(f"episode exact RNG replay mismatch: row {index}")
            if any(not _integer(row[key]) for key in ("step", "episode")) or any(
                not _integer(i)
                for key in ("support_positions", "query_positions")
                for i in row[key]
            ):
                raise ValueError("episode integer positions/counters required")
            observed.update((db, i) for i in positions)
            index += 1
    return len(observed)


def _endpoints(rows, panels, caches):
    expected, groups = {}, {}
    for db in (*TRAIN, *VALIDATION, *HOLDOUT):
        cache = caches[db]
        cohort = "P" if db in TRAIN else "V" if db in VALIDATION else "H"
        positions = (
            panels[db]
            if db in TRAIN
            else _select(
                db,
                cache["query_hash"],
                "s8-bridge-v1" if db in VALIDATION else "s8-smoke-h-v1",
                256 if db in VALIDATION else 32,
            )
        )
        groups[(db, cohort)] = []
        for i in positions:
            expected[(db, cohort, cache["query_hash"][i])] = i
    fields = {
        "database",
        "cohort",
        "query_hash",
        "true",
        "pg",
        "n_tables",
        "predictions",
    }
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("invalid endpoint fields")
        if any(
            not isinstance(row[k], str) for k in ("database", "cohort", "query_hash")
        ):
            raise ValueError("invalid endpoint identity type")
        key = row["database"], row["cohort"], row["query_hash"]
        if key not in expected or key in seen:
            raise ValueError("extra/duplicate/mismatched endpoint identity")
        seen.add(key)
        cache, i = caches[key[0]], expected[key]
        for field in ("true", "pg", "n_tables"):
            _number(row[field], f"endpoint {field}")
            # Compare Python scalars without rounding large integer labels/PG.
            if (
                row[field] != cache[field][i].item()
                or field == "n_tables"
                and not _integer(row[field])
            ):
                raise ValueError(f"endpoint source value mismatch: {field}")
        pred = row["predictions"]
        if not isinstance(pred, dict) or set(pred) != set(ARMS):
            raise ValueError("endpoint predictions must contain exactly five arms")
        target = math.log(max(row["true"], 1))
        numeric = {}
        for arm in ARMS:
            log_pred = _number(pred[arm], "endpoint prediction")
            try:
                # Validate the un-clipped raw exp too, including small cells.
                raw = math.exp(log_pred)
                q = math.exp(abs(max(log_pred, 0) - target))
                error = log_pred - target
                mse = error * error
                if not all(math.isfinite(v) for v in (raw, q, error, mse)):
                    raise ValueError("endpoint nonfinite numeric result")
            except OverflowError as exc:
                raise ValueError("endpoint raw/Q exponent numeric overflow") from exc
            numeric[arm] = (q, abs(error), mse)
        groups[key[:2]].append((row["n_tables"], numeric))
    if seen != set(expected):
        raise ValueError("missing endpoint identity/full smoke denominator")
    return groups


def _quantile(values, fraction):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _metrics(document, groups):
    if (
        not isinstance(document, dict)
        or "rows" not in document
        or set(document) - {"rows", "purpose"}
        or "purpose" in document
        and document["purpose"] != "smoke-not-research"
        or not isinstance(document["rows"], list)
    ):
        raise ValueError("invalid metric document")
    expected = {
        (db, cohort, bucket) for db, cohort in groups for bucket in ("all", "four_plus")
    }
    seen = set()
    for row in document["rows"]:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(k), str) for k in ("database", "cohort", "bucket")
        ):
            raise ValueError("invalid metric identity")
        key = row["database"], row["cohort"], row["bucket"]
        if key not in expected or key in seen:
            raise ValueError("extra/duplicate metric cell")
        seen.add(key)
        cell = [
            values
            for tables, values in groups[key[:2]]
            if key[2] == "all" or tables >= 4
        ]
        n = len(cell)
        status = "descriptive" if n >= 200 else "below_min_cell_n"
        fields = {"database", "cohort", "bucket", "n", "status"} | (
            {"arms"} if n >= 200 else set()
        )
        if (
            set(row) != fields
            or not _integer(row.get("n"))
            or row["n"] != n
            or row.get("status") != status
        ):
            raise ValueError(
                "metric count/status/fields mismatch; small cells omit details"
            )
        if n < 200:
            continue
        if not isinstance(row["arms"], dict) or set(row["arms"]) != set(ARMS):
            raise ValueError("metric arm set mismatch")
        for arm in ARMS:
            claimed = row["arms"][arm]
            if not isinstance(claimed, dict) or set(claimed) != set(METRIC_NAMES):
                raise ValueError("metric names mismatch")
            q = [v[arm][0] for v in cell]
            try:
                computed = {
                    "q50": _quantile(q, 0.5),
                    "q95": _quantile(q, 0.95),
                    "raw_log_mae": math.fsum(v[arm][1] for v in cell) / n,
                    "raw_log_mse": math.fsum(v[arm][2] for v in cell) / n,
                }
            except OverflowError as exc:
                raise ValueError("metric numeric aggregate overflow") from exc
            for name, value in computed.items():
                actual = _number(claimed[name], f"metric {arm}/{name}")
                if not math.isfinite(value) or not math.isclose(
                    actual, value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"metric independent recomputation mismatch: {key}/{arm}/{name}"
                    )
    if seen != expected:
        raise ValueError("missing metric cell")
    return len(seen)


def verify_smoke(run_dir, bindings, panels, caches):
    """Authenticate the exact G1 smoke contract; raise on any missing/bad evidence."""
    run_dir = Path(run_dir)
    _bindings(bindings)
    numeric_checks = _numeric_records(run_dir, bindings)
    nomination_sha, loads = _events(run_dir)
    _caches(caches)
    panel_sha = _panels(bindings, panels, caches)
    episodes = _read(run_dir / "episodes.jsonl", lines=True)
    exposed = _episodes(episodes, panels, caches)
    groups = _endpoints(_read(run_dir / "endpoints.jsonl", lines=True), panels, caches)
    cells = _metrics(_read(run_dir / "metrics.json"), groups)
    if (
        hashlib.sha256((run_dir / "nomination.json").read_bytes()).hexdigest()
        != nomination_sha
    ):
        raise ValueError("nomination file changed during independent verification")
    counts = {
        role: sum(len(rows) for (_, cohort), rows in groups.items() if cohort == role)
        for role in ("P", "V", "H")
    }
    return {
        **numeric_checks,
        "passed": True,
        "purpose": "smoke-not-research",
        "mode": "smoke",
        "formal_ready": False,
        "verified_counts": dict(
            databases=26,
            episodes=len(episodes),
            metric_cells=cells,
            training_unique_exposed=exposed,
            **counts,
        ),
        "query_count": sum(counts.values()),
        "panel_identity_digest": panel_sha,
        "nomination_sha256": nomination_sha,
        "nomination_event_order_verified": True,
        "H_semantic_load_events": loads,
        "actual_H_access_order_verified": False,
        "full_linear_optimality_verified": False,
        "source_file_provenance_verified": False,
        "units": {
            "predictions": "natural_log_cardinality",
            "q50": "dimensionless_q_error",
            "q95": "dimensionless_q_error",
            "raw_log_mae": "natural_log_cardinality",
            "raw_log_mse": "squared_natural_log_cardinality",
        },
    }
