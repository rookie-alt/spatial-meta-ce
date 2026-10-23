"""S8 metadata and V nomination seals; standard library only, never trains.

This module authorizes no H I/O itself. The runner must pin the complete seal's
file SHA256, call require_nomination before opening H semantic data, then retain
its fixed report even if H vetoes a formal candidate. Smoke is not selection.
"""

import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path

# Literal S8 boundaries, independent of the torch-dependent capture module.
T19 = (
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
H4 = ("accidents", "grants", "employee", "hockey")
V3 = ("chembl", "legalacts", "telstra")
_TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))
_ARMS = ("A_h_adam", "B_h_ls", "C_x_adam", "D_x_ls", "F0")
_CANDIDATES = _ARMS[1:4]
_PURPOSES = {"smoke": "s8-engineering-smoke-not-research", "formal": "s8-research"}
_REQUIRED = {
    "schema",
    "purpose",
    "mode",
    "protocol_sha256",
    "source_binding_sha256",
    "panel_identity_digest",
    "train_databases",
    "holdout_databases",
    "validation_databases",
    "model_config",
    "seed",
    "train_steps",
    "meta_batch",
    "query_size",
    "support_sizes",
}
_SMOKE_RESOURCES = {
    "batch_size": 128,
    "threads": 4,
    "probe_fit_per_database": 64,
    "probe_adam_steps": 2,
    "timeout_seconds": 1800,
    "address_space_bytes": 12884901888,
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _check_sha(value, field):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"S8 invalid SHA256: {field}")


def _json_safe(value):
    """Accept JSON values only; reject hidden target identities even in paths."""
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("S8 nonfinite metadata")
        return
    if isinstance(value, str):
        if _TARGETS.intersection(re.split(r"[^a-z0-9_]+", value.lower())):
            raise ValueError("S8 target metadata forbidden")
        return
    if type(value) is list:
        for item in value:
            _json_safe(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("S8 JSON keys must be strings")  # noqa: TRY004 - public ValueError contract
            _json_safe(key)
            _json_safe(item)
            if key.lower() in ("db", "database", "databases", "dbs") or (
                key.lower().endswith(("_db", "_database", "_databases", "_dbs"))
                and not key.lower().endswith(("_per_database", "_per_db"))
            ):
                databases = item if type(item) is list else [item]
                if not databases or any(
                    type(db) is not str or db not in (*T19, *H4, *V3)
                    for db in databases
                ):
                    raise ValueError("S8 unknown auxiliary database metadata")
        return
    raise ValueError("S8 bindings require JSON values")


def signature(bindings):
    """Validate the fixed boundary/config and SHA256 every canonical JSON field.

    Required model_config: arch='sum', width=64, use_pg=true,
    pg_output_scale=1; optional layers=2, heads=4. Auxiliary JSON metadata is
    included in the signature, not certification of its scientific correctness.
    """
    if type(bindings) is not dict or not _REQUIRED.issubset(bindings):
        raise ValueError("S8 missing bindings fields")
    try:
        _json_safe(bindings)
    except RecursionError as exc:
        raise ValueError("S8 cyclic/deep metadata forbidden") from exc
    mode = bindings["mode"]
    if type(mode) is not str or mode not in _PURPOSES:
        raise ValueError("S8 unknown mode")
    if bindings["purpose"] != _PURPOSES[mode]:
        raise ValueError("S8 mode/purpose mismatch")
    schema = bindings["schema"]
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError("S8 frozen schema mismatch")
    if schema == 1:
        if {
            "numerical_acceptance_policy",
            "acceptance_protocol_sha256",
        } & bindings.keys():
            raise ValueError("S8 schema1 forbids v2 acceptance fields")
    else:
        if (
            mode != "smoke"
            or bindings.get("numerical_acceptance_policy")
            != "s8-g1-resource-engineering-v2"
        ):
            raise ValueError("S8 schema2 requires smoke engineering v2 policy")
        _check_sha(
            bindings.get("acceptance_protocol_sha256"), "acceptance_protocol_sha256"
        )
    for key, value in (
        ("seed", 42),
        ("meta_batch", 4),
        ("query_size", 64),
        ("train_steps", 2 if mode == "smoke" else 4000),
    ):
        if type(bindings[key]) is not int or bindings[key] != value:
            raise ValueError(f"S8 frozen {key} mismatch")
    for key in ("protocol_sha256", "source_binding_sha256", "panel_identity_digest"):
        _check_sha(bindings[key], key)
    for key, databases in (
        ("train_databases", T19),
        ("holdout_databases", H4),
        ("validation_databases", V3),
    ):
        if type(bindings[key]) is not list or bindings[key] != list(databases):
            raise ValueError(f"S8 literal {key} boundary/order mismatch")
    if mode == "smoke":
        resources = bindings.get("resourcefixedfields")
        if type(resources) is not dict or resources.keys() != _SMOKE_RESOURCES.keys():
            raise ValueError("S8 smoke resourcefixedfields required")
        for key, value in _SMOKE_RESOURCES.items():
            if type(resources[key]) is not int or resources[key] != value:
                raise ValueError(f"S8 frozen smoke resource {key} mismatch")
    sizes = bindings["support_sizes"]
    if (
        type(sizes) is not list
        or any(type(k) is not int for k in sizes)
        or sizes != [5, 10, 20, 50, 100]
    ):
        raise ValueError("S8 frozen support sizes mismatch")
    config = bindings["model_config"]
    fixed = {"arch": "sum", "width": 64, "use_pg": True, "pg_output_scale": 1.0}
    if (
        type(config) is not dict
        or not fixed.keys() <= config.keys()
        or config.keys() - (fixed.keys() | {"layers", "heads"})
    ):
        raise ValueError("S8 model_config fields mismatch")
    if (
        config["arch"] != "sum"
        or type(config["width"]) is not int
        or config["width"] != 64
        or config["use_pg"] is not True
        or type(config["pg_output_scale"]) not in (int, float)
        or config["pg_output_scale"] != 1
    ):
        raise ValueError("S8 requires sum64 PG anchor1")
    for key, value in (("layers", 2), ("heads", 4)):
        if key in config and (type(config[key]) is not int or config[key] != value):
            raise ValueError(f"S8 frozen model {key} mismatch")
    try:
        return _sha(_canonical(bindings))
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError("S8 invalid canonical bindings") from exc


def select_candidate(metrics):
    """V-only gate over complete paired summaries; invalid input is not no-candidate.

    Input: {arm: {Vdb: {all|four_plus: {n, q50, q95}}}}. Counts must match
    across all five arms for each db/cohort, both cohorts n>=200, Q>=1 and
    q50<=q95. This checks summaries, not query identities or LS certificates;
    the runner must authenticate those before supplying formal metrics.
    """
    if type(metrics) is not dict or set(metrics) != set(_ARMS):
        raise ValueError("S8 requires exactly four probe arms and F0")
    counts = {}
    macros = {}
    for arm in _ARMS:
        by_db = metrics[arm]
        if type(by_db) is not dict or set(by_db) != set(V3):
            raise ValueError("S8 selection requires exactly the three V databases")
        for db in V3:
            cohorts = by_db[db]
            if type(cohorts) is not dict or set(cohorts) != {"all", "four_plus"}:
                raise ValueError("S8 missing/extra selection cohort")
            for cohort in ("all", "four_plus"):
                summary = cohorts[cohort]
                if type(summary) is not dict or set(summary) != {"n", "q50", "q95"}:
                    raise ValueError("S8 requires n/q50/q95 selection fields")
                n = summary["n"]
                if type(n) is not int or n < 200:
                    raise ValueError("S8 sparse/invalid selection denominator")
                key = (db, cohort)
                if n != counts.setdefault(key, n):
                    raise ValueError("S8 unpaired selection counts")
                for field in ("q50", "q95"):
                    q = summary[field]
                    try:
                        valid = type(q) in (int, float) and q >= 1 and math.isfinite(q)
                    except OverflowError as exc:
                        raise ValueError("S8 overflowing Q summary") from exc
                    if not valid:
                        raise ValueError("S8 nonfinite/invalid Q summary")
                if summary["q50"] > summary["q95"]:
                    raise ValueError("S8 inverted Q quantiles")
            if cohorts["four_plus"]["n"] > cohorts["all"]["n"]:
                raise ValueError("S8 four_plus exceeds all denominator")
        macros[arm] = math.fsum(by_db[db]["four_plus"]["q95"] / 3 for db in V3)
    eligible = []
    for arm in _CANDIDATES:
        macro_ok = all(macros[arm] <= 0.9 * macros[ref] for ref in ("A_h_adam", "F0"))
        protection_ok = all(
            metrics[arm][db][cohort][field] <= 1.1 * metrics[ref][db][cohort][field]
            for db in V3
            for cohort in ("all", "four_plus")
            for field in ("q50", "q95")
            for ref in ("A_h_adam", "F0")
        )
        if macro_ok and protection_ok:
            eligible.append(arm)
    eligible.sort(key=lambda arm: (macros[arm], _CANDIDATES.index(arm)))
    return {
        "decision": "development_promising" if eligible else "no_development_candidate",
        "candidate": eligible[0] if eligible else None,
        "macro_four_plus_q95": macros,
        "eligible_candidates": eligible,
    }


def _payload(metrics, bindings):
    binding_sha = signature(bindings)
    # Snapshot via canonical JSON; never inspect or persist metrics for smoke.
    snapshot = json.loads(_canonical(bindings))
    selection = select_candidate(metrics) if snapshot["mode"] == "formal" else None
    body = {
        "schema": 1,
        "mode": snapshot["mode"],
        "purpose": snapshot["purpose"],
        "decision": selection["decision"] if selection else "smoke_not_selection",
        "candidate": selection["candidate"] if selection else None,
        "bindings_sha256": binding_sha,
        "bindings": snapshot,
        "metrics": json.loads(_canonical(metrics)) if selection else None,
        "selection": selection,
    }
    return {**body, "body_digest": _sha(_canonical(body))}


@contextmanager
def _parent_fd(path):
    """Walk held directory descriptors; reject symlinks and '..' at every level."""
    path = Path(path)
    if ".." in path.parts or not path.name:
        raise ValueError("S8 unsafe seal path")
    if not path.is_absolute():
        path = Path.cwd() / path
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        yield fd, path.name
    finally:
        os.close(fd)


def seal_nomination(path, metrics, bindings):
    """Create canonical seal with O_EXCL, fsync and no symlink traversal.

    Parent must already exist; all I/O failures become ValueError. Failed writes
    can leave an invalid, non-overwritable seal (fail closed, no automatic retry).
    Pin SHA256 of the resulting complete file externally before any H access.
    """
    payload = _payload(metrics, bindings)
    try:
        with _parent_fd(path) as (parent, name):
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            with os.fdopen(fd, "wb") as output:
                output.write(_canonical(payload))
                output.flush()
                os.fsync(output.fileno())
            os.fsync(parent)
    except (OSError, TypeError) as exc:
        raise ValueError(
            "S8 nomination seal create failed (no overwrite/symlinks)"
        ) from exc
    return payload


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("S8 duplicate JSON body key")
        result[key] = value
    return result


def require_nomination(path, expected_file_sha, bindings):
    """Authenticate file bytes/body/bindings/mode and formal gate before H loads.

    expected_file_sha must be pinned at seal time outside the mutable file, not
    recalculated from a potentially tampered file at H-load time. body_digest is
    integrity redundancy, not an adversary-resistant signature or authorization.
    """
    _check_sha(expected_file_sha, "expected_file_sha")
    signature(bindings)
    try:
        with _parent_fd(path) as (parent, name):
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
            )
            with os.fdopen(fd, "rb") as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError("S8 nomination is not a regular seal file")
                raw = source.read()
        if _sha(raw) != expected_file_sha:
            raise ValueError("S8 nomination file SHA mismatch")
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        if type(payload) is not dict or _canonical(payload) != raw:
            raise ValueError("S8 nomination must have canonical JSON body")
        if "body_digest" not in payload:
            raise ValueError("S8 missing nomination body digest")
        body = {key: value for key, value in payload.items() if key != "body_digest"}
        if _sha(_canonical(body)) != payload["body_digest"]:
            raise ValueError("S8 nomination body digest mismatch")
        # Reconstruct every field; rejects extra fields, inconsistent decisions,
        # changed controls/source/mode and formal candidate tampering as well.
        if raw != _canonical(_payload(payload.get("metrics"), bindings)):
            raise ValueError("S8 nomination binding/mode/decision mismatch")
        return payload
    except (OSError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError("S8 nomination read/authentication failed") from exc
