"""Read-only verifier for a protocol-122 P0 certificate.

Format validation checks structure only; truth validation checks the signed
state; verify() additionally recomputes every bound file's SHA256. No model,
training, GPU or network access. This module never writes the certificate.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

TRUTH_FIELDS = (
    "p0_gate_passed",
    "formal_training_authorized",
    "research_training_started",
    "protocol_122_executed",
)
BINDING_FIELDS = (
    "frozen_code",
    "a_training_configs",
    "training_plans",
    "cache_manifests",
    "source_preflight",
    "source_eval_plan",
    "data_config_122",
)
LIST_FIELDS = frozenset({"a_training_configs", "training_plans", "cache_manifests"})
HEX = frozenset("0123456789abcdef")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in HEX for char in value)
    )


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Certificate must be a JSON object: {path}")
    return value


def _check_entry(label: str, entry: object, diff: list[str]) -> None:
    if not isinstance(entry, dict):
        diff.append(f"{label}: expected an object")
        return
    for field in ("path", "sha256"):
        if field not in entry:
            diff.append(f"{label}: missing field '{field}'")
    if "path" in entry and not isinstance(entry["path"], str):
        diff.append(f"{label}.path: expected a string")
    if "sha256" in entry and not _is_sha256(entry["sha256"]):
        diff.append(f"{label}.sha256: expected a 64-char lowercase hex digest")


def verify_format(path: Path) -> list[str]:
    """Structure-only validation. Does not touch any bound file or truth value."""
    diff: list[str] = []
    try:
        cert = _load(Path(path))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"certificate: unreadable or invalid ({error})"]
    for field in (
        "schema",
        "protocol",
        "purpose",
        "status",
        "issued_at",
        "truth",
        "bindings",
    ):
        if field not in cert:
            diff.append(f"certificate: missing field '{field}'")
    if "protocol" in cert and cert["protocol"] != 122:
        diff.append("certificate.protocol: expected 122")
    if "schema" in cert and cert["schema"] not in (1, 2):
        diff.append("certificate.schema: expected 1 or 2")
    if cert.get("purpose") != "p0-certificate":
        diff.append("certificate.purpose: expected 'p0-certificate'")
    if "status" in cert and cert["status"] not in ("signed", "unsigned"):
        diff.append("certificate.status: expected 'signed' or 'unsigned'")
    if (
        "issued_at" in cert
        and not isinstance(cert["issued_at"], str)
        and not (
            cert.get("schema") == 2
            and cert.get("status") == "unsigned"
            and cert["issued_at"] is None
        )
    ):
        diff.append("certificate.issued_at: expected a string")
    truth = cert.get("truth")
    if not isinstance(truth, dict):
        diff.append("certificate.truth: expected an object")
    else:
        for field in TRUTH_FIELDS:
            if field not in truth:
                diff.append(f"certificate.truth: missing field '{field}'")
            elif truth[field] is not None and type(truth[field]) is not bool:
                diff.append(
                    f"certificate.truth.{field}: expected boolean or null "
                    f"(got {truth[field]!r})"
                )
    bindings = cert.get("bindings")
    if not isinstance(bindings, dict):
        diff.append("certificate.bindings: expected an object")
        return diff
    for field in BINDING_FIELDS:
        if field not in bindings:
            diff.append(f"certificate.bindings: missing field '{field}'")
    for field in LIST_FIELDS:
        if field in bindings and not isinstance(bindings[field], list):
            diff.append(f"certificate.bindings.{field}: expected a list")
    for field in BINDING_FIELDS:
        if field not in bindings:
            continue
        value = bindings[field]
        if field in LIST_FIELDS and isinstance(value, list):
            if not value:
                diff.append(f"certificate.bindings.{field}: list must not be empty")
            for index, entry in enumerate(value):
                _check_entry(f"certificate.bindings.{field}[{index}]", entry, diff)
                if field == "cache_manifests" and isinstance(entry, dict):
                    if "content_hash" not in entry:
                        diff.append(
                            f"certificate.bindings.{field}[{index}]: "
                            "missing field 'content_hash'"
                        )
        else:
            _check_entry(f"certificate.bindings.{field}", value, diff)
    if "source_eval_plan" in bindings and isinstance(
        bindings["source_eval_plan"], dict
    ):
        if "plan_sha256" not in bindings["source_eval_plan"]:
            diff.append(
                "certificate.bindings.source_eval_plan: missing field 'plan_sha256'"
            )
    return diff


def verify_truth(path: Path) -> list[str]:
    """Signed-state validation. An unsigned template must fail here."""
    diff: list[str] = []
    try:
        cert = _load(Path(path))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return [f"certificate: unreadable or invalid ({error})"]
    if cert.get("status") != "signed":
        diff.append(
            f"certificate.status: expected 'signed' (got {cert.get('status')!r})"
        )
    truth = cert.get("truth")
    if not isinstance(truth, dict):
        diff.append("certificate.truth: expected an object")
        return diff
    for field in TRUTH_FIELDS:
        expected = (
            True if cert.get("schema") == 1 or field == "p0_gate_passed" else False
        )
        if truth.get(field) is not expected:
            diff.append(
                f"certificate.truth.{field}: expected {expected!r} (got {truth.get(field)!r})"
            )
    if cert.get("schema") == 2:
        identity = cert.get("issued_by")
        if not isinstance(identity, str) or not identity.strip():
            diff.append(
                "certificate.issued_by: expected a non-empty approving identity"
            )
        try:
            timestamp = datetime.fromisoformat(cert.get("issued_at", ""))
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timezone required")
        except (ValueError, TypeError):
            diff.append("certificate.issued_at: expected ISO 8601 with a timezone")
    return diff


def _entries(bindings: dict) -> list[tuple[str, dict]]:
    result: list[tuple[str, dict]] = []
    for field in BINDING_FIELDS:
        if field not in bindings:
            continue
        value = bindings[field]
        if field in LIST_FIELDS and isinstance(value, list):
            for index, entry in enumerate(value):
                if isinstance(entry, dict):
                    result.append((f"bindings.{field}[{index}]", entry))
        elif isinstance(value, dict):
            result.append((f"bindings.{field}", value))
    return result


def _recompute(bindings: dict, diff: list[str]) -> None:
    for label, entry in _entries(bindings):
        if "path" not in entry or "sha256" not in entry:
            continue
        if not isinstance(entry.get("path"), str):
            continue
        candidate = Path(entry["path"])
        if not candidate.is_file():
            diff.append(f"{label}.path: file does not exist ({candidate})")
            continue
        try:
            raw = candidate.read_bytes()
        except OSError as error:
            diff.append(f"{label}.path: unreadable ({error})")
            continue
        actual = hashlib.sha256(raw).hexdigest()
        if actual != entry["sha256"]:
            diff.append(
                f"{label}.sha256: mismatch (declared {entry['sha256']}, "
                f"actual {actual})"
            )
        if "content_hash" in entry and isinstance(entry["content_hash"], str):
            try:
                declared = json.loads(raw).get("content_hash")
            except (ValueError, AttributeError, json.JSONDecodeError):
                declared = None
            if declared != entry["content_hash"]:
                diff.append(
                    f"{label}.content_hash: mismatch "
                    f"(declared {entry['content_hash']}, file {declared!r})"
                )
        if "plan_sha256" in entry and isinstance(entry["plan_sha256"], str):
            try:
                inner = json.loads(raw).get("plan_sha256")
            except (ValueError, AttributeError, json.JSONDecodeError):
                inner = None
            if inner != entry["plan_sha256"]:
                diff.append(
                    f"{label}.plan_sha256: mismatch "
                    f"(declared {entry['plan_sha256']}, file {inner!r})"
                )
    frozen = bindings.get("frozen_code", {})
    if isinstance(frozen, dict) and "code_files" in frozen:
        entries = frozen["code_files"]
        if not isinstance(entries, list) or not entries:
            diff.append("bindings.frozen_code.code_files: expected a non-empty list")
            return
        code_hashes = {}
        for index, entry in enumerate(entries):
            label = f"bindings.frozen_code.code_files[{index}]"
            _check_entry(label, entry, diff)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in code_hashes:
                diff.append(f"{label}.name: missing or duplicate module")
                continue
            code_hashes[name] = entry.get("sha256")
            try:
                actual = hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
            except OSError as error:
                diff.append(f"{label}.path: unreadable ({error})")
                continue
            if actual != entry.get("sha256"):
                diff.append(f"{label}.sha256: mismatch")
        digest = hashlib.sha256(
            json.dumps(code_hashes, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        if frozen.get("manifest_digest") != digest:
            diff.append("bindings.frozen_code.manifest_digest: mismatch")
        try:
            recorded = _load(Path(frozen["path"]))
        except (OSError, ValueError, KeyError):
            recorded = None
        if recorded != code_hashes:
            diff.append("bindings.frozen_code: code_files differ from digest file")


def verify(path: Path) -> list[str]:
    """Full verification: format, signed truth, and recomputed bindings."""
    diff = verify_format(Path(path))
    diff.extend(verify_truth(Path(path)))
    try:
        cert = _load(Path(path))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return diff
    bindings = cert.get("bindings")
    if isinstance(bindings, dict):
        _recompute(bindings, diff)
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a protocol-122 P0 certificate (read-only)."
    )
    parser.add_argument("certificate", type=Path, help="Path to the certificate JSON")
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Validate structure only; skip signed-truth and SHA recomputation",
    )
    args = parser.parse_args()
    diff = (
        verify_format(args.certificate)
        if args.format_only
        else verify(args.certificate)
    )
    if diff:
        for line in diff:
            print(f"DIFF {line}")
        return 1
    print("OK certificate verified" if not args.format_only else "OK format only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
