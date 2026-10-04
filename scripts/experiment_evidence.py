"""Deterministic, bounded recorded JSON evidence (no execution claims).

CLI: python scripts/experiment_evidence.py --root PATH
Token counts are UTF-8 byte/4 estimates, not model tokenizer measurements.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_FIELDS = 256
MAX_VALUE_BYTES = 16384
MAX_ARRAY_ITEMS = 128
MAX_DEPTH = 16
RESULT_ROOT = ("实验文档", "results")
EXCLUDED = frozenset(
    {
        "history",
        "histories",
        "manifest",
        "manifests",
        "progress",
        "log",
        "logs",
        "stale",
    }
)
RESULT_NAMES = frozenset(
    {
        "config.json",
        "run_config.json",
        "protocol.json",
        "summary.json",
        "decision.json",
        "status.json",
        "result.json",
        "price-lr-search-result.json",
    }
)
REFERENCES = frozenset({"data_config", "config", "model_config"})
ROW_FIELDS = frozenset(
    {"results", "runs", "reports", "queries", "predictions", "rows", "history"}
)
SUMMARY_ARRAYS = frozenset(
    {"hypothesis", "decision", "limitations", "checks", "risk_flags", "risks"}
)


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _root(root: Path) -> Path:
    if not isinstance(root, Path):
        raise TypeError("root must be a Path")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("root must be an existing non-symlink directory")
    return root.absolute()


def _safe_path(root: Path, path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    candidate = path if path.is_absolute() else root / path
    if ".." in candidate.parts:
        raise ValueError("path traversal is not permitted")
    try:
        relative = candidate.relative_to(root)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError) as error:
        raise ValueError("path must exist within root") from error
    if any(
        (root / Path(*relative.parts[:index])).is_symlink()
        for index in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("symlink path is not permitted")
    if not candidate.is_file():
        raise ValueError("path must be a regular file")
    return candidate


def _kind(relative: str) -> str | None:
    parts = PurePosixPath(relative).parts
    if not parts:
        return None
    name = parts[-1]
    if parts[0] in {
        "core",
        "spatial_meta_ce",
        "scripts",
        "tests",
        "vendor",
    } and name.endswith(".py"):
        return "code"
    if len(parts) == 2 and parts[0] == "reports" and name.endswith(".py"):
        return "code"
    if relative in {"README.md", "reports/STATUS.md"}:
        return "document"
    if name.endswith(".md") and (
        parts[0] == "docs" or (parts[0] == "实验文档" and len(parts) == 2)
    ):
        return "document"
    if len(parts) == 2 and parts[0] == "configs" and name.endswith(".json"):
        return "json"
    if parts[:2] != RESULT_ROOT or not name.endswith(".json"):
        return None
    if any(
        EXCLUDED.intersection(re.split(r"[-_.]", part.lower())) for part in parts[2:]
    ):
        return None
    if (
        name in RESULT_NAMES
        or name.endswith(("-summary.json", "-run_config.json"))
        or name.startswith("select")
    ):
        return "json"
    return None


def _walk(directory: Path) -> list[Path]:
    with os.scandir(directory) as stream:
        entries = sorted(stream, key=lambda entry: entry.name)
    return [
        path
        for entry in entries
        if not entry.is_symlink()
        for path in (
            _walk(Path(entry.path))
            if entry.is_dir(follow_symlinks=False) and not entry.name.startswith(".")
            else [Path(entry.path)]
            if entry.is_file(follow_symlinks=False)
            else []
        )
    ]


def select_files(root: Path) -> dict[str, list[Path]]:
    """Select only whitelisted regular files, never traversing symlinks."""
    root = _root(root)
    classified = [
        (path, _kind(path.relative_to(root).as_posix())) for path in _walk(root)
    ]
    return {
        kind: sorted(
            (_safe_path(root, path) for path, found in classified if found == kind),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        for kind in ("code", "document", "json")
    }


def node_id(relative_path: str, pointer: str = "") -> str:
    """Hash a length-unambiguous path/pointer pair; never use stems as identity."""
    if not isinstance(relative_path, str) or not isinstance(pointer, str):
        raise TypeError("path and pointer must be strings")
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative_path
    ):
        raise ValueError("path must be a normalized root-relative path")
    if pointer and (not pointer.startswith("/") or re.search(r"~(?![01])", pointer)):
        raise ValueError("pointer must be an RFC 6901 JSON Pointer")
    return "evidence_" + hashlib.sha256(_encoded([relative_path, pointer])).hexdigest()


def _pointer(parent: str, key: str) -> str:
    return parent + "/" + key.replace("~", "~0").replace("/", "~1")


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError("duplicate JSON object key")
    return dict(pairs)


def _nonfinite(value: str) -> None:
    raise ValueError("nonfinite JSON number: " + value)


def _load(path: Path) -> tuple[dict, str, int]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON size limit exceeded: {path}")
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError("JSON size limit exceeded")
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (ValueError, UnicodeError, RecursionError, OSError) as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must be an object: {path}")
    if len(value) > MAX_FIELDS:
        raise ValueError(f"JSON field limit exceeded: {path}")
    return value, hashlib.sha256(raw).hexdigest(), (len(raw) + 3) // 4


def _date(value: Any) -> str | None:
    if not isinstance(value, str) or not re.match(r"^\d{4}-\d{2}-\d{2}(?:T|$)", value):
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _metadata(relative: str, pointer: str, digest: str, payload: dict) -> dict:
    nested = payload.get("metadata")
    explicit = {**(nested if isinstance(nested, dict) else {}), **payload}
    date = next(
        (
            parsed
            for key in ("date", "completed_at")
            if (parsed := _date(explicit.get(key))) is not None
        ),
        None,
    )
    return {
        "source_file": relative,
        "source_location": pointer,
        "sha256": digest,
        "date": date,
        "branch": explicit.get("branch"),
        "commit": explicit.get("commit"),
        "status": explicit.get("status"),
        "validation_state": "recorded_not_reexecuted",
    }


def _entity(relative: str) -> str:
    name = PurePosixPath(relative).name
    if name == "protocol.json":
        return "Protocol"
    if name == "run_config.json" or name.endswith("-run_config.json"):
        return "Run"
    if relative.startswith("configs/") or name == "config.json":
        return "Config"
    return "Result"


def _omitted(value: Any, pointer: str) -> dict:
    return {
        "source_pointer": pointer,
        "sha256": hashlib.sha256(_encoded(value)).hexdigest(),
    }


def _compact(value: Any, pointer: str, depth: int = 0) -> tuple[Any, list[dict], bool]:
    """Copy bounded summaries; drop row arrays without creating per-row nodes."""
    name = pointer.rsplit("/", 1)[-1]
    if depth > MAX_DEPTH:
        raise ValueError("JSON nesting depth limit exceeded")
    if isinstance(value, list):
        if value and (
            name in ROW_FIELDS
            or len(value) > MAX_ARRAY_ITEMS
            or (
                name not in SUMMARY_ARRAYS
                and any(isinstance(item, (dict, list)) for item in value)
            )
        ):
            return None, [_omitted(value, pointer)], False
        if len(_encoded(value)) > MAX_VALUE_BYTES:
            return None, [_omitted(value, pointer)], False
        if name in SUMMARY_ARRAYS:
            parts = [
                _compact(item, _pointer(pointer, str(index)), depth + 1)
                for index, item in enumerate(value)
            ]
            return (
                [item[0] for item in parts if item[2]],
                [field for item in parts for field in item[1]],
                True,
            )
        return json.loads(_encoded(value)), [], True
    if isinstance(value, dict):
        if len(value) > MAX_FIELDS:
            raise ValueError("JSON nested field limit exceeded")
        compacted = [
            (key, _compact(item, _pointer(pointer, key), depth + 1))
            for key, item in sorted(value.items())
        ]
        copied = {key: result[0] for key, result in compacted if result[2]}
        omitted = [field for _, result in compacted for field in result[1]]
        if len(_encoded(copied)) > MAX_VALUE_BYTES:
            return None, [_omitted(value, pointer)], False
        return copied, omitted, True
    if len(_encoded(value)) > MAX_VALUE_BYTES:
        return None, [_omitted(value, pointer)], False
    return value, [], True


def _edge(
    source: str, target: str, relation: str, metadata: dict, inferred: bool = False
) -> dict:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "confidence": "INFERRED" if inferred else "EXTRACTED",
        "confidence_score": 0.75 if inferred else 1.0,
        "weight": 1.0,
        **{key: metadata[key] for key in ("source_file", "source_location", "sha256")},
    }


def _artifact(relative: str, digest: str, payload: dict) -> dict:
    return {
        "id": node_id(relative),
        "type": "artifact",
        "file_type": "document",
        "entity_type": _entity(relative),
        "label": relative,
        "metadata": _metadata(relative, "", digest, payload),
    }


def _fields(relative: str, payload: dict, digest: str) -> tuple[list[dict], list[dict]]:
    entries = [
        (key, _pointer("", key), value)
        for key, value in sorted(payload.items())
        if key not in {"model", "metadata"} or not isinstance(value, dict)
    ]
    entries = entries + [
        (key, _pointer("/" + parent, key), value)
        for parent in ("model", "metadata")
        if isinstance(payload.get(parent), dict)
        for key, value in sorted(payload[parent].items())
    ]
    if len(entries) > MAX_FIELDS:
        raise ValueError("JSON evidence field limit exceeded")
    processed = [
        (key, pointer, _compact(value, pointer)) for key, pointer, value in entries
    ]
    nodes = [
        {
            "id": node_id(relative, pointer),
            "type": "evidence",
            "file_type": "document",
            "entity_type": "DecisionRule"
            if key in {"decision_rule", "decision_rules"}
            else ("DecisionRule" if _entity(relative) == "Protocol" else "Conclusion")
            if key == "decision"
            else "Evidence",
            "label": key,
            "value": result[0],
            "metadata": _metadata(relative, pointer, digest, payload),
        }
        for key, pointer, result in processed
        if result[2]
    ]
    return nodes, sorted(
        [field for _, _, result in processed for field in result[1]],
        key=lambda field: field["source_pointer"],
    )


def _reference_path(root: Path, source: Path, value: str) -> Path | None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        return None
    for candidate in (root / path, source.parent / path):
        try:
            resolved = _safe_path(root, candidate)
            if _kind(resolved.relative_to(root).as_posix()) != "json":
                continue
            if resolved.stat().st_size > MAX_JSON_BYTES:
                continue
            return resolved
        except ValueError:
            continue
    return None


def _references(
    root: Path, path: Path, artifact: dict, fields: list[dict]
) -> tuple[list[dict], list[dict], list[dict]]:
    references = [
        field
        for field in fields
        if field["label"] in REFERENCES and isinstance(field["value"], str)
    ]
    checked = [
        (field, _reference_path(root, path, field["value"])) for field in references
    ]
    edges = [
        _edge(
            artifact["id"],
            node_id(target.relative_to(root).as_posix()),
            "references",
            field["metadata"],
        )
        for field, target in checked
        if target is not None
    ]
    endpoints = [
        _artifact(
            target.relative_to(root).as_posix(),
            hashlib.sha256(target.read_bytes()).hexdigest(),
            {},
        )
        for _, target in checked
        if target is not None
    ]
    unresolved = [
        {
            "source_pointer": field["metadata"]["source_location"],
            "value": field["value"],
        }
        for field, target in checked
        if target is None
    ]
    return edges, endpoints, unresolved


def _record(root: Path, path: Path) -> dict:
    payload, digest, tokens = _load(path)
    relative = path.relative_to(root).as_posix()
    artifact = _artifact(relative, digest, payload)
    fields, omitted = _fields(relative, payload, digest)
    refs, endpoints, unresolved = _references(root, path, artifact, fields)
    artifact = {
        **artifact,
        "omitted_fields": omitted,
        "unresolved_reference": unresolved,
    }
    return {
        "artifact": artifact,
        "fields": fields,
        "endpoints": endpoints,
        "edges": [
            _edge(artifact["id"], field["id"], "has_evidence", field["metadata"])
            for field in fields
        ]
        + refs,
        "tokens": tokens,
    }


def _experiments(records: list[dict]) -> tuple[list[dict], list[dict]]:
    directories = sorted(
        {
            str(PurePosixPath(record["artifact"]["label"]).parent)
            for record in records
            if record["artifact"]["label"].startswith("实验文档/results/")
        }
    )
    groups = [
        (
            directory,
            [
                record["artifact"]
                for record in records
                if str(PurePosixPath(record["artifact"]["label"]).parent) == directory
            ],
        )
        for directory in directories
    ]
    groups = [
        (directory, artifacts) for directory, artifacts in groups if len(artifacts) > 1
    ]
    nodes = [
        {
            "id": node_id(directory),
            "type": "experiment",
            "file_type": "document",
            "entity_type": "Experiment",
            "label": directory,
            "metadata": {
                **_metadata(
                    directory,
                    "",
                    hashlib.sha256(
                        _encoded(
                            [artifact["metadata"]["sha256"] for artifact in artifacts]
                        )
                    ).hexdigest(),
                    {},
                ),
                "status": "unknown",
            },
        }
        for directory, artifacts in groups
    ]
    edges = [
        _edge(
            node_id(directory),
            artifact["id"],
            "related_artifact",
            artifact["metadata"],
            True,
        )
        for directory, artifacts in groups
        for artifact in artifacts
    ]
    return nodes, edges


def extract_json(root: Path, paths: list[Path]) -> dict:
    """Rebuild from supplied whitelisted JSON files, without caches or input mutation."""
    root = _root(root)
    if not isinstance(paths, list):
        raise TypeError("paths must be a list of Path objects")
    safe = sorted(
        {_safe_path(root, path) for path in paths},
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if any(_kind(path.relative_to(root).as_posix()) != "json" for path in safe):
        raise ValueError("JSON path is outside the evidence whitelist")
    records = [_record(root, path) for path in safe]
    experiments, related = _experiments(records)
    endpoints = [node for record in records for node in record["endpoints"]]
    evidence = [
        node for record in records for node in [record["artifact"]] + record["fields"]
    ]
    nodes = sorted(
        {node["id"]: node for node in endpoints + evidence + experiments}.values(),
        key=lambda node: node["id"],
    )
    edges = sorted(
        [edge for record in records for edge in record["edges"]] + related,
        key=lambda edge: (
            edge["source"],
            edge["target"],
            edge["relation"],
            edge["source_location"],
        ),
    )
    graph = {"nodes": nodes, "edges": edges, "hyperedges": []}
    return {
        **graph,
        "input_tokens": sum(record["tokens"] for record in records),
        "output_tokens": (len(_encoded(graph)) + 3) // 4 if records else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        graph = extract_json(args.root, select_files(args.root)["json"])
    except (ValueError, OSError, TypeError) as error:
        parser.exit(2, f"experiment evidence: {error}\n")
    sys.stdout.write(_encoded(graph).decode("utf-8") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
