import json
import math
from collections import Counter
from pathlib import Path

from .manifests import file_hash

MAPPING_MODES = ("mapping", "shuffled_mapping")
MAPPING_KEYS = ("join_column_tables", "filter_column_tables", "fanout_join_columns")
TARGETS = frozenset(("imdb", "stats", "genome", "ergastf1"))


def validate_mapping(mapping: dict, counts: list[int]) -> None:
    if (
        not isinstance(counts, list)
        or len(counts) != 4
        or any(type(v) is not int or v < 0 for v in counts)
        or counts[2] < 1
        or counts[1] % 2
    ):
        raise ValueError("Invalid mapping counts or incomplete fanout pairs")
    if not isinstance(mapping, dict) or mapping.get("counts") != counts:
        raise ValueError("Mapping counts disagree with tokens")
    for key, length, limit in zip(
        MAPPING_KEYS,
        (counts[0], counts[3], counts[1]),
        (counts[2], counts[2], counts[0]),
    ):
        values = mapping.get(key)
        if (
            not isinstance(values, list)
            or len(values) != length
            or any(type(v) is not int or v < 0 or v >= limit for v in values)
        ):
            raise ValueError(f"Invalid mapping indices: {key}")


class MappingStore:
    def __init__(
        self,
        path: str | Path,
        mode: str = "mapping",
        expected_database: str | None = None,
        expected_split: str = "pretrain",
        expected_cache_hash: str | None = None,
    ) -> None:
        if mode not in MAPPING_MODES:
            raise ValueError(f"Unknown mapping mode: {mode}")
        self.path = Path(path)
        self.mode = mode
        self.manifest_hash = file_hash(self.path / "manifest.json")
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        database = self.manifest.get("database")
        if self.manifest.get("schema") != 1 or not isinstance(database, str):
            raise ValueError("Unsupported mapping sidecar schema/database")
        if database in TARGETS or expected_split != "pretrain":
            raise ValueError("Mapping sidecars are source pretrain-only")
        if expected_database is not None and database != expected_database:
            raise ValueError("Mapping database mismatch")
        if self.manifest.get("split") != expected_split:
            raise ValueError("Mapping split mismatch")
        if (
            expected_cache_hash is not None
            and self.manifest.get("cache_content_hash") != expected_cache_hash
        ):
            raise ValueError("Mapping cache manifest mismatch")
        jsonl = self.path / "mapping.jsonl"
        if file_hash(jsonl) != self.manifest.get("jsonl_sha256"):
            raise ValueError("Mapping JSONL hash mismatch")
        records = {}
        with jsonl.open() as stream:
            for line in stream:
                record = json.loads(line)
                self._validate_record(record)
                key = record["query_hash"]
                if key in records:
                    raise ValueError(f"Duplicate mapping query_hash: {key}")
                selected = record[mode]
                records[key] = {
                    "source_lines": record["source_lines"],
                    "source_line": record.get("source_line"),
                    "id": record.get("id"),
                    "exact_hash": record.get("exact_hash"),
                    "true": record.get("true"),
                    "pg": record.get("pg"),
                    "indices": {
                        k: list(selected[k]) for k in (*MAPPING_KEYS, "counts")
                    },
                }
        self.records = records

    @staticmethod
    def _validate_record(record: dict) -> None:
        if not isinstance(record, dict):
            raise ValueError("Mapping record must be an object")
        key, lines = record.get("query_hash"), record.get("source_lines")
        if not isinstance(key, str) or not key:
            raise ValueError("Invalid mapping query_hash")
        if (
            not isinstance(lines, list)
            or not lines
            or any(type(v) is not int or v < 1 for v in lines)
            or len(set(lines)) != len(lines)
        ):
            raise ValueError("Invalid mapping source_lines")
        true = record.get("mapping")
        shuffled = record.get("shuffled_mapping")
        if not isinstance(true, dict) or not isinstance(shuffled, dict):
            raise ValueError("Both mapping modes must be present")
        counts = true.get("counts")
        validate_mapping(true, counts)
        validate_mapping(shuffled, counts)
        for key in MAPPING_KEYS:
            slices = (
                (slice(None),)
                if key != "fanout_join_columns"
                else (slice(0, None, 2), slice(1, None, 2))
            )
            if any(Counter(true[key][s]) != Counter(shuffled[key][s]) for s in slices):
                raise ValueError(f"Shuffled mapping marginal mismatch: {key}")

    def bind_cache(self, store) -> None:
        hashes = self.manifest.get("input_sha256", {})
        files = [
            store.path / "manifest.json",
            store.path / "cards.npy",
            store.path / "offsets.npy",
            *[store.path / f"group{i}.npy" for i in range(4)],
        ]
        for path in files:
            if hashes.get(f"cache/{path.name}") != file_hash(path):
                raise ValueError(f"Mapping cache file hash mismatch: {path.name}")
        coverage = {}
        for index, row in enumerate(store.rows):
            key = row["query_hash"]
            if key not in self.records:
                raise ValueError("Mapping sidecar is missing a cache query")
            record = self.records[key]
            counts = (store.offsets[index + 1] - store.offsets[index]).tolist()
            validate_mapping(record["indices"], counts)
            for field, value in zip(("true", "pg"), store.cards[index]):
                actual = record[field]
                if (
                    type(actual) not in (int, float)
                    or not math.isfinite(actual)
                    or actual < 0
                    or actual != float(value)
                ):
                    raise ValueError(f"Mapping/cache cardinality mismatch: {field}")
            if row["source_line"] == record["source_line"]:
                if (
                    row["id"] != record["id"]
                    or row.get("exact_hash") != record["exact_hash"]
                ):
                    raise ValueError("Mapping/cache first-row identity mismatch")
            coverage.setdefault(key, []).append(row["source_line"])
        if set(coverage) != set(self.records):
            raise ValueError("Mapping sidecar query coverage mismatch")
        if any(
            sorted(lines) != sorted(self.records[key]["source_lines"])
            or self.records[key]["source_line"] != lines[0]
            for key, lines in coverage.items()
        ):
            raise ValueError("Mapping sidecar source line coverage mismatch")

    def __getitem__(self, query_hash: str) -> dict:
        return {
            key: list(values)
            for key, values in self.records[query_hash]["indices"].items()
        }

    def __len__(self) -> int:
        return len(self.records)

    def provenance(self) -> dict:
        return {
            "manifest_hash": self.manifest_hash,
            "jsonl_hash": self.manifest["jsonl_sha256"],
            "mode": self.mode,
        }
