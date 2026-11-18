"""Explicit official 26/4 mapping access; legacy source-only stores stay intact."""

from __future__ import annotations

import ast
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .data import FeatureStore, batch_samples
from .manifests import file_hash
from .mappings import MAPPING_KEYS, MAPPING_MODES, MappingStore
from .official_split import TEST_DATABASES, validate_official_splits
from .price_capacity import OFFICIAL_TRAIN_DATABASES

OFFICIAL_MAPPING_SCOPE = "official_26_4"
CACHE_ARRAYS = tuple(f"group{i}.npy" for i in range(4)) + (
    "offsets.npy",
    "cards.npy",
)


def validate_official_pair(split: str, database: str) -> None:
    allowed = {"pretrain": OFFICIAL_TRAIN_DATABASES, "test": TEST_DATABASES}
    if (
        not isinstance(split, str)
        or not isinstance(database, str)
        or split not in allowed
        or database not in allowed[split]
    ):
        raise ValueError(
            "Official mappings require a 26/pretrain or 4/test database pair"
        )


def plain_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Reject symlinks before resolving, including directory ancestors."""
    candidate = Path(path).absolute()
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError(f"Official input/output path contains a symlink: {path}")
    resolved = candidate.resolve()
    if root is not None and not resolved.is_relative_to(plain_path(root)):
        raise ValueError("Official input path escapes its root")
    return resolved


def _identity(record: dict, split: str, database: str) -> None:
    line = record.get("source_line")
    if (
        type(line) is not int
        or line < 1
        or record.get("id") != f"{database}/{split}/{line}"
    ):
        raise ValueError("Official mapping/cache row database, split or id mismatch")
    for name in ("query_hash", "exact_hash"):
        value = record.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"Invalid official row hash: {name}")


def builder_compatibility(historical_source: str | Path) -> dict:
    """Bind real historical bytes and compare Python ASTs, never execute them."""
    historical = plain_path(historical_source)
    current = Path(__file__).resolve().parents[1] / "scripts/build_mapping_sidecar.py"
    old_ast = ast.dump(ast.parse(historical.read_bytes()), include_attributes=False)
    new_ast = ast.dump(ast.parse(current.read_bytes()), include_attributes=False)
    if old_ast != new_ast:
        raise ValueError(
            "Historical mapping builder is not AST-equivalent to current code"
        )
    return {
        "historical_source": str(historical),
        "historical_sha256": file_hash(historical),
        "current_source": str(current),
        "current_sha256": file_hash(current),
        "ast_sha256": hashlib.sha256(old_ast.encode()).hexdigest(),
        "ast_equal": True,
        "comparison": "Python AST without source positions; every input mapping is recomputed independently",
    }


def validate_optional_structural_values(path) -> None:
    """Check every optional value against the batch consumer, without a model."""
    store = path if isinstance(path, FeatureStore) else FeatureStore(plain_path(path))
    values = store.struct_log
    if values is None:
        return
    if values.dtype.kind not in ("i", "u", "f"):
        raise ValueError("Official struct_log must contain finite real numeric values")
    # __getitem__ converts to Python float (float64), then batch_samples casts
    # to float32. Certify both consumer precisions, not only the stored dtype.
    with np.errstate(over="ignore", invalid="ignore"):
        as_double = np.asarray(values, dtype=np.float64)
        as_float = as_double.astype(np.float32)
    if not np.isfinite(as_double).all() or not np.isfinite(as_float).all():
        raise ValueError(
            "Official struct_log must remain finite in float64 and float32"
        )


class OfficialMappingStore(MappingStore):
    """Independent scope constructor reusing validated legacy record/binding logic."""

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
        database = (
            expected_database if expected_database is not None else Path(path).name
        )
        validate_official_pair(expected_split, database)
        self.path = plain_path(path)
        if self.path.name != database or self.path.parent.name != expected_split:
            raise ValueError("Official mapping path must be <root>/<split>/<database>")
        manifest_path = plain_path(self.path / "manifest.json", self.path)
        jsonl = plain_path(self.path / "mapping.jsonl", self.path)
        self.manifest_hash = file_hash(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        manifest = self.manifest
        if (
            type(manifest.get("schema")) is not int
            or manifest["schema"] != 1
            or manifest.get("scope") != OFFICIAL_MAPPING_SCOPE
        ):
            raise ValueError("Unsupported official mapping schema/scope")
        if (
            manifest.get("database") != database
            or manifest.get("split") != expected_split
        ):
            raise ValueError("Official mapping database/split mismatch")
        if (
            expected_cache_hash is not None
            and manifest.get("cache_content_hash") != expected_cache_hash
        ):
            raise ValueError("Official mapping cache manifest mismatch")
        hashes = manifest.get("input_sha256")
        arrays = manifest.get("array_sha256")
        if (
            not isinstance(hashes, dict)
            or not isinstance(arrays, dict)
            or set(arrays) != set(CACHE_ARRAYS)
            or any(arrays[name] != hashes.get(f"cache/{name}") for name in CACHE_ARRAYS)
        ):
            raise ValueError(
                "Official mapping array hashes are missing or inconsistent"
            )
        if file_hash(jsonl) != manifest.get("jsonl_sha256"):
            raise ValueError("Official mapping JSONL hash mismatch")
        self.mode = mode
        self.split, self.database = expected_split, database
        self.records = {}
        with jsonl.open() as stream:
            for line in stream:
                record = json.loads(line)
                self._validate_record(record)
                _identity(record, self.split, self.database)
                key = record["query_hash"]
                if key in self.records:
                    raise ValueError(f"Duplicate mapping query_hash: {key}")
                self.records[key] = {
                    **{
                        name: record.get(name)
                        for name in (
                            "source_lines",
                            "source_line",
                            "id",
                            "exact_hash",
                            "true",
                            "pg",
                        )
                    },
                    "indices": {
                        name: list(record[mode][name])
                        for name in (*MAPPING_KEYS, "counts")
                    },
                }

    def bind_cache(self, store: FeatureStore) -> None:
        if store.path.name != self.database or store.path.parent.name != self.split:
            raise ValueError("Official mapping/cache path split or database mismatch")
        if self.manifest.get("cache_content_hash") != store.manifest.get(
            "content_hash"
        ):
            raise ValueError("Official mapping cache manifest mismatch")
        for name in ("manifest.json", *CACHE_ARRAYS):
            plain_path(store.path / name, store.path)
        if (
            not np.issubdtype(store.offsets.dtype, np.integer)
            or np.any(store.offsets[0] != 0)
            or np.any(np.diff(store.offsets, axis=0) < 0)
        ):
            raise ValueError("Invalid official cache offsets")
        for row in store.rows:
            _identity(row, self.split, self.database)
        validate_optional_structural_values(store)
        super().bind_cache(store)

    def provenance(self) -> dict:
        return {
            **super().provenance(),
            "scope": OFFICIAL_MAPPING_SCOPE,
            "split": self.split,
        }


class OfficialFeatureStore(FeatureStore):
    def __init__(self, path, mapping_path, mapping_mode="mapping"):
        path = plain_path(path)
        for name in (
            "manifest.json",
            *CACHE_ARRAYS,
            "struct_log.npy",
            "struct_manifest.json",
        ):
            plain_path(path / name, path)
        validate_official_pair(path.parent.name, path.name)
        # Deliberately do not pass mapping_path to the legacy constructor.
        super().__init__(path)
        self.mapping = OfficialMappingStore(
            mapping_path,
            mapping_mode,
            path.name,
            path.parent.name,
            self.manifest.get("content_hash"),
        )
        self.mapping.bind_cache(self)

    def batch(self, indices, device="cpu"):
        result = batch_samples([self[int(index)] for index in indices], device)
        result["mapping_scope"] = OFFICIAL_MAPPING_SCOPE
        result["mapping_split"] = self.path.parent.name
        return result


class OfficialStorePool:
    def __init__(self, cache_root, mapping_root, max_open=26, mapping_mode="mapping"):
        if type(max_open) is not int or max_open < 1:
            raise ValueError("Official store max_open must be a positive integer")
        if mapping_mode not in MAPPING_MODES:
            raise ValueError(f"Unknown mapping mode: {mapping_mode}")
        if mapping_root is None:
            raise ValueError("Official mappings require an explicit mapping_root")
        self.root, self.mapping_root = plain_path(cache_root), plain_path(mapping_root)
        self.max_open, self.mapping_mode = max_open, mapping_mode
        self.open = OrderedDict()

    def get(self, split, db):
        validate_official_pair(split, db)
        cache = plain_path(self.root / split / db, self.root)
        mapping = plain_path(self.mapping_root / split / db, self.mapping_root)
        key = (split, db)
        if key not in self.open:
            self.open[key] = OfficialFeatureStore(cache, mapping, self.mapping_mode)
        self.open.move_to_end(key)
        while len(self.open) > self.max_open:
            self.open.popitem(last=False)
        return self.open[key]


def verify_input_report(report: dict, splits=("pretrain", "test")) -> dict:
    """Recheck selected inputs only; pretrain verification never opens test files.

    The caller binds the root report's file hash in its experiment spec. This
    function checks the certificate and all selected sidecar/input file bytes
    against that report, including the original source provenance paths.
    """
    if (
        not isinstance(splits, (tuple, list))
        or not splits
        or len(set(splits)) != len(splits)
        or any(split not in ("pretrain", "test") for split in splits)
    ):
        raise ValueError("Select distinct official pretrain/test input splits")
    if (
        not isinstance(report, dict)
        or report.get("schema") != 1
        or report.get("scope") != OFFICIAL_MAPPING_SCOPE
        or report.get("state") != "completed"
        or report.get("model_predictions_run") is not False
        or report.get("training_run") is not False
        or report.get("unsupported_count") != 0
        or report.get("dedup_conflict_count") != 0
        or report.get("unsupported_count_complete") is not True
        or report.get("failed_databases") != []
        or type(report.get("include_test")) is not bool
    ):
        raise ValueError("Official input report is incomplete or failed")
    validate_official_splits(report.get("splits", {}))
    included = report["include_test"]
    if report.get("target_data_read") != (list(TEST_DATABASES) if included else []):
        raise ValueError("Official input report target-read inventory mismatch")
    databases = report.get("databases", {})
    if (
        not isinstance(databases, dict)
        or set(databases) != {"pretrain", "test"}
        or set(databases["pretrain"]) != set(OFFICIAL_TRAIN_DATABASES)
        or set(databases["test"]) != (set(TEST_DATABASES) if included else set())
        or ("test" in splits and not included)
    ):
        raise ValueError("Official input report database coverage mismatch")
    root = plain_path(report["mapping_root"])
    cache = plain_path(report["cache_root"])
    source = plain_path(report["data_root"])
    expected_target_files = set()
    for db, entry in databases["test"].items():
        for name in entry.get("input_sha256", {}):
            if name == "extractor":
                continue
            if name == "workload":
                path = source / "datas/workloads/test" / db / "workloads.sql"
            elif name == "source_features":
                path = source / "setup/features/test" / db / "features40.pkl"
            elif name.startswith("cache/"):
                path = cache / "test" / db / name.removeprefix("cache/")
            elif name.startswith("statistics/"):
                path = (
                    source
                    / "datas/statistics/test"
                    / db
                    / name.removeprefix("statistics/")
                )
            else:
                raise ValueError("Unknown target input report provenance key")
            expected_target_files.add(str(path))
    target_files = report.get("target_files_read")
    if (
        not isinstance(target_files, list)
        or any(not isinstance(value, str) for value in target_files)
        or len(set(target_files)) != len(target_files)
        or set(target_files) != expected_target_files
    ):
        raise ValueError("Official input report target file inventory mismatch")
    project = Path(__file__).resolve().parents[1]
    code = report.get("code_sha256", {})
    required_code = {
        "scripts/prepare_official_module_inputs.py",
        "spatial_meta_ce/official_mappings.py",
        "scripts/audit_input_mapping.py",
        "scripts/build_mapping_sidecar.py",
        "spatial_meta_ce/data.py",
        "spatial_meta_ce/features.py",
        "spatial_meta_ce/manifests.py",
        "spatial_meta_ce/mappings.py",
        "spatial_meta_ce/official_split.py",
        "spatial_meta_ce/price_capacity.py",
        "vendor/price_features/features_tool.py",
    }
    if "preparation_executor" in report:
        executor = report["preparation_executor"]
        parallel_source = "scripts/prepare_official_module_inputs_parallel.py"
        if (
            not isinstance(executor, dict)
            or set(executor) != {"kind", "workers", "source_sha256"}
            or executor["kind"] != "cpu-process-pool"
            or type(executor["workers"]) is not int
            or not 1 <= executor["workers"] <= 4
            or executor["source_sha256"] != code.get(parallel_source)
        ):
            raise ValueError("Official input parallel executor binding changed")
        required_code.add(parallel_source)
    if set(code) != required_code:
        raise ValueError("Official input report code closure mismatch")
    for name, expected in code.items():
        if file_hash(plain_path(project / name, project)) != expected:
            # Preserve actual producer bytes; recheck with today's stricter
            # verifier and optional-value guard, without rewriting old reports.
            snapshots = {
                "9b70883188be122ae101aa2e30e663f67b3fcbb87a66492e9b7ba201831172a4": "input-producer-official-mappings.py.txt",
                "79b9ff74eefec4ed47f8c5e19f8e0530ac8571dbf65a19d741442be54e339804": "input-producer-parallel-v3-mappings.py.txt",
            }
            if (
                name != "spatial_meta_ce/official_mappings.py"
                or expected not in snapshots
            ):
                raise ValueError(f"Official input code hash changed: {name}")
            producer = plain_path(
                project / "reports/official-module-prep-20261118" / snapshots[expected],
                project,
            )
            if file_hash(producer) != expected:
                raise ValueError("Frozen input producer snapshot hash changed")
            trees = [
                ast.parse(producer.read_bytes()),
                ast.parse((project / name).read_bytes()),
            ]
            for tree in trees:
                tree.body = [
                    node
                    for node in tree.body
                    if not (
                        isinstance(node, ast.FunctionDef)
                        and node.name == "verify_input_report"
                    )
                ]
            # The sole producer-side addition is the common value guard. Strip
            # exactly that definition and call for the historical comparison;
            # the current bound code still executes it on every selected store.
            current = trees[1]
            helper = [
                node
                for node in current.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "validate_optional_structural_values"
            ]
            classes = [
                node
                for node in current.body
                if isinstance(node, ast.ClassDef)
                and node.name == "OfficialMappingStore"
            ]
            if len(helper) != 1 or len(classes) != 1:
                raise ValueError("Current input value guard identity changed")
            methods = [
                node
                for node in classes[0].body
                if isinstance(node, ast.FunctionDef) and node.name == "bind_cache"
            ]
            guard = ast.dump(
                ast.parse("validate_optional_structural_values(store)").body[0],
                include_attributes=False,
            )
            calls = (
                [
                    node
                    for node in methods[0].body
                    if ast.dump(node, include_attributes=False) == guard
                ]
                if len(methods) == 1
                else []
            )
            if len(calls) != 1:
                raise ValueError("Current input value guard call changed")
            current.body.remove(helper[0])
            methods[0].body.remove(calls[0])
            if ast.dump(trees[0], include_attributes=False) != ast.dump(
                trees[1], include_attributes=False
            ):
                raise ValueError("Frozen input producer executable AST changed")
    for split in splits:
        expected_databases = (
            OFFICIAL_TRAIN_DATABASES if split == "pretrain" else TEST_DATABASES
        )
        for db in expected_databases:
            entry = databases[split][db]
            if (
                entry.get("state") != "completed"
                or entry.get("database") != db
                or entry.get("split") != split
                or entry.get("unsupported_count") != 0
                or entry.get("dedup_conflict_count") != 0
                or entry.get("unsupported_count_complete") is not True
                or entry.get("inputs_unchanged") is not True
                or type(entry.get("rows")) is not int
                or entry["rows"] < 1
                or entry.get("checked_rows") != entry["rows"]
                or entry.get("mapping_recoverable") != entry["rows"]
                or entry.get("sql_mapping_recoverable") != entry["rows"]
                or entry.get("recreated_rows")
                != (0 if entry.get("source_sidecar") else entry["rows"])
                or (split == "test" and entry.get("source_sidecar") is not None)
            ):
                raise ValueError(
                    f"Official input database certificate is incomplete: {split}/{db}"
                )
            folder = plain_path(root / split / db, root)
            for name, key in (
                ("manifest.json", "manifest_sha256"),
                ("mapping.jsonl", "jsonl_sha256"),
            ):
                if file_hash(plain_path(folder / name, root)) != entry.get(key):
                    raise ValueError(
                        f"Official sidecar report hash mismatch: {split}/{db}/{name}"
                    )
            manifest = json.loads((folder / "manifest.json").read_text())
            certification = manifest.get("certification", {})
            if any(
                entry.get(name) != value for name, value in certification.items()
            ) or not {
                "checked_rows",
                "mapping_recoverable",
                "sql_mapping_recoverable",
                "recreated_rows",
                "unsupported_count",
                "dedup_conflict_count",
                "unsupported_count_complete",
            }.issubset(certification):
                raise ValueError(
                    "Official input report/manifest certification mismatch"
                )
            for name in ("input_sha256", "array_sha256", "cache_content_hash"):
                if manifest.get(name) != entry.get(name):
                    raise ValueError(
                        f"Official input report/manifest binding mismatch: {name}"
                    )
            paths, hashes = manifest.get("input_paths", {}), manifest["input_sha256"]
            if set(paths) != set(hashes):
                raise ValueError(
                    "Official input provenance paths/hash inventory mismatch"
                )
            cache_folder = plain_path(cache / split / db, cache)
            actual_cache_files = {
                "cache/manifest.json",
                *(f"cache/{path.name}" for path in cache_folder.glob("*.npy")),
            }
            if (cache_folder / "struct_manifest.json").exists():
                actual_cache_files.add("cache/struct_manifest.json")
            declared_cache_files = {name for name in paths if name.startswith("cache/")}
            if actual_cache_files != declared_cache_files:
                raise ValueError(
                    "Official cache file inventory/presence changed after certification"
                )
            required_inputs = {
                "workload",
                "source_features",
                "extractor",
                "cache/manifest.json",
                *(f"cache/{name}" for name in CACHE_ARRAYS),
                *(
                    f"statistics/{name}"
                    for name in (
                        "abbrev_col_type.pkl",
                        "fanout40.pkl",
                        "histogram40.pkl",
                        "size.pkl",
                        "summary40.pkl",
                    )
                ),
            }
            if not required_inputs.issubset(paths):
                raise ValueError("Official input provenance file coverage mismatch")
            for name, value in paths.items():
                if name == "extractor":
                    # Keep the actual producer path, requiring both its frozen
                    # bytes and today's identically bound extractor bytes.
                    expected_path = plain_path(value)
                    suffix = ("vendor", "price_features", "features_tool.py")
                    if (
                        expected_path.parts[-3:] != suffix
                        or hashes[name]
                        != code["vendor/price_features/features_tool.py"]
                        or file_hash(project / "vendor/price_features/features_tool.py")
                        != hashes[name]
                    ):
                        raise ValueError("Official producer extractor binding changed")
                elif name == "workload":
                    expected_path = (
                        source / "datas/workloads" / split / db / "workloads.sql"
                    )
                elif name == "source_features":
                    expected_path = (
                        source / "setup/features" / split / db / "features40.pkl"
                    )
                elif name.startswith("statistics/") and Path(
                    name
                ).name == name.removeprefix("statistics/"):
                    expected_path = (
                        source / "datas/statistics" / split / db / Path(name).name
                    )
                elif name.startswith("cache/") and Path(name).name == name.removeprefix(
                    "cache/"
                ):
                    expected_path = cache / split / db / Path(name).name
                else:
                    raise ValueError("Unknown official input provenance file")
                if plain_path(value) != plain_path(expected_path):
                    raise ValueError(
                        "Official input provenance path split/database mismatch"
                    )
                if file_hash(expected_path) != hashes[name]:
                    raise ValueError(
                        f"Official input file hash changed: {split}/{db}/{name}"
                    )
            import_binding = entry.get("source_sidecar")
            if import_binding is not None:
                if split != "pretrain" or import_binding != manifest.get(
                    "source_sidecar"
                ):
                    raise ValueError("Official source sidecar provenance mismatch")
                legacy_root = plain_path(report["source_mapping_root"])
                for name, key in (
                    ("manifest.json", "manifest"),
                    ("mapping.jsonl", "jsonl"),
                ):
                    expected_path = plain_path(legacy_root / db / name, legacy_root)
                    if (
                        plain_path(import_binding[f"{key}_path"]) != expected_path
                        or file_hash(expected_path) != import_binding[f"{key}_sha256"]
                    ):
                        raise ValueError("Imported legacy sidecar hash/path changed")
                proof = import_binding.get("builder_compatibility")
                current_builder = project / "scripts/build_mapping_sidecar.py"
                if proof is not None:
                    if not isinstance(proof, dict):
                        raise ValueError("Historical builder proof must be an object")
                    current_proof = builder_compatibility(
                        proof.get("historical_source")
                    )
                    producer_builder = plain_path(proof.get("current_source"))
                    if (
                        producer_builder.parts[-2:]
                        != ("scripts", "build_mapping_sidecar.py")
                        or file_hash(producer_builder) != proof.get("current_sha256")
                        or {**current_proof, "current_source": str(producer_builder)}
                        != proof
                        or proof["historical_sha256"]
                        != import_binding.get("builder_sha256")
                    ):
                        raise ValueError(
                            "Historical source sidecar builder compatibility changed"
                        )
                elif import_binding.get("builder_sha256") != file_hash(current_builder):
                    raise ValueError(
                        "Imported sidecar builder has no valid compatibility proof"
                    )
            store = OfficialFeatureStore(cache / split / db, folder)
            if len(store) != entry["rows"] or len(store.mapping) != entry.get(
                "unique_queries"
            ):
                raise ValueError("Official input report cache/query coverage mismatch")
    return report
