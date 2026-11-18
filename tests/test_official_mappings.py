import contextlib
import copy
import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.build_mapping_sidecar import build_records
from scripts.prepare_official_module_inputs import InputPreparationError, prepare_inputs
from spatial_meta_ce.features import COUNT_KEYS, extractor, write_cache
from spatial_meta_ce.manifests import digest, file_hash, read_workload
from spatial_meta_ce.mappings import MappingStore
from spatial_meta_ce.official_mappings import (
    OfficialMappingStore,
    OfficialStorePool,
    verify_input_report,
)
from spatial_meta_ce.official_split import TEST_DATABASES
from spatial_meta_ce.price_capacity import OFFICIAL_TRAIN_DATABASES

ROOT = Path(__file__).resolve().parents[1]
SQL = "select count(*) from beta b, alpha a where a.id=b.aid and b.v>2 and a.x<9"
ARRAYS = [*(f"group{i}.npy" for i in range(4)), "offsets.npy", "cards.npy"]


def config(root):
    return {
        "data_root": str(root / "price"),
        "cache_root": str(root / "cache"),
        "splits": {
            "train": list(OFFICIAL_TRAIN_DATABASES),
            "validation": [],
            "test": list(TEST_DATABASES),
        },
    }


def fixture(root, split="pretrain", db="accidents", sql=SQL):
    data = config(root)
    source = Path(data["data_root"])
    workload = source / "datas/workloads" / split / db / "workloads.sql"
    feature = source / "setup/features" / split / db / "features40.pkl"
    stats = source / "datas/statistics" / split / db
    for path in (workload.parent, feature.parent, stats):
        path.mkdir(parents=True, exist_ok=True)
    hist = {
        alias: {
            column: {
                "hist": np.arange(1, 41, dtype=np.float64) + index + (alias == "b"),
                "bin_edges": np.arange(41, dtype=np.float64),
                "len": 1000,
            }
            for index, column in enumerate(columns)
        }
        for alias, columns in (("a", ("id", "x")), ("b", ("aid", "v")))
    }
    statistics = {
        "histogram40.pkl": hist,
        "fanout40.pkl": {("a.id", "b.aid"): [np.arange(40) + 1, np.arange(40) + 3]},
        "summary40.pkl": {},
        "size.pkl": {"a": {"size": 100}, "b": {"size": 200}},
        "abbrev_col_type.pkl": {"col_type": {"a": {"dsct": []}, "b": {"dsct": []}}},
    }
    for name, value in statistics.items():
        with (stats / name).open("wb") as stream:
            pickle.dump(value, stream)
    workload.write_text(f"{sql} || 10 || 2\n{sql.upper()} || 10 || 2\n")
    rows = read_workload(workload, db, split)
    # Unsupported SQL fixtures retain baseline tokens: preparation must report
    # every unsupported row instead of filtering it out.
    engine = extractor(source, db, split)
    with contextlib.redirect_stdout(io.StringIO()):
        parts, *counts = engine.create_sql_features(SQL)
    original = {
        "data_features": [parts, parts],
        "true_cards": [10.0, 10.0],
        "pg_est_cards": [2.0, 2.0],
        **{key: [value, value] for key, value in zip(COUNT_KEYS, counts)},
    }
    with feature.open("wb") as stream:
        pickle.dump(original, stream)
    provenance = {
        "source": {
            "database": db,
            "split": split,
            "workload_sha256": file_hash(workload),
            "feature_sha256": file_hash(feature),
            "extractor_sha256": file_hash(
                ROOT / "vendor/price_features/features_tool.py"
            ),
            "statistics": {name: file_hash(stats / name) for name in statistics},
            "sqlglot": "18.17.0",
        },
        "regenerated": False,
        "excluded": [],
    }
    cache = Path(data["cache_root"]) / split / db
    write_cache(cache, rows, original, provenance)
    paths = {
        "workload": workload,
        "source_features": feature,
        "extractor": ROOT / "vendor/price_features/features_tool.py",
        **{f"statistics/{name}": stats / name for name in statistics},
        **{f"cache/{name}": cache / name for name in ["manifest.json", *ARRAYS]},
    }
    return data, rows, paths


def sidecar(root, rows, paths, official=True):
    db, split, _ = rows[0]["id"].split("/")
    path = root / split / db if official else root / db
    path.mkdir(parents=True)
    records = build_records(rows)
    (path / "mapping.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = {
        "schema": 1,
        "scope": "official_26_4" if official else "CPU source-only preflight",
        "database": db,
        "split": split,
        "shuffle_seed": 42,
        "sqlglot": "18.17.0",
        "cache_content_hash": json.loads(paths["cache/manifest.json"].read_text())[
            "content_hash"
        ],
        "input_sha256": {name: file_hash(path) for name, path in paths.items()},
        "input_paths": {name: str(path) for name, path in paths.items()},
        "builder_sha256": file_hash(ROOT / "scripts/build_mapping_sidecar.py"),
        "jsonl_sha256": file_hash(path / "mapping.jsonl"),
    }
    if official:
        manifest["array_sha256"] = {
            name: file_hash(paths[f"cache/{name}"]) for name in ARRAYS
        }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


def edit_manifest(path, **changes):
    value = json.loads((path / "manifest.json").read_text())
    value.update(changes)
    (path / "manifest.json").write_text(json.dumps(value))


def edit_records(path, change):
    records = [
        json.loads(line) for line in (path / "mapping.jsonl").read_text().splitlines()
    ]
    change(records)
    (path / "mapping.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    edit_manifest(path, jsonl_sha256=file_hash(path / "mapping.jsonl"))


class OfficialStoreTests(unittest.TestCase):
    def test_both_splits_and_modes_preserve_database_and_batch_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for split, db in (("pretrain", "accidents"), ("test", "stats")):
                data, rows, paths = fixture(root, split, db)
                sidecar(root / "mapping", rows, paths)
                for mode in ("mapping", "shuffled_mapping"):
                    pool = OfficialStorePool(
                        data["cache_root"], root / "mapping", mapping_mode=mode
                    )
                    store = pool.get(split, db)
                    batch = store.batch([0, 1], "cpu")
                    self.assertEqual(batch["mapping_scope"], "official_26_4")
                    self.assertEqual(batch["mapping_split"], split)
                    self.assertEqual(batch["mapping_mode"], mode)
                    self.assertEqual(batch["databases"], [db, db])
                    self.assertEqual(batch["ids"], [r["id"] for r in rows])
                    self.assertEqual(store.unique_indices().tolist(), [0])

    def test_legacy_store_keeps_rejecting_target_and_official_store_requires_scope(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _, rows, paths = fixture(root, "test", "stats")
            path = sidecar(root / "mapping", rows, paths)
            with self.assertRaisesRegex(ValueError, "pretrain-only"):
                MappingStore(path, expected_database="stats", expected_split="test")
            edit_manifest(path, scope="legacy")
            with self.assertRaisesRegex(ValueError, "scope"):
                OfficialMappingStore(
                    path, expected_database="stats", expected_split="test"
                )

    def test_pair_names_modes_and_limits_fail_before_file_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            pool = OfficialStorePool(root / "cache", root / "mapping")
            for split, db in (
                ("pretrain", "stats"),
                ("test", "accidents"),
                ("finetune", "stats"),
                ("pretrain", "unknown"),
                ("pretrain", "../accidents"),
                ("../pretrain", "accidents"),
                ("pretrain", "/accidents"),
            ):
                with self.subTest(split=split, db=db), self.assertRaises(ValueError):
                    pool.get(split, db)
            for limit in (0, -1, True):
                with self.assertRaises(ValueError):
                    OfficialStorePool(root / "cache", root / "mapping", max_open=limit)
            with self.assertRaises(ValueError):
                OfficialStorePool(
                    root / "cache", root / "mapping", mapping_mode="missing"
                )
            self.assertEqual(list(root.iterdir()), [])

    def test_no_fallback_to_legacy_root_database_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data, rows, paths = fixture(root)
            sidecar(root / "mapping", rows, paths, official=False)
            with self.assertRaises(FileNotFoundError):
                OfficialStorePool(data["cache_root"], root / "mapping").get(
                    "pretrain", "accidents"
                )

    def test_jsonl_tampering_and_missing_mode_or_query_coverage_are_rejected(self):
        for mutation in ("bytes", "mode", "coverage", "cards", "counts", "identity"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                data, rows, paths = fixture(root)
                path = sidecar(root / "mapping", rows, paths)
                if mutation == "bytes":
                    with (path / "mapping.jsonl").open("a") as stream:
                        stream.write("\n")
                elif mutation == "mode":
                    edit_records(
                        path, lambda records: records[0].pop("shuffled_mapping")
                    )
                elif mutation == "coverage":
                    edit_records(
                        path, lambda records: records[0].update(source_lines=[1])
                    )
                elif mutation == "cards":
                    edit_records(path, lambda records: records[0].update(pg=99))
                elif mutation == "counts":
                    edit_records(
                        path,
                        lambda records: records[0]["mapping"].update(
                            counts=[2, 2, 2, 0]
                        ),
                    )
                else:
                    edit_records(
                        path, lambda records: records[0].update(exact_hash="wrong")
                    )
                with self.assertRaises(ValueError):
                    OfficialStorePool(data["cache_root"], root / "mapping").get(
                        "pretrain", "accidents"
                    )

    def test_array_bytes_and_declared_array_hashes_are_bound(self):
        for mutation in ("array", "declared", "manifest"):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                data, rows, paths = fixture(root)
                path = sidecar(root / "mapping", rows, paths)
                if mutation == "array":
                    array = np.load(paths["cache/group0.npy"]).copy()
                    array[0, 0] += 1
                    np.save(paths["cache/group0.npy"], array)
                elif mutation == "declared":
                    edit_manifest(path, array_sha256={})
                else:
                    edit_manifest(path, cache_content_hash="0" * 64)
                with self.assertRaises(ValueError):
                    OfficialStorePool(data["cache_root"], root / "mapping").get(
                        "pretrain", "accidents"
                    )

    def test_symlinks_at_root_split_database_or_file_are_rejected(self):
        for component in ("root", "split", "database", "file"):
            with (
                self.subTest(component=component),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary).resolve()
                data, rows, paths = fixture(root)
                path = sidecar(root / "mapping", rows, paths)
                selected = {
                    "root": root / "mapping",
                    "split": path.parent,
                    "database": path,
                    "file": path / "mapping.jsonl",
                }[component]
                real = selected.with_name(selected.name + "-real")
                selected.rename(real)
                selected.symlink_to(real, target_is_directory=real.is_dir())
                with self.assertRaisesRegex(ValueError, "symlink"):
                    OfficialStorePool(data["cache_root"], root / "mapping").get(
                        "pretrain", "accidents"
                    )


class OfficialPreparationTests(unittest.TestCase):
    def make_sources(self, root):
        for db in OFFICIAL_TRAIN_DATABASES:
            data, rows, paths = fixture(root, db=db)
            sidecar(root / "legacy", rows, paths, official=False)
        return data

    def test_invalid_split_or_incomplete_inputs_never_create_output_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = config(root)
            bad = copy.deepcopy(data)
            bad["splits"]["train"][0] = "stats"
            for candidate in (bad, data):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prepare_inputs(candidate, root / "new-parent/mappings")
                self.assertFalse((root / "new-parent").exists())

    def test_imports_all_26_without_reading_or_requiring_any_target_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            report = prepare_inputs(data, root / "official", root / "legacy")
            self.assertEqual(report["state"], "completed")
            self.assertEqual(
                list(report["databases"]["pretrain"]), list(OFFICIAL_TRAIN_DATABASES)
            )
            self.assertEqual(report["databases"]["test"], {})
            self.assertEqual(report["target_data_read"], [])
            self.assertEqual(report["target_files_read"], [])
            self.assertFalse(report["model_predictions_run"])
            verify_input_report(report, splits=("pretrain",))
            pool = OfficialStorePool(data["cache_root"], root / "official")
            for db in OFFICIAL_TRAIN_DATABASES:
                self.assertEqual(len(pool.get("pretrain", db)), 2)
                self.assertEqual(
                    file_hash(root / "legacy" / db / "mapping.jsonl"),
                    file_hash(root / "official/pretrain" / db / "mapping.jsonl"),
                )

    def test_explicit_target_preparation_reconstructs_every_row_and_binds_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            for db in TEST_DATABASES:
                fixture(root, "test", db)
            inputs = {
                p: file_hash(p)
                for parent in ("price", "cache", "legacy")
                for p in (root / parent).rglob("*")
                if p.is_file()
            }
            report = prepare_inputs(
                data, root / "official", root / "legacy", include_test=True
            )
            self.assertEqual(report["target_data_read"], list(TEST_DATABASES))
            self.assertFalse(report["model_predictions_run"])
            self.assertEqual(report["data_config_hash"], digest(data))
            self.assertEqual(report["unsupported_count"], 0)
            self.assertTrue(report["unsupported_count_complete"])
            pool = OfficialStorePool(data["cache_root"], root / "official")
            for db in TEST_DATABASES:
                entry = report["databases"]["test"][db]
                self.assertEqual(entry["recreated_rows"], 2)
                self.assertEqual(entry["mapping_recoverable"], 2)
                self.assertEqual(entry["duplicate_rows"], 1)
                self.assertEqual(len(entry["array_sha256"]), 6)
                self.assertEqual(
                    entry["manifest_sha256"],
                    file_hash(root / "official/test" / db / "manifest.json"),
                )
                self.assertEqual(pool.get("test", db).batch([1])["databases"], [db])
            self.assertTrue(report["target_files_read"])
            self.assertTrue(
                all("/test/" in path for path in report["target_files_read"])
            )
            self.assertEqual(inputs, {p: file_hash(p) for p in inputs})
            self.assertEqual(
                json.loads((root / "official/official_module_inputs.json").read_text()),
                report,
            )
            verify_input_report(report, splits=("test",))

            # A training verification must succeed after all four target trees
            # disappear. Its only target access is the already-bound root JSON.
            for parent in (
                root / "official",
                root / "cache",
                root / "price/datas/workloads",
                root / "price/datas/statistics",
                root / "price/setup/features",
            ):
                (parent / "test").rename(parent / "test-not-readable")
            verify_input_report(report, splits=("pretrain",))
            with self.assertRaises((ValueError, FileNotFoundError)):
                verify_input_report(report, splits=("test",))

    def test_fresh_sources_without_import_reconstruct_all_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for db in OFFICIAL_TRAIN_DATABASES:
                data, _, _ = fixture(root, db=db)
            report = prepare_inputs(data, root / "official")
            self.assertEqual(report["state"], "completed")
            self.assertEqual(
                sum(
                    r["recreated_rows"]
                    for r in report["databases"]["pretrain"].values()
                ),
                52,
            )
            self.assertEqual(report["target_data_read"], [])
            verify_input_report(report, splits=("pretrain",))

    def test_legacy_builder_requires_hash_bound_ast_equivalent_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            builder = root / "historical-builder.py"
            builder.write_text(
                "# historical formatting\n"
                + (ROOT / "scripts/build_mapping_sidecar.py").read_text()
            )
            for db in OFFICIAL_TRAIN_DATABASES:
                edit_manifest(root / "legacy" / db, builder_sha256=file_hash(builder))
            report = prepare_inputs(
                data, root / "official", root / "legacy", legacy_builder_source=builder
            )
            proof = report["databases"]["pretrain"]["accidents"]["source_sidecar"][
                "builder_compatibility"
            ]
            self.assertEqual(proof["historical_sha256"], file_hash(builder))
            self.assertTrue(proof["ast_equal"])
            verify_input_report(report, splits=("pretrain",))
            builder.write_text(builder.read_text() + "\nCOMPATIBILITY_BROKEN = True\n")
            with self.assertRaises(ValueError):
                verify_input_report(report, splits=("pretrain",))
            with self.assertRaises(ValueError):
                prepare_inputs(
                    data,
                    root / "must-not-create",
                    root / "legacy",
                    legacy_builder_source=builder,
                )
            self.assertFalse((root / "must-not-create").exists())

    def test_bound_producer_paths_survive_training_code_relocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            old_builder = root / "historical-builder.py"
            old_builder.write_text(
                "# historical\n"
                + (ROOT / "scripts/build_mapping_sidecar.py").read_text()
            )
            for db in OFFICIAL_TRAIN_DATABASES:
                edit_manifest(
                    root / "legacy" / db, builder_sha256=file_hash(old_builder)
                )
            report = prepare_inputs(
                data,
                root / "official",
                root / "legacy",
                legacy_builder_source=old_builder,
            )
            producer = root / "frozen-producer"
            extractor_path = producer / "vendor/price_features/features_tool.py"
            builder_path = producer / "scripts/build_mapping_sidecar.py"
            for source, destination in (
                (ROOT / "vendor/price_features/features_tool.py", extractor_path),
                (ROOT / "scripts/build_mapping_sidecar.py", builder_path),
            ):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            for db, entry in report["databases"]["pretrain"].items():
                folder = root / "official/pretrain" / db
                manifest_path = folder / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["input_paths"]["extractor"] = str(extractor_path)
                proof = manifest["source_sidecar"]["builder_compatibility"]
                proof["current_source"] = str(builder_path)
                entry["source_sidecar"] = copy.deepcopy(manifest["source_sidecar"])
                manifest_path.write_text(json.dumps(manifest))
                entry["manifest_sha256"] = file_hash(manifest_path)
            before = copy.deepcopy(report)
            verify_input_report(report, splits=("pretrain",))
            self.assertEqual(report, before)
            extractor_path.write_bytes(extractor_path.read_bytes() + b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "hash|changed"):
                verify_input_report(report, splits=("pretrain",))

    def test_optional_structural_sidecars_added_after_legacy_are_bound_in_new_report(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            for db in OFFICIAL_TRAIN_DATABASES:
                folder = root / "cache/pretrain" / db
                manifest = json.loads((folder / "manifest.json").read_text())
                np.save(folder / "struct_log.npy", np.zeros(2))
                (folder / "struct_manifest.json").write_text(
                    json.dumps(
                        {
                            "cache_content_hash": manifest["content_hash"],
                            "rows": 2,
                        }
                    )
                )
            report = prepare_inputs(data, root / "official", root / "legacy")
            entry = report["databases"]["pretrain"]["accidents"]
            self.assertIn("cache/struct_log.npy", entry["input_sha256"])
            self.assertIn("cache/struct_manifest.json", entry["input_sha256"])
            verify_input_report(report, splits=("pretrain",))
            np.save(root / "cache/pretrain/accidents/struct_log.npy", np.ones(2))
            with self.assertRaises(ValueError):
                verify_input_report(report, splits=("pretrain",))

    def test_certified_absence_rejects_new_optional_cache_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            report = prepare_inputs(data, root / "official", root / "legacy")
            folder = root / "cache/pretrain/accidents"
            manifest = json.loads((folder / "manifest.json").read_text())
            np.save(folder / "struct_log.npy", np.full(2, float("nan")))
            (folder / "struct_manifest.json").write_text(
                json.dumps(
                    {
                        "cache_content_hash": manifest["content_hash"],
                        "rows": 2,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "inventory|presence|changed"):
                verify_input_report(report, splits=("pretrain",))

    def test_preexisting_nonfinite_structural_values_never_certify_or_load(self):
        from spatial_meta_ce.official_mappings import (
            validate_optional_structural_values,
        )

        for value in (
            float("nan"),
            float("inf"),
            -float("inf"),
            np.timedelta64(1, "D"),
            np.longdouble("1e400"),
            np.float64(1e100),
            np.complex128(1),
            np.str_("1"),
        ):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                data = self.make_sources(root)
                folder = root / "cache/pretrain/accidents"
                manifest = json.loads((folder / "manifest.json").read_text())
                np.save(folder / "struct_log.npy", np.full(2, value))
                (folder / "struct_manifest.json").write_text(
                    json.dumps(
                        {
                            "cache_content_hash": manifest["content_hash"],
                            "rows": 2,
                        }
                    )
                )
                # A legacy-but-hash-bound sidecar represents exactly this input.
                rows = read_workload(
                    root / "price/datas/workloads/pretrain/accidents/workloads.sql",
                    "accidents",
                    "pretrain",
                )
                from scripts.prepare_official_module_inputs import _input_paths

                paths = _input_paths(data, "pretrain", "accidents")
                with self.assertRaisesRegex(ValueError, "finite"):
                    validate_optional_structural_values(folder)
                with self.assertRaises(InputPreparationError) as raised:
                    prepare_inputs(data, root / "official", root / "legacy")
                self.assertIn(
                    "finite",
                    raised.exception.report["databases"]["pretrain"]["accidents"][
                        "error"
                    ],
                )
                self.assertEqual(raised.exception.report["state"], "failed")
                with patch(
                    "spatial_meta_ce.official_mappings.validate_optional_structural_values"
                ):
                    historical = prepare_inputs(
                        data, root / "historical", root / "legacy"
                    )
                self.assertEqual(historical["state"], "completed")
                with self.assertRaisesRegex(ValueError, "finite"):
                    verify_input_report(historical, splits=("pretrain",))
                sidecar(root / "direct", rows, paths)
                with self.assertRaisesRegex(ValueError, "finite"):
                    OfficialStorePool(data["cache_root"], root / "direct").get(
                        "pretrain", "accidents"
                    )

    def test_original_producer_bytes_allow_verifier_only_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            report = prepare_inputs(data, root / "official", root / "legacy")
            original = copy.deepcopy(report)
            producer = ROOT / (
                "reports/official-module-prep-20261118/"
                "input-producer-official-mappings.py.txt"
            )
            name = "spatial_meta_ce/official_mappings.py"
            report["code_sha256"][name] = file_hash(producer)
            producer_report = copy.deepcopy(report)
            verify_input_report(report, splits=("pretrain",))
            self.assertEqual(report, producer_report)
            self.assertNotEqual(report["code_sha256"], original["code_sha256"])
            bad = copy.deepcopy(report)
            bad["code_sha256"][name] = "0" * 64
            with self.assertRaisesRegex(ValueError, "code hash changed"):
                verify_input_report(bad, splits=("pretrain",))
            bad = copy.deepcopy(report)
            bad["code_sha256"]["spatial_meta_ce/data.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "code hash changed"):
                verify_input_report(bad, splits=("pretrain",))
            real_read = Path.read_bytes

            def altered_snapshot(path):
                value = real_read(path)
                return (
                    value + b"\nBROKEN_PRODUCER = True\n" if path == producer else value
                )

            with patch.object(Path, "read_bytes", altered_snapshot):
                with self.assertRaisesRegex(ValueError, "producer|code hash changed"):
                    verify_input_report(report, splits=("pretrain",))
            # Even a matching snapshot digest cannot admit a producer whose
            # executable AST (outside the verifier) differs from current code.
            real_hash = file_hash

            def claimed_hash(path):
                return (
                    real_hash(producer) if Path(path) == producer else real_hash(path)
                )

            with (
                patch.object(Path, "read_bytes", altered_snapshot),
                patch(
                    "spatial_meta_ce.official_mappings.file_hash",
                    side_effect=claimed_hash,
                ),
                self.assertRaisesRegex(ValueError, "producer|AST"),
            ):
                verify_input_report(report, splits=("pretrain",))
            folder = root / "cache/pretrain/accidents"
            np.save(folder / "struct_log.npy", np.full(2, float("nan")))
            with self.assertRaisesRegex(ValueError, "inventory|presence|changed"):
                verify_input_report(report, splits=("pretrain",))

    def test_frozen_parallel_producer_is_rechecked_without_report_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            report = prepare_inputs(data, root / "official", root / "legacy")
            name = "spatial_meta_ce/official_mappings.py"
            producer = ROOT / (
                "reports/official-module-prep-20261118/"
                "input-producer-parallel-v3-mappings.py.txt"
            )
            report["code_sha256"][name] = file_hash(producer)
            before = copy.deepcopy(report)
            verify_input_report(report, splits=("pretrain",))
            self.assertEqual(report, before)

    def test_report_verification_rejects_input_drift_and_partial_certificates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            report = prepare_inputs(data, root / "official", root / "legacy")
            for field, value in (
                ("state", "running"),
                ("model_predictions_run", True),
                ("unsupported_count_complete", False),
            ):
                bad = copy.deepcopy(report)
                bad[field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    verify_input_report(bad, splits=("pretrain",))
            bad = copy.deepcopy(report)
            bad["databases"]["pretrain"]["accidents"]["recreated_rows"] = 1
            with self.assertRaises(ValueError):
                verify_input_report(bad, splits=("pretrain",))
            bad = copy.deepcopy(report)
            bad["target_files_read"] = [
                str(root / "price/datas/workloads/test/stats/workloads.sql")
            ]
            with self.assertRaises(ValueError):
                verify_input_report(bad, splits=("pretrain",))
            bad = copy.deepcopy(report)
            bad["databases"]["pretrain"].pop("chembl")
            with self.assertRaises(ValueError):
                verify_input_report(bad, splits=("pretrain",))
            for selected in ((), ("finetune",), ("pretrain", "pretrain")):
                with self.assertRaises(ValueError):
                    verify_input_report(report, splits=selected)
            path = root / "price/datas/workloads/pretrain/accidents/workloads.sql"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash|changed"):
                verify_input_report(report, splits=("pretrain",))

    def test_output_overlap_existing_and_symlink_paths_refuse_before_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            (root / "existing").mkdir()
            (root / "alias").symlink_to(root / "existing", target_is_directory=True)
            for output in (
                root / "price/new",
                root / "cache/new",
                root / "legacy/new",
                root / "existing",
                root / "alias/new",
            ):
                with (
                    self.subTest(output=output),
                    self.assertRaises((ValueError, FileExistsError)),
                ):
                    prepare_inputs(data, output, root / "legacy")
                self.assertFalse(output.name == "new" and output.exists())

    def test_target_missing_input_or_source_symlink_refuses_before_output_parents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            with self.assertRaises(FileNotFoundError):
                prepare_inputs(
                    data, root / "new/output", root / "legacy", include_test=True
                )
            self.assertFalse((root / "new").exists())
            path = root / "price/datas/workloads/pretrain/accidents/workloads.sql"
            real = path.with_name("real.sql")
            path.rename(real)
            path.symlink_to(real)
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_inputs(data, root / "new/output", root / "legacy")
            self.assertFalse((root / "new").exists())

    def test_duplicate_query_pg_conflicts_are_explicit_and_not_silently_merged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            for db in TEST_DATABASES:
                _, _, paths = fixture(root, "test", db)
                lines = paths["workload"].read_text().splitlines()
                paths["workload"].write_text(
                    lines[0] + "\n" + lines[1].replace("|| 2", "|| 3") + "\n"
                )
                manifest = json.loads(paths["cache/manifest.json"].read_text())
                manifest["provenance"]["source"]["workload_sha256"] = file_hash(
                    paths["workload"]
                )
                manifest["content_hash"] = digest(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key != "content_hash"
                    }
                )
                paths["cache/manifest.json"].write_text(json.dumps(manifest))
            with self.assertRaises(InputPreparationError) as raised:
                prepare_inputs(
                    data, root / "official", root / "legacy", include_test=True
                )
            self.assertEqual(raised.exception.report["dedup_conflict_count"], 4)
            self.assertEqual(raised.exception.report["unsupported_count"], 4)

    def test_source_provenance_drift_is_rechecked_during_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            path = root / "price/datas/workloads/pretrain/accidents/workloads.sql"
            path.write_text(path.read_text().replace("|| 10", "|| 11"))
            with self.assertRaises(InputPreparationError) as raised:
                prepare_inputs(data, root / "official", root / "legacy")
            self.assertEqual(raised.exception.report["state"], "failed")
            self.assertIn("accidents", str(raised.exception))
            self.assertFalse(
                (root / "official/pretrain/accidents/manifest.json").exists()
            )

    def test_unsupported_target_rows_are_all_counted_and_never_filtered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            for db in TEST_DATABASES:
                fixture(
                    root,
                    "test",
                    db,
                    sql="select count(*) from alpha a join beta b on a.id=b.aid",
                )
            with self.assertRaises(InputPreparationError) as raised:
                prepare_inputs(
                    data, root / "official", root / "legacy", include_test=True
                )
            report = raised.exception.report
            self.assertEqual(report["unsupported_count"], 8)
            self.assertTrue(report["unsupported_count_complete"])
            self.assertEqual(report["target_data_read"], list(TEST_DATABASES))
            self.assertFalse(report["model_predictions_run"])
            for db in TEST_DATABASES:
                self.assertFalse(
                    (root / "official/test" / db / "manifest.json").exists()
                )

    def test_target_reconstruction_cannot_certify_reordered_feature_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            data = self.make_sources(root)
            for db in TEST_DATABASES:
                fixture(root, "test", db)
            real_factory = extractor

            def wrong_factory(source, db, split):
                engine = real_factory(source, db, split)
                create = engine.create_sql_features

                def reordered(sql):
                    parts, *counts = create(sql)
                    return (
                        parts[0].reshape(2, 40).flip(0).reshape(-1),
                        *parts[1:],
                    ), *counts

                engine.create_sql_features = reordered
                return engine

            with patch(
                "scripts.prepare_official_module_inputs.extractor",
                side_effect=wrong_factory,
            ):
                with self.assertRaises(InputPreparationError) as raised:
                    prepare_inputs(
                        data, root / "official", root / "legacy", include_test=True
                    )
            self.assertEqual(raised.exception.report["unsupported_count"], 8)


if __name__ == "__main__":
    unittest.main()
