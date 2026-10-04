import concurrent.futures
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts/experiment_evidence.py"


class ExperimentGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(MODULE.exists(), "experiment evidence adapter not implemented")
        spec = importlib.util.spec_from_file_location("experiment_evidence", MODULE)
        self.adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.adapter)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, relative: str, value: object) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
        path.write_text(text, encoding="utf-8")
        return path

    def build(self, *paths: Path) -> dict:
        return self.adapter.extract_json(self.root, list(paths))

    def field(self, graph: dict, path: Path, pointer: str) -> dict:
        identifier = self.adapter.node_id(
            path.relative_to(self.root).as_posix(), pointer
        )
        return next(node for node in graph["nodes"] if node["id"] == identifier)

    def test_stable_ids_same_stem_and_pointer_boundary(self) -> None:
        paths = ["configs/summary.json", "实验文档/results/summary.json", "a", "a/b"]
        ids = [self.adapter.node_id(path, "/seed") for path in paths]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertRegex(ids[0], r"^[a-z]+_[0-9a-f]{64}$")
        self.assertEqual(ids[0], self.adapter.node_id(paths[0], "/seed"))
        self.assertNotEqual(
            self.adapter.node_id("a", "/b"), self.adapter.node_id("a/b")
        )

    def test_whitelist_exclusions_and_sensitive_names(self) -> None:
        accepted = {
            "code": [
                "core/nested/a.py",
                "spatial_meta_ce/a.py",
                "scripts/a.py",
                "tests/a.py",
                "vendor/nested/a.py",
                "reports/a.py",
            ],
            "document": [
                "README.md",
                "docs/a/b.md",
                "实验文档/a.md",
                "reports/STATUS.md",
            ],
            "json": [
                "configs/token.json",
                "configs/password.json",
                "configs/secret.json",
                "实验文档/results/config.json",
                "实验文档/results/a/run_config.json",
                "实验文档/results/protocol.json",
                "实验文档/results/summary.json",
                "实验文档/results/decision.json",
                "实验文档/results/status.json",
                "实验文档/results/result.json",
                "实验文档/results/a-summary.json",
                "实验文档/results/a-run_config.json",
                "实验文档/results/select-best.json",
                "实验文档/results/price-lr-search-result.json",
                "实验文档/results/secret-summary.json",
            ],
        }
        rejected = [
            "reports/summary.json",
            "reports/nested/a.py",
            "configs/nested/a.json",
            "实验文档/nested/a.md",
            "实验文档/results/metrics.json",
            "runs/summary.json",
            "实验文档/results/stale-summary.json",
            "other/a.py",
        ]
        rejected += [
            f"实验文档/results/{name}/summary.json"
            for name in ["history", "manifests", "progress", "log", "stale"]
        ]
        for path in sum(accepted.values(), []) + rejected:
            self.write(path, {})
        selected = self.adapter.select_files(self.root)
        self.assertEqual(set(selected), {"code", "document", "json"})
        for kind, expected in accepted.items():
            self.assertEqual(
                [p.relative_to(self.root).as_posix() for p in selected[kind]],
                sorted(expected),
            )

    def test_empty_inputs(self) -> None:
        self.assertEqual(
            self.adapter.select_files(self.root),
            {"code": [], "document": [], "json": []},
        )
        self.assertEqual(
            self.build(),
            {
                "nodes": [],
                "edges": [],
                "hyperedges": [],
                "input_tokens": 0,
                "output_tokens": 0,
            },
        )

    def test_symlink_selection_and_extract_traversal(self) -> None:
        real = self.write("configs/a.json", {})
        link = self.root / "configs/link.json"
        link.symlink_to(real)
        (self.root / "docs").symlink_to(real.parent, target_is_directory=True)
        self.assertEqual(self.adapter.select_files(self.root)["json"], [real])
        self.assertEqual(self.adapter.select_files(self.root)["document"], [])
        for path in [
            link,
            self.root / "../escape.json",
            self.root / "configs/../configs/a.json",
        ]:
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "path|root|symlink"),
            ):
                self.build(path)

    def test_invalid_api_types(self) -> None:
        for root in [None, "root", self.root / "missing"]:
            with self.subTest(root=root), self.assertRaises((TypeError, ValueError)):
                self.adapter.select_files(root)
        for paths in [None, "paths", [None]]:
            with self.subTest(paths=paths), self.assertRaises((TypeError, ValueError)):
                self.adapter.extract_json(self.root, paths)
        for path, pointer in [
            (None, ""),
            ("", ""),
            ("../a", ""),
            ("/a", ""),
            ("a", None),
            ("a", "seed"),
        ]:
            with self.subTest(path=path), self.assertRaises((TypeError, ValueError)):
                self.adapter.node_id(path, pointer)

    def test_references_only_read_bounded_whitelisted_json(self) -> None:
        self.write("configs/model.json", {"seed": 42})
        self.write(".git", {"secret": "never read"})
        source = self.write("实验文档/results/run_config.json", {"data_config": ".git"})
        graph = self.build(source)
        artifact = self.field(graph, source, "")
        self.assertEqual(artifact["unresolved_reference"][0]["value"], ".git")
        self.assertFalse(
            any(
                node.get("metadata", {}).get("source_file") == ".git"
                for node in graph["nodes"]
            )
        )

    def test_unknown_metadata_and_actual_parsed_field_values(self) -> None:
        value = {
            "seed": 0,
            "empty": "",
            "null": None,
            "empty_array": [],
            "a/b~c": "中文",
            "model": {"arch": "price", "width": 64},
            "metadata": {
                "method": "erm",
                "train_databases": ["a"],
                "validation_databases": ["v"],
                "training_signature": {"seed": 42},
                "manifest_hashes": {"a": "hash"},
            },
        }
        path = self.write("实验文档/results/summary.json", value)
        graph = self.build(path)
        artifact = self.field(graph, path, "")
        self.assertEqual(artifact["type"], "artifact")
        self.assertEqual(artifact["file_type"], "document")
        self.assertEqual(artifact["entity_type"], "Result")
        self.assertEqual(
            artifact["metadata"]["source_file"], "实验文档/results/summary.json"
        )
        self.assertEqual(artifact["metadata"]["source_location"], "")
        self.assertEqual(
            artifact["metadata"]["sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            artifact["metadata"]["validation_state"], "recorded_not_reexecuted"
        )
        for key in ["date", "branch", "commit", "status"]:
            self.assertIsNone(artifact["metadata"][key])
        for pointer, expected in [
            ("/seed", 0),
            ("/empty", ""),
            ("/null", None),
            ("/empty_array", []),
            ("/a~1b~0c", "中文"),
            ("/model/arch", "price"),
            ("/metadata/training_signature", {"seed": 42}),
            ("/metadata/manifest_hashes", {"a": "hash"}),
        ]:
            self.assertEqual(self.field(graph, path, pointer)["value"], expected)
        self.assertEqual(json.loads(path.read_text()), value)

    def test_dates_preserve_offset_only_explicit_metadata(self) -> None:
        value = {
            "metadata": {
                "completed_at": "2026-10-02T12:30:00+08:00",
                "branch": "experiment",
                "commit": "abc",
                "status": "completed",
            }
        }
        path = self.write("configs/a.json", value)
        metadata = self.field(self.build(path), path, "")["metadata"]
        self.assertEqual(metadata["date"], value["metadata"]["completed_at"])
        for key in ["branch", "commit", "status"]:
            self.assertEqual(metadata[key], value["metadata"][key])
        path.write_text('{"date":"yesterday"}', encoding="utf-8")
        self.assertIsNone(self.field(self.build(path), path, "")["metadata"]["date"])

    def test_false_target_evaluation_limitations_checks_and_risk_retained(self) -> None:
        value = {
            "hypothesis": "possible improvement",
            "limitations": ["not rerun"],
            "target_evaluation_performed": False,
            "primary_macro_ratio": 0.92,
            "checks": {"mapping": False},
            "checked_rows": 123,
            "mapping_recoverable": False,
            "risk_flags": ["unverified"],
        }
        path = self.write("实验文档/results/result.json", value)
        graph = self.build(path)
        for key, expected in value.items():
            self.assertEqual(self.field(graph, path, "/" + key)["value"], expected)
        self.assertFalse(any(e["relation"] == "evaluates_on" for e in graph["edges"]))

    def test_protocol_rules_distinct_from_recorded_conclusions(self) -> None:
        protocol = self.write(
            "实验文档/results/protocol.json", {"decision": {"accept_if": "ratio < 1"}}
        )
        result = self.write("实验文档/results/decision.json", {"decision": "rejected"})
        run = self.write("实验文档/results/run_config.json", {"seed": 7})
        config = self.write("configs/a.json", {})
        graph = self.build(protocol, result, run, config)
        for path, expected in [
            (protocol, "Protocol"),
            (run, "Run"),
            (config, "Config"),
        ]:
            self.assertEqual(self.field(graph, path, "")["entity_type"], expected)
        self.assertEqual(
            self.field(graph, protocol, "/decision")["entity_type"], "DecisionRule"
        )
        conclusion = self.field(graph, result, "/decision")
        self.assertEqual(conclusion["entity_type"], "Conclusion")
        self.assertEqual(
            conclusion["metadata"]["validation_state"], "recorded_not_reexecuted"
        )

    def test_10000_query_rows_omitted_hash_pointer_and_controlled_nodes(self) -> None:
        rows = [
            {"query_id": i, "raw_query": "select private_value"} for i in range(10000)
        ]
        path = self.write(
            "实验文档/results/summary.json",
            {
                "results": rows,
                "runs": rows,
                "reports": rows,
                "metadata": {"queries": rows, "seed": 5},
                "limitations": ["limited"],
            },
        )
        graph = self.build(path)
        omitted = self.field(graph, path, "")["omitted_fields"]
        self.assertEqual(
            {v["source_pointer"] for v in omitted},
            {"/results", "/runs", "/reports", "/metadata/queries"},
        )
        for field in omitted:
            self.assertRegex(field["sha256"], r"^[a-f0-9]{64}$")
        self.assertNotIn("raw_query", json.dumps(graph))
        self.assertLess(len(graph["nodes"]), 12)
        self.assertLess(len(json.dumps(graph)), 20000)

    def test_nested_checks_preserved_while_large_values_are_bounded(self) -> None:
        path = self.write(
            "实验文档/results/summary.json",
            {
                "checks": {
                    "checked_rows": 10,
                    "risk": False,
                    "queries": [{"query": "private"}] * 10000,
                },
                "metadata": {"seed": 7},
                "blob": "x" * 50000,
            },
        )
        graph = self.build(path)
        self.assertEqual(
            self.field(graph, path, "/checks")["value"],
            {"checked_rows": 10, "risk": False},
        )
        self.assertEqual(self.field(graph, path, "/metadata/seed")["value"], 7)
        self.assertEqual(
            {
                v["source_pointer"]
                for v in self.field(graph, path, "")["omitted_fields"]
            },
            {"/checks/queries", "/blob"},
        )

    def test_explicit_references_only_existing_safe_paths(self) -> None:
        target = self.write("configs/model.json", {"arch": "price"})
        source = self.write(
            "实验文档/results/summary.json",
            {
                "model_config": "configs/model.json",
                "data_config": "../../outside.json",
                "config": "configs/missing.json",
                "checkpoint": "configs/model.json",
            },
        )
        graph = self.build(source, target)
        refs = [e for e in graph["edges"] if e["relation"] == "references"]
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["target"], self.adapter.node_id("configs/model.json"))
        self.assertEqual(refs[0]["source_location"], "/model_config")
        self.assertEqual(
            {
                v["source_pointer"]
                for v in self.field(graph, source, "")["unresolved_reference"]
            },
            {"/config", "/data_config"},
        )

    def test_local_reference_endpoint_and_symlink_unresolved(self) -> None:
        target = self.write("实验文档/results/a/config.json", {"seed": 7})
        (target.parent / "link.json").symlink_to(target)
        source = self.write(
            "实验文档/results/a/summary.json",
            {"config": "config.json", "model_config": "link.json"},
        )
        graph = self.build(source)
        identifiers = {n["id"] for n in graph["nodes"]}
        for edge in graph["edges"]:
            self.assertIn(edge["source"], identifiers)
            self.assertIn(edge["target"], identifiers)
        self.assertEqual(
            len([e for e in graph["edges"] if e["relation"] == "references"]), 1
        )
        self.assertEqual(
            self.field(graph, source, "")["unresolved_reference"][0]["value"],
            "link.json",
        )

    def test_experiment_related_edges_inferred_never_produces(self) -> None:
        a = self.write("实验文档/results/a/summary.json", {"seed": 1})
        b = self.write("实验文档/results/a/protocol.json", {"seed": 1})
        graph = self.build(a, b)
        experiments = [n for n in graph["nodes"] if n["entity_type"] == "Experiment"]
        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0]["metadata"]["status"], "unknown")
        related = [e for e in graph["edges"] if e["relation"] == "related_artifact"]
        self.assertEqual(len(related), 2)
        self.assertTrue(
            all(
                e["confidence"] == "INFERRED" and e["confidence_score"] == 0.75
                for e in related
            )
        )
        self.assertFalse(any(e["relation"] == "produces" for e in graph["edges"]))

    def test_edges_provenance_and_weight(self) -> None:
        path = self.write("configs/a.json", {"seed": 1, "model": {"width": 64}})
        graph = self.build(path)
        self.assertTrue(graph["edges"])
        for edge in graph["edges"]:
            for key in [
                "source",
                "target",
                "relation",
                "confidence",
                "confidence_score",
                "source_file",
                "source_location",
                "sha256",
                "weight",
            ]:
                self.assertIn(key, edge)
            self.assertEqual(edge["confidence"], "EXTRACTED")
            self.assertEqual(edge["confidence_score"], 1.0)
            self.assertEqual(edge["weight"], 1.0)
            self.assertEqual(
                edge["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_duplicates_repeated_builds_and_concurrency_deterministic(self) -> None:
        a = self.write("configs/a.json", {"seed": 1})
        b = self.write("configs/b.json", {"seed": 2})
        paths = [b, a, b]
        graph = self.adapter.extract_json(self.root, paths)
        self.assertEqual(paths, [b, a, b])
        self.assertEqual(graph, self.build(a, b))
        self.assertEqual(len({n["id"] for n in graph["nodes"]}), len(graph["nodes"]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.build(a, b), range(3)))
        self.assertTrue(all(result == graph for result in results))

    def test_source_change_clears_old_fields_and_same_stem_not_merged(self) -> None:
        a = self.write(
            "实验文档/results/a/summary.json", {"seed": 1, "old_field": "old"}
        )
        b = self.write("实验文档/results/b/summary.json", {"seed": 2})
        first = self.build(a, b)
        self.assertEqual(self.field(first, a, "/seed")["value"], 1)
        self.assertEqual(self.field(first, b, "/seed")["value"], 2)
        a.write_text('{"seed":3}', encoding="utf-8")
        second = self.build(a, b)
        self.assertNotEqual(first, second)
        self.assertNotIn("old_field", json.dumps(second))
        self.assertEqual(self.field(second, a, "/seed")["value"], 3)

    def test_invalid_json_nonfinite_duplicate_keys_and_nonobject(self) -> None:
        for text in ["{bad}", '{"seed":NaN}', '{"seed":1,"seed":2}', "[]", "null"]:
            path = self.write("configs/a.json", text)
            with (
                self.subTest(text=text),
                self.assertRaisesRegex(ValueError, "JSON|object"),
            ):
                self.build(path)

    def test_size_and_field_limits_fail_clearly(self) -> None:
        path = self.write("configs/a.json", "x" * (self.adapter.MAX_JSON_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "size|limit"):
            self.build(path)
        path.write_text(
            json.dumps({f"field{i}": i for i in range(10000)}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "field|limit"):
            self.build(path)

    def test_unlisted_input_rejected(self) -> None:
        path = self.write("logs/summary.json", {})
        with self.assertRaisesRegex(ValueError, "whitelist"):
            self.build(path)

    def test_structured_limitations_and_decision_rules_retained(self) -> None:
        value = {
            "limitations": [{"scope": "source only", "risk": True}],
            "decision_rules": {"accept_if": "ratio < 1"},
        }
        path = self.write("实验文档/results/summary.json", value)
        graph = self.build(path)
        self.assertEqual(
            self.field(graph, path, "/limitations")["value"], value["limitations"]
        )
        self.assertEqual(
            self.field(graph, path, "/decision_rules")["entity_type"], "DecisionRule"
        )

    def test_exponent_overflow_fails_with_json_context(self) -> None:
        path = self.write("configs/a.json", '{"ratio":1e999}')
        with self.assertRaisesRegex(ValueError, "JSON"):
            self.build(path)

    def test_cli_integration(self) -> None:
        self.write("configs/a.json", {"seed": 1})
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        args = [sys.executable, str(MODULE), "--root", str(self.root)]
        result = subprocess.run(args, capture_output=True, text=True, env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)
        graph = json.loads(result.stdout)
        self.assertEqual(
            graph,
            self.adapter.extract_json(
                self.root, self.adapter.select_files(self.root)["json"]
            ),
        )
        self.assertGreater(graph["input_tokens"], 0)
        self.assertGreater(graph["output_tokens"], 0)
        self.write("configs/a.json", "{bad}")
        result = subprocess.run(args, capture_output=True, text=True, env=environment)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("JSON", result.stderr)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
