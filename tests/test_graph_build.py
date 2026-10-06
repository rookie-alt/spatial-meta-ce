import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_graph import (
    atomic_json,
    merge_fragments,
    project_relations,
    validate_semantic,
)


class GraphBuildTests(unittest.TestCase):
    def test_preserves_multiple_relations_and_deduplicates_identical_evidence(self):
        nodes = [{"id": n, "label": n, "source_file": "a.md"} for n in ("a", "b")]
        first = {
            "source": "a",
            "target": "b",
            "relation": "supports",
            "source_file": "a.md",
            "source_location": "4",
            "sha256": "abc",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
        }
        fragment = {
            "nodes": nodes,
            "edges": [first, dict(first, relation="references"), first],
        }
        original = copy.deepcopy(fragment)
        result = merge_fragments([fragment])
        self.assertEqual(fragment, original)
        self.assertEqual(len(result["edges"]), 2)
        self.assertEqual(len({e["id"] for e in result["edges"]}), 2)
        graph = project_relations(result)
        self.assertTrue(graph["directed"])
        self.assertEqual(len(graph["edges"]), 4)
        self.assertEqual(len(graph["nodes"]), 4)

    def test_rejects_dangling_edge_and_conflicting_node_id(self):
        with self.assertRaises(ValueError):
            merge_fragments([{"nodes": [], "edges": [{"source": "x", "target": "y"}]}])
        with self.assertRaises(ValueError):
            merge_fragments(
                [
                    {
                        "nodes": [
                            {"id": "x", "source_file": "a.md"},
                            {"id": "x", "source_file": "b.md"},
                        ],
                        "edges": [],
                    }
                ]
            )

    def test_atomic_json_replaces_instead_of_growing_old_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "graph.json"
            atomic_json(path, {"nodes": ["old", "new"], "edges": []})
            atomic_json(path, {"nodes": ["new"], "edges": []})
            self.assertEqual(json.loads(path.read_text())["nodes"], ["new"])
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_semantic_sources_must_match_current_content_and_cover_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "a.md"
            path.write_text("unverified\n", encoding="utf-8")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            fragment = {
                "source_hashes": {"a.md": sha},
                "nodes": [
                    {
                        "id": "a",
                        "source_file": "a.md",
                        "source_location": "1",
                        "sha256": sha,
                        "status": "unverified",
                    }
                ],
                "edges": [],
            }
            validate_semantic(root, fragment, [path])
            self.assertEqual(fragment["nodes"][0]["status"], "unverified")
            path.write_text("completed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale"):
                validate_semantic(root, fragment, [path])
            with self.assertRaises(ValueError):
                validate_semantic(
                    root, {"source_hashes": {}, "nodes": [], "edges": []}, [path]
                )

    def test_rejects_unsafe_semantic_paths_and_invalid_confidence(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            validate_semantic(Path(folder), {"source_hashes": {"../x.md": "x"}}, [])
        nodes = [{"id": n, "source_file": "a.md"} for n in ("a", "b")]
        with self.assertRaises(ValueError):
            merge_fragments(
                [
                    {
                        "nodes": nodes,
                        "edges": [
                            {
                                "source": "a",
                                "target": "b",
                                "relation": "supports",
                                "confidence": "EXTRACTED",
                                "confidence_score": 0.5,
                                "source_file": "a.md",
                                "source_location": "1",
                            }
                        ],
                    }
                ]
            )

    def test_html_evidence_is_script_safe_and_has_source_metadata(self):
        from scripts.build_experiment_graph import enhance_html

        template = (
            "<html><head></head><body><script>"
            'return `<span class="neighbor-link" onclick="focusNode(${JSON.stringify(nid)})">'
            "${esc(nb ? nb.label : nid)}</span>`;"
            "</script></body></html>"
        )
        content = enhance_html(
            template,
            {
                "nodes": [
                    {
                        "id": "a",
                        "label": "</script><script>bad()</script>",
                        "source_file": "a.md",
                        "source_location": "L4",
                        "sha256": "abc",
                        "status": "unverified",
                    }
                ],
                "edges": [],
            },
        )
        self.assertNotIn("</script><script>bad()", content)
        self.assertNotIn('onclick="focusNode(', content)
        self.assertIn("data-node-id=", content)
        self.assertIn("addEventListener('click'", content)
        self.assertIn(
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            content,
        )
        self.assertEqual(content.count('name="viewport"'), 1)
        self.assertIn("unverified", content)
        self.assertIn("sha256", content)
        self.assertIn("textContent", content)
        with self.assertRaisesRegex(ValueError, "neighbor-link template changed"):
            enhance_html(
                "<html><head></head><body></body></html>", {"nodes": [], "edges": []}
            )

    def test_ast_ids_are_portable_between_roots(self):
        from scripts.build_experiment_graph import ast_fragment

        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            results = []
            for name in ("first", "second"):
                root = parent / name
                root.mkdir()
                code = root / "a.py"
                code.write_text("def example():\n    return 1\n")
                results.append(ast_fragment(root, [code], root / "graphify-out"))
            self.assertEqual(results[0]["nodes"], results[1]["nodes"])
            self.assertEqual(results[0]["edges"], results[1]["edges"])

    def test_json_build_export_and_rebuild_clears_old_fields(self):
        from scripts.build_experiment_graph import build

        workspace = Path(__file__).resolve().parents[1]
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as inputs,
            tempfile.TemporaryDirectory(dir=workspace) as outputs,
            patch(
                "scripts.build_experiment_graph.git_snapshot",
                return_value={"branch": "fixture", "head": "fixture", "dirty": True},
            ),
        ):
            root, out = Path(inputs), Path(outputs)
            (root / "configs").mkdir()
            source = root / "configs/model.json"
            source.write_text(
                json.dumps(
                    {
                        "model": {"arch": "sum"},
                        "target_evaluation_performed": False,
                        "old": 1,
                    }
                )
            )
            first = build(root, out, None)
            self.assertTrue((out / "graph.html").is_file())
            self.assertTrue(json.loads((out / "graph.json").read_text())["directed"])
            self.assertIn(
                "target_evaluation_performed", {n["label"] for n in first["nodes"]}
            )
            source.write_text(json.dumps({"seed": 42}))
            second = build(root, out, None)
            self.assertNotIn("old", {n["label"] for n in second["nodes"]})
            self.assertEqual(second, build(root, out, None))
            self.assertTrue(all("id" in e for e in second["edges"]))
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(
                manifest["sources"]["configs/model.json"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )

    def test_rejects_unsafe_ids_external_claims_and_output(self):
        from scripts.build_experiment_graph import artifact_id, build

        with self.assertRaisesRegex(ValueError, "ASCII"):
            merge_fragments(
                [
                    {
                        "nodes": [{"id": 'x\\" onload=bad', "source_file": "a"}],
                        "edges": [],
                    }
                ]
            )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "worktree"):
                build(root, root / "output", None)
            with self.assertRaises(ValueError):
                validate_semantic(
                    root,
                    {
                        "source_hashes": {},
                        "nodes": [
                            {
                                "id": artifact_id("configs/a.json"),
                                "source_file": "configs/a.json",
                                "entity_type": "Artifact",
                                "external_reference": True,
                                "sha256": None,
                                "status": "completed",
                            }
                        ],
                        "edges": [],
                    },
                    [],
                )

    def test_duplicate_non_placeholder_nodes_are_rejected(self):
        first = {
            "nodes": [{"id": "same", "source_file": "a.md", "status": "planned"}],
            "edges": [],
        }
        second = {
            "nodes": [{"id": "same", "source_file": "a.md", "status": "completed"}],
            "edges": [],
        }
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            merge_fragments([first, second])

    def test_file_limit_allows_206_and_rejects_251_before_output(self):
        from unittest.mock import patch

        from scripts.build_experiment_graph import build

        workspace = Path(__file__).resolve().parents[1]
        with (
            tempfile.TemporaryDirectory() as inputs,
            tempfile.TemporaryDirectory(dir=workspace) as outputs,
            patch("scripts.build_experiment_graph.git_snapshot", return_value={}),
        ):
            root, parent = Path(inputs), Path(outputs)
            configs = root / "configs"
            configs.mkdir()
            for index in range(206):
                (configs / f"c{index}.json").write_text('{"seed":42}')
            out = parent / "accepted"
            build(root, out, None)
            self.assertEqual(
                len(json.loads((out / "manifest.json").read_text())["sources"]), 206
            )
            for index in range(206, 251):
                (configs / f"c{index}.json").write_text('{"seed":42}')
            rejected = parent / "rejected"
            with self.assertRaisesRegex(ValueError, "Corpus exceeds limit: 251 files"):
                build(root, rejected, None)
            self.assertFalse(rejected.exists())

    def test_graph_is_deterministic_and_inputs_are_immutable(self):
        fragment = {
            "nodes": [{"id": "b", "source_file": "a"}, {"id": "a", "source_file": "a"}],
            "edges": [],
        }
        self.assertEqual(merge_fragments([fragment]), merge_fragments([fragment]))
        self.assertEqual([n["id"] for n in fragment["nodes"]], ["b", "a"])


if __name__ == "__main__":
    unittest.main()
