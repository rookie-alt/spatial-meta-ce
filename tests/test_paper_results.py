"""Paper export regression tests: recorded evidence only, no model dependencies."""

import copy
import csv
import importlib.util
import json
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "paper_results", ROOT / "scripts/build_paper_results.py"
)
paper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper)


class PaperResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = paper.load_evidence(ROOT)

    def test_frozen_36_sources_checked(self):
        self.assertEqual(len(self.data["original_36_sha256"]), 36)
        self.assertEqual(len(self.data["targets"]), 108)
        self.assertEqual(len(self.data["initializations"]), 252)

    def test_all_k_and_uncertainty_are_separate(self):
        for database in paper.DATABASES:
            for method in paper.METHODS:
                rows = [
                    r
                    for r in self.data["targets"]
                    if r["database"] == database and r["method"] == method
                ]
                self.assertEqual([r["k"] for r in rows], list(paper.SHOTS))
                for row in rows:
                    if method == "PRICE":
                        self.assertIsNone(row["q95_training_seed_sd"])
                        self.assertIsNotNone(row["q95_support_sd"])
                        self.assertEqual(
                            row["uncertainty"], "support_sd_single_checkpoint"
                        )
                    elif method == "PG":
                        self.assertIsNone(row["q95_training_seed_sd"])
                        self.assertIsNone(row["q95_support_sd"])
                        self.assertEqual(row["uncertainty"], "none_fixed_pg")
                    else:
                        self.assertIsNotNone(row["q95_training_seed_sd"])
                        self.assertIsNone(row["q95_support_sd"])

    def test_target_values_match_frozen_table(self):
        row = next(
            r
            for r in self.data["targets"]
            if (r["method"], r["database"], r["k"]) == ("FOMAML-sum", "genome", 20)
        )
        self.assertAlmostEqual(row["q95_mean"], 129.76132833982388, places=5)

    def test_source_macro_is_not_pooled_and_has_no_sd(self):
        rows = self.data["source_tradeoffs"]
        for study in ("S4", "S5", "S6"):
            group = [r for r in rows if r["study"] == study]
            macro = next(r for r in group if r["database"] == "macro")
            dbs = [r for r in group if r["database"] != "macro"]
            self.assertAlmostEqual(macro["b_q95"], sum(r["b_q95"] for r in dbs) / 3)
            self.assertEqual(macro["decision"], "no_candidate")
            self.assertEqual(macro["training_seed"], 42)
            self.assertEqual(macro["aggregation"], "equal_database_arithmetic_macro")
            self.assertNotIn("sd", macro)
        s6 = next(r for r in rows if r["study"] == "S6" and r["database"] == "macro")
        self.assertAlmostEqual(s6["b_q95"], 39.98662687913146)
        self.assertAlmostEqual(s6["relative_change"], -0.09418411090532353)

    def test_output_ancestor_symlink_cannot_overwrite_historical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            historical = root / "historical" / "paper"
            historical.mkdir(parents=True)
            sentinel = historical / "target-q95.csv"
            sentinel.write_text("HISTORICAL_SENTINEL")
            alias = root / "alias"
            alias.symlink_to(historical.parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink|alias"):
                paper.export(self.data, alias / "paper")
            self.assertEqual(sentinel.read_text(), "HISTORICAL_SENTINEL")
            with self.assertRaisesRegex(ValueError, "symlink|alias"):
                paper.export(self.data, alias / "new" / "paper")
            self.assertFalse((historical.parent / "new").exists())

    def test_deterministic_export_and_valid_svg(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp).resolve() / "result"
            paper.export(self.data, out)
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            paper.export(self.data, out)
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})
            self.assertEqual(len(list(out.glob("*.svg"))), 4)
            for p in out.glob("*.svg"):
                svg = ET.fromstring(p.read_text())
                self.assertEqual(svg.tag, "{http://www.w3.org/2000/svg}svg")
                self.assertIn("desc", p.read_text())
                self.assertNotIn("nan", p.read_text().lower())
            with (out / "target-q95.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 108)
            manuscript = paper.render_chapter(self.data)
            for db in paper.DATABASES:
                self.assertIn(f"### {db}", manuscript)
            self.assertIn("IMDB", manuscript)
            self.assertIn("P2", manuscript)
            self.assertIn("support SD", manuscript)

    def test_reject_duplicate_and_incomplete_target_episodes(self):
        raw = json.loads((ROOT / paper.TARGET_FILES["FOMAML-sum"][0][1]).read_text())
        for mutate in (
            lambda x: x["results"].append(copy.deepcopy(x["results"][0])),
            lambda x: x["results"].pop(),
        ):
            damaged = copy.deepcopy(raw)
            mutate(damaged)
            with self.assertRaises(ValueError):
                paper.validate_target_summary(damaged, "FOMAML-sum", 42)

    def test_reject_invalid_metrics_and_support_identity(self):
        raw = json.loads((ROOT / paper.TARGET_FILES["FOMAML-sum"][0][1]).read_text())
        for value in (math.nan, math.inf, -1.0, None, True):
            damaged = copy.deepcopy(raw)
            damaged["results"][0]["unique_query_metrics"]["q95"] = value
            with self.assertRaises(ValueError):
                paper.validate_target_summary(damaged, "FOMAML-sum", 42)
        damaged = copy.deepcopy(raw)
        row = next(r for r in damaged["results"] if r["k"] == 5)
        row["support_ids"][1] = row["support_ids"][0]
        with self.assertRaises(ValueError):
            paper.validate_target_summary(damaged, "FOMAML-sum", 42)

    def test_reject_duplicate_json_keys(self):
        with self.assertRaises(ValueError):
            paper.parse_json('{"q95":1,"q95":2}')

    def test_reject_missing_or_tampered_hash_input_and_no_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for relative in self.data["input_sha256"]:
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT / relative).read_bytes())
            key = next(iter(self.data["original_36_sha256"]))
            (root / key).write_bytes((root / key).read_bytes() + b"\n")
            out = root / "paper-output"
            with self.assertRaises(ValueError):
                paper.build(root, out)
            self.assertFalse(out.exists())
            (root / key).unlink()
            with self.assertRaises(ValueError):
                paper.build(root, out)
            self.assertFalse(out.exists())

    def test_live_status_and_agents_are_not_authenticated_inputs(self):
        self.assertNotIn("AGENTS.md", paper.INPUT_SHA256)
        for number in ("00", "23", "99"):
            self.assertFalse(
                any(
                    Path(p).name.startswith(number + "-")
                    for p in self.data["input_sha256"]
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for relative in self.data["input_sha256"]:
                dest = root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((ROOT / relative).read_bytes())
            # Deliberately absent AGENTS and arbitrary live status: never read/hash them.
            (root / "实验文档/23-当前任务清单与源侧改进总结.md").write_text(
                "live update"
            )
            rebuilt = paper.load_evidence(root)
            self.assertEqual(rebuilt["targets"], self.data["targets"])

    def test_reject_missing_target_metric_and_mismatched_aggregate(self):
        raw = json.loads((ROOT / paper.TARGET_FILES["FOMAML-sum"][0][1]).read_text())
        damaged = copy.deepcopy(raw)
        damaged["results"][0]["unique_query_metrics"].pop("q95")
        with self.assertRaises((ValueError, KeyError)):
            paper.validate_target_summary(damaged, "FOMAML-sum", 42)
        damaged = copy.deepcopy(raw)
        damaged["aggregate"][0]["across_support_runs"]["unique_query_q95"]["mean"] += 1
        with self.assertRaises(ValueError):
            paper.validate_target_summary(damaged, "FOMAML-sum", 42)

    def test_target_figures_keep_all_points_and_different_sd_labels(self):
        for db in paper.DATABASES:
            svg = paper.target_svg(self.data, db)
            self.assertEqual(svg.count("<circle"), 24)
            self.assertEqual(svg.count("<polyline"), 4)
            self.assertIn("PRICE support SD", svg)
            self.assertIn("training seed SD", svg)
            self.assertIn("no SD", svg)
        svg = paper.source_svg(self.data)
        self.assertEqual(svg.count("<circle"), 12)
        self.assertIn("no SD", svg)

    def test_reject_csv_duplicate_and_missing_row(self):
        rows = [{"method": "X", "k": "0"}, {"method": "X", "k": "0"}]
        with self.assertRaises(ValueError):
            paper.index_rows(rows, ("method", "k"), {("X", "0")})
        with self.assertRaises(ValueError):
            paper.index_rows([], ("method", "k"), {("X", "0")})

    def test_reject_source_duplicate_incomplete_and_nonfinite(self):
        raw = json.loads(
            (ROOT / paper.RESULTS / paper.STUDIES["S6"] / "summary.json").read_text()
        )
        for mutate in (
            lambda x: x["runs"].append(copy.deepcopy(x["runs"][0])),
            lambda x: x["runs"].pop(),
            lambda x: x["runs"][0]["endpoint"]["per_database"].pop("telstra"),
            lambda x: x["runs"][0]["endpoint"]["per_database"]["legalacts"][
                "four_plus_tables"
            ].__setitem__("q95", math.inf),
        ):
            damaged = copy.deepcopy(raw)
            mutate(damaged)
            with self.assertRaises(ValueError):
                paper.validate_source_summary(damaged, "S6")


if __name__ == "__main__":
    unittest.main()
