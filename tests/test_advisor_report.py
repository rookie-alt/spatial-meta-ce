import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "advisor_report", ROOT / "scripts/build_advisor_report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdvisorReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.html"
            subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/build_advisor_report.py"),
                    "--repo-root",
                    str(ROOT),
                    "--git-branch",
                    "main",
                    "--lit-branch",
                    "main",
                    "--out",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            cls.report = output.read_text(encoding="utf-8")

    def test_paper_legend_matches_our_win_coloring(self) -> None:
        self.assertIn(
            "绿色表示我们在该库上优于该方法，红色表示该方法优于我们。",
            self.report,
        )
        self.assertNotIn("绿色表示该方法在该库上优于我们", self.report)

    def test_paper_table_colors_agree_with_the_legend(self) -> None:
        """Green means we are better, and lower Q95 is better.

        So a `win` cell must hold a value LARGER than our row in the same
        column. This locks the legend to the colouring logic.
        """
        table = next(
            t
            for t in re.findall(r"<table\b.*?</table>", self.report, re.S)
            if "同一张表（all-rows" in t
        )
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table, re.S)
        cells = []
        for row in rows:
            parsed = re.findall(r"<td(?: class=\"([^\"]*)\")?>(.*?)</td>", row, re.S)
            if parsed:
                cells.append(parsed)
        ours_row = next(row for row in cells if "本次实测" in row[0][1])
        ours = [float(re.sub(r"<[^>]+>", "", v)) for _, v in ours_row[1:]]
        checked = 0
        for row in cells:
            if "本次实测" in row[0][1]:
                continue
            for i, (cls, value) in enumerate(row[1:]):
                value = re.sub(r"<[^>]+>", "", value).strip()
                if value == "—":
                    continue
                if cls == "win":
                    self.assertGreater(float(value), ours[i])
                elif cls == "lose":
                    self.assertLess(float(value), ours[i])
                checked += 1
        self.assertGreater(checked, 20)

    def test_stats_loss_is_not_described_as_a_tie(self) -> None:
        self.assertIn("stats 的 PG/我们为 0.904", self.report)
        self.assertIn("不能称为打平", self.report)
        self.assertNotIn("属于打平而非胜负", self.report)

    def test_three_seed_note_is_scoped_and_traceable(self) -> None:
        self.assertIn("三 seed 复核（2026-10-23）", self.report)
        self.assertIn("12/12 个 seed×库组合均落后 PRICE", self.report)
        self.assertIn("不替换前面的单 seed 表格", self.report)
        self.assertIn("不外推到源库或 K&gt;0", self.report)
        self.assertIn("单一 PRICE checkpoint", self.report)
        self.assertRegex(self.report, r"sha256 <code>[0-9a-f]{64}</code>")

    def test_three_seed_note_rejects_inconsistent_evidence(self) -> None:
        module = _load_module()
        original = (ROOT / module.THREE_SEED_PATH).read_text(encoding="utf-8")
        for defect in ("headline", "nonfinite", "seeds"):
            with self.subTest(defect=defect):
                summary = json.loads(original)
                if defect == "headline":
                    summary["headline"]["seed_database_pairs_beating_pg"] = 8
                elif defect == "nonfinite":
                    summary["databases"]["stats"]["ours_k0"]["per_seed_q95"]["42"] = (
                        float("nan")
                    )
                else:
                    summary["seeds"] = [42]
                with self.assertRaises(ValueError):
                    module.build_three_seed_note(summary, module.THREE_SEED_SHA256)

    def test_existing_single_seed_tables_are_preserved(self) -> None:
        self.assertIn("single seed · 冻结 checkpoint · 全量 population", self.report)
        self.assertTrue(re.search(r"83\.72.*39\.52.*75\.65", self.report))
        self.assertNotRegex(self.report, r"\{\{[A-Z0-9_]+\}\}")


if __name__ == "__main__":
    unittest.main()
