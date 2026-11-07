"""Tests for `scripts/g2_oracle.py` (protocol 116 §8)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g2_oracle as g2  # noqa: E402

SQL_TYPES = {"int": "integer", "str": "varchar(45)", "float": "decimal(10,2)"}


def write_dataset(root: Path, db: str, tables: dict) -> Path:
    directory = root / "datasets" / db
    directory.mkdir(parents=True, exist_ok=True)
    ddl, copies = [], []
    for table, (columns, rows) in tables.items():
        ddl.append(f"drop table if exists {table};")
        ddl.append(f"create table {table} (")
        for index, (name, kind) in enumerate(columns):
            tail = "," if index < len(columns) - 1 else ""
            ddl.append(f"  {name} {SQL_TYPES[kind]} {tail}")
        ddl.append(");")
        copies.append(
            f"\\copy {table} from './{table}.csv' csv delimiter '|' null 'NULL' header;"
        )
        with (directory / f"{table}.csv").open("w") as handle:
            handle.write("|".join(name for name, _ in columns) + "\n")
            for row in rows:
                handle.write(
                    "|".join("NULL" if value is None else str(value) for value in row)
                    + "\n"
                )
    (directory / f"postgres_create_{db}.sql").write_text("\n".join(ddl + copies) + "\n")
    return directory


def write_workload(
    root: Path, db: str, queries: list[tuple[str, float, float]]
) -> Path:
    directory = root / "workloads" / db
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "workloads.sql"
    path.write_text("".join(f"{sql}||{true}||{pg}||\n" for sql, true, pg in queries))
    return path


def frames(left, middle, right):
    return {
        "l": pd.DataFrame(left),
        "m": pd.DataFrame(middle),
        "r": pd.DataFrame(right),
    }


SPEC = g2.Spec3("l", "l", "a", "m", "m", "a", "b", "r", "r", "b", ())


class AnalyzeTests(unittest.TestCase):
    def test_accepts_three_table_path(self):
        spec = g2.analyze3(
            "SELECT COUNT(*) FROM t1 a, t2 b, t3 c WHERE (a.k = b.x) AND (b.y = c.k)"
        )
        self.assertEqual(spec.middle_table, "t2")
        self.assertEqual(spec.left_table, "t1")
        self.assertEqual(spec.right_table, "t3")
        self.assertEqual((spec.mid_key_left, spec.left_key), ("x", "k"))
        self.assertEqual((spec.mid_key_right, spec.right_key), ("y", "k"))

    def test_orients_left_right_by_table_name(self):
        spec = g2.analyze3(
            "SELECT COUNT(*) FROM zz a, mm b, aa c WHERE (a.k = b.x) AND (b.y = c.k)"
        )
        self.assertEqual(spec.left_table, "aa")
        self.assertEqual(spec.right_table, "zz")

    def test_rejects_two_tables(self):
        with self.assertRaises(g2.Unsupported) as caught:
            g2.analyze3("SELECT COUNT(*) FROM t1 a, t2 b WHERE (a.k = b.k)")
        self.assertEqual(str(caught.exception), "table_count_2")

    def test_rejects_triangle(self):
        with self.assertRaises(g2.Unsupported) as caught:
            g2.analyze3(
                "SELECT COUNT(*) FROM t1 a, t2 b, t3 c "
                "WHERE (a.k = b.k) AND (b.k = c.k) AND (a.k = c.k)"
            )
        self.assertEqual(str(caught.exception), "join_predicate_count_3")

    def test_rejects_disconnected(self):
        with self.assertRaises(g2.Unsupported) as caught:
            g2.analyze3(
                "SELECT COUNT(*) FROM t1 a, t2 b, t3 c "
                "WHERE (a.k = b.k) AND (a.j = b.j)"
            )
        self.assertEqual(str(caught.exception), "not_a_path")

    def test_rejects_same_middle_key(self):
        with self.assertRaises(g2.Unsupported) as caught:
            g2.analyze3(
                "SELECT COUNT(*) FROM t1 a, t2 b, t3 c "
                "WHERE (a.k = b.id) AND (b.id = c.k)"
            )
        self.assertEqual(str(caught.exception), "same_middle_key")

    def test_rejects_left_join(self):
        with self.assertRaises(g2.Unsupported) as caught:
            g2.analyze3(
                "SELECT COUNT(*) FROM t1 a LEFT JOIN t2 b ON a.k = b.x "
                "LEFT JOIN t3 c ON b.y = c.k"
            )
        self.assertEqual(str(caught.exception), "outer_join")

    def test_records_join_key_filter(self):
        spec = g2.analyze3(
            "SELECT COUNT(*) FROM t1 a, t2 b, t3 c "
            "WHERE (a.k = b.x) AND (b.y = c.k) AND (b.x = 3)"
        )
        self.assertTrue(spec.filters_join_key())


class EstimatorTests(unittest.TestCase):
    def test_u_joint_equals_exact_without_filters(self):
        tables = frames(
            {"a": ["A"] * 10 + ["B"]},
            {"a": ["A", "B"], "b": ["X", "Y"]},
            {"b": ["X"] * 10 + ["Y"]},
        )
        scored = g2.score_query3(SPEC, tables, truth=101.0)
        self.assertEqual(scored["exact"], 101)
        self.assertAlmostEqual(scored["u_joint"], 101.0)

    def test_u_indep_equals_u_joint_on_independent_keys(self):
        tables = frames(
            {"a": ["A"] * 10 + ["B"]},
            {"a": ["A", "A", "B", "B"], "b": ["X", "Y", "X", "Y"]},
            {"b": ["X"] * 10 + ["Y"]},
        )
        scored = g2.score_query3(SPEC, tables, truth=121.0)
        self.assertAlmostEqual(scored["u_indep"], scored["u_joint"])
        self.assertAlmostEqual(scored["f_indep"], scored["f_joint"])

    def test_f_indep_differs_from_f_joint_when_correlated(self):
        tables = frames(
            {"a": ["A"] * 10 + ["B"]},
            {"a": ["A", "B"], "b": ["X", "Y"]},
            {"b": ["X"] * 10 + ["Y"]},
        )
        scored = g2.score_query3(SPEC, tables, truth=101.0)
        self.assertEqual(scored["f_joint"], 101.0)
        self.assertAlmostEqual(scored["f_indep"], 60.5)

    def test_null_population_counterexample(self):
        # Protocol 116 §5: with no filter, NULL keys must not shrink the
        # population and manufacture a phantom conditioning effect.
        tables = frames(
            {"a": ["A", None]},
            {"a": ["A", None], "b": ["X", None]},
            {"b": ["X", None]},
        )
        scored = g2.score_query3(SPEC, tables, truth=1.0)
        self.assertEqual((scored["M1"], scored["M2"], scored["M3"]), (1, 1, 1))
        for name in ("u_indep", "u_joint", "f_indep", "f_joint"):
            self.assertAlmostEqual(float(scored[name]), 1.0, msg=name)
        self.assertEqual(scored["exact"], 1)

    def test_zero_population_branch(self):
        tables = frames(
            {"a": [None]},
            {"a": ["A"], "b": ["X"]},
            {"b": ["X"]},
        )
        scored = g2.score_query3(SPEC, tables, truth=0.0)
        self.assertTrue(scored["zero_population"])
        self.assertEqual(scored["M1"], 0)
        for name in ("u_indep", "u_joint", "f_indep", "f_joint"):
            self.assertEqual(float(scored[name]), 0.0)
        self.assertNotEqual(scored["sel_left"], scored["sel_left"])  # NaN

    def test_empty_filtered_side_sets_all_to_zero(self):
        tables = frames(
            {"a": ["A", "B"], "x": [1, 2]},
            {"a": ["A", "B"], "b": ["X", "Y"]},
            {"b": ["X", "Y"]},
        )
        spec = g2.Spec3(
            "l",
            "l",
            "a",
            "m",
            "m",
            "a",
            "b",
            "r",
            "r",
            "b",
            (g2.Predicate("l", "x", "=", "99", False),),
        )
        scored = g2.score_query3(spec, tables, truth=0.0)
        self.assertEqual(scored["m1"], 0)
        self.assertFalse(scored["zero_population"])
        for name in ("u_indep", "u_joint", "f_indep", "f_joint"):
            self.assertEqual(float(scored[name]), 0.0)

    def test_independent_recheck_agrees(self):
        tables = frames(
            {"a": ["A"] * 10 + ["B"]},
            {"a": ["A", "B"], "b": ["X", "Y"]},
            {"b": ["X"] * 10 + ["Y"]},
        )
        scored = g2.score_query3(SPEC, tables, truth=101.0)
        g2.independent_recheck3(SPEC, tables, scored)


class CounterexampleTests(unittest.TestCase):
    def test_positive_control(self):
        result = g2.counterexample_correlation()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["u_indep"][0], result["u_indep"][1])
        self.assertNotEqual(result["f_joint"][0], result["f_joint"][1])

    def test_negative_control(self):
        result = g2.counterexample_constant_marginal()
        self.assertTrue(result["passed"], result)


class PlanRunTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        write_dataset(
            root,
            "tiny",
            {
                "t1": ([("k", "int"), ("f", "int")], [[1, 1], [2, 2]]),
                "t2": ([("x", "int"), ("y", "int")], [[1, 10], [2, 20]]),
                "t3": ([("k", "int")], [[10], [20]]),
            },
        )
        write_workload(
            root,
            "tiny",
            [
                (
                    "SELECT COUNT(*) FROM t1 a, t2 b, t3 c "
                    "WHERE (a.k = b.x) AND (b.y = c.k)",
                    2.0,
                    2.0,
                ),
                (
                    "SELECT COUNT(*) FROM t1 a, t2 b WHERE (a.k = b.x)",
                    1.0,
                    1.0,
                ),
            ],
        )

    def test_plan_accepts_only_three_table_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._fixture(root)
            out = root / "out"
            out.mkdir()
            g2.main(
                [
                    "plan",
                    "--workload-dir",
                    str(root / "workloads"),
                    "--databases",
                    "tiny",
                    "--out",
                    str(out),
                ]
            )
            plan = json.loads((out / "plan.json").read_text())
            block = plan["databases"]["tiny"]
            self.assertEqual(block["eligible"], 1)
            self.assertEqual(block["table_count_histogram"].get("2"), 1)
            self.assertTrue((out / "plan.sha256").exists())

    def test_run_scores_and_verifies(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._fixture(root)
            out = root / "out"
            out.mkdir()
            g2.main(
                [
                    "plan",
                    "--workload-dir",
                    str(root / "workloads"),
                    "--databases",
                    "tiny",
                    "--out",
                    str(out),
                ]
            )
            g2.main(
                [
                    "run",
                    "--plan",
                    str(out / "plan.json"),
                    "--dataset-dir",
                    str(root / "datasets"),
                    "--out",
                    str(out),
                ]
            )
            verdict = json.loads((out / "go_no_go.json").read_text())
            self.assertTrue(verdict["gates"]["1_correctness"]["passed"])
            self.assertIn("forbidden_inferences", verdict)

    def test_plan_rejects_target_database(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "out"
            out.mkdir()
            with self.assertRaises(ValueError):
                g2.build_plan(Path(raw), ["imdb"], 200, out)


if __name__ == "__main__":
    unittest.main()
