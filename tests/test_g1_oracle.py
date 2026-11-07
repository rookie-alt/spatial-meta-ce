"""Tests for `scripts/g1_oracle.py` (protocol §9.2 synthetic fixtures)."""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import g1_oracle as g1  # noqa: E402

SQL_TYPES = {"int": "integer", "str": "varchar(45)", "float": "decimal(10,2)"}


def write_dataset(root: Path, db: str, tables: dict) -> Path:
    """Build a synthetic `datasets/<db>/` with DDL, CSVs and a `\\copy` map."""
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


class AnalyzeTests(unittest.TestCase):
    def test_accepts_two_table_single_join(self):
        spec = g1.analyze(
            "SELECT COUNT(*) FROM r a, s b "
            "WHERE (a.k = b.k) AND (a.x = '1') AND (b.y >= 3)"
        )
        self.assertEqual((spec.left_table, spec.left_key), ("r", "k"))
        self.assertEqual((spec.right_table, spec.right_key), ("s", "k"))
        self.assertEqual(len(spec.filters), 2)
        self.assertFalse(spec.filters_join_key())

    def test_rejects_three_tables(self):
        with self.assertRaises(g1.Unsupported) as caught:
            g1.analyze(
                "SELECT COUNT(*) FROM r a, s b, t c WHERE (a.k = b.k) AND (b.k = c.k)"
            )
        self.assertEqual(str(caught.exception), "table_count_3")

    def test_rejects_or(self):
        with self.assertRaises(g1.Unsupported):
            g1.analyze(
                "SELECT COUNT(*) FROM r a, s b "
                "WHERE (a.k = b.k) AND ((a.x = 1) OR (a.y = 2))"
            )

    def test_rejects_like(self):
        with self.assertRaises(g1.Unsupported):
            g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x LIKE 'foo%')"
            )

    def test_rejects_self_join(self):
        with self.assertRaises(g1.Unsupported) as caught:
            g1.analyze("SELECT COUNT(*) FROM r a, r b WHERE (a.k = b.k)")
        self.assertEqual(str(caught.exception), "self_join")

    def test_rejects_two_join_predicates(self):
        with self.assertRaises(g1.Unsupported) as caught:
            g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.j = b.j)"
            )
        self.assertEqual(str(caught.exception), "join_predicate_count_2")

    def test_records_filter_on_join_key(self):
        spec = g1.analyze(
            "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.k > 5)"
        )
        self.assertTrue(spec.filters_join_key())

    def test_flips_operator_for_reversed_operands(self):
        spec = g1.analyze(
            "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (5 < a.k)"
        )
        self.assertEqual(spec.filters[0].op, ">")

    def test_handles_negative_literal(self):
        spec = g1.analyze(
            "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.k > -5)"
        )
        self.assertEqual(spec.filters[0].raw, "-5")

    def test_rejects_explicit_left_join(self):
        with self.assertRaises(g1.Unsupported) as caught:
            g1.analyze("SELECT COUNT(*) FROM r a LEFT JOIN s b ON a.k = b.k")
        self.assertEqual(str(caught.exception), "outer_join")

    def test_accepts_explicit_inner_join(self):
        spec = g1.analyze("SELECT COUNT(*) FROM r a JOIN s b ON a.k = b.k")
        self.assertEqual(spec.left_key, "k")

    def test_cross_join_without_on_does_not_crash(self):
        # A CROSS JOIN carries no `on`; it must behave like a comma join.
        spec = g1.analyze("SELECT COUNT(*) FROM r a CROSS JOIN s b WHERE (a.k = b.k)")
        self.assertEqual(spec.left_key, "k")

    def test_on_clause_range_predicate_becomes_a_filter(self):
        # Previously the ON path kept only exp.EQ nodes, so `a.x > 5` vanished.
        spec = g1.analyze("SELECT COUNT(*) FROM r a JOIN s b ON a.k = b.k AND a.x > 5")
        self.assertEqual(len(spec.filters), 1)
        self.assertEqual((spec.filters[0].column, spec.filters[0].op), ("x", ">"))

    def test_on_clause_second_equality_is_rejected(self):
        with self.assertRaises(g1.Unsupported) as caught:
            g1.analyze("SELECT COUNT(*) FROM r a JOIN s b ON a.k = b.k AND a.j = b.j")
        self.assertEqual(str(caught.exception), "join_predicate_count_2")

    def test_on_clause_unsupported_predicate_is_rejected(self):
        with self.assertRaises(g1.Unsupported):
            g1.analyze(
                "SELECT COUNT(*) FROM r a JOIN s b ON a.k = b.k AND a.x LIKE 'q%'"
            )

    def test_on_clause_equality_filter_is_kept(self):
        spec = g1.analyze("SELECT COUNT(*) FROM r a JOIN s b ON a.k = b.k AND a.x = 5")
        self.assertEqual(len(spec.filters), 1)
        self.assertEqual(spec.filters[0].column, "x")

    def test_rejects_with_clause(self):
        with self.assertRaises(g1.Unsupported) as caught:
            g1.analyze(
                "WITH t AS (SELECT k FROM s) "
                "SELECT COUNT(*) FROM r a, t b WHERE (a.k = b.k)"
            )
        self.assertEqual(str(caught.exception), "with_clause")

    def test_rejects_having(self):
        with self.assertRaises(g1.Unsupported):
            g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) "
                "GROUP BY a.k HAVING COUNT(*) > 1"
            )


class EstimatorTests(unittest.TestCase):
    def test_two_world_counterexample(self):
        result = g1.two_world_counterexample()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["c_cond"], [1.0, 0.0])
        # |R|^P=1, |S|^P=1, collision = (1/2)(1/1) = 0.5 in both worlds.
        self.assertEqual(result["c_uncond"], [0.5, 0.5])

    def test_scores_match_exact_on_synthetic(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": (
                        [("id", "int"), ("k", "int"), ("x", "int")],
                        [[1, 1, 1], [2, 1, 0], [3, 2, 1], [4, 3, 1]],
                    ),
                    "s": (
                        [("id", "int"), ("k", "int")],
                        [[10, 1], [11, 1], [12, 1], [13, 3]],
                    ),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x = 1)"
            )
            scored = g1.score_query(spec, tables, truth=6.0)
            # x=1 leaves keys {1, 2, 3}; s has 3 rows on key 1 and 1 on key 3.
            self.assertEqual(scored["exact"], 4)
            self.assertEqual(scored["c_cond"], 4.0)
            # 3 filtered left rows x 4 right rows x collision 0.4375.
            self.assertEqual(scored["c_uncond"], 5.25)
            g1.independent_recheck(spec, tables, scored)

    def test_null_keys_never_join(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int")], [[1], [None], [None]]),
                    "s": ([("k", "int")], [[1], [None]]),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze("SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k)")
            scored = g1.score_query(spec, tables, truth=1.0)
            self.assertEqual(scored["exact"], 1)
            self.assertEqual(scored["c_cond"], 1.0)

    def test_string_literal_on_integer_column(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int"), ("x", "int")], [[1, 6], [2, 7]]),
                    "s": ([("k", "int")], [[1], [2]]),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x = '6')"
            )
            scored = g1.score_query(spec, tables, truth=1.0)
            self.assertEqual(scored["exact"], 1)
            self.assertEqual(scored["c_cond"], 1.0)

    def test_empty_filter_result(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int"), ("x", "int")], [[1, 1], [2, 1]]),
                    "s": ([("k", "int")], [[1], [2]]),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x = 99)"
            )
            scored = g1.score_query(spec, tables, truth=0.0)
            self.assertEqual(scored["exact"], 0)
            self.assertEqual(scored["c_cond"], 0.0)
            self.assertEqual(scored["q_cond"], 1.0)

    def test_shuffled_keys_keep_c_cond_equal_to_exact(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            rows = [[index, (index * 7) % 5 + 1] for index in range(40)]
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("id", "int"), ("k", "int")], rows),
                    "s": (
                        [("id", "int"), ("k", "int")],
                        [[i, i % 5 + 1] for i in range(25)],
                    ),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze("SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k)")
            base = g1.score_query(spec, tables, truth=1.0)
            shuffled = dict(tables)
            shuffled["r"] = tables["r"].copy()
            shuffled["r"]["k"] = pd.array(
                tables["r"]["k"].sample(frac=1.0, random_state=0).tolist(),
                dtype="Int64",
            )
            after = g1.score_query(spec, shuffled, truth=1.0)
            self.assertEqual(base["c_cond"], base["exact"])
            self.assertEqual(after["c_cond"], after["exact"])

    def test_fractional_literal_on_int_column_is_unsupported(self):
        # PostgreSQL rejects `integer_col = 1.5`; truncating would silently
        # compare a different value.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int"), ("x", "int")], [[1, 1], [2, 2]]),
                    "s": ([("k", "int")], [[1], [2]]),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze(
                "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x = 1.5)"
            )
            with self.assertRaises(g1.Unsupported):
                g1.score_query(spec, tables, truth=0.0)

    def test_cross_kind_join_keys_are_unsupported(self):
        # int 6 must not be merged with str "6", nor split from float 6.0.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int")], [[1], [2]]),
                    "s": ([("k", "str")], [["1"], ["2"]]),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze("SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k)")
            with self.assertRaises(g1.Unsupported):
                g1.score_query(spec, tables, truth=2.0)

    def test_int_and_float_keys_are_unified(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            dataset = write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int")], [[1], [2]]),
                    "s": ([("k", "float")], [[1.0], [2.0]]),
                },
            )
            kinds, files = g1.parse_ddl(dataset / "postgres_create_tiny.sql")
            tables = g1.load_tables(dataset, kinds, files)
            spec = g1.analyze("SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k)")
            scored = g1.score_query(spec, tables, truth=2.0)
            self.assertEqual(scored["exact"], 2)
            self.assertEqual(scored["c_cond"], 2.0)
            # The checker must not raise a false alarm on a correct int/float join.
            g1.independent_recheck(spec, tables, scored)

    def test_aligned_reads_key_dtype_from_the_index(self):
        # `value_counts()` puts keys in the index and counts in the values.
        ints = pd.Series([1, 1], index=pd.Index([6, 7], dtype="Int64"))
        strings = pd.Series([1, 1], index=pd.Index(["6", "7"], dtype="string"))
        with self.assertRaises(g1.Unsupported):
            g1._aligned(ints, strings)

    def test_aligned_unifies_int_and_float_index(self):
        ints = pd.Series([1, 1], index=pd.Index([1995, 1996], dtype="Int64"))
        floats = pd.Series([1, 1], index=pd.Index([1995.0, 1996.0]))
        a, b = g1._aligned(ints, floats)
        self.assertEqual(float((a * b).sum()), 2.0)


class DdlTests(unittest.TestCase):
    def test_parse_ddl_preserves_quoted_case(self):
        with tempfile.TemporaryDirectory() as raw:
            ddl = Path(raw) / "d.sql"
            ddl.write_text(
                'create table "MyTable" (\n'
                "  id integer ,\n"
                "  name varchar(10)\n"
                ");\n"
                "\\copy \"MyTable\" from './t.csv' csv delimiter '|' null 'NULL' header;\n"
            )
            kinds, files = g1.parse_ddl(ddl)
            self.assertIn("MyTable", kinds)
            self.assertIn("MyTable", files)
            self.assertEqual(kinds["MyTable"]["id"], "int")

    def test_parse_ddl_accepts_uppercase_and_if_not_exists(self):
        with tempfile.TemporaryDirectory() as raw:
            ddl = Path(raw) / "d.sql"
            ddl.write_text(
                "CREATE TABLE IF NOT EXISTS Foo (\n"
                "  id integer ,\n"
                "  amount numeric(10,2) ,\n"
                "  label text\n"
                ");\n"
                "\\copy Foo from './f.csv' csv delimiter '|' null 'NULL' header;\n"
            )
            kinds, files = g1.parse_ddl(ddl)
            self.assertIn("Foo", kinds)
            self.assertEqual(kinds["Foo"]["id"], "int")
            self.assertEqual(kinds["Foo"]["amount"], "float")
            self.assertEqual(kinds["Foo"]["label"], "str")


class PlanRunTests(unittest.TestCase):
    def _fixture(self, root: Path) -> None:
        write_dataset(
            root,
            "tiny",
            {
                "r": (
                    [("id", "int"), ("k", "int"), ("x", "str")],
                    [[1, 1, "a"], [2, 2, "b"]],
                ),
                "s": ([("id", "int"), ("k", "int")], [[1, 1], [2, 2]]),
            },
        )
        write_workload(
            root,
            "tiny",
            [
                (
                    "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x = 'a')",
                    1.0,
                    2.0,
                ),
                (
                    "SELECT COUNT(*) FROM r a, s b, r c "
                    "WHERE (a.k = b.k) AND (a.k = c.k)",
                    1.0,
                    1.0,
                ),
            ],
        )

    def test_plan_dedupes_and_counts_unsupported(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._fixture(root)
            out = root / "out"
            out.mkdir()
            g1.main(
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
            self.assertEqual(block["rows"], 2)
            self.assertEqual(block["eligible"], 1)
            self.assertEqual(block["unsupported"], {"table_count_3": 1})
            self.assertEqual(len(block["queries"]), 1)
            self.assertTrue((out / "plan.sha256").exists())

    def test_plan_rejects_target_database(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "out"
            out.mkdir()
            with self.assertRaises(ValueError):
                g1.build_plan(Path(raw), ["imdb"], 200, out)

    def test_plan_rejects_inconsistent_labels(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_workload(
                root,
                "tiny",
                [
                    ("SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k)", 1.0, 1.0),
                    ("SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k)", 2.0, 1.0),
                ],
            )
            out = root / "out"
            out.mkdir()
            with self.assertRaises(ValueError):
                g1.build_plan(root / "workloads", ["tiny"], 200, out)

    def test_run_verifies_plan_hash(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self._fixture(root)
            out = root / "out"
            out.mkdir()
            g1.build_plan(root / "workloads", ["tiny"], 200, out)
            (out / "plan.sha256").write_text("0" * 64 + "\n")
            code = None
            try:
                g1.main(
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
            except SystemExit as error:
                code = str(error)
            self.assertIn("Plan SHA mismatch", code or "")


class RunVerdictTests(unittest.TestCase):
    def test_run_rejection_forces_blocked(self):
        # 114 §8: gates 1/2/4 failing must yield `blocked`, not a signal verdict.
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int"), ("x", "int")], [[1, 1], [2, 1]]),
                    "s": ([("k", "int")], [[1], [2]]),
                },
            )
            write_workload(
                root,
                "tiny",
                [
                    (
                        "SELECT COUNT(*) FROM r a, s b WHERE (a.k = b.k) AND (a.x = 1)",
                        2.0,
                        2.0,
                    ),
                    (
                        "SELECT COUNT(*) FROM r a, s b "
                        "WHERE (a.k = b.k) AND (a.x = 1.5)",
                        0.0,
                        1.0,
                    ),
                ],
            )
            out = root / "out"
            out.mkdir()
            g1.main(
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
            g1.main(
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
            rejected = json.loads((out / "rejected.json").read_text())
            self.assertEqual(len(rejected), 1)
            self.assertEqual(verdict["gates"]["2_coverage"]["passed"], False)
            self.assertEqual(verdict["verdict"], "blocked")

    def test_all_rejected_fails_loudly(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_dataset(
                root,
                "tiny",
                {
                    "r": ([("k", "int"), ("x", "int")], [[1, 1], [2, 1]]),
                    "s": ([("k", "int")], [[1], [2]]),
                },
            )
            write_workload(
                root,
                "tiny",
                [
                    (
                        "SELECT COUNT(*) FROM r a, s b "
                        "WHERE (a.k = b.k) AND (a.x = 1.5)",
                        0.0,
                        1.0,
                    )
                ],
            )
            out = root / "out"
            out.mkdir()
            g1.main(
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
            with self.assertRaises(SystemExit) as caught:
                g1.main(
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
            self.assertIn("All sampled queries were rejected", str(caught.exception))


class SamplingTests(unittest.TestCase):
    def test_evenly_spaced_returns_all_when_small(self):
        self.assertEqual(g1.evenly_spaced(["a", "b"], 10), [0, 1])

    def test_evenly_spaced_is_deterministic_and_spread(self):
        items = [str(i) for i in range(1000)]
        picks = g1.evenly_spaced(items, 200)
        self.assertEqual(len(picks), 200)
        self.assertEqual(picks, g1.evenly_spaced(items, 200))
        self.assertEqual(picks[0], 0)
        self.assertEqual(picks[-1], 999)
        gaps = {b - a for a, b in zip(picks, picks[1:])}
        self.assertLessEqual(max(gaps) - min(gaps), 1)


class MetricTests(unittest.TestCase):
    def test_q_error_is_symmetric_and_clips_at_one(self):
        self.assertTrue(math.isclose(g1.q_error(10, 5), 2.0))
        self.assertTrue(math.isclose(g1.q_error(5, 10), 2.0))
        self.assertEqual(g1.q_error(0, 0), 1.0)
        self.assertEqual(g1.q_error(0, 1), 1.0)

    def test_selectivity_buckets(self):
        self.assertEqual(g1._selectivity_bucket(0.0001), "<1e-3")
        self.assertEqual(g1._selectivity_bucket(0.005), "1e-3..1e-2")
        self.assertEqual(g1._selectivity_bucket(0.05), "1e-2..1e-1")
        self.assertEqual(g1._selectivity_bucket(0.2), "1e-1..0.5")
        self.assertEqual(g1._selectivity_bucket(1.0), ">=0.5")

    def test_breakdown_splits_by_join_key_filter(self):
        frame = pd.DataFrame(
            {
                "database": ["a", "a", "b"],
                "filters_join_key": [False, False, True],
                "sel_left": [0.5, 0.01, 0.9],
                "sel_right": [0.5, 0.02, 0.9],
                "q_uncond": [1.0, 2.0, 3.0],
                "q_cond": [1.0, 1.0, 1.0],
            }
        )
        result = g1.breakdown(frame)
        self.assertEqual(sum(row["n"] for row in result["rows"]), 3)
        self.assertEqual({row["database"] for row in result["rows"]}, {"a", "b"})
        self.assertEqual(
            {row["filters_join_key"] for row in result["rows"]}, {False, True}
        )


if __name__ == "__main__":
    unittest.main()
