"""Tests for `scripts/tree_identity.py`."""

from __future__ import annotations

import itertools
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tree_identity as ti  # noqa: E402


def brute_force(tables: dict[str, pd.DataFrame], predicates) -> int:
    """Enumerate every row combination that satisfies all join predicates."""
    names = list(tables)
    total = 0
    for combo in itertools.product(*[tables[n].to_dict("records") for n in names]):
        rows = dict(zip(names, combo))
        ok = True
        for alias_a, col_a, alias_b, col_b in predicates:
            left = rows[alias_a].get(col_a)
            right = rows[alias_b].get(col_b)
            if left is None or right is None or left != right:
                ok = False
                break
        if ok:
            total += 1
    return total


def score(tables, predicates, filters=(), truth=0.0):
    aliases = list(tables)
    spec = ti.TreeSpec(
        tables=tuple(aliases),
        names=tuple(aliases),
        predicates=tuple(predicates),
        filters=tuple(filters),
    )
    return ti.score_query_tree(spec, tables, truth)


class TreeTests(unittest.TestCase):
    def test_two_table_matches_brute_force(self):
        tables = {
            "l": pd.DataFrame({"k": [1, 1, 2]}),
            "r": pd.DataFrame({"k": [1, 2, 2]}),
        }
        predicates = [("l", "k", "r", "k")]
        expected = brute_force(tables, predicates)
        scored = score(tables, predicates, truth=float(expected))
        self.assertEqual(scored["exact"], expected)
        self.assertAlmostEqual(scored["c_cond"], expected)

    def test_three_table_path_matches_brute_force(self):
        tables = {
            "a": pd.DataFrame({"k": [1, 2]}),
            "b": pd.DataFrame({"x": [1, 1, 2], "y": [10, 20, 10]}),
            "c": pd.DataFrame({"k": [10, 10, 20]}),
        }
        predicates = [("a", "k", "b", "x"), ("b", "y", "c", "k")]
        expected = brute_force(tables, predicates)
        scored = score(tables, predicates, truth=float(expected))
        self.assertEqual(scored["exact"], expected)
        self.assertAlmostEqual(scored["c_cond"], expected)

    def test_repeated_middle_column_matches_brute_force(self):
        # The shape that produced G2a's carcinogenesis coverage gap.
        tables = {
            "hub": pd.DataFrame({"id": [1, 2, 3]}),
            "p": pd.DataFrame({"drug": [1, 1, 3]}),
            "q": pd.DataFrame({"drug": [1, 3, 3, 3]}),
        }
        predicates = [("hub", "id", "p", "drug"), ("hub", "id", "q", "drug")]
        expected = brute_force(tables, predicates)
        scored = score(tables, predicates, truth=float(expected))
        self.assertEqual(scored["exact"], expected)
        self.assertAlmostEqual(scored["c_cond"], expected)

    def test_four_table_chain_matches_brute_force(self):
        tables = {
            "a": pd.DataFrame({"k": [1, 2]}),
            "b": pd.DataFrame({"x": [1, 1, 2], "y": [7, 8, 7]}),
            "c": pd.DataFrame({"p": [7, 7, 8], "q": [5, 6, 5]}),
            "d": pd.DataFrame({"k": [5, 6, 6]}),
        }
        predicates = [
            ("a", "k", "b", "x"),
            ("b", "y", "c", "p"),
            ("c", "q", "d", "k"),
        ]
        expected = brute_force(tables, predicates)
        scored = score(tables, predicates, truth=float(expected))
        self.assertEqual(scored["exact"], expected)
        self.assertAlmostEqual(scored["c_cond"], expected)

    def test_four_table_star_matches_brute_force(self):
        tables = {
            "hub": pd.DataFrame({"id": [1, 2, 3]}),
            "p": pd.DataFrame({"a": [1, 1, 3]}),
            "q": pd.DataFrame({"b": [1, 2, 2]}),
            "s": pd.DataFrame({"c": [2, 3, 3]}),
        }
        predicates = [
            ("hub", "id", "p", "a"),
            ("hub", "id", "q", "b"),
            ("hub", "id", "s", "c"),
        ]
        expected = brute_force(tables, predicates)
        scored = score(tables, predicates, truth=float(expected))
        self.assertEqual(scored["exact"], expected)
        self.assertAlmostEqual(scored["c_cond"], expected)

    def test_filters_change_the_answer(self):
        tables = {
            "a": pd.DataFrame({"k": [1, 2], "f": [0, 1]}),
            "b": pd.DataFrame({"k": [1, 2, 2]}),
        }
        predicates = [("a", "k", "b", "k")]
        predicate = ti.Predicate("a", "f", "=", "1", False)
        scored = score(tables, predicates, filters=(predicate,), truth=2.0)
        self.assertEqual(scored["exact"], 2)

    def test_null_keys_never_join(self):
        tables = {
            "a": pd.DataFrame({"k": [1, None]}),
            "b": pd.DataFrame({"k": [1, None]}),
        }
        predicates = [("a", "k", "b", "k")]
        scored = score(tables, predicates, truth=1.0)
        self.assertEqual(scored["exact"], 1)

    def test_unconditional_equals_closed_form_on_two_tables(self):
        tables = {
            "a": pd.DataFrame({"k": [1, 1, 2, 3]}),
            "b": pd.DataFrame({"k": [1, 1, 1, 3]}),
        }
        predicates = [("a", "k", "b", "k")]
        scored = score(tables, predicates, truth=7.0)
        # sum_k c_a(k) c_b(k) = 2*3 + 0 + 1*1 = 7
        self.assertAlmostEqual(scored["c_uncond"], 7.0)

    def test_unconditional_scales_once_under_a_filter(self):
        # The filter must enter through the selectivity, not through the
        # marginals as well; using filtered counts here double-counts s.
        tables = {
            "a": pd.DataFrame({"k": [1, 1, 2, 3], "f": [0, 0, 1, 1]}),
            "b": pd.DataFrame({"k": [1, 1, 1, 3]}),
        }
        predicates = [("a", "k", "b", "k")]
        predicate = ti.Predicate("a", "f", "=", "1", False)
        scored = score(tables, predicates, filters=(predicate,), truth=1.0)
        # s_a = 2/4, sum_k c_a c_b = 7  ->  3.5
        self.assertAlmostEqual(scored["c_uncond"], 3.5)


class AnalyzeTests(unittest.TestCase):
    def test_rejects_cycle(self):
        with self.assertRaises(ti.Unsupported) as caught:
            ti.analyze_tree(
                "SELECT COUNT(*) FROM a x, b y, c z "
                "WHERE (x.k = y.k) AND (y.k = z.k) AND (x.k = z.k)",
                min_tables=3,
            )
        self.assertEqual(str(caught.exception), "join_predicate_count_3")

    def test_rejects_below_min_tables(self):
        with self.assertRaises(ti.Unsupported) as caught:
            ti.analyze_tree(
                "SELECT COUNT(*) FROM a x, b y WHERE (x.k = y.k)", min_tables=4
            )
        self.assertEqual(str(caught.exception), "table_count_2")

    def test_accepts_seven_table_tree(self):
        sql = (
            "SELECT COUNT(*) FROM t1 a, t2 b, t3 c, t4 d, t5 e, t6 f, t7 g "
            "WHERE (a.k = b.k) AND (b.k = c.k) AND (c.k = d.k) "
            "AND (d.k = e.k) AND (e.k = f.k) AND (f.k = g.k)"
        )
        spec = ti.analyze_tree(sql, min_tables=4)
        self.assertEqual(len(spec.tables), 7)

    def test_records_join_key_filter(self):
        spec = ti.analyze_tree(
            "SELECT COUNT(*) FROM a x, b y WHERE (x.k = y.k) AND (x.k > 5)",
            min_tables=2,
        )
        self.assertTrue(spec.filters_join_key())


if __name__ == "__main__":
    unittest.main()
