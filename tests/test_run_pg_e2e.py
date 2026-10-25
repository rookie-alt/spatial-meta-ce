import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

AVAILABLE = importlib.util.find_spec("spatial_meta_ce.data") is not None
if AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

    import run_pg_e2e as e2e

TOP = "select count(*) from badges as st_b, users as st_u where st_b.userid = st_u.id;"
JOINED = (
    "select count(*) from a t1 join b t2 on t1.id = t2.id join c t3 on t3.id = t2.id;"
)
NESTED = "select count(*) from (select id from a) s, b t2 where s.id = t2.id;"


def workload(root, lines):
    path = Path(root) / "workloads_all.sql"
    path.write_text("\n".join(lines) + "\n")
    return path


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_groups_and_orders_subqueries(self):
        with tempfile.TemporaryDirectory() as d:
            path = workload(
                d,
                [
                    f"{TOP}||10||@0-0@",
                    "/* (st_b) */ select count(*) from badges st_b;||5||@0-1@",
                    "/* (st_b, st_u) */ select count(*) from badges st_b, users st_u||9||@0-3@",
                    f"{TOP}||11||@1-0@",
                ],
            )
            groups = e2e.parse_workload_all(path)
        self.assertEqual(sorted(groups), [0, 1])
        self.assertEqual([row["subquery"] for row in groups[0]], [0, 1, 3])
        self.assertEqual(groups[0][1]["aliases"], ("st_b",))
        self.assertEqual(groups[0][2]["aliases"], ("st_b", "st_u"))
        self.assertEqual(groups[0][0]["aliases"], ())
        self.assertEqual(groups[0][0]["line"], 1)

    def test_rejects_wrong_field_count(self):
        with tempfile.TemporaryDirectory() as d:
            path = workload(d, ["select 1||10||2||@0-0@"])
            with self.assertRaises(ValueError):
                e2e.parse_workload_all(path)

    def test_rejects_missing_tag(self):
        with tempfile.TemporaryDirectory() as d:
            path = workload(d, ["select 1||10||nope"])
            with self.assertRaises(ValueError):
                e2e.parse_workload_all(path)


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_comma_join_aliases(self):
        self.assertEqual(e2e.top_level_aliases(TOP), ("st_b", "st_u"))

    def test_join_chain_aliases(self):
        self.assertEqual(e2e.top_level_aliases(JOINED), ("t1", "t2", "t3"))

    def test_nested_relation_is_not_a_top_level_alias(self):
        self.assertNotIn("a", e2e.top_level_aliases(NESTED))


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def rows(self, prefixes):
        return [
            {"aliases": prefix, "subquery": index, "true": 1.0, "line": index}
            for index, prefix in enumerate(prefixes)
        ]

    def test_prefix_subset_is_supported(self):
        rows = self.rows([(), ("st_b",), ("st_b", "st_u")])
        self.assertTrue(e2e.supported(rows, ("st_b", "st_u")))

    def test_single_relation_top_level_is_not_supported(self):
        self.assertFalse(e2e.supported(self.rows([(), ("st_b",)]), ("st_b",)))

    def test_prefix_outside_the_block_is_not_supported(self):
        rows = self.rows([(), ("st_b", "st_x")])
        self.assertFalse(e2e.supported(rows, ("st_b", "st_u")))

    def test_duplicate_top_level_alias_is_not_supported(self):
        rows = self.rows([(), ("st_b", "st_b")])
        self.assertFalse(e2e.supported(rows, ("st_b", "st_b")))


class HintTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def rows(self):
        return [
            {"aliases": (), "subquery": 0},
            {"aliases": ("st_b",), "subquery": 1},
            {"aliases": ("st_b", "st_u"), "subquery": 3},
        ]

    def test_single_table_prefix_is_not_hinted(self):
        estimates = {"ours": {0: 10.0, 1: 5.0, 3: 7.0}}
        text = e2e.hint(self.rows(), estimates, "ours")
        self.assertEqual(text, "/*+ Rows(st_b st_u #7) */ ")
        self.assertNotIn("Rows(st_b #", text)

    def test_missing_value_is_skipped(self):
        estimates = {"ours": {3: 7.0}}
        self.assertEqual(
            e2e.hint(self.rows(), estimates, "ours"), "/*+ Rows(st_b st_u #7) */ "
        )

    def test_no_hint_when_nothing_is_injectable(self):
        self.assertEqual(e2e.hint(self.rows(), {"ours": {}}, "ours"), "")

    def test_values_are_rounded_and_floored_at_one(self):
        estimates = {"ours": {3: 0.2}}
        self.assertEqual(
            e2e.hint(self.rows(), estimates, "ours"), "/*+ Rows(st_b st_u #1) */ "
        )


class JoinRowsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_reads_the_join_below_a_plain_aggregate(self):
        plan = {
            "Node Type": "Aggregate",
            "Plan Rows": 1,
            "Plans": [{"Node Type": "Hash Join", "Plan Rows": 512.0}],
        }
        self.assertEqual(e2e.join_rows(plan), 512.0)

    def test_skips_gather_and_partial_aggregate(self):
        plan = {
            "Node Type": "Aggregate",
            "Plan Rows": 1,
            "Plans": [
                {
                    "Node Type": "Gather",
                    "Plan Rows": 1,
                    "Plans": [
                        {
                            "Node Type": "Partial Aggregate",
                            "Plan Rows": 1,
                            "Plans": [{"Node Type": "Nested Loop", "Plan Rows": 42.0}],
                        }
                    ],
                }
            ],
        }
        self.assertEqual(e2e.join_rows(plan), 42.0)

    def test_falls_back_when_no_join_is_present(self):
        plan = {
            "Node Type": "Aggregate",
            "Plan Rows": 1,
            "Plans": [{"Node Type": "Seq Scan", "Plan Rows": 7.0}],
        }
        self.assertEqual(e2e.join_rows(plan), 7.0)


class QErrorTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_symmetric(self):
        self.assertAlmostEqual(e2e.q_error(10.0, 40.0), 4.0)
        self.assertAlmostEqual(e2e.q_error(40.0, 10.0), 4.0)

    def test_exact_estimate_is_one(self):
        self.assertAlmostEqual(e2e.q_error(7.0, 7.0), 1.0)

    def test_floors_the_denominator_at_one(self):
        self.assertAlmostEqual(e2e.q_error(0.0, 5.0), 5.0)


class EstimatesTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def test_single_table_prefixes_need_no_estimate(self):
        import numpy as np

        rows = [
            {"aliases": (), "subquery": 0, "sql": TOP, "true": 10.0, "line": 1},
            {
                "aliases": ("st_b",),
                "subquery": 1,
                "sql": "select 1",
                "true": 5.0,
                "line": 2,
            },
        ]
        table, missing = e2e.estimates_for(rows, {}, np.zeros(0), np.zeros(0))
        self.assertEqual(missing, [])
        self.assertEqual(table["true_card"], {0: 10.0, 1: 5.0})

    def test_absent_from_cache_is_recorded(self):
        import numpy as np

        rows = [
            {
                "aliases": ("st_b", "st_u"),
                "subquery": 3,
                "sql": TOP,
                "true": 9.0,
                "line": 7,
            }
        ]
        table, missing = e2e.estimates_for(rows, {}, np.zeros(0), np.zeros(0))
        self.assertEqual(missing, [7])
        self.assertNotIn(3, table["ours"])


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(AVAILABLE, "spatial_meta_ce not importable")

    def base(self, root, **overrides):
        values = {
            "database": "stats",
            "allow_database": ["stats"],
            "workload_all": str(root / "workloads_all.sql"),
            "cache_root": str(root),
            "checkpoint": str(root / "none.pt"),
            "price_checkpoint": str(root / "price.pth"),
            "output": str(root / "out.json"),
        }
        values.update(overrides)
        args = []
        for key, value in values.items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, list):
                args += [flag] + [str(item) for item in value]
            else:
                args += [flag, str(value)]
        return args

    def test_unapproved_database_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                e2e.main(self.base(Path(d), allow_database=["ergastf1"]))

    def test_single_run_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                e2e.main(self.base(Path(d), runs=1))

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            existing = root / "out.json"
            existing.write_text("keep me")
            with self.assertRaises(SystemExit):
                e2e.main(self.base(root))
            self.assertEqual(existing.read_text(), "keep me")


if __name__ == "__main__":
    unittest.main()
