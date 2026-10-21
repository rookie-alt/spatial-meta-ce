import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_SPEC = importlib.util.spec_from_file_location(
    "broad_distribution_summary", _ROOT / "scripts/broad_distribution_summary.py"
)
broad = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(broad)


def row(ours, price, pg, ours4=None, price4=None, pg4=None):
    return {
        "q95": {"ours_k0": ours, "price_k0": price, "pg": pg},
        "bucket_4plus_q95": {
            "ours_k0": ours if ours4 is None else ours4,
            "price_k0": price if price4 is None else price4,
            "pg": pg if pg4 is None else pg4,
        },
    }


class ClassifyTests(unittest.TestCase):
    def test_beats_both(self):
        self.assertEqual(broad.classify(row(10.0, 20.0, 30.0)), "beats_pg_and_price")

    def test_beats_pg_only(self):
        self.assertEqual(broad.classify(row(10.0, 5.0, 30.0)), "beats_pg_loses_price")

    def test_loses_pg_even_when_beating_price(self):
        self.assertEqual(broad.classify(row(40.0, 5.0, 30.0)), "loses_pg")

    def test_tie_with_pg_counts_as_loss(self):
        self.assertEqual(broad.classify(row(30.0, 5.0, 30.0)), "loses_pg")

    def test_missing_metric_is_an_error(self):
        with self.assertRaises(ValueError):
            broad.classify(row(None, 20.0, 30.0))


class RatioTests(unittest.TestCase):
    def test_ratio_above_one_means_we_win(self):
        self.assertAlmostEqual(broad.ratio(30.0, 10.0), 3.0)

    def test_zero_denominator_is_undefined(self):
        self.assertIsNone(broad.ratio(30.0, 0.0))

    def test_none_input_is_undefined(self):
        self.assertIsNone(broad.ratio(None, 1.0))

    def test_describe_ignores_undefined(self):
        stats = broad.describe([1.0, 3.0, None])
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["median"], 2.0)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 3.0)

    def test_describe_handles_empty(self):
        stats = broad.describe([None])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["median"])


class BuildDistributionTests(unittest.TestCase):
    def setUp(self):
        self.databases = {
            "a": row(10.0, 20.0, 30.0, ours4=10.0, price4=40.0, pg4=50.0),
            "b": row(10.0, 5.0, 30.0, ours4=10.0, price4=2.0, pg4=50.0),
            "c": row(40.0, 5.0, 30.0, ours4=40.0, price4=2.0, pg4=50.0),
        }

    def test_counts_all_three_tiers(self):
        dist = broad.build_distribution(self.databases, ["a", "b", "c"])
        self.assertEqual(dist["n_databases"], 3)
        self.assertEqual(dist["ours_better_than_pg"], 2)  # a and b
        # Only a (ours 10) is below PRICE; b and c face PRICE 5.  Here the
        # pairwise count and the "beats both" tier happen to agree.
        self.assertEqual(dist["ours_better_than_price"], 1)
        self.assertEqual(
            dist["tier_counts"],
            {"beats_pg_and_price": 1, "beats_pg_loses_price": 1, "loses_pg": 1},
        )
        self.assertEqual(dist["tier_by_database"]["c"], "loses_pg")

    def test_winning_price_while_losing_pg_counts_as_a_price_win(self):
        # tpc_h in the real sweep: our Q95 is far below PRICE but just above
        # PG.  "ours_better_than_price" must include it even though its tier
        # is loses_pg; only the tier counts exclude it.
        databases = {
            "a": row(10.0, 20.0, 30.0),  # beats both
            "tpc_h_like": row(23.0, 68.0, 20.0),  # beats PRICE, loses PG
        }
        dist = broad.build_distribution(databases, ["a", "tpc_h_like"])
        self.assertEqual(dist["ours_better_than_pg"], 1)
        self.assertEqual(dist["ours_better_than_price"], 2)
        self.assertEqual(
            dist["tier_counts"],
            {"beats_pg_and_price": 1, "beats_pg_loses_price": 0, "loses_pg": 1},
        )

    def test_ratio_distribution_is_reported(self):
        dist = broad.build_distribution(self.databases, ["a", "b", "c"])
        self.assertEqual(dist["q95_ratio_price_over_ours"]["n"], 3)
        # price/ours = 2.0, 0.5, 0.125
        self.assertAlmostEqual(dist["q95_ratio_price_over_ours"]["max"], 2.0)
        self.assertAlmostEqual(dist["q95_ratio_price_over_ours"]["min"], 0.125)
        self.assertAlmostEqual(dist["q95_ratio_pg_over_ours"]["median"], 3.0)
        self.assertAlmostEqual(dist["bucket_4plus_ratio_price_over_ours"]["max"], 4.0)

    def test_missing_database_is_an_error(self):
        with self.assertRaises(ValueError):
            broad.build_distribution(self.databases, ["a", "b"])

    def test_unexpected_database_is_an_error(self):
        with self.assertRaises(ValueError):
            broad.build_distribution(self.databases, ["a", "b", "c", "d"])

    def test_row_order_follows_the_expected_list(self):
        dist = broad.build_distribution(self.databases, ["c", "a", "b"])
        self.assertEqual(list(dist["tier_by_database"]), ["c", "a", "b"])

    def test_empty_four_plus_buckets_are_named_not_dropped(self):
        databases = {
            "a": row(10.0, 20.0, 30.0, ours4=10.0, price4=40.0, pg4=50.0),
            "b": {
                "q95": {"ours_k0": 10.0, "price_k0": 20.0, "pg": 30.0},
                "bucket_4plus_q95": {
                    "ours_k0": None,
                    "price_k0": None,
                    "pg": None,
                },
            },
        }
        dist = broad.build_distribution(databases, ["a", "b"])
        self.assertEqual(dist["bucket_4plus_empty"]["databases"], ["b"])
        self.assertEqual(dist["bucket_4plus_empty"]["n"], 1)
        # Only the non-empty database contributes a 4+ ratio.
        self.assertEqual(dist["bucket_4plus_ratio_price_over_ours"]["n"], 1)
        # But both still count in the headline tier classification.
        self.assertEqual(dist["n_databases"], 2)


class FlattenTests(unittest.TestCase):
    """The raw sweep writes full database reports, not headline records."""

    def test_full_database_report_is_read(self):
        full = {
            "n": 50000,
            "ours_k0": {"q50": 1.0, "q95": 10.0, "log_q95": 2.0},
            "price_k0": {"q50": 1.0, "q95": 20.0, "log_q95": 3.0},
            "pg": {"q50": 1.0, "q95": 30.0, "log_q95": 4.0},
            "bucket_4plus": {
                "n": 7,
                "ours_k0": {"q50": 1.0, "q95": 11.0, "log_q95": 2.0},
                "price_k0": {"q50": 1.0, "q95": 21.0, "log_q95": 3.0},
                "pg": {"q50": 1.0, "q95": 31.0, "log_q95": 4.0},
            },
        }
        flat = broad.flatten(full)
        self.assertEqual(flat["q95"], {"ours_k0": 10.0, "price_k0": 20.0, "pg": 30.0})
        self.assertEqual(
            flat["bucket_4plus_q95"],
            {"ours_k0": 11.0, "price_k0": 21.0, "pg": 31.0},
        )

    def test_full_report_with_empty_four_plus_is_null(self):
        full = {
            "n": 50000,
            "ours_k0": {"q50": 1.0, "q95": 10.0, "log_q95": 2.0},
            "price_k0": {"q50": 1.0, "q95": 20.0, "log_q95": 3.0},
            "pg": {"q50": 1.0, "q95": 30.0, "log_q95": 4.0},
            "bucket_4plus": {"n": 0, "ours_k0": None, "price_k0": None, "pg": None},
        }
        flat = broad.flatten(full)
        self.assertEqual(
            flat["bucket_4plus_q95"],
            {"ours_k0": None, "price_k0": None, "pg": None},
        )

    def test_headline_record_is_read_too(self):
        flat = broad.flatten(row(1.0, 2.0, 3.0, ours4=4.0, price4=5.0, pg4=6.0))
        self.assertEqual(flat["q95"]["ours_k0"], 1.0)
        self.assertEqual(flat["bucket_4plus_q95"]["pg"], 6.0)

    def test_both_shapes_classify_identically(self):
        headline = row(10.0, 20.0, 30.0)
        full = {
            "ours_k0": {"q95": 10.0},
            "price_k0": {"q95": 20.0},
            "pg": {"q95": 30.0},
            "bucket_4plus": {"n": 0, "ours_k0": None, "price_k0": None, "pg": None},
        }
        self.assertEqual(broad.classify(headline), broad.classify(full))


class SummarizeTests(unittest.TestCase):
    def _protocol(self, read):
        return {
            "target_databases_read": read,
            "price": {
                "maxima": [12, 12, 7, 10],
                "checkpoint_sha256": "abc",
                "maxima_rule": "explicit",
            },
            "inputs": {"caches": {}},
        }

    def test_target_reading_run_is_refused(self):
        with self.assertRaises(ValueError):
            broad.summarize({"a": row(1, 2, 3)}, ["a"], self._protocol(["imdb"]))

    def test_scope_is_labelled_in_sample_fit(self):
        summary = broad.summarize({"a": row(1, 2, 3)}, ["a"], self._protocol([]))
        self.assertEqual(summary["scope"], "in_sample_fit_diagnostic")
        self.assertEqual(summary["target_databases_read"], [])
        self.assertIn("fit", summary["scope_note"])
        self.assertEqual(summary["price"]["maxima"], [12, 12, 7, 10])


if __name__ == "__main__":
    unittest.main()
