import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import build_support_coverage_plan as planner
from scripts import diagnostic_90_a_support as stage_a


class FakeStore:
    """Minimal stand-in for FeatureStore covering what stage A reads."""

    def __init__(self, table_counts, true, pg, group_lengths, cartesian):
        n = len(table_counts)
        self.path = Path("/tmp/fakestore")
        self.rows = [{"id": "fake/pretrain/%d" % (index + 1),
                      "query_hash": "h%04d" % index} for index in range(n)]
        self.cards = np.stack([np.asarray(true, dtype=np.float64),
                               np.asarray(pg, dtype=np.float64)], axis=1)
        self.offsets = np.concatenate(
            [np.zeros((1, 4), dtype=np.int64),
             np.cumsum(np.asarray(group_lengths, dtype=np.int64), axis=0)], axis=0)
        self.offsets[:, 2] = np.concatenate(
            [[0], np.cumsum(np.asarray(table_counts, dtype=np.int64))])
        self.groups = [
            np.zeros((int(self.offsets[-1, index]), dim), dtype=np.float32)
            for index, dim in enumerate((40, 40, 4, 43))]
        for index in range(n):
            start, stop = self.offsets[index, 2], self.offsets[index + 1, 2]
            self.groups[2][start:stop, 0] = cartesian[index]
        self.table_counts = np.asarray(table_counts, dtype=np.int64)

    def tables(self):
        return self.table_counts


def simple_store(n=200):
    rng = np.random.default_rng(0)
    tables = np.asarray([4] * 50 + [5] * 50 + [6] * 50 + [7] * 50)
    true = rng.integers(1, 500, n).astype(np.float64)
    true[:50] = rng.uniform(1, 90, 50)
    pg = true * rng.uniform(0.5, 2.0, n)
    lengths = np.ones((n, 4), dtype=np.int64)
    lengths[:, 2] = tables
    cartesian = rng.uniform(0, 4, n)
    return FakeStore(tables, true, pg, lengths, cartesian)


class PlanBuilderTests(unittest.TestCase):
    def setUp(self):
        self.indices = list(range(400))
        self.hashes = ["h%04d" % index for index in self.indices]
        self.identity = dict(zip(self.indices, self.hashes))

    def test_draw_satisfies_every_invariant(self):
        plan = planner.draw(self.indices, self.hashes, "chembl", 7)
        support = plan["support_max"]
        self.assertEqual(len(support), 100)
        self.assertEqual(len(set(support)), 100)
        self.assertEqual(plan["supports"]["20"], support[:20])
        self.assertFalse(set(support) & set(plan["query"]))
        self.assertEqual(plan["query"],
                         [i for i in self.indices if i not in set(support)])
        self.assertEqual(len(plan["query"]), len(self.indices) - 100)

    def test_draw_accepts_sparse_cache_row_indices(self):
        # Real caches keep row numbers after first-unique deduplication, so the
        # drawn indices are sparse and exceed the population size.
        sparse = [index * 3 + 7 for index in range(400)]
        hashes = ["h%04d" % index for index in sparse]
        plan = planner.draw(sparse, hashes, "chembl", 11)
        self.assertTrue(set(plan["support_max"]) <= set(sparse))
        self.assertEqual(plan["query"],
                         [i for i in sparse if i not in set(plan["support_max"])])

    def test_draws_are_deterministic_and_run_dependent(self):
        first = planner.draw(self.indices, self.hashes, "chembl", 5)
        again = planner.draw(self.indices, self.hashes, "chembl", 5)
        other = planner.draw(self.indices, self.hashes, "chembl", 6)
        self.assertEqual(first, again)
        self.assertNotEqual(first["support_max"], other["support_max"])

    def test_draw_does_not_reuse_the_protocol_81_seed_scheme(self):
        import random

        from scripts import meta65_holdout_contract as contract

        seed_81 = contract.FROZEN["evaluation"]["support_seed"]
        self.assertEqual(seed_81, 2718)
        plan = planner.draw(self.indices, self.hashes, "legalacts", 5)
        legacy = random.Random("%d:legalacts:5" % seed_81).sample(self.indices, 100)
        self.assertNotEqual(plan["support_max"], legacy)

    def test_certify_rejects_query_that_reuses_a_support_index(self):
        plan = planner.draw(self.indices, self.hashes, "telstra", 9)
        broken = copy.deepcopy(plan)
        broken["query"] = broken["query"] + [plan["support_max"][0]]
        broken["query_digest"] = planner.contract.digest(
            [(i, self.identity[i]) for i in broken["query"]])
        with self.assertRaises(ValueError):
            planner.certify_draw(broken, self.indices, self.hashes, "telstra", 9)

    def test_certify_rejects_a_shortened_query(self):
        plan = planner.draw(self.indices, self.hashes, "telstra", 9)
        broken = copy.deepcopy(plan)
        broken["query"] = broken["query"][:-1]
        broken["query_digest"] = planner.contract.digest(
            [(i, self.identity[i]) for i in broken["query"]])
        with self.assertRaises(ValueError):
            planner.certify_draw(broken, self.indices, self.hashes, "telstra", 9)

    def test_guard_rejects_target_holdout_and_unknown_databases(self):
        for name in planner.FORBIDDEN:
            with self.assertRaises(ValueError):
                planner.guard_name(name)
        for name in planner.EXCLUDED:
            with self.assertRaises(ValueError):
                planner.guard_name(name)
        with self.assertRaises(ValueError):
            planner.guard_name("accidents")
        for name in planner.DATABASES:
            planner.guard_name(name)

    def test_guard_rejects_target_database_inside_a_path(self):
        with self.assertRaises(ValueError):
            planner.guard_path("/gly/weilin/runs/imdb/out.json")
        planner.guard_path("/gly/weilin/runs/support-coverage-20261019/out.json")

    def test_run_range_covers_only_the_45_new_draws(self):
        self.assertEqual(planner.RUNS, tuple(range(5, 50)))
        self.assertNotIn(0, planner.RUNS)
        self.assertEqual(len(planner.RUNS), 45)


class OutcomeTests(unittest.TestCase):
    def test_better_share_and_q95_ratio_match_manual_arithmetic(self):
        true = np.array([10.0, 20.0, 30.0, 400.0])
        tables = np.array([4, 5, 6, 4])
        none = np.log(true) + np.array([1.0, -1.0, 0.5, 2.0])
        head3 = np.log(true) + np.array([0.2, -0.3, 0.9, 1.0])
        result = stage_a.outcomes(none, head3, true, tables)
        mask = stage_a.low4_mask(true, tables)
        # The 400-row is not low4, so the denominator is three queries:
        # errors fall 1.0->0.2 and 0.5->0.9, so two improve and one worsens.
        self.assertEqual(result["low4_count"], int(mask.sum()))
        self.assertEqual(result["better"], 2)
        self.assertEqual(result["worse"], 1)
        self.assertAlmostEqual(result["better_share"], 2 / 3, places=15)
        q_none = np.exp(np.abs(np.maximum(none, 0) - np.log(np.maximum(true, 1))))
        q_head = np.exp(np.abs(np.maximum(head3, 0) - np.log(np.maximum(true, 1))))
        expected = (np.percentile(q_head[mask], 95, method="linear")
                    / np.percentile(q_none[mask], 95, method="linear"))
        self.assertAlmostEqual(result["q95_ratio"], expected, places=15)

    def test_low4_definition_excludes_large_and_three_table_queries(self):
        true = np.array([50.0, 500.0, 10.0])
        tables = np.array([4, 4, 3])
        self.assertEqual(stage_a.low4_mask(true, tables).tolist(),
                         [True, False, False])

    def test_empty_low4_returns_null_outcomes(self):
        true = np.array([500.0, 600.0])
        tables = np.array([4, 5])
        result = stage_a.outcomes(np.zeros(2), np.zeros(2), true, tables)
        self.assertEqual(result["low4_count"], 0)
        self.assertIsNone(result["better_share"])
        self.assertIsNone(result["q95_ratio"])

    def test_nonfinite_prediction_is_rejected(self):
        true = np.array([10.0, 20.0])
        tables = np.array([4, 5])
        head3 = np.array([1.0, np.nan])
        with self.assertRaises(ValueError):
            stage_a.outcomes(np.zeros(2), head3, true, tables)


class MetricGateTests(unittest.TestCase):
    def setUp(self):
        self.true = np.array([10.0, 20.0, 30.0, 40.0])
        self.tables = np.array([4, 5, 6, 7])
        self.none = np.log(self.true) + np.array([1.0, -1.0, 0.5, 2.0])
        self.head3 = np.log(self.true) + np.array([0.2, -0.3, 0.9, 1.0])
        measured = stage_a.outcomes(self.none, self.head3, self.true, self.tables)
        self.reference = {("chembl", 0): {
            "better_share": measured["better_share"],
            "q95_ratio": measured["q95_ratio"],
            "low4_count": measured["low4_count"],
            "query_count": 4}}

    def test_metric_gate_passes_at_exact_agreement(self):
        result = stage_a.check_metric_path(
            self.reference, {"none": self.none, "head3": self.head3},
            self.true, self.tables, "chembl", 0)
        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(max(result["deltas"].values()), 1e-9)

    def test_metric_gate_fails_above_one_e_minus_nine(self):
        reference = copy.deepcopy(self.reference)
        reference[("chembl", 0)]["better_share"] += 1e-8
        with self.assertRaises(ValueError):
            stage_a.check_metric_path(
                reference, {"none": self.none, "head3": self.head3},
                self.true, self.tables, "chembl", 0)

    def test_metric_gate_rejects_a_missing_reference_run(self):
        with self.assertRaises(ValueError):
            stage_a.check_metric_path(
                self.reference, {"none": self.none, "head3": self.head3},
                self.true, self.tables, "chembl", 3)

    def test_forward_gate_accepts_the_measured_cpu_gpu_gap(self):
        archived = {"none": self.none, "head3": self.head3}
        predicted = {key: values + 1.9e-6 for key, values in archived.items()}
        result = stage_a.check_forward_path(archived, predicted, "chembl", 0)
        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(max(result["max_abs_diff"].values()), 1e-4)

    def test_forward_gate_rejects_a_gap_above_the_tolerance(self):
        archived = {"none": self.none, "head3": self.head3}
        predicted = {key: values + 2e-4 for key, values in archived.items()}
        with self.assertRaises(ValueError):
            stage_a.check_forward_path(archived, predicted, "chembl", 0)

    def test_forward_gate_rejects_a_length_mismatch(self):
        archived = {"none": self.none, "head3": self.head3}
        predicted = {"none": self.none[:-1], "head3": self.head3}
        with self.assertRaises(ValueError):
            stage_a.check_forward_path(archived, predicted, "chembl", 0)


class FeatureTests(unittest.TestCase):
    def test_table_shares_cover_the_whole_population(self):
        store = simple_store()
        population = stage_a.population_context(store)
        features = stage_a.support_features(store, list(range(20)), population)
        self.assertAlmostEqual(
            features["tables_1_3_share"] + features["tables_4_share"]
            + features["tables_5_share"] + features["tables_6p_share"], 1.0, places=12)
        self.assertEqual(features["tables_4_share"], 1.0)
        self.assertEqual(features["tables_1_3_share"], 0.0)

    def test_one_to_three_table_share_is_reported(self):
        # Real supports contain 1-3 table queries; the four shares must still sum
        # to one, which the earlier three-share version silently violated.
        tables = np.array([2, 3, 4, 5, 6, 7, 4, 5, 6, 7,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5])
        lengths = np.ones((20, 4), dtype=np.int64)
        lengths[:, 2] = tables
        store = FakeStore(tables, np.linspace(1, 90, 20),
                          np.linspace(1, 90, 20) * 1.1, lengths, np.zeros(20))
        population = stage_a.population_context(store)
        features = stage_a.support_features(store, list(range(20)), population)
        self.assertAlmostEqual(features["tables_1_3_share"], 2 / 20, places=12)
        self.assertAlmostEqual(
            features["tables_1_3_share"] + features["tables_4_share"]
            + features["tables_5_share"] + features["tables_6p_share"], 1.0, places=12)

    def test_tercile_shares_are_distributions(self):
        store = simple_store()
        population = stage_a.population_context(store)
        features = stage_a.support_features(store, list(range(0, 60, 3)), population)
        for key in ("true_tercile_share", "pg_dev_tercile_share"):
            self.assertEqual(len(features[key]), 3)
            self.assertAlmostEqual(sum(features[key]), 1.0, places=12)

    def test_group_length_and_cartesian_means_match_manual_values(self):
        n = 50
        tables = np.full(n, 4)
        lengths = np.ones((n, 4), dtype=np.int64)
        lengths[:, 0] = 2  # group 0 join/filter tokens
        lengths[:, 3] = 2  # group 3 filter tokens; mean is 2 per row
        lengths[:, 2] = tables
        cartesian = np.asarray([0.0 if index % 2 == 0 else 3.0
                                for index in range(n)], dtype=np.float64)
        # Keep every row low4-eligible so the population terciles are defined.
        store = FakeStore(tables, np.linspace(1, 90, n),
                          np.linspace(1, 90, n) * 1.1, lengths, cartesian)
        population = stage_a.population_context(store)
        features = stage_a.support_features(store, list(range(20)), population)
        self.assertAlmostEqual(features["group_len_mean"], 2.0, places=12)
        # FakeStore writes the cartesian value into every group-2 token of the
        # row, and a 4-table row holds 4 such tokens.
        expected = float(np.mean([4 * (np.log1p(0.0) if index % 2 == 0 else np.log1p(3.0))
                                  for index in range(20)]))
        # The cache stores features as float32, so compare at float32 precision.
        self.assertAlmostEqual(features["log_cartesian_mean"], expected, places=6)

    def test_low4_share_uses_the_shared_definition(self):
        store = simple_store()
        population = stage_a.population_context(store)
        support = list(range(20))
        features = stage_a.support_features(store, support, population)
        expected = float(stage_a.low4_mask(
            np.asarray(store.cards[support, 0]),
            store.tables()[support]).mean())
        self.assertAlmostEqual(features["low4_share"], expected, places=12)

    def test_population_terciles_are_database_wide(self):
        store = simple_store()
        population = stage_a.population_context(store)
        tables = np.diff(store.offsets[:, 2])
        mask = stage_a.low4_mask(store.cards[:, 0], tables)
        low, high = stage_a.tercile_thresholds(store.cards[mask, 0])
        self.assertEqual(population["true_terciles"], (low, high))

    def test_pg_over_share_counts_overestimates(self):
        tables = np.full(20, 4)
        true = np.full(20, 10.0)
        pg = np.asarray([5.0] * 7 + [15.0] * 13)
        store = FakeStore(tables, true, pg, np.ones((20, 4), dtype=np.int64),
                          np.zeros(20))
        population = stage_a.population_context(store)
        features = stage_a.support_features(store, list(range(20)), population)
        self.assertAlmostEqual(features["pg_over_share"], 13 / 20, places=12)

    def test_invalid_support_size_is_rejected(self):
        store = simple_store()
        population = stage_a.population_context(store)
        with self.assertRaises(ValueError):
            stage_a.support_features(store, list(range(19)), population)
        with self.assertRaises(ValueError):
            stage_a.support_features(store, [0] * 20, population)

    def test_guard_rejects_target_holdout_and_unknown_databases(self):
        for name in stage_a.FORBIDDEN:
            with self.assertRaises(ValueError):
                stage_a.guard_database(name)
        for name in stage_a.EXCLUDED:
            with self.assertRaises(ValueError):
                stage_a.guard_database(name)
        with self.assertRaises(ValueError):
            stage_a.guard_database("accidents")
        for name in stage_a.DATABASES:
            stage_a.guard_database(name)


class StatisticsTests(unittest.TestCase):
    def test_spearman_matches_perfect_orders(self):
        left = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        right = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertAlmostEqual(stage_a.spearman(left, right), 1.0, places=12)
        self.assertAlmostEqual(stage_a.spearman(left, -right), -1.0, places=12)

    def test_spearman_matches_the_rank_correlation_formula(self):
        rng = np.random.default_rng(3)
        left = rng.normal(size=40)
        right = left * 0.5 + rng.normal(size=40)
        expected = np.corrcoef(stage_a.rank(left), stage_a.rank(right))[0, 1]
        self.assertAlmostEqual(stage_a.spearman(left, right), expected, places=12)

    def test_rank_shares_ties(self):
        self.assertEqual(stage_a.rank(np.array([5.0, 5.0, 9.0])).tolist(),
                         [1.5, 1.5, 3.0])

    def test_spearman_rejects_nonfinite_and_degenerate_input(self):
        with self.assertRaises(ValueError):
            stage_a.spearman(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]))
        self.assertIsNone(stage_a.spearman(np.ones(5), np.arange(5.0)))

    def test_distribution_percentiles_are_ordered(self):
        summary = stage_a.distribution(list(range(1, 51)))
        self.assertEqual(summary["count"], 50)
        ordered = [summary["min"], summary["p5"], summary["p25"], summary["p50"],
                   summary["p75"], summary["p95"], summary["max"]]
        self.assertEqual(ordered, sorted(ordered))
        self.assertEqual(summary["p50"], 25.5)

    def test_distribution_ignores_none_and_reports_empty(self):
        self.assertEqual(stage_a.distribution([])["count"], 0)
        summary = stage_a.distribution([None, 0.5, 0.5, 1.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["distinct_values"], 2)

    def test_percentile_position_counts_the_sample_at_or_below(self):
        self.assertAlmostEqual(stage_a.percentile_position(2.0, [1.0, 2.0, 3.0, 4.0]),
                               50.0, places=12)
        self.assertIsNone(stage_a.percentile_position(None, [1.0, 2.0]))

    def test_bootstrap_is_reproducible_with_a_fixed_seed(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        first = stage_a.bootstrap_repetitions(values)
        again = stage_a.bootstrap_repetitions(values)
        self.assertEqual(first, again)
        self.assertIn(first["status"], {"reached", "not_reached_within_bound"})
        if first["status"] == "reached":
            self.assertGreaterEqual(first["coverage"], stage_a.BOOTSTRAP_CONFIDENCE)

    def test_bootstrap_marks_inadequate_samples(self):
        self.assertEqual(stage_a.bootstrap_repetitions([0.5])["status"],
                         "insufficient_sample")
        self.assertEqual(stage_a.bootstrap_repetitions([0.0, 0.0, 0.0])["status"],
                         "zero_mean")


class VerdictTests(unittest.TestCase):
    def correlations(self, values):
        return {"spearman": {
            "better_share": {"tables_4_share": values[0],
                             "low4_share": values[1]},
            "q95_ratio": {"tables_4_share": values[2] if len(values) > 2 else None,
                          "low4_share": values[3] if len(values) > 3 else None}}}

    def test_thresholds_split_the_three_bands(self):
        self.assertEqual(stage_a.spearman_verdict(self.correlations([0.9, 0.1]))[0],
                         "predictable_by_support_composition")
        self.assertEqual(stage_a.spearman_verdict(self.correlations([0.49, -0.2]))[0],
                         "not_predictable_from_support")
        self.assertEqual(stage_a.spearman_verdict(self.correlations([0.7, 0.1]))[0],
                         "inconclusive")

    def test_threshold_boundaries(self):
        self.assertEqual(stage_a.spearman_verdict(self.correlations([0.8, 0.0]))[0],
                         "predictable_by_support_composition")
        self.assertEqual(stage_a.spearman_verdict(self.correlations([0.5, 0.0]))[0],
                         "inconclusive")

    def test_no_finite_correlation_is_inconclusive(self):
        self.assertEqual(stage_a.spearman_verdict(self.correlations([None, None]))[0],
                         "inconclusive")

    def test_overall_verdict_requires_every_database(self):
        predictable = {"verdict": {"status": "predictable_by_support_composition"}}
        middle = {"verdict": {"status": "inconclusive"}}
        negative = {"verdict": {"status": "not_predictable_from_support"}}
        self.assertEqual(
            stage_a.overall_verdict({"a": predictable, "b": predictable})["status"],
            "predictable_by_support_composition")
        self.assertEqual(
            stage_a.overall_verdict({"a": predictable, "b": middle})["status"],
            "inconclusive")
        self.assertEqual(
            stage_a.overall_verdict({"a": predictable, "b": negative})["status"],
            "not_predictable_from_support")

    def test_correlation_block_expands_terciles_and_skips_constant_outcomes(self):
        runs = tuple(range(6))
        features = {}
        for run in runs:
            features[run] = {name: float(run) for name in stage_a.FEATURES
                             if name not in ("true_tercile_share",
                                             "pg_dev_tercile_share")}
            features[run]["true_tercile_share"] = [0.2, 0.3, 0.5]
            features[run]["pg_dev_tercile_share"] = [0.1, 0.1, 0.8]
        outcomes = {run: {"better_share": 1.0 / (run + 1), "q95_ratio": 1.0}
                    for run in runs}
        block = stage_a.correlation_block(features, outcomes, tuple(range(5)))
        self.assertEqual(block["n"], 5)
        self.assertIn("tables_4_share", block["spearman"]["better_share"])
        self.assertIn("true_tercile_share_1", block["spearman"]["better_share"])
        self.assertEqual(block["spearman"]["q95_ratio"], {"skipped": "degenerate outcome"})

    def test_n5_and_n50_are_reported_separately(self):
        runs = tuple(range(8))
        features = {}
        for run in runs:
            features[run] = {name: float(run) for name in stage_a.FEATURES
                             if name not in ("true_tercile_share",
                                             "pg_dev_tercile_share")}
            features[run]["true_tercile_share"] = [0.2, 0.3, 0.5]
            features[run]["pg_dev_tercile_share"] = [0.1, 0.1, 0.8]
        outcomes = {run: {"better_share": float(run), "q95_ratio": 2.0 - run}
                    for run in runs}
        self.assertEqual(stage_a.correlation_block(features, outcomes, runs[:5])["n"], 5)
        self.assertEqual(stage_a.correlation_block(features, outcomes, runs)["n"], 8)


class ConsumerCertificationTests(unittest.TestCase):
    """The consumer must re-prove plan-90 invariants, not trust the builder."""

    def setUp(self):
        self.indices = list(range(400))
        self.hashes = ["h%04d" % index for index in self.indices]
        self.entry = planner.draw(self.indices, self.hashes, "chembl", 5)

    def test_a_clean_entry_certifies(self):
        support, query = stage_a.certify_draw_entry(
            self.entry, self.indices, self.hashes, "chembl", 5)
        self.assertEqual(support, self.entry["supports"]["20"])
        self.assertEqual(query, self.entry["query"])

    def test_a_late_support_row_smuggled_into_query_is_rejected(self):
        # Only supports["20"] is used for adaptation, so a query that contains
        # one of support_max[20:100] would otherwise pass unnoticed.
        smuggled = self.entry["support_max"][50]
        broken = copy.deepcopy(self.entry)
        broken["query"] = [smuggled] + broken["query"][:-1]
        broken["query_digest"] = planner.contract.digest(
            [(i, dict(zip(self.indices, self.hashes))[i]) for i in broken["query"]])
        with self.assertRaises(ValueError):
            stage_a.certify_draw_entry(broken, self.indices, self.hashes, "chembl", 5)

    def test_a_tampered_query_digest_is_rejected(self):
        broken = copy.deepcopy(self.entry)
        broken["query_digest"] = "0" * 64
        with self.assertRaises(ValueError):
            stage_a.certify_draw_entry(broken, self.indices, self.hashes, "chembl", 5)

    def test_a_tampered_support_prefix_is_rejected(self):
        broken = copy.deepcopy(self.entry)
        broken["supports"]["20"] = list(reversed(broken["supports"]["20"]))
        with self.assertRaises(ValueError):
            stage_a.certify_draw_entry(broken, self.indices, self.hashes, "chembl", 5)

    def test_short_support_max_is_rejected(self):
        broken = copy.deepcopy(self.entry)
        broken["support_max"] = broken["support_max"][:50]
        with self.assertRaises(ValueError):
            stage_a.certify_draw_entry(broken, self.indices, self.hashes, "chembl", 5)


class PlanLoadingTests(unittest.TestCase):
    def write(self, value):
        path = Path(tempfile.mkdtemp()) / "plan.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_new_plan_entry_is_found_by_run(self):
        path = self.write({"seed_new": 20261019, "support_size": 100,
                           "databases": {"chembl": [{"run": 5, "query": []}]}})
        self.assertEqual(stage_a.load_draw_plan(path, "chembl", 5)["run"], 5)

    def test_plan_header_change_is_rejected(self):
        path = self.write({"seed_new": 1729, "support_size": 100,
                           "databases": {"chembl": [{"run": 5}]}})
        with self.assertRaises(ValueError):
            stage_a.load_draw_plan(path, "chembl", 5)

    def test_missing_run_or_database_is_rejected(self):
        path = self.write({"seed_new": 20261019, "support_size": 100,
                           "databases": {"chembl": [{"run": 5}]}})
        with self.assertRaises(ValueError):
            stage_a.load_draw_plan(path, "chembl", 6)
        with self.assertRaises(ValueError):
            stage_a.load_draw_plan(path, "telstra", 5)

    def test_too_few_runs_is_refused_by_the_guarded_cli(self):
        # Spearman needs three paired points; --runs 2 used to evaluate every
        # cell and only then crash in the summary. Set the CPU environment so
        # the failure comes from the run-count guard, not the earlier checks.
        import os
        from unittest import mock

        env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError) as caught:
                stage_a.main([
                    "--run-dir", "/tmp/x", "--d1", "/tmp/y", "--plan-90", "/tmp/z",
                    "--cache-root", "/tmp/c", "--out", "/tmp/o", "--runs", "2"])
        self.assertIn("--runs must be at least 3", str(caught.exception))

    def test_protocol_81_plan_entry_rejects_out_of_range_runs(self):
        plan = {"chembl": [{"run": run} for run in range(5)]}
        self.assertEqual(stage_a.plan_entry(plan, "chembl", 4)["run"], 4)
        with self.assertRaises(ValueError):
            stage_a.plan_entry(plan, "chembl", 5)


class D1ReferenceTests(unittest.TestCase):
    def test_reference_is_keyed_by_database_and_run(self):
        path = Path(tempfile.mkdtemp()) / "d1.json"
        path.write_text(json.dumps({"databases": {"chembl": [
            {"run": 0, "low4_count": 10, "query_count": 20,
             "paired_low4": {"better_share": 0.25},
             "q95_ratio_low4": {"q95_ratio": 1.5}}]}}), encoding="utf-8")
        reference = stage_a.load_d1_reference(path)
        self.assertEqual(reference[("chembl", 0)],
                         {"better_share": 0.25, "q95_ratio": 1.5,
                          "low4_count": 10, "query_count": 20})

    def test_reference_rejects_a_forbidden_database(self):
        path = Path(tempfile.mkdtemp()) / "d1.json"
        path.write_text(json.dumps({"databases": {"imdb": []}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            stage_a.load_d1_reference(path)


if __name__ == "__main__":
    unittest.main()
