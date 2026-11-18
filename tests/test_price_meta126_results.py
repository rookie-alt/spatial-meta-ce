import importlib
import math
import unittest

DATABASES = ("imdb", "stats", "ergastf1", "genome")
FEWSHOT_DATABASES = ("stats", "ergastf1", "genome")
SEEDS = (42, 43, 44)
SEEDS_STR = ("42", "43", "44")
DENOMINATORS = {"imdb": 521, "stats": 2603, "ergastf1": 1258, "genome": 1699}

# Per-cell means of the five support draws, by arm/stream/database.
K20_MEAN = {
    "M": {
        "imdb": {42: 6.0, 43: 6.3, 44: 6.8},
        "stats": {42: 4.0, 43: 4.4, 44: 5.0},
        "ergastf1": {42: 3.2, 43: 3.6, 44: 4.4},
        "genome": {42: 5.0, 43: 5.2, 44: 6.0},
    },
    "E": {
        "imdb": {42: 9.0, 43: 9.4, 44: 10.0},
        "stats": {42: 6.0, 43: 6.6, 44: 7.8},
        "ergastf1": {42: 5.6, 43: 6.0, 44: 7.2},
        "genome": {42: 7.5, 43: 8.0, 44: 9.0},
    },
    "P": {
        "imdb": {0: 7.0},
        "stats": {0: 5.0},
        "ergastf1": {0: 5.0},
        "genome": {0: 5.0},
    },
}
K5_MEAN = {
    "M": {
        "imdb": {42: 7.0, 43: 7.3, 44: 7.8},
        "stats": {42: 5.0, 43: 5.3, 44: 5.8},
        "ergastf1": {42: 4.0, 43: 4.3, 44: 5.0},
        "genome": {42: 6.0, 43: 6.3, 44: 7.0},
    },
    "E": {
        "imdb": {42: 10.0, 43: 10.4, 44: 11.0},
        "stats": {42: 7.0, 43: 7.5, 44: 8.5},
        "ergastf1": {42: 6.0, 43: 6.4, 44: 7.5},
        "genome": {42: 8.0, 43: 8.5, 44: 9.5},
    },
    "P": {
        "imdb": {0: 8.0},
        "stats": {0: 5.5},
        "ergastf1": {0: 5.5},
        "genome": {0: 5.5},
    },
}
K0_VALUE = {
    "P": {"imdb": 10.0, "stats": 12.0, "ergastf1": 14.0, "genome": 16.0},
    "E": {
        "imdb": {42: 9.4, 43: 9.6, 44: 9.8},
        "stats": {42: 11.4, 43: 11.6, 44: 11.8},
        "ergastf1": {42: 13.4, 43: 13.6, 44: 13.8},
        "genome": {42: 15.4, 43: 15.6, 44: 15.8},
    },
    "M": {
        "imdb": {42: 9.6, 43: 9.8, 44: 10.0},
        "stats": {42: 11.6, 43: 11.8, 44: 12.0},
        "ergastf1": {42: 13.6, 43: 13.8, 44: 14.0},
        "genome": {42: 15.6, 43: 15.8, 44: 16.0},
    },
    "PG": {"imdb": 15.0, "stats": 16.0, "ergastf1": 17.0, "genome": 18.0},
}
RUN_MULTIPLIER = {
    ("M", 20): (0.7, 0.9, 1.0, 1.1, 1.3),
    ("E", 20): (0.95, 0.98, 1.0, 1.02, 1.05),
    ("P", 20): (1.0, 1.0, 1.0, 1.0, 1.0),
    ("M", 5): (0.8, 0.9, 1.0, 1.05, 1.25),
    ("E", 5): (0.96, 0.99, 1.0, 1.01, 1.04),
    ("P", 5): (1.0, 1.0, 1.0, 1.0, 1.0),
}


def q95_value(arm, stream, database, k, run):
    if k == 0:
        value = K0_VALUE[arm][database]
        if isinstance(value, dict):
            return value[stream]
        return value
    mean = (K20_MEAN if k == 20 else K5_MEAN)[arm][database][stream]
    return mean * RUN_MULTIPLIER[(arm, k)][run]


def make_unit(arm, stream, database, k, run, q95=None, n=None):
    value = q95_value(arm, stream, database, k, run) if q95 is None else q95
    return {
        "arm": arm,
        "stream": stream,
        "database": database,
        "k": k,
        "run": run,
        "lr": 0.0 if k == 0 else 1e-6,
        "n": DENOMINATORS[database] if n is None else n,
        "status": "completed",
        "metrics": {
            "q50": value / 2.0,
            "q90": value * 0.8,
            "q95": value,
            "q99": value * 1.5,
        },
    }


def full_matrix():
    units = []
    for arm, streams in (("P", (0,)), ("E", SEEDS), ("M", SEEDS)):
        for stream in streams:
            for database in DATABASES:
                units.append(make_unit(arm, stream, database, 0, 0))
                if database in FEWSHOT_DATABASES:
                    for k in (5, 20):
                        for run in range(5):
                            units.append(make_unit(arm, stream, database, k, run))
    for database in DATABASES:
        units.append(make_unit("PG", 0, database, 0, 0))
    return units


def draw_mean(arm, seed, database, k):
    return sum(q95_value(arm, seed, database, k, run) for run in range(5)) / 5


def expected_main_ratio():
    """Independent hand computation: means over draws, then logs, then seeds."""
    per_stream = []
    for seed in SEEDS:
        total = 0.0
        for database in FEWSHOT_DATABASES:
            total += math.log(draw_mean("M", seed, database, 20)) - math.log(
                draw_mean("E", seed, database, 20)
            )
        per_stream.append(total / len(FEWSHOT_DATABASES))
    return math.exp(sum(per_stream) / len(per_stream))


def pooled_ratio_of_means():
    """Rejected alternative: pool every cell before taking the log."""
    m_values = [
        draw_mean("M", seed, database, 20)
        for seed in SEEDS
        for database in FEWSHOT_DATABASES
    ]
    e_values = [
        draw_mean("E", seed, database, 20)
        for seed in SEEDS
        for database in FEWSHOT_DATABASES
    ]
    return (sum(m_values) / len(m_values)) / (sum(e_values) / len(e_values))


def logs_before_draw_mean():
    """Rejected alternative: log every draw, then average the logs."""
    per_stream = []
    for seed in SEEDS:
        total = 0.0
        for database in FEWSHOT_DATABASES:
            m = (
                sum(
                    math.log(q95_value("M", seed, database, 20, run))
                    for run in range(5)
                )
                / 5
            )
            e = (
                sum(
                    math.log(q95_value("E", seed, database, 20, run))
                    for run in range(5)
                )
                / 5
            )
            total += m - e
        per_stream.append(total / len(FEWSHOT_DATABASES))
    return math.exp(sum(per_stream) / len(per_stream))


def expected_k0_ratio(numerator_arm, denominator_arm):
    per_database = []
    for database in DATABASES:
        numerator_seeds = SEEDS if numerator_arm not in ("P", "PG") else (0,)
        numerator = sum(
            q95_value(numerator_arm, seed, database, 0, 0) for seed in numerator_seeds
        ) / len(numerator_seeds)
        if denominator_arm in ("P", "PG"):
            denominator = q95_value(denominator_arm, 0, database, 0, 0)
        else:
            denominator = sum(
                q95_value(denominator_arm, seed, database, 0, 0) for seed in SEEDS
            ) / len(SEEDS)
        per_database.append(numerator / denominator)
    return math.exp(sum(math.log(v) for v in per_database) / len(per_database))


class ResultTests(unittest.TestCase):
    def module(self):
        return importlib.import_module("spatial_meta_ce.price_meta126_results")

    def test_complete_matrix_aggregates_with_full_cell_count_and_coverage(self):
        module = self.module()
        units = full_matrix()
        self.assertEqual(len(units), 242)
        result = module.aggregate(units, DENOMINATORS)
        self.assertEqual(result["coverage"]["cells_expected"], 242)
        self.assertEqual(result["coverage"]["cells_present"], 242)
        self.assertEqual(result["coverage"]["fewshot_coverage"], "3/4")
        self.assertEqual(result["coverage"]["k0_coverage"], "4/4")
        self.assertEqual(result["coverage"]["imdb_fewshot"], "not_evaluated")
        self.assertEqual(result["denominators"], DENOMINATORS)
        self.assertEqual(
            set(result["per_database"]["stats"]["P"]["q95"]), {"0", "5", "20"}
        )
        self.assertEqual(set(result["per_database"]["stats"]["PG"]["q95"]), {"0"})
        self.assertEqual(
            sorted(result["per_database"]["genome"]["E"]["streams"]),
            list(SEEDS_STR),
        )
        self.assertEqual(
            len(
                result["per_database"]["genome"]["E"]["streams"]["42"]["20"]["q95_runs"]
            ),
            5,
        )

    def test_primary_ratio_is_paired_mean_then_log_and_never_pooled(self):
        module = self.module()
        result = module.aggregate(full_matrix(), DENOMINATORS)
        ratio = result["primary_metric"]["ratio"]
        self.assertAlmostEqual(ratio, expected_main_ratio(), places=10)
        self.assertNotAlmostEqual(ratio, pooled_ratio_of_means(), places=6)
        self.assertNotAlmostEqual(ratio, logs_before_draw_mean(), places=6)
        self.assertEqual(len(result["primary_metric"]["per_stream_deltas"]), 3)

    def test_k0_headline_ratios_are_geometric_means_over_four_databases(self):
        module = self.module()
        result = module.aggregate(full_matrix(), DENOMINATORS)
        ratios = result["k0_geometric_mean"]
        self.assertAlmostEqual(
            ratios["M0/P0"]["ratio"], expected_k0_ratio("M", "P"), places=10
        )
        self.assertAlmostEqual(
            ratios["E0/P0"]["ratio"], expected_k0_ratio("E", "P"), places=10
        )
        self.assertAlmostEqual(
            ratios["M0/E0"]["ratio"], expected_k0_ratio("M", "E"), places=10
        )
        self.assertAlmostEqual(
            result["pg_reference"]["M0/PG0"]["ratio"],
            expected_k0_ratio("M", "PG"),
            places=10,
        )
        self.assertAlmostEqual(
            result["pg_reference"]["P0/PG0"]["ratio"],
            expected_k0_ratio("P", "PG"),
            places=10,
        )
        self.assertEqual(sorted(ratios["M0/E0"]["per_database"]), sorted(DATABASES))
        three_database = math.exp(
            sum(
                math.log(ratios["M0/E0"]["per_database"][database])
                for database in FEWSHOT_DATABASES
            )
            / len(FEWSHOT_DATABASES)
        )
        self.assertNotAlmostEqual(ratios["M0/E0"]["ratio"], three_database, places=6)

    def test_per_database_protection_keeps_q95_and_q50_numbers(self):
        module = self.module()
        result = module.aggregate(full_matrix(), DENOMINATORS)
        protection = result["per_database_protection"]
        self.assertIn("stats/q95/M20/E20", protection)
        self.assertIn("imdb/q50/M0/P0", protection)
        self.assertIn("genome/q95/M20/M0", protection)
        self.assertNotIn("imdb/q95/M20/E20", protection)
        self.assertNotIn("imdb/q95/M0/PG0", protection)
        self.assertIn("imdb", result["pg_reference"]["M0/PG0"]["per_database"])
        for key, value in protection.items():
            self.assertTrue(math.isfinite(value), key)
            self.assertGreater(value, 0.0, key)
        self.assertIsInstance(result["gates"]["per_database"], bool)

    def test_k0_protection_gate_and_m20_against_own_k0_cells(self):
        module = self.module()
        result = module.aggregate(full_matrix(), DENOMINATORS)
        self.assertTrue(result["gates"]["k0_protect"])
        protection = result["k0_protection"]
        self.assertEqual(protection["by_seed_database"]["total"], 9)
        self.assertEqual(protection["by_seed_database"]["improved"], 9)
        self.assertEqual(protection["by_block"]["total"], 45)
        self.assertEqual(protection["by_block"]["worsened"], 0)

    def test_k0_protection_counts_cells_and_reports_degraded_ones(self):
        module = self.module()
        units = full_matrix()
        for unit in units:
            if unit["arm"] == "M" and unit["k"] == 20 and unit["database"] == "genome":
                unit["metrics"]["q95"] = 900.0
                unit["metrics"]["q50"] = 450.0
                unit["metrics"]["q90"] = 720.0
                unit["metrics"]["q99"] = 1350.0
        result = module.aggregate(units, DENOMINATORS)
        protection = result["k0_protection"]
        self.assertEqual(protection["by_seed_database"]["total"], 9)
        self.assertEqual(protection["by_seed_database"]["improved"], 6)
        self.assertEqual(protection["by_seed_database"]["degraded"], 3)
        self.assertEqual(protection["by_block"]["total"], 45)
        self.assertEqual(protection["by_block"]["worsened"], 15)
        self.assertTrue(result["gates"]["k0_protect"])
        self.assertFalse(result["gates"]["relative_gain"])

    def test_single_pretraining_model_is_not_duplicated_into_independent_models(self):
        module = self.module()
        result = module.aggregate(full_matrix(), DENOMINATORS)
        p_streams = result["per_database"]["stats"]["P"]["streams"]
        self.assertEqual(sorted(p_streams), ["0"])
        self.assertNotIn("across_seeds_q95", result["per_database"]["stats"]["P"])
        self.assertIn("across_seeds_q95", result["per_database"]["stats"]["E"])
        self.assertIn(
            "not independent pretrained models",
            " ".join(result["notes"]),
        )

    def test_decision_is_target_scoped_and_never_claims_four_database_fewshot(self):
        module = self.module()
        result = module.aggregate(full_matrix(), DENOMINATORS)
        self.assertEqual(result["decision"], "target_fewshot_gain")
        notes = " ".join(result["notes"])
        self.assertIn("3 of 4", notes)
        self.assertIn("no target-side hyper-parameter selection", notes)
        self.assertIn("not replicated", notes)
        self.assertEqual(result["selection"]["alpha"], 1e-6)
        self.assertEqual(result["selection"]["target_side_selection"], "none")

    def test_missing_cell_is_rejected(self):
        module = self.module()
        units = full_matrix()
        with self.assertRaises(ValueError):
            module.aggregate(units[:-1], DENOMINATORS)

    def test_duplicate_cell_is_rejected(self):
        module = self.module()
        units = full_matrix()
        with self.assertRaises(ValueError):
            module.aggregate(units + [units[0]], DENOMINATORS)

    def test_nonfinite_and_missing_metrics_are_rejected(self):
        module = self.module()
        units = full_matrix()
        units[0]["metrics"]["q95"] = float("nan")
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)
        units = full_matrix()
        del units[0]["metrics"]["q99"]
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)
        units = full_matrix()
        units[0]["metrics"]["q50"] = float("inf")
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)

    def test_denominator_drift_is_rejected(self):
        module = self.module()
        units = full_matrix()
        units[3]["n"] = DENOMINATORS["imdb"] - 1
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)
        units = full_matrix()
        with self.assertRaises(ValueError):
            module.aggregate(units, dict(DENOMINATORS, stats=2604))

    def test_failed_cell_is_rejected_and_never_filtered(self):
        module = self.module()
        units = full_matrix()
        units[0]["status"] = "failed"
        units[0]["error"] = "nonfinite"
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)

    def test_unexpected_imdb_fewshot_cell_is_rejected(self):
        module = self.module()
        units = full_matrix()
        units.append(make_unit("M", 42, "imdb", 20, 0))
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)

    def test_invalid_cell_identity_is_rejected(self):
        module = self.module()
        for mutate in (
            lambda u: u.update(arm="PG2"),
            lambda u: u.update(stream=7),
            lambda u: u.update(database="chembl"),
            lambda u: u.update(k=10),
            lambda u: u.update(run=9),
            lambda u: u.update(n=0),
        ):
            units = full_matrix()
            mutate(units[0])
            with self.assertRaises(ValueError):
                module.aggregate(units, DENOMINATORS)

    def test_learning_rate_must_match_the_fixed_alpha(self):
        module = self.module()
        units = full_matrix()
        for unit in units:
            if unit["k"] == 20 and unit["arm"] == "M":
                unit["lr"] = 1e-5
                break
        with self.assertRaises(ValueError):
            module.aggregate(units, DENOMINATORS)

    def test_no_gain_case_is_not_reported_as_gain(self):
        module = self.module()
        units = full_matrix()
        for unit in units:
            if unit["arm"] == "M" and unit["k"] == 20:
                unit["metrics"]["q95"] = 12.0
                unit["metrics"]["q50"] = 6.0
                unit["metrics"]["q90"] = 9.6
                unit["metrics"]["q99"] = 18.0
        result = module.aggregate(units, DENOMINATORS)
        self.assertEqual(result["decision"], "no_target_gain")
        self.assertFalse(result["gates"]["relative_gain"])


if __name__ == "__main__":
    unittest.main()
