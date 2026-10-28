import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "scripts/target_three_seed_summary.py"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_SPEC = importlib.util.spec_from_file_location("target_three_seed_summary", _SOURCE)
summary_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(summary_module)

DATABASES = ("stats", "genome", "ergastf1", "imdb")
SEEDS = (42, 43, 44)


def seed_payload(
    seed, ours_q95, price_q95=10.0, pg_q95=100.0, ours_q50=1.5, databases=DATABASES
):
    """Build a synthetic per-seed summary payload.

    ``price_q95`` and ``pg_q95`` default to the same value in every seed,
    which is what the real runs produce: PRICE ships one checkpoint and the
    PostgreSQL estimate is fixed.
    """
    payload_databases = {}
    for database in databases:
        payload_databases[database] = {
            "n": 100,
            "ours_k0": {"q95": ours_q95[database], "q50": ours_q50},
            "price_k0": {"q95": price_q95, "q50": 1.1},
            "pg": {"q95": pg_q95, "q50": 1.2},
            "bucket_4plus": {
                "n": 40,
                "ours_k0": {"q95": ours_q95[database] * 2},
                "price_k0": {"q95": price_q95 * 2},
                "pg": {"q95": pg_q95 * 2},
            },
        }
    return {
        "target_databases_read": list(databases),
        "checkpoint": {"seed": seed},
        "databases": payload_databases,
    }


def protocol_payload(seed):
    return {
        "metric": "unique-query population Q95 (single pass, NOT episode mean)",
        "metric_definitions": {"q95": "percentile of the multiplier error"},
        "estimators": {"ours_k0": "frozen checkpoint, zero-shot"},
        "checkpoint": {
            "path": f"/runs/seed{seed}/last.pt",
            "sha256": f"{seed:064x}",
            "seed": seed,
            "method": "erm",
            "train_databases": ["accidents"],
            "validation_databases": ["chembl"],
            "model_config": {"arch": "sum"},
            "transform": "price40-group-logfanout-v1",
        },
        "price": {"checkpoint": "/price/model_params.pth", "maxima": [12, 14, 8, 10]},
        "inputs": {"split": "test", "caches": {}},
    }


def write_seed_files(
    directory, ours_by_seed, price_q95=10.0, pg_q95=100.0, databases=DATABASES
):
    """Write the six per-seed files the aggregator consumes; return their paths."""
    summary_paths = []
    protocol_paths = []
    for seed in SEEDS:
        summary_path = Path(directory) / f"seed{seed}-summary.json"
        protocol_path = Path(directory) / f"seed{seed}-protocol.json"
        summary_path.write_text(
            json.dumps(
                seed_payload(
                    seed,
                    ours_by_seed[seed],
                    price_q95,
                    pg_q95,
                    databases=databases,
                )
            )
        )
        protocol_path.write_text(json.dumps(protocol_payload(seed)))
        summary_paths.append(str(summary_path))
        protocol_paths.append(str(protocol_path))
    return summary_paths, protocol_paths


class TargetThreeSeedSummaryTests(unittest.TestCase):
    def run_aggregator(
        self,
        ours_by_seed,
        price_q95=10.0,
        pg_q95=100.0,
        databases=DATABASES,
        audit=None,
    ):
        with tempfile.TemporaryDirectory() as directory:
            summary_paths, protocol_paths = write_seed_files(
                directory, ours_by_seed, price_q95, pg_q95, databases
            )
            argv = []
            for path in summary_paths:
                argv += ["--seed-summary", path]
            for path in protocol_paths:
                argv += ["--seed-protocol", path]
            output = Path(directory) / "summary.json"
            protocol = Path(directory) / "protocol.json"
            argv += ["--output", str(output), "--protocol", str(protocol)]
            argv += ["--databases", *databases]
            if audit is not None:
                argv += ["--audit", audit]
            summary_module.main(argv)
            return json.loads(output.read_text()), json.loads(protocol.read_text())

    def test_reports_mean_and_sample_sd_over_seeds(self):
        # seed 42 -> 10, 43 -> 11, 44 -> 12 (seed % 3 is 0/1/2)
        ours = {
            seed: {database: 10.0 + seed % 3 for database in DATABASES}
            for seed in SEEDS
        }
        summary, _ = self.run_aggregator(ours)
        stats = summary["databases"]["stats"]["ours_k0"]["q95"]
        self.assertEqual(stats["n"], 3)
        self.assertAlmostEqual(stats["mean"], 11.0)
        # sample SD with ddof=1 over (10, 11, 12)
        self.assertAlmostEqual(stats["std"], 1.0)
        self.assertAlmostEqual(stats["min"], 10.0)
        self.assertAlmostEqual(stats["max"], 12.0)

    def test_never_reports_a_best_of_seeds_value_as_the_central_number(self):
        ours = {seed: {database: 10.0 for database in DATABASES} for seed in SEEDS}
        ours[42] = {database: 99.0 for database in DATABASES}
        summary, _ = self.run_aggregator(ours)
        stats = summary["databases"]["stats"]["ours_k0"]["q95"]
        self.assertAlmostEqual(stats["mean"], 39.666666666666664)
        self.assertNotAlmostEqual(stats["mean"], 99.0)

    def test_counts_seeds_beating_price_per_database(self):
        ours = {seed: {database: 20.0 for database in DATABASES} for seed in SEEDS}
        # One seed on one database is good enough to beat PRICE there.
        ours[44] = dict(ours[44])
        ours[44]["stats"] = 5.0
        summary, _ = self.run_aggregator(ours, price_q95=10.0)
        self.assertEqual(summary["databases"]["stats"]["seeds_beating_price_on_q95"], 1)
        self.assertEqual(
            summary["databases"]["genome"]["seeds_beating_price_on_q95"], 0
        )
        self.assertFalse(summary["headline"]["every_seed_loses_to_price_on_q95"])

    def test_headline_counts_every_seed_database_pair(self):
        ours = {seed: {database: 20.0 for database in DATABASES} for seed in SEEDS}
        summary, _ = self.run_aggregator(ours, price_q95=10.0, pg_q95=100.0)
        headline = summary["headline"]
        self.assertEqual(headline["total_seed_database_pairs"], 12)
        self.assertEqual(headline["seed_database_pairs_beating_price"], 0)
        self.assertEqual(headline["seed_database_pairs_beating_pg"], 12)
        self.assertTrue(headline["every_seed_loses_to_price_on_q95"])

    def test_refuses_to_aggregate_when_price_moves_between_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_paths, protocol_paths = write_seed_files(
                directory, {seed: {db: 20.0 for db in DATABASES} for seed in SEEDS}
            )
            # Simulate a corrupted run: PRICE's Q95 differs in one seed.
            payload = json.loads(Path(summary_paths[1]).read_text())
            payload["databases"]["stats"]["price_k0"]["q95"] = 11.0
            Path(summary_paths[1]).write_text(json.dumps(payload))

            argv = []
            for path in summary_paths:
                argv += ["--seed-summary", path]
            for path in protocol_paths:
                argv += ["--seed-protocol", path]
            argv += [
                "--output",
                str(Path(directory) / "summary.json"),
                "--protocol",
                str(Path(directory) / "protocol.json"),
            ]
            with self.assertRaises(ValueError) as caught:
                summary_module.main(argv)
            self.assertIn("differs across seeds", str(caught.exception))

    def test_refuses_to_aggregate_when_pg_moves_between_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_paths, protocol_paths = write_seed_files(
                directory, {seed: {db: 20.0 for db in DATABASES} for seed in SEEDS}
            )
            payload = json.loads(Path(summary_paths[2]).read_text())
            payload["databases"]["imdb"]["pg"]["q95"] = 101.0
            Path(summary_paths[2]).write_text(json.dumps(payload))

            argv = []
            for path in summary_paths:
                argv += ["--seed-summary", path]
            for path in protocol_paths:
                argv += ["--seed-protocol", path]
            argv += [
                "--output",
                str(Path(directory) / "summary.json"),
                "--protocol",
                str(Path(directory) / "protocol.json"),
            ]
            with self.assertRaises(ValueError):
                summary_module.main(argv)

    def test_rejects_a_summary_whose_checkpoint_seed_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_paths, protocol_paths = write_seed_files(
                directory, {seed: {db: 20.0 for db in DATABASES} for seed in SEEDS}
            )
            payload = json.loads(Path(summary_paths[0]).read_text())
            payload["checkpoint"]["seed"] = 999
            Path(summary_paths[0]).write_text(json.dumps(payload))

            argv = []
            for path in summary_paths:
                argv += ["--seed-summary", path]
            for path in protocol_paths:
                argv += ["--seed-protocol", path]
            argv += [
                "--output",
                str(Path(directory) / "summary.json"),
                "--protocol",
                str(Path(directory) / "protocol.json"),
            ]
            with self.assertRaises(ValueError) as caught:
                summary_module.main(argv)
            self.assertIn("expected 42", str(caught.exception))

    def test_rejects_a_summary_that_read_a_different_target_set(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_paths, protocol_paths = write_seed_files(
                directory, {seed: {db: 20.0 for db in DATABASES} for seed in SEEDS}
            )
            payload = json.loads(Path(summary_paths[0]).read_text())
            payload["target_databases_read"] = ["stats", "genome"]
            Path(summary_paths[0]).write_text(json.dumps(payload))

            argv = []
            for path in summary_paths:
                argv += ["--seed-summary", path]
            for path in protocol_paths:
                argv += ["--seed-protocol", path]
            argv += [
                "--output",
                str(Path(directory) / "summary.json"),
                "--protocol",
                str(Path(directory) / "protocol.json"),
            ]
            with self.assertRaises(ValueError) as caught:
                summary_module.main(argv)
            self.assertIn("expected", str(caught.exception))

    def test_requires_exactly_three_seed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_paths, protocol_paths = write_seed_files(
                directory, {seed: {db: 20.0 for db in DATABASES} for seed in SEEDS}
            )
            argv = []
            for path in summary_paths[:2]:
                argv += ["--seed-summary", path]
            for path in protocol_paths:
                argv += ["--seed-protocol", path]
            argv += [
                "--output",
                str(Path(directory) / "summary.json"),
                "--protocol",
                str(Path(directory) / "protocol.json"),
            ]
            with self.assertRaises(ValueError):
                summary_module.main(argv)

    def test_protocol_records_every_checkpoint_hash_and_the_guards(self):
        ours = {seed: {database: 20.0 for database in DATABASES} for seed in SEEDS}
        _, protocol = self.run_aggregator(ours)
        for seed in SEEDS:
            self.assertEqual(protocol["checkpoints"][str(seed)]["seed"], seed)
            self.assertEqual(
                protocol["checkpoints"][str(seed)]["sha256"], f"{seed:064x}"
            )
        self.assertTrue(protocol["comparability"]["price_and_pg_fixed_across_seeds"])
        self.assertEqual(protocol["seeds"], [42, 43, 44])

    def test_four_plus_bucket_is_aggregated_separately(self):
        ours = {seed: {database: 20.0 for database in DATABASES} for seed in SEEDS}
        summary, _ = self.run_aggregator(ours)
        self.assertEqual(summary["bucket_4plus"]["stats"]["n"], 40)
        self.assertAlmostEqual(
            summary["bucket_4plus"]["stats"]["ours_k0"]["q95"]["mean"], 40.0
        )

    def test_databases_and_audit_options_restrict_the_aggregate(self):
        databases = ("stats", "ergastf1", "imdb")
        ours = {seed: {database: 20.0 for database in databases} for seed in SEEDS}
        summary, protocol = self.run_aggregator(
            ours, databases=databases, audit="p2-2-fomaml-sum-target-3seed"
        )
        self.assertEqual(summary["audit"], "p2-2-fomaml-sum-target-3seed")
        self.assertEqual(protocol["audit"], "p2-2-fomaml-sum-target-3seed")
        self.assertEqual(summary["target_databases_read"], list(databases))
        self.assertEqual(protocol["target_databases_read"], list(databases))
        self.assertEqual(list(summary["databases"]), list(databases))
        self.assertEqual(summary["headline"]["total_seed_database_pairs"], 9)
        self.assertFalse(any("genome" in note for note in protocol["limitations"]))


if __name__ == "__main__":
    unittest.main()
