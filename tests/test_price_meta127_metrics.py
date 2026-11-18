import csv
import importlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PINNED_UPSTREAM = (
    REPO_ROOT
    / "reports"
    / "price-meta127-official-eval-20261118"
    / "upstream"
    / "utils"
    / "model"
    / "qerror.py"
)


def _module():
    return importlib.import_module("spatial_meta_ce.price_meta127_metrics")


def _pinned_upstream():
    """Load the pinned upstream ``qerror.py`` as an independent oracle."""
    if not PINNED_UPSTREAM.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "pinned_price_qerror", PINNED_UPSTREAM
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal_upstream_qerrors(encoded, encoded_labels):
    """Literal transcription of the pinned upstream ``utils/model/qerror.py``.

    Deliberately independent of the project module so the test is a real
    cross-check of the upstream expression rather than a restatement of the
    implementation under test.
    """
    preds = np.asarray(encoded).astype(np.float32)
    labels = np.asarray(encoded_labels).astype(np.float32)
    preds = np.exp(preds - 1) - 1
    labels = np.exp(labels - 1) - 1
    preds = preds.astype(np.float64) + 1.0
    labels = labels.astype(np.float64) + 1.0
    q_errors = np.maximum(labels / preds, preds / labels)
    q_errors = np.nan_to_num(q_errors, nan=np.inf)
    return q_errors


def literal_upstream_get_qerror(encoded, encoded_labels, percentile_list):
    """Literal transcription of upstream ``get_qerror`` (rounded to 4 dp)."""
    q_errors = literal_upstream_qerrors(encoded, encoded_labels)
    return [round(float(np.percentile(q_errors, p)), 4) for p in percentile_list]


def literal_standard_log_prediction(encoded):
    """Literal transcription of the 126-protocol decode."""
    a = np.asarray(encoded, dtype=np.float64) - 1
    result = np.zeros_like(a)
    positive = a > np.log(2)
    result[positive] = a[positive] + np.log1p(-np.exp(-a[positive]))
    return result


def literal_standard_qerrors(encoded, true_cards):
    """Literal transcription of the 126-protocol Q-error."""
    log_pred = literal_standard_log_prediction(encoded)
    log_true = np.log(np.maximum(np.asarray(true_cards, dtype=np.float64), 1.0))
    with np.errstate(over="ignore"):
        return np.exp(np.abs(np.maximum(log_pred, 0.0) - log_true))


def float64_rewrite_qerrors(encoded, encoded_labels):
    """The same formula rewritten in float64; upstream is float32, so this
    must not be used as a substitute for the official computation."""
    preds = np.asarray(encoded, dtype=np.float64)
    labels = np.asarray(encoded_labels, dtype=np.float64)
    preds = np.exp(preds - 1) - 1
    labels = np.exp(labels - 1) - 1
    preds = preds.astype(np.float64) + 1.0
    labels = labels.astype(np.float64) + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        q_errors = np.maximum(labels / preds, preds / labels)
    return np.nan_to_num(q_errors, nan=np.inf)


class OfficialQErrorTest(unittest.TestCase):
    def test_matches_literal_upstream_expression(self):
        module = _module()
        rng = np.random.default_rng(20261118)
        encoded = rng.uniform(-4.0, 12.0, 500).astype(np.float64)
        labels = rng.uniform(1.0, 14.0, 500).astype(np.float64)
        expected = literal_upstream_qerrors(encoded, labels)
        actual = module.official_qerrors(encoded, labels)
        self.assertEqual(actual.dtype, np.float64)
        self.assertTrue(np.array_equal(actual, expected))

    def test_official_arithmetic_is_float32(self):
        module = _module()
        encoded = np.array([1.0 + 1e-7, 1.0 + 1e-7, 3.0], dtype=np.float64)
        labels = np.array([np.log(1e6) + 1.0, np.log(1e6) + 1.0, 2.0])
        official = module.official_qerrors(encoded, labels)
        literal = literal_upstream_qerrors(encoded, labels)
        self.assertTrue(np.array_equal(official, literal))
        rewritten = float64_rewrite_qerrors(encoded, labels)
        self.assertFalse(np.allclose(official, rewritten, rtol=0, atol=1e-9))
        self.assertGreater(abs(official[0] - rewritten[0]), 1e-3)

    def test_no_clipping_of_extreme_predictions(self):
        module = _module()
        encoded = np.array([1.0 + 60.0, 1.0 + 60.0], dtype=np.float64)
        labels = np.array([np.log(10.0) + 1.0, np.log(10.0) + 1.0])
        decoded = module.decode_official(encoded)
        # the upstream decode runs in float32, so the reference is the float32
        # exponential and not the float64 one
        reference = float(np.exp(np.float32(60.0)) - np.float32(1.0))
        self.assertAlmostEqual(float(decoded[0]), reference, delta=0.0, places=3)
        clipped = float(np.exp(np.float32(50.0)) - np.float32(1.0))
        self.assertGreater(float(decoded[0]), clipped)
        self.assertGreater(
            float(decoded[0]),
            math.exp(50.0) / 1.0001,
            msg="the decode must not be bounded by exp(50)",
        )
        self.assertTrue(
            np.array_equal(
                module.official_qerrors(encoded, labels),
                literal_upstream_qerrors(encoded, labels),
            )
        )

    def test_percentiles_are_linear_and_rounded_to_four_decimals(self):
        module = _module()
        rng = np.random.default_rng(11)
        encoded = rng.uniform(1.0, 10.0, 101).astype(np.float64)
        labels = rng.uniform(1.0, 10.0, 101).astype(np.float64)
        q = literal_upstream_qerrors(encoded, labels)
        summary = module.summarize_official(encoded, labels)
        for percentile in (50, 80, 90, 95, 99):
            key = f"q{percentile}"
            self.assertEqual(
                summary[key], round(float(np.percentile(q, percentile)), 4)
            )
        manual = float(np.percentile(q, 95, method="linear"))
        self.assertEqual(summary["q95"], round(manual, 4))

    def test_matches_upstream_get_qerror_round_four(self):
        module = _module()
        rng = np.random.default_rng(101)
        encoded = rng.uniform(0.5, 14.0, 777).astype(np.float64)
        labels = rng.uniform(1.0, 14.0, 777).astype(np.float64)
        percentile_list = [30, 50, 80, 90, 95, 99]
        expected = literal_upstream_get_qerror(encoded, labels, percentile_list)
        actual = module.official_percentiles(encoded, labels)
        self.assertEqual(set(actual), {"q50", "q80", "q90", "q95", "q99"})
        for percentile, value in zip((50, 80, 90, 95, 99), expected[1:]):
            self.assertEqual(actual[f"q{percentile}"], value)

    def test_upstream_rounding_overflows_to_infinity_like_numpy(self):
        # upstream calls round() on a numpy float64, which overflows to +inf
        # for magnitudes above ~1e15; the Python built-in round() does not.
        module = _module()
        encoded = np.array([1e16, 1e16, 1e16], dtype=np.float64)
        labels = np.array([3.0, 3.0, 3.0], dtype=np.float64)
        actual = module.official_percentiles(encoded, labels)
        self.assertTrue(np.isinf(actual["q50"]))
        self.assertEqual(actual["q50"], math.inf)
        raw = float(np.percentile(module.official_qerrors(encoded, labels), 50))
        self.assertTrue(np.isfinite(raw))
        self.assertNotEqual(actual["q50"], raw)
        self.assertEqual(round(raw, 4), raw)

    def test_duplicate_rows_are_counted_with_multiplicity(self):
        module = _module()
        # decoded value d is encoded as log(d + 1) + 1 by the upstream scheme
        same = math.log(2.0) + 1.0
        four = math.log(5.0) + 1.0
        encoded = np.array([same, same, same, four], dtype=np.float64)
        labels = np.array([same, same, same, same], dtype=np.float64)
        summary = module.summarize_official(encoded, labels)
        q = literal_upstream_qerrors(encoded, labels)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["q50"], round(float(np.percentile(q, 50)), 4))
        unique_q = np.unique(q)
        self.assertNotEqual(
            summary["q50"], round(float(np.percentile(unique_q, 50)), 4)
        )

    def test_zero_true_cardinalities(self):
        module = _module()
        encoded = np.array([math.log(101.0) + 1.0], dtype=np.float64)
        labels = module.encode_labels_official(np.array([0.0]))
        self.assertEqual(float(labels[0]), 1.0)
        q = module.official_qerrors(encoded, labels)
        expected = literal_upstream_qerrors(encoded, labels)
        self.assertTrue(np.array_equal(q, expected))
        self.assertAlmostEqual(float(q[0]), 101.0, places=4)

    def test_nan_ratio_becomes_positive_infinity(self):
        module = _module()
        encoded = np.array([-1e30], dtype=np.float64)
        labels = np.array([-1e30], dtype=np.float64)
        q = module.official_qerrors(encoded, labels)
        self.assertTrue(np.isinf(q[0]))
        self.assertEqual(float(q[0]), math.inf)

    def test_infinite_ratio_becomes_float64_maximum(self):
        module = _module()
        encoded = np.array([-1e30], dtype=np.float64)
        labels = np.array([np.log(1000.0) + 1.0], dtype=np.float64)
        q = module.official_qerrors(encoded, labels)
        self.assertFalse(np.isinf(q[0]))
        self.assertEqual(float(q[0]), np.finfo(np.float64).max)

    def test_summary_flags_nonfinite_values_and_stays_json_serializable(self):
        module = _module()
        encoded = np.array([-1e30, 2.0], dtype=np.float64)
        labels = np.array([-1e30, 3.0], dtype=np.float64)
        summary = module.summarize_official(encoded, labels)
        self.assertEqual(summary["metadata"]["nonfinite_q"]["inf"], 1)
        self.assertFalse(summary["metadata"]["official_comparable"])
        self.assertFalse(summary["metadata"]["percentiles_finite"])
        text = json.dumps(summary)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)

    def test_summary_stays_json_safe_for_every_nonfinite_case(self):
        module = _module()
        cases = [
            (np.array([1e16, 1e16, 1e16]), np.array([3.0, 3.0, 3.0])),
            (np.array([-1e30, 2.0]), np.array([-1e30, 3.0])),
            (np.array([1e39, 2.0]), np.array([3.0, 3.0])),
            (np.array([200.0, 2.0]), np.array([5.0, 3.0])),
        ]
        for encoded, labels in cases:
            summary = module.summarize_official(encoded, labels)
            text = json.dumps(summary)
            self.assertNotIn("Infinity", text)
            self.assertNotIn("NaN", text)
            self.assertFalse(summary["metadata"]["percentiles_finite"])
            self.assertEqual(
                summary["metadata"]["nonfinite_percentiles"],
                ["q50", "q80", "q90", "q95", "q99"],
            )

    def test_summary_exposes_unrounded_percentiles(self):
        module = _module()
        rng = np.random.default_rng(31)
        encoded = rng.uniform(1.0, 12.0, 300).astype(np.float64)
        labels = rng.uniform(1.0, 12.0, 300).astype(np.float64)
        summary = module.summarize_official(encoded, labels)
        unrounded = summary["unrounded"]
        self.assertEqual(set(unrounded), {"q50", "q80", "q90", "q95", "q99"})
        q = literal_upstream_qerrors(encoded, labels)
        for percentile in (50, 80, 90, 95, 99):
            expected = float(np.percentile(q, percentile))
            self.assertAlmostEqual(unrounded[f"q{percentile}"], expected, places=9)
            self.assertNotEqual(
                unrounded[f"q{percentile}"],
                summary[f"q{percentile}"],
                msg="the rounded display value must differ from the raw one",
            )

    def test_unrounded_mapping_is_json_safe(self):
        module = _module()
        encoded = np.array([-1e30, 2.0], dtype=np.float64)
        labels = np.array([-1e30, 3.0], dtype=np.float64)
        summary = module.summarize_official(encoded, labels)
        text = json.dumps(summary)
        self.assertNotIn("Infinity", text)
        self.assertNotIn("NaN", text)
        self.assertIsNone(summary["unrounded"]["q95"])

    def test_decode_diagnostics_report_overflow_and_zero_denominator(self):
        module = _module()
        # exp overflows to +inf in the float32 decode
        overflow = module.summarize_official(
            np.array([200.0, 2.0]), np.array([3.0, 3.0])
        )
        self.assertEqual(overflow["metadata"]["decode"]["n_overflow"], 1)
        self.assertEqual(overflow["metadata"]["decode"]["n_prediction_overflow"], 1)
        self.assertFalse(overflow["metadata"]["official_comparable"])

        # a vanishing decode yields a zero denominator after the +1 shift
        zero = module.summarize_official(np.array([-1e30, 2.0]), np.array([3.0, 3.0]))
        self.assertGreaterEqual(zero["metadata"]["decode"]["n_zero_denominator"], 1)
        self.assertFalse(zero["metadata"]["official_comparable"])

    def test_official_comparable_is_true_for_a_clean_sample(self):
        module = _module()
        rng = np.random.default_rng(41)
        encoded = rng.uniform(1.0, 12.0, 200).astype(np.float64)
        labels = rng.uniform(1.0, 12.0, 200).astype(np.float64)
        summary = module.summarize_official(encoded, labels)
        self.assertTrue(summary["metadata"]["official_comparable"])
        self.assertTrue(summary["metadata"]["percentiles_finite"])
        self.assertEqual(summary["metadata"]["decode"]["n_overflow"], 0)
        self.assertEqual(summary["metadata"]["decode"]["n_zero_denominator"], 0)

    def test_validation_rejects_invalid_inputs(self):
        module = _module()
        good = np.array([2.0, 3.0])
        with self.assertRaises(ValueError):
            module.official_qerrors(np.array([]), np.array([]))
        with self.assertRaises(ValueError):
            module.official_qerrors(good, np.array([2.0]))
        with self.assertRaises(ValueError):
            module.official_qerrors(np.array([2.0, np.nan]), good)
        with self.assertRaises(ValueError):
            module.official_qerrors(np.array([2.0, np.inf]), good)
        with self.assertRaises(ValueError):
            module.summarize_official(good, np.array([2.0, 3.0, 4.0]))

    def test_float32_overflow_input_follows_upstream_semantics(self):
        module = _module()
        encoded = np.array([1e39, 2.0], dtype=np.float64)
        labels = np.array([3.0, 3.0], dtype=np.float64)
        q = module.official_qerrors(encoded, labels)
        # 1e39 overflows the float32 tensor to +inf; exp(+inf) stays +inf and
        # nan_to_num maps the +inf ratio onto the float64 maximum, exactly as
        # the upstream file does.
        self.assertEqual(float(q[0]), np.finfo(np.float64).max)
        self.assertTrue(np.isfinite(q[0]))
        summary = module.summarize_official(encoded, labels)
        self.assertEqual(summary["metadata"]["float32_overflow_inputs"], 1)
        self.assertEqual(summary["metadata"]["nonfinite_q"]["inf"], 0)
        # the overflow is a degenerate decode, so the sample is not comparable
        self.assertEqual(summary["metadata"]["decode"]["n_overflow"], 1)
        self.assertFalse(summary["metadata"]["official_comparable"])

    def test_standard_companion_matches_the_126_protocol_formula(self):
        module = _module()
        rng = np.random.default_rng(5)
        encoded = rng.uniform(1.0, 12.0, 200).astype(np.float64)
        true_cards = rng.uniform(1.0, 5000.0, 200).astype(np.float64)
        expected = literal_standard_qerrors(encoded, true_cards)
        actual = module.standard_raw_qerrors(encoded, true_cards=true_cards)
        self.assertTrue(np.allclose(actual, expected, rtol=0, atol=0))

    def test_standard_decode_matches_the_project_baseline(self):
        module = _module()
        try:
            from spatial_meta_ce.baselines import price_log_predictions
        except ImportError:  # pragma: no cover - torch is optional here
            self.skipTest("torch is unavailable")
        rng = np.random.default_rng(3)
        encoded = rng.uniform(-2.0, 15.0, 500).astype(np.float64)
        self.assertTrue(
            np.array_equal(
                module.decode_standard_log_prediction(encoded),
                price_log_predictions(encoded),
            )
        )

    def test_encode_labels_matches_the_torch_reference(self):
        module = _module()
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is optional here
            self.skipTest("torch is unavailable")
        true_cards = np.array(
            [0.0, 1.0, 2.0, 10.0, 12345.678, 1e6, 2.0**24], dtype=np.float64
        )
        reference = (
            torch.log(torch.from_numpy(true_cards.astype(np.float32)) + 1) + 1
        ).numpy()
        encoded = module.encode_labels_official(true_cards)
        self.assertEqual(encoded.dtype, np.float32)
        self.assertTrue(np.array_equal(encoded, reference))

    def test_summary_reports_the_uncapped_difference_to_the_standard_metric(self):
        module = _module()
        rng = np.random.default_rng(77)
        encoded = rng.uniform(1.0, 12.0, 120).astype(np.float64)
        true_cards = rng.uniform(1.0, 5000.0, 120).astype(np.float64)
        summary = module.summarize_official(
            encoded,
            module.encode_labels_official(true_cards),
            true_cards=true_cards,
        )
        companion = summary["standard_companion"]
        self.assertEqual(companion["n"], 120)
        self.assertGreater(companion["n_differing"], 0)
        self.assertGreater(companion["max_abs_difference"], 0.0)
        official = module.official_qerrors(
            encoded, module.encode_labels_official(true_cards)
        )
        standard = module.standard_raw_qerrors(encoded, true_cards=true_cards)
        self.assertEqual(
            companion["n_differing"], int(np.sum(np.abs(official - standard) > 1e-12))
        )


class PinnedUpstreamOracleTest(unittest.TestCase):
    """Cross-check against the pinned upstream file itself."""

    def setUp(self):
        self.upstream = _pinned_upstream()
        if self.upstream is None:
            self.skipTest("pinned upstream copy is not present")

    def _compare(self, encoded, labels):
        module = _module()
        predictions = np.asarray(encoded).astype(np.float32)
        truths = np.asarray(labels).astype(np.float32)
        upstream_list = self.upstream.get_qerror(
            predictions,
            truths,
            cuda=False,
            do_scale=True,
            percentile_list=[30, 50, 80, 90, 95, 99],
        )
        mine = module.official_percentiles(encoded, labels)
        for percentile, expected in zip((50, 80, 90, 95, 99), upstream_list[1:]):
            actual = mine[f"q{percentile}"]
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual))
            else:
                self.assertEqual(actual, expected)

    def test_matches_pinned_upstream_on_random_inputs(self):
        rng = np.random.default_rng(20261118)
        for _ in range(4):
            self._compare(rng.uniform(0.5, 14.0, 400), rng.uniform(1.0, 14.0, 400))

    def test_matches_pinned_upstream_on_extreme_inputs(self):
        cases = [
            (np.array([-1e30, 1.0, 200.0]), np.array([-1e30, 3.0, 5.0])),
            (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
            (np.array([-89.0, 89.0, 0.5]), np.array([-89.0, 89.0, 12.0])),
            (np.array([1e39, 2.0]), np.array([3.0, 3.0])),
            (np.array([1e16, 1e15, 2.0]), np.array([3.0, 3.0, 3.0])),
        ]
        for encoded, labels in cases:
            self._compare(encoded, labels)


class VerifierScriptTest(unittest.TestCase):
    """Smoke test of the standalone verifier on a synthetic 127 layout."""

    def setUp(self):
        self.script = (
            REPO_ROOT
            / "reports"
            / "price-meta127-official-eval-20261118"
            / "verify_metrics.py"
        )
        if not self.script.exists():
            self.skipTest("the verifier script is not present")

    def _build_fixture(self, root, rows=10):
        import gzip
        import hashlib
        import pickle

        output = root / "outputs" / "zero"
        output.mkdir(parents=True)
        (root / "inputs").mkdir(parents=True)
        rng = np.random.default_rng(0)
        truth = rng.integers(1, 1000, rows).astype(np.float32)
        pg = rng.integers(1, 1000, rows).astype(np.float32)
        with (root / "features.pkl").open("wb") as stream:
            pickle.dump(
                {"true_cards": truth.tolist(), "pg_est_cards": pg.tolist()}, stream
            )
        identities = [f"hash{index:03d}" for index in range(rows)]
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is optional here
            self.skipTest("torch is unavailable")
        label = (torch.log(torch.as_tensor(truth, dtype=torch.float32) + 1) + 1).numpy()
        encoded = (torch.log(torch.as_tensor(pg, dtype=torch.float32) + 1) + 1).numpy()
        fields = (
            "model",
            "database",
            "row",
            "query_hash",
            "true",
            "pg",
            "encoded",
            "label",
        )
        csv_path = output / "PG-stats.csv.gz"
        with gzip.open(csv_path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(rows):
                writer.writerow(
                    dict(
                        zip(
                            fields,
                            (
                                "PG",
                                "stats",
                                index + 1,
                                identities[index],
                                repr(float(truth[index])),
                                repr(float(pg[index])),
                                repr(float(encoded[index])),
                                repr(float(label[index])),
                            ),
                        )
                    )
                )

        def file_sha(path):
            hasher = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        def digest(value):
            payload = json.dumps(
                value, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            return hashlib.sha256(payload.encode()).hexdigest()

        official = _module().summarize_official(
            encoded.astype(np.float64),
            label.astype(np.float64),
            true_cards=truth,
        )
        unit = {
            "model": "PG",
            "database": "stats",
            "n": rows,
            "official": official,
            "prediction_file": csv_path.name,
            "prediction_sha256": file_sha(csv_path),
        }
        (output / "units.json").write_text(json.dumps([unit]))
        binding = {
            "schema": 1,
            "protocol": 127,
            "root": str(root),
            "data": {
                "test/stats": {
                    "available": True,
                    "path": str(root / "features.pkl"),
                    "identities": identities,
                }
            },
        }
        binding["binding_sha256"] = digest(binding)
        (root / "inputs" / "binding.json").write_text(json.dumps(binding))
        completion = {
            "schema": 1,
            "protocol": 127,
            "stage": "zero",
            "status": "completed",
            "binding_sha256": binding["binding_sha256"],
            "selection_forbidden": True,
            "files": {"units.json": file_sha(output / "units.json")},
        }
        (output / "completion.json").write_text(json.dumps(completion))
        (output.with_name(output.name + ".supervisor.json")).write_text(
            json.dumps({"status": "completed", "completion": completion})
        )
        return unit, file_sha

    def _run(self, root):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "-B", str(self.script), "--root", str(root)],
            capture_output=True,
            check=False,
            text=True,
        )

    def test_verifier_accepts_a_consistent_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_fixture(root)
            result = self._run(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["problems"], [])
            self.assertEqual(report["cells_verified"], 1)

    def test_verifier_detects_a_tampered_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unit, _ = self._build_fixture(root)
            path = root / "outputs" / "zero" / "units.json"
            unit["official"]["q95"] = 99.0
            path.write_text(json.dumps([unit]))
            result = self._run(root)
            self.assertEqual(result.returncode, 1)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any(
                    "q95 differs" in problem or "evidence changed" in problem
                    for problem in report["problems"]
                ),
                msg=report["problems"],
            )

    def test_all_stage_ignores_supervisor_sidecars(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_fixture(root)
            output = root / "outputs"
            (output / "finetune-P-0-stats.supervisor.json").write_text("{}")
            (output / "finetune-P-0-stats.supervisor.log").write_text("not JSON")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(self.script),
                    "--root",
                    str(root),
                    "--stage",
                    "all",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value["cells_verified"], 1)
            self.assertFalse(value["matrix_complete"])
            self.assertFalse(value["report_checked"])

    def test_all_53_cells_distinguish_zero_from_finetuned_report_rows(self):
        import contextlib
        import io
        from unittest import mock

        spec = importlib.util.spec_from_file_location("independent127", self.script)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        keys = ("PG", "P-0", "E-42", "E-43", "E-44", "M-42", "M-43", "M-44")
        databases = ("imdb", "stats", "ergastf1", "genome")
        zero = [
            {"model": key, "database": db, "prediction_file": f"{key}-{db}.csv.gz"}
            for key in keys
            for db in databases
        ]
        fine = {
            f"finetune-{key}-{db}": [
                {"model": key, "database": db, "prediction_file": "predictions.csv.gz"}
            ]
            for key in keys[1:]
            for db in databases[1:]
        }
        all_report = zero + [row for rows in fine.values() for row in rows]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_fixture(root)
            report_dir = root / "outputs/report"
            report_dir.mkdir()
            (report_dir / "completion.json").write_text("{}")
            units = {"zero": zero, "report": all_report, **fine}
            text = io.StringIO()
            with (
                mock.patch.object(
                    verifier, "finished_stages", return_value=[*sorted(fine), "zero"]
                ),
                mock.patch.object(
                    verifier,
                    "collect_units",
                    side_effect=lambda root, stage: (units[stage], []),
                ),
                mock.patch.object(
                    verifier, "check_cell", side_effect=lambda unit, *args: unit
                ),
                contextlib.redirect_stdout(text),
            ):
                self.assertEqual(
                    verifier.main(["--root", str(root), "--stage", "all"]), 0
                )
            value = json.loads(text.getvalue())
            self.assertEqual(value["cells_verified"], 53)
            self.assertTrue(value["matrix_complete"])
            self.assertTrue(value["report_checked"])


if __name__ == "__main__":
    unittest.main()
