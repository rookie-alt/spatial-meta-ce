"""Handwritten synthetic records only; no probe import, solver, or real smoke."""

import ast
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_s8_independent as legacy

from scripts import s8_independent_verify as verifier

POLICY = "s8-g1-resource-engineering-v2"
ACCEPTANCE_SHA = "a" * 64  # Synthetic binding, not a claim about a protocol file.
HARD_FLAGS = (
    "metadata_pass",
    "normal_residual_pass",
    "objective_pass",
    "reference_objective_pass",
    "reference_prediction_pass",
    "reference_spectrum_pass",
    "svd_backend_pass",
    "retained_subspace_pass",
    "reference_retained_subspace_pass",
    "observed_rank_agreement",
)
STRICT = {
    "status": "unknown",
    "scopes": [
        "true_rank",
        "rigorous_retained_subspace",
        "strict_linear_optimality",
    ],
    "reason": "no_verified_arithmetic_certificate",
}
ORIGINAL_BINDINGS = verifier._bindings


def certificate(columns, stability="stable", rank=None):
    """Construct records directly: do not ask production code to certify itself."""
    rank = columns - 1 if rank is None else rank
    passed = stability == "stable"
    return {
        **dict.fromkeys(HARD_FLAGS, True),
        "engineering_verified": True,
        "observed_ranks": dict.fromkeys(("reported_ls", "svd", "reference_ls"), rank),
        "rank_stability": stability,
        "pass": passed,
        "solver_verified": passed,
        "rank_limited": rank < columns,
        "numerical_rank_limited": rank < columns,
        "fullrank_numeric_solver": passed and rank == columns,
        "full_linear_optimality_verified": False,
        "full_linear_optimality": False,
        "reference_pass": True,
        "spectrum_pass": True,
        "certification_linalg_calls": 2,
        "strict_certificate": copy.deepcopy(STRICT),
    }


def numeric_records():
    profiles, resources = {}, {}
    for feature, columns in (("h", 65), ("x", 775)):
        profiles[feature] = {
            "fit_rows": 1216,
            "parameters": columns,
            "adam_updates": 2,
            "seconds": 0.2,
            "certificate": certificate(columns),
        }
        resources[feature] = {
            "resource_only": True,
            "not_used_for_predictions_or_selection": True,
            "rows": 19456,
            "columns": columns,
            "matrix_bytes": 8 * 19456 * columns,
            "linalg_calls": 3,
            "rank": columns - 1,
            "standardization_seconds": 0.1,
            "solver_certification_seconds": 0.2,
            "seconds": 0.3,
            "full_linear_optimality_verified": False,
            "certificate": certificate(columns, "unknown"),
        }
    return {
        "schema": 2,
        "numerical_acceptance_policy": POLICY,
        "acceptance_protocol_sha256": ACCEPTANCE_SHA,
        "purpose": "smoke-not-research",
        "probe_fit_rows": 1216,
        "profiles": profiles,
        "resource_profile": resources,
    }


class IndependentV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.IndependentSmokeTests.setUpClass()

    def setUp(self):
        # Reuse only the existing synthetic identity/episode/endpoint fixture.
        self.base = legacy.IndependentSmokeTests(methodName="runTest")
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.base.write()
        self.root = self.base.root
        self.bindings = {
            **self.base.bindings,
            "schema": 2,
            "numerical_acceptance_policy": POLICY,
            "acceptance_protocol_sha256": ACCEPTANCE_SHA,
        }
        self.record = numeric_records()

    def write_numeric(self):
        (self.root / "probe-smoke.json").write_text(json.dumps(self.record))
        for feature in ("h", "x"):
            for stage, group in (("probe", "profiles"), ("resource", "resource_profile")):
                (self.root / f"{feature}-{stage}-certificate.json").write_text(
                    json.dumps(self.record[group][feature]["certificate"])
                )

    def verify(self):
        return verifier.verify_smoke(
            self.root, self.bindings, self.base.panels, self.base.caches
        )

    def verify_records(self):
        # Isolate record validation from the separately tested schema2 entrance.
        # Otherwise every malformed record could 'pass' simply because v2 is absent.
        def validated_v1_binding(bindings):
            binding = dict(bindings, schema=1)
            binding.pop("numerical_acceptance_policy", None)
            binding.pop("acceptance_protocol_sha256", None)
            ORIGINAL_BINDINGS(binding)

        with patch.object(verifier, "_bindings", side_effect=validated_v1_binding):
            return self.verify()

    def reject_record(self):
        self.write_numeric()
        with self.assertRaises(ValueError):
            self.verify_records()

    def test_v2_unknown_resource_accepts_without_summary_and_preserves_limits(self):
        self.write_numeric()
        self.assertFalse((self.root / "summary.json").exists())
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = self.verify()
        self.assertIs(result["passed"], True)
        self.assertIs(result["engineering_verified"], True)
        self.assertEqual(result["strict_certificate_status"], "unknown")
        self.assertIs(result["formal_ready"], False)
        self.assertIs(result["full_linear_optimality_verified"], False)
        self.assertIs(result["actual_H_access_order_verified"], False)
        self.assertEqual(result["query_count"], 20352)
        self.assertEqual(result["verified_counts"]["metric_cells"], 52)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_v2_stable_resource_and_fullrank_records_accept(self):
        for feature, columns in (("h", 65), ("x", 775)):
            self.record["resource_profile"][feature]["certificate"] = certificate(
                columns, rank=columns
            )
            self.record["resource_profile"][feature]["rank"] = columns
        self.write_numeric()
        self.assertIs(self.verify()["engineering_verified"], True)

    def test_v1_still_accepts_original_evidence_without_numeric_artifacts(self):
        self.bindings = self.base.bindings
        self.assertIs(self.verify()["passed"], True)

    def test_v1_rejects_either_v2_binding_key(self):
        for key, value in (
            ("numerical_acceptance_policy", POLICY),
            ("acceptance_protocol_sha256", ACCEPTANCE_SHA),
        ):
            with self.subTest(key=key):
                binding = {**self.base.bindings, key: value}
                with self.assertRaises(ValueError):
                    ORIGINAL_BINDINGS(binding)

    def test_v2_binding_contract(self):
        for key, value in (
            ("schema", True),
            ("schema", 2.0),
            ("schema", 3),
            ("mode", "formal"),
            ("numerical_acceptance_policy", None),
            ("numerical_acceptance_policy", "unknown"),
            ("acceptance_protocol_sha256", None),
            ("acceptance_protocol_sha256", "a" * 63),
            ("acceptance_protocol_sha256", "g" * 64),
            ("acceptance_protocol_sha256", 0),
        ):
            with self.subTest(key=key, value=value):
                binding = dict(self.bindings)
                if value is None:
                    binding.pop(key)
                else:
                    binding[key] = value
                with self.assertRaises(ValueError):
                    ORIGINAL_BINDINGS(binding)
        ORIGINAL_BINDINGS(self.bindings)

    def test_source_binding_requires_exact_schema_policy_and_sha(self):
        for key, value in (
            ("schema", 1),
            ("schema", 2.0),
            ("numerical_acceptance_policy", "other"),
            ("acceptance_protocol_sha256", "b" * 64),
            ("acceptance_protocol_sha256", "invalid"),
        ):
            with self.subTest(key=key):
                self.record = numeric_records()
                self.record[key] = value
                self.reject_record()
        for key in ("schema", "numerical_acceptance_policy", "acceptance_protocol_sha256"):
            with self.subTest(missing=key):
                self.record = numeric_records()
                del self.record[key]
                self.reject_record()

    def test_missing_artifacts_rejected(self):
        for name in (
            "probe-smoke.json",
            "h-probe-certificate.json",
            "h-resource-certificate.json",
            "x-probe-certificate.json",
            "x-resource-certificate.json",
        ):
            with self.subTest(name=name):
                self.write_numeric()
                (self.root / name).unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.verify_records()

    def test_each_stage_certificate_must_equal_combined(self):
        for feature in ("h", "x"):
            for stage, group in (("probe", "profiles"), ("resource", "resource_profile")):
                with self.subTest(feature=feature, stage=stage):
                    self.write_numeric()
                    changed = copy.deepcopy(self.record[group][feature]["certificate"])
                    changed["metadata_pass"] = False
                    (self.root / f"{feature}-{stage}-certificate.json").write_text(
                        json.dumps(changed)
                    )
                    with self.assertRaises(ValueError):
                        self.verify_records()

    def test_hard_flags_must_be_present_and_strictly_true(self):
        for flag in HARD_FLAGS:
            for value in (False, 1, "true", None):
                with self.subTest(flag=flag, value=value):
                    self.record = numeric_records()
                    cert = self.record["resource_profile"]["h"]["certificate"]
                    if value is None:
                        del cert[flag]
                    else:
                        cert[flag] = value
                    self.reject_record()

    def test_engineering_boolean_cannot_be_false_missing_or_truthy(self):
        for group in ("profiles", "resource_profile"):
            for value in (False, 1, "true", None):
                with self.subTest(group=group, value=value):
                    self.record = numeric_records()
                    cert = self.record[group]["x"]["certificate"]
                    if value is None:
                        del cert["engineering_verified"]
                    else:
                        cert["engineering_verified"] = value
                    self.reject_record()

    def test_observed_ranks_must_be_integer_equal_and_bounded(self):
        for value in (None, [], {}, {"reported_ls": 64, "svd": 64},
                      {"reported_ls": 63, "svd": 64, "reference_ls": 64}):
            with self.subTest(value=value):
                self.record = numeric_records()
                self.record["resource_profile"]["h"]["certificate"]["observed_ranks"] = value
                self.reject_record()
        for rank in (True, 64.0, "64", -1, 66):
            with self.subTest(rank=rank):
                self.record = numeric_records()
                self.record["resource_profile"]["h"]["certificate"]["observed_ranks"] = (
                    dict.fromkeys(("reported_ls", "svd", "reference_ls"), rank)
                )
                self.reject_record()

    def test_legacy_pass_solver_and_rank_stability_consistency(self):
        for key, value in (
            ("pass", True), ("solver_verified", True),
            ("pass", 0), ("solver_verified", 0),
            ("rank_stability", "verified"), ("rank_stability", None),
        ):
            with self.subTest(key=key, value=value):
                self.record = numeric_records()
                self.record["resource_profile"]["h"]["certificate"][key] = value
                self.reject_record()

    def test_small_probe_old_gate_cannot_be_replaced_by_engineering(self):
        for feature, columns in (("h", 65), ("x", 775)):
            with self.subTest(feature=feature):
                self.record = numeric_records()
                self.record["profiles"][feature]["certificate"] = certificate(columns, "unknown")
                self.reject_record()

    def test_legacy_rank_and_scientific_flags_remain_consistent(self):
        for key, value in (
            ("rank_limited", False), ("numerical_rank_limited", False),
            ("fullrank_numeric_solver", True), ("rank_limited", 1),
            ("full_linear_optimality_verified", True),
            ("full_linear_optimality", True),
        ):
            with self.subTest(key=key):
                self.record = numeric_records()
                self.record["resource_profile"]["h"]["certificate"][key] = value
                self.reject_record()

    def test_strict_unknown_scope_and_reason_cannot_be_forged(self):
        for value in (
            None, {},
            {**STRICT, "status": "verified"},
            {**STRICT, "scopes": STRICT["scopes"][:-1]},
            {**STRICT, "scopes": STRICT["scopes"] + ["other"]},
            {**STRICT, "reason": "engineering_verified"},
        ):
            with self.subTest(value=value):
                self.record = numeric_records()
                self.record["resource_profile"]["x"]["certificate"]["strict_certificate"] = value
                self.reject_record()

    def test_both_profile_sets_and_certificates_required(self):
        for group in ("profiles", "resource_profile"):
            for change in ("missing_group", "missing_h", "extra", "certificate"):
                with self.subTest(group=group, change=change):
                    self.record = numeric_records()
                    if change == "missing_group":
                        del self.record[group]
                    elif change == "missing_h":
                        del self.record[group]["h"]
                    elif change == "extra":
                        self.record[group]["extra"] = self.record[group]["h"]
                    else:
                        del self.record[group]["x"]["certificate"]
                    # Write only the combined record; keep valid stage files.
                    baseline = self.record
                    self.record = numeric_records()
                    self.write_numeric()
                    self.record = baseline
                    (self.root / "probe-smoke.json").write_text(json.dumps(baseline))
                    with self.assertRaises(ValueError):
                        self.verify_records()

    def test_profile_shapes_updates_counts_and_resource_scope(self):
        cases = (
            ("profiles", "fit_rows", 1215), ("profiles", "parameters", 64),
            ("profiles", "adam_updates", 1), ("profiles", "adam_updates", 2.0),
            ("resource_profile", "rows", 19455), ("resource_profile", "columns", 64),
            ("resource_profile", "matrix_bytes", 8 * 19456 * 65 - 1),
            ("resource_profile", "linalg_calls", 2), ("resource_profile", "linalg_calls", 3.0),
            ("resource_profile", "resource_only", 1),
            ("resource_profile", "not_used_for_predictions_or_selection", False),
            ("resource_profile", "rank", 63), ("resource_profile", "rank", 64.0),
            ("resource_profile", "full_linear_optimality_verified", True),
        )
        for group, key, value in cases:
            with self.subTest(group=group, key=key):
                self.record = numeric_records()
                self.record[group]["h"][key] = value
                self.reject_record()
        self.record = numeric_records()
        self.record["resource_profile"]["x"]["columns"] = 65
        self.reject_record()

    def test_resource_times_finite_nonnegative_and_total_consistent(self):
        for key in ("standardization_seconds", "solver_certification_seconds", "seconds"):
            for value in (-1, float("nan"), float("inf"), True, "0.1", None):
                with self.subTest(key=key, value=value):
                    self.record = numeric_records()
                    profile = self.record["resource_profile"]["h"]
                    if value is None:
                        del profile[key]
                    else:
                        profile[key] = value
                    self.reject_record()
        self.record = numeric_records()
        self.record["resource_profile"]["h"]["seconds"] = 0.4
        self.reject_record()

    def test_no_production_probe_import_or_math_proof_claim(self):
        source = Path(verifier.__file__).read_text()
        imports = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
        self.assertEqual(set(imports), {"hashlib", "json", "math", "pathlib", "numpy"})
        self.write_numeric()
        result = self.verify_records()
        self.assertIs(result["full_linear_optimality_verified"], False)
        self.assertIs(result["source_file_provenance_verified"], False)
        self.assertIs(result["actual_H_access_order_verified"], False)


if __name__ == "__main__":
    unittest.main()
