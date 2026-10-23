"""Pure record fixtures: do not solve, train, or open actual source matrices."""

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

from test_s8_independent_v2 import ACCEPTANCE_SHA, POLICY, numeric_records

from scripts import s8_independent_verify as verifier


class NumericRecordIntegrityTests(unittest.TestCase):
    def verify(self, record):
        documents = {"probe-smoke.json": record}
        for group, stage in [("profiles", "probe"), ("resource_profile", "resource")]:
            for feature in ("h", "x"):
                documents[f"{feature}-{stage}-certificate.json"] = copy.deepcopy(
                    record[group][feature]["certificate"]
                )
        with patch.object(
            verifier, "_read", side_effect=lambda path: documents[Path(path).name]
        ):
            return verifier._numeric_records(
                Path("/synthetic-no-I-O"),
                {
                    "schema": 2,
                    "numerical_acceptance_policy": POLICY,
                    "acceptance_protocol_sha256": ACCEPTANCE_SHA,
                },
            )

    def reject_mutation(self, key, value, *, remove=False):
        for group in ("profiles", "resource_profile"):
            with self.subTest(group=group, key=key, value=value):
                records = numeric_records()
                cert = records[group]["h"]["certificate"]
                if remove:
                    del cert[key]
                else:
                    cert[key] = value
                # Mirror corruption into BOTH records; file equality isn't enough.
                with self.assertRaises(ValueError):
                    self.verify(records)

    def test_missing_existing_diagnostic_fields_is_rejected(self):
        fields = (
            "spectrum_matrix_shape",
            "spectrum_dtype",
            "spectrum_specification",
            "spectrum_numpy_version",
            "engineering_scope",
            "spectrum_epsilon",
            "spectrum_gamma",
            "spectrum_error_bound",
            "spectrum_comparison_bound",
            "svd_scaled_reconstruction_residual",
            "svd_u_orthogonality_residual",
            "svd_v_orthogonality_residual",
            "svd_u_normalized_orthogonality_residual",
            "svd_v_normalized_orthogonality_residual",
            "svd_normalized_orthogonality_bound",
            "rank_cutoff_interval",
            "rank_cutoff_intervals",
            "rank_interval_margin",
            "objective_adam",
            "objective_ls",
            "reference_objective",
            "normalized_normal_residual",
        )
        for key in fields:
            self.reject_mutation(key, None, remove=True)

    def test_wrong_stage_shape_dtype_or_scope_is_rejected(self):
        for key, value in [
            ("spectrum_matrix_shape", [19456, 2]),
            ("spectrum_matrix_shape", [1216.0, 65]),
            ("spectrum_dtype", "float32"),
            ("engineering_scope", "strict_optimality"),
            ("spectrum_specification", "unknown"),
            ("spectrum_numpy_version", None),
        ]:
            self.reject_mutation(key, value)

    def test_true_flags_cannot_contradict_recorded_numeric_gates(self):
        for key, value in [
            ("normalized_normal_residual", 1e99),
            ("objective_ls", 1e99),
            ("reference_objective", 1e99),
            ("svd_scaled_reconstruction_residual", 1e99),
            ("svd_u_normalized_orthogonality_residual", 1e99),
            ("svd_v_normalized_orthogonality_residual", 1e99),
            ("svd_u_orthogonality_residual", 1e99),
            ("spectrum_gamma", 0.0),
            ("spectrum_epsilon", 0.0),
            ("spectrum_comparison_bound", 1e99),
            ("svd_normalized_orthogonality_bound", 1e99),
        ]:
            self.reject_mutation(key, value)

    def test_malformed_or_nonfinite_numbers_and_intervals_rejected(self):
        for key, value in [
            ("objective_ls", float("nan")),
            ("objective_adam", float("inf")),
            ("reference_objective", -1.0),
            ("spectrum_error_bound", -1.0),
            ("rank_interval_margin", float("inf")),
            ("svd_v_orthogonality_residual", True),
            ("rank_cutoff_interval", [2.0, 1.0]),
            ("rank_cutoff_intervals", {"svd": [0.0, 1.0]}),
        ]:
            self.reject_mutation(key, value)

    def test_complete_consistent_records_preserve_strict_unknown(self):
        result = self.verify(numeric_records())
        self.assertTrue(result["engineering_verified"])
        self.assertEqual(result["strict_certificate_status"], "unknown")
        self.assertEqual(
            result["numerical_record_checks"]["scope"],
            "record_consistency_not_independent_mathematical_proof",
        )


if __name__ == "__main__":
    unittest.main()
