"""Independent handwritten record negatives; no real P or target data reads."""

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import test_s8_readonly_p as readonly_fixtures
from test_s8_independent_v2 import ACCEPTANCE_SHA, POLICY, certificate, numeric_records

from scripts import s8_independent_verify as iv
from scripts import verify_s8_readonly_p as ro


def retained_check(rank=64):
    eps = 2.0**-52
    sigma1, sigma_r = (1.0, 0.5) if rank else (0.0, 0.0)
    bound = {
        "tol": 2e-12,
        "sigma1": sigma1,
        "sigma_r": sigma_r,
        "sigma_cutoff": 1e-12 * sigma1,
        "kappa_retained": 2.0 if rank else 0.0,
        "projection_norm": 1.0,
        "target_scale": 2.0,
        "amplification_scale": 2.0,
        "engineering_floor": 2e-12,
        "reachable_bound": 32 * eps if rank else 0.0,
        "accumulation_constant": 8.0,
        "eps": eps,
        "scope": (
            "max_engineering_floor_and_fp64_reachable_bound"
            if rank
            else "rank_zero_engineering_floor"
        ),
    }
    if not rank:
        bound.update(tol=1e-12, projection_norm=0.0, engineering_floor=1e-12)
    return {
        "scope": "real_retained_condition_number_fp64_reachable_bound",
        "reported_residual": 0.0,
        "reference_residual": 0.0,
        "reported_bound": copy.deepcopy(bound),
        "reference_bound": copy.deepcopy(bound),
    }


class RetainedRecordTests(unittest.TestCase):
    def valid(self):
        cert = certificate(65)
        cert["retained_subspace_check"] = retained_check()
        return cert

    def reject(self, cert):
        self.assertFalse(ro.certificate_complete(cert, 1216, 65))
        with self.assertRaises(ValueError):
            iv._certificate_record(cert, 1216, 65, "synthetic")

    def test_valid_handwritten_formula_accepts_both_verifiers(self):
        cert = self.valid()
        self.assertTrue(ro.certificate_complete(cert, 1216, 65))
        self.assertEqual(
            iv._certificate_record(cert, 1216, 65, "synthetic"), (64, "stable")
        )

    def test_missing_records_scopes_fields_rejected(self):
        for path in (
            ["retained_subspace_check"],
            *[["retained_subspace_check", k] for k in retained_check()],
            *[
                ["retained_subspace_check", "reported_bound", k]
                for k in retained_check()["reported_bound"]
            ],
        ):
            cert = self.valid()
            record = cert
            for key in path[:-1]:
                record = record[key]
            del record[path[-1]]
            with self.subTest(path=path):
                self.reject(cert)
        for path in (
            ["scope"],
            ["reported_bound", "scope"],
            ["reference_bound", "scope"],
        ):
            cert = self.valid()
            record = cert["retained_subspace_check"]
            for key in path[:-1]:
                record = record[key]
            record[path[-1]] = "strict_optimality"
            with self.subTest(path=path):
                self.reject(cert)

    def test_every_numeric_field_must_be_finite_nonnegative_number(self):
        for side in ("reported", "reference"):
            for field in [
                "residual",
                *[k for k in retained_check()[f"{side}_bound"] if k != "scope"],
            ]:
                for invalid in (
                    None,
                    True,
                    "0",
                    -1.0,
                    float("nan"),
                    float("inf"),
                    10**400,
                ):
                    cert = self.valid()
                    check = cert["retained_subspace_check"]
                    if field == "residual":
                        check[f"{side}_residual"] = invalid
                    else:
                        check[f"{side}_bound"][field] = invalid
                    with self.subTest(side=side, field=field, value=str(invalid)):
                        self.reject(cert)

    def test_formula_damage_cannot_be_hidden_by_two_matching_records(self):
        for key, value in (
            ("eps", 1e-8),
            ("accumulation_constant", 10.0),
            ("tol", 1.0),
            ("engineering_floor", 1.0),
            ("reachable_bound", 1.0),
            ("sigma_cutoff", 0.0),
            ("sigma_r", 1e-14),
            ("sigma_r", 2.0),
            ("kappa_retained", 1e12),
            ("amplification_scale", 100.0),
            ("projection_norm", 100.0),
            ("target_scale", 100.0),
        ):
            cert = self.valid()
            for side in ("reported", "reference"):
                cert["retained_subspace_check"][f"{side}_bound"][key] = value
            with self.subTest(key=key):
                self.reject(cert)
        cert = self.valid()
        cert["retained_subspace_check"]["reported_residual"] = 3e-12
        self.reject(cert)
        cert = self.valid()
        cert["retained_subspace_check"]["reference_residual"] = 3e-12
        self.reject(cert)

    def test_residual_is_checked_against_formula_not_rounding_larger_stored_tol(self):
        cert = self.valid()
        for side in ("reported", "reference"):
            check = cert["retained_subspace_check"]
            check[f"{side}_bound"]["tol"] *= 1 + 1e-13
            check[f"{side}_residual"] = check[f"{side}_bound"]["tol"]
        # Stored tol differs within metadata closeness, but cannot extend acceptance.
        self.reject(cert)

    def test_sigma1_must_match_existing_cutoff_interval_not_just_its_tolerance(self):
        cert = self.valid()
        for side in ("reported", "reference"):
            cert["retained_subspace_check"][f"{side}_bound"].update(
                sigma1=2.0, sigma_r=1.0, sigma_cutoff=2e-12
            )
        self.reject(cert)

    def test_rank_zero_requires_zero_spectrum_and_projection(self):
        cert = certificate(65, rank=0)
        cert["retained_subspace_check"] = retained_check(0)
        self.assertTrue(ro.certificate_complete(cert, 1216, 65))
        iv._certificate_record(cert, 1216, 65, "zero")
        for field in (
            "sigma1",
            "sigma_r",
            "sigma_cutoff",
            "kappa_retained",
            "projection_norm",
        ):
            damaged = copy.deepcopy(cert)
            for side in ("reported", "reference"):
                damaged["retained_subspace_check"][f"{side}_bound"][field] = 1.0
            with self.subTest(field=field):
                self.reject(damaged)

    def test_aggregate_and_single_file_simultaneous_damage_rejected(self):
        records = numeric_records()
        documents = {}
        for group, stage in (("profiles", "probe"), ("resource_profile", "resource")):
            for feature in ("h", "x"):
                cert = records[group][feature]["certificate"]
                cert["retained_subspace_check"] = retained_check(
                    cert["observed_ranks"]["svd"]
                )
                cert["retained_subspace_check"]["reported_bound"]["tol"] = 1.0
                documents[f"{feature}-{stage}-certificate.json"] = copy.deepcopy(cert)
        documents["probe-smoke.json"] = records
        with (
            patch.object(
                iv, "_read", side_effect=lambda path: documents[Path(path).name]
            ),
            self.assertRaises(ValueError),
        ):
            iv._numeric_records(
                Path("/synthetic"),
                {
                    "schema": 2,
                    "numerical_acceptance_policy": POLICY,
                    "acceptance_protocol_sha256": ACCEPTANCE_SHA,
                },
            )

    def test_readonly_aggregate_and_stage_sync_damage_rejected(self):
        fixture = readonly_fixtures.ReadOnlyPTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        proof = fixture.execute()
        output = fixture.root / "new-proof"
        proof["features"]["h"]["full_P_resource"]["certificate"][
            "retained_subspace_check"
        ]["reported_bound"]["tol"] = 1.0
        (output / "h-full_P_resource.json").write_text(
            json.dumps(proof["features"]["h"]["full_P_resource"])
        )
        self.assertFalse(ro.proof_complete(proof, fixture.contract, output))


if __name__ == "__main__":
    unittest.main()
