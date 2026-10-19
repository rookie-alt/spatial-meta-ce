"""Synthetic v2 engineering-permission bindings, not formal or real smoke.

Approval permits the new smoke metadata policy only. These standard-library
checks do not authorize or execute training, resource probes, H I/O or selection.
"""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from test_s8_state import bindings, canonical, file_sha

from spatial_meta_ce.s8_state import require_nomination, seal_nomination, signature

POLICY = "s8-g1-resource-engineering-v2"
POLICY_KEY = "numerical_acceptance_policy"
ACCEPTANCE_KEY = "acceptance_protocol_sha256"


def v2_bindings(mode="smoke"):
    return {
        **bindings(mode),
        "schema": 2,
        POLICY_KEY: POLICY,
        ACCEPTANCE_KEY: "4" * 64,
    }


class EngineeringV2SignatureTests(unittest.TestCase):
    def test_legacy_schema1_smoke_and_formal_keep_canonical_semantics(self):
        for mode in ("smoke", "formal"):
            data = bindings(mode)
            with self.subTest(mode=mode):
                self.assertEqual(
                    signature(data), hashlib.sha256(canonical(data)).hexdigest()
                )
                data["runtime"] = {"python": "synthetic-only"}
                self.assertEqual(
                    signature(data), hashlib.sha256(canonical(data)).hexdigest()
                )

    def test_schema2_smoke_accepts_only_engineering_permission_and_signs_all_keys(self):
        data = v2_bindings()
        original = copy.deepcopy(data)
        expected = hashlib.sha256(canonical(data)).hexdigest()
        self.assertEqual(signature(data), expected)
        self.assertEqual(signature(dict(reversed(list(data.items())))), expected)
        self.assertEqual(data, original)
        self.assertNotEqual(signature(bindings()), expected)
        for key in (
            ACCEPTANCE_KEY,
            "protocol_sha256",
            "source_binding_sha256",
            "panel_identity_digest",
        ):
            with self.subTest(key=key):
                self.assertNotEqual(signature({**data, key: "a" * 64}), expected)

    def test_schema1_rejects_either_or_both_v2_fields_even_without_valid_policy(self):
        for mode in ("smoke", "formal"):
            for extra in (
                {POLICY_KEY: POLICY},
                {ACCEPTANCE_KEY: "4" * 64},
                {POLICY_KEY: POLICY, ACCEPTANCE_KEY: "4" * 64},
                {POLICY_KEY: None},
                {POLICY_KEY: "unknown"},
                {ACCEPTANCE_KEY: None},
            ):
                with (
                    self.subTest(mode=mode, extra=extra),
                    self.assertRaises(ValueError),
                ):
                    signature({**bindings(mode), **extra})

    def test_schema2_rejects_formal_even_with_matching_formal_purpose_and_steps(self):
        with self.assertRaises(ValueError):
            signature(v2_bindings("formal"))

    def test_schema2_requires_each_policy_and_acceptance_field(self):
        for key in (POLICY_KEY, ACCEPTANCE_KEY):
            data = v2_bindings()
            del data[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                signature(data)
        data = v2_bindings()
        del data[POLICY_KEY]
        del data[ACCEPTANCE_KEY]
        with self.assertRaises(ValueError):
            signature(data)

    def test_schema2_rejects_unknown_wrong_or_non_string_policy(self):
        for value in (
            "",
            "unknown",
            "s8-g1-resource-engineering-v1",
            POLICY + " ",
            None,
            True,
            2,
            [],
            {},
        ):
            with self.subTest(policy=value), self.assertRaises(ValueError):
                signature({**v2_bindings(), POLICY_KEY: value})

    def test_schema2_acceptance_requires_canonical_sha256(self):
        for value in (
            "",
            "g" * 64,
            "A" * 64,
            "4" * 63,
            "4" * 65,
            "4" * 64 + "\n",
            None,
            True,
            4,
            [],
            {},
        ):
            with self.subTest(acceptance=value), self.assertRaises(ValueError):
                signature({**v2_bindings(), ACCEPTANCE_KEY: value})

    def test_schema_is_exact_integer_version_not_coercible_or_unknown(self):
        for value in (True, False, 1.0, 2.0, "2", None, 0, 3):
            with self.subTest(schema=value), self.assertRaises(ValueError):
                signature({**v2_bindings(), "schema": value})

    def test_schema2_keeps_all_original_required_fields(self):
        self.assertEqual(len(signature(v2_bindings())), 64)
        for key in bindings():
            data = v2_bindings()
            del data[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                signature(data)

    def test_schema2_preserves_data_model_resource_and_step_gates(self):
        self.assertEqual(len(signature(v2_bindings())), 64)
        mutations = [
            ("mode", "formal"),
            ("purpose", "s8-research"),
            ("train_steps", 4000),
            ("train_steps", True),
            ("seed", 43),
            ("meta_batch", 8),
            ("query_size", 128),
            ("support_sizes", [5, 10, 20, 50]),
            ("protocol_sha256", "g" * 64),
            ("source_binding_sha256", None),
            ("panel_identity_digest", "3" * 63),
            ("checkpoint", "/runs/imdb/last.pt"),
            ("database", "unknown"),
            ("runtime", {"value": float("nan")}),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                signature({**v2_bindings(), key: value})
        for key in ("train_databases", "holdout_databases", "validation_databases"):
            for value in (list(reversed(v2_bindings()[key])), ["imdb"], ["unknown"]):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    signature({**v2_bindings(), key: value})
        for key, value in (
            ("arch", "set"),
            ("width", 128),
            ("use_pg", False),
            ("pg_output_scale", 0),
            ("layers", 3),
            ("heads", 8),
        ):
            data = v2_bindings()
            data["model_config"][key] = value
            with self.subTest(model=key), self.assertRaises(ValueError):
                signature(data)
        for key, value in v2_bindings()["resourcefixedfields"].items():
            data = v2_bindings()
            data["resourcefixedfields"][key] = value + 1
            with self.subTest(resource=key), self.assertRaises(ValueError):
                signature(data)


class EngineeringV2SealTests(unittest.TestCase):
    def test_v2_seal_round_trip_keeps_policy_hash_and_smoke_not_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "v2-seal.json"
            data = v2_bindings()
            # Non-JSON sentinel: smoke must never inspect selection metrics.
            created = seal_nomination(path, object(), data)
            verified = require_nomination(path, file_sha(path), data)
            self.assertEqual(verified, created)
            self.assertEqual(verified["bindings"], data)
            self.assertEqual(verified["bindings_sha256"], signature(data))
            self.assertEqual(verified["decision"], "smoke_not_selection")
            self.assertIsNone(verified["candidate"])
            self.assertIsNone(verified["metrics"])
            self.assertIsNone(verified["selection"])
            self.assertEqual(verified["mode"], "smoke")

    def test_seal_h_gate_rejects_schema1_schema2_in_both_directions(self):
        for sealed, supplied in (
            (bindings(), v2_bindings()),
            (v2_bindings(), bindings()),
        ):
            with (
                self.subTest(sealed_schema=sealed["schema"]),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory).resolve() / "seal.json"
                seal_nomination(path, None, sealed)
                with self.assertRaises(ValueError):
                    require_nomination(path, file_sha(path), supplied)

    def test_v2_seal_h_gate_rejects_changed_or_removed_revision_and_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "seal.json"
            data = v2_bindings()
            seal_nomination(path, None, data)
            expected_file_sha = file_sha(path)
            for key, value in (
                (ACCEPTANCE_KEY, "a" * 64),
                ("protocol_sha256", "b" * 64),
                (POLICY_KEY, "unknown"),
                ("schema", 1),
            ):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    require_nomination(path, expected_file_sha, {**data, key: value})
            for key in (POLICY_KEY, ACCEPTANCE_KEY):
                changed = copy.deepcopy(data)
                del changed[key]
                with self.subTest(missing=key), self.assertRaises(ValueError):
                    require_nomination(path, expected_file_sha, changed)

    def test_rehashed_seal_revision_tampering_cannot_match_pinned_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "seal.json"
            data = v2_bindings()
            seal_nomination(path, None, data)
            payload = json.loads(path.read_text())
            payload["bindings"][ACCEPTANCE_KEY] = "a" * 64
            payload["bindings_sha256"] = signature(payload["bindings"])
            body = {
                key: value for key, value in payload.items() if key != "body_digest"
            }
            payload["body_digest"] = hashlib.sha256(canonical(body)).hexdigest()
            path.write_bytes(canonical(payload))
            with self.assertRaises(ValueError):
                require_nomination(path, file_sha(path), data)


if __name__ == "__main__":
    unittest.main()
