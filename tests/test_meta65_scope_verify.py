"""Independent scope certification: synthetic source-only evidence, never targets."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import meta65_scope_runner as runner
from scripts import meta65_scope_supervisor as supervisor
from scripts import meta65_scope_verify as verify_module
from scripts.meta65_scope_decision import decide

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("R_head3", "T_all3")
DBS = ("chembl", "legalacts", "telstra")
HEAD_NAMES = [
    f"head.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def metrics(prediction: np.ndarray, identity: dict) -> dict:
    true, tables = identity["true"], identity["n_tables"]
    q = np.exp(np.abs(np.maximum(prediction, 0) - np.log(np.maximum(true, 1))))

    def aggregate(mask: np.ndarray) -> dict:
        values = q[mask]
        n = len(values)
        return {
            "count": n,
            "q50": float(np.quantile(values, 0.5)) if n else None,
            "q95": float(np.quantile(values, 0.95)) if n else None,
            "status": "ok" if n >= 30 else "below_min_n" if n else "empty",
        }

    edges = (0, 100, 1000, 10000, 100000, 1000000, np.inf)
    labels = ("<100", "100-1e3", "1e3-1e4", "1e4-1e5", "1e5-1e6", ">=1e6")
    joint = {
        f"{label}|{table}": aggregate(
            (true >= edges[i]) & (true < edges[i + 1]) & (np.minimum(tables, 5) == j)
        )
        for i, label in enumerate(labels)
        for j, table in enumerate(("1", "2", "3", "4", "5+"), 1)
    }
    return {
        "all": aggregate(np.ones(len(true), dtype=bool)),
        "4+": aggregate(tables >= 4),
        "joint": joint,
        "low4": aggregate((true < 100) & (tables >= 4)),
    }


class ScopeVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.counts = {db: 64 for db in DBS}
        self.count_patch = patch.object(verify_module, "COUNTS", self.counts)
        self.count_patch.start()
        self.addCleanup(self.count_patch.stop)
        self.make_fixture()

    def read(self, name: str) -> dict:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def save(self, name: str, payload: dict) -> None:
        write_json(self.root / name, payload)

    def make_fixture(
        self, *, formal: bool = True, treatment_factor: float = 1.5
    ) -> None:
        mode = "formal" if formal else "cpu_preflight"
        old = json.loads(
            (
                ROOT / "实验文档/results/meta65-source-screen-20261009/B-metadata.json"
            ).read_text(encoding="utf-8")
        )
        old_path = self.root / "old-B-metadata.json"
        write_json(old_path, old)
        records = []
        for arm, scope in zip(ARMS, ("head", "all")):
            config = {**old["config"], "adapt_scope": scope, "purpose": "meta65-scope"}
            if not formal:
                config = {**config, "steps": 2}
            signature = {
                k: v for k, v in config.items() if k not in ("steps", "data_config")
            }
            record = {
                **old,
                "arm": arm,
                "config": config,
                "metadata": {
                    **old["metadata"],
                    "training_signature": signature,
                    "purpose": config["purpose"],
                },
            }
            if not formal:
                record = {
                    **record,
                    "last_step": 2,
                    "optimizer_steps": 2,
                    "sample_plan_episodes": 8,
                    "support_sample_count": 100,
                    "counts": {
                        "optimizer_steps": 2,
                        "support_presentations": 300,
                        "query_presentations": 512,
                        "backward_calls": 32,
                    },
                }
            path = self.root / f"{arm}-metadata.json"
            write_json(path, record)
            records.append(
                {**record, "metadata_file": path.name, "metadata_sha256": digest(path)}
            )
        self.save("training-records.json", {"schema": 1, "records": records})
        cells, anchors = [], []
        for db in DBS:
            for run in range(5 if formal else 1):
                n = self.counts[db]
                identity = {
                    "query_hash": np.array(
                        [f"{i:064x}" for i in range(n)], dtype="S64"
                    ),
                    "query_id": np.array(
                        [f"{db}/pretrain/{i}" for i in range(n)], dtype="S128"
                    ),
                    "true": np.where(np.arange(n) < n // 2, 10.0, 1000.0).astype(
                        np.float64
                    ),
                    "n_tables": np.where(np.arange(n) % 2, 5, 4).astype(np.int16),
                }
                identity_path = self.root / f"identity-{db}-{run}.npz"
                np.savez(identity_path, **identity)
                old_pred = np.log(identity["true"]) + np.log(2.0)
                anchor = {**identity, "log_pred": old_pred}
                anchor_path = self.root / f"B-{db}-{run}.npz"
                # Original B files are unsorted, Unicode identities, int64 tables.
                np.savez(
                    anchor_path,
                    **{
                        k: v[::-1].astype(str)
                        if k in ("query_hash", "query_id")
                        else v[::-1].astype(np.int64)
                        if k == "n_tables"
                        else v[::-1]
                        for k, v in anchor.items()
                    },
                )
                anchors.append(
                    {
                        "database": db,
                        "run": run,
                        "path": str(anchor_path),
                        "sha256": digest(anchor_path),
                    }
                )
                view = identity if formal else {k: v[:32] for k, v in identity.items()}
                for arm in ARMS:
                    for operator in ("none", "head3"):
                        factor = {
                            ("R_head3", "head3"): 2.0,
                            ("R_head3", "none"): 3.0,
                            ("T_all3", "head3"): treatment_factor,
                            # Deployment baseline tracks the treatment so the
                            # low4 negative-transfer guard never masks the
                            # primary gate under test.
                            ("T_all3", "none"): max(1.6, treatment_factor),
                        }[arm, operator]
                        pred = np.log(view["true"]) + np.log(factor)
                        pred_path = (
                            self.root / f"prediction-{arm}-{operator}-{db}-{run}.npz"
                        )
                        np.savez(pred_path, log_pred=pred)
                        cell = {
                            "arm": arm,
                            "operator": operator,
                            "database": db,
                            "run": run,
                            "k20": 20,
                            "count": len(pred),
                            "identity_file": identity_path.name,
                            "identity_sha256": digest(identity_path),
                            "prediction_file": pred_path.name,
                            "prediction_sha256": digest(pred_path),
                            "metrics": metrics(pred, view),
                            "steps": 3 if operator == "head3" else 0,
                            "selected_parameter_count": 107521
                            if operator == "head3"
                            else 0,
                            "selected_parameter_names": HEAD_NAMES
                            if operator == "head3"
                            else [],
                            "changed_parameter_names": ["head.4.bias"]
                            if operator == "head3"
                            else [],
                            "state_before": hashlib.sha256(arm.encode()).hexdigest(),
                            "state_after": hashlib.sha256(arm.encode()).hexdigest(),
                        }
                        cells.append(
                            cell
                            if formal
                            else {**cell, "query_positions": list(range(32))}
                        )
        self.save(
            "reproduction.json",
            {
                "status": "verified" if formal else "not_applicable",
                "reproduction_verified": formal,
                "anchors": anchors,
                "old_b_metadata_file": str(old_path),
                "old_b_metadata_sha256": digest(old_path),
            },
        )
        inputs = {
            str(old_path): digest(old_path),
            **{a["path"]: a["sha256"] for a in anchors},
        }
        self.save(
            "input-binding.json",
            {
                "schema": 1,
                "mode": mode,
                "input_sha256": inputs,
                "code_sha256": {
                    "scripts/meta65_scope_verify.py": digest(
                        ROOT / "scripts/meta65_scope_verify.py"
                    )
                },
            },
        )
        self.save(
            "cells.json" if formal else "microcells.json",
            {"schema": 1, "mode": mode, "cells": cells},
        )

    def update_cell(self, index: int, **fields: object) -> None:
        payload = self.read("cells.json")
        payload = {
            **payload,
            "cells": [
                {**cell, **fields} if i == index else cell
                for i, cell in enumerate(payload["cells"])
            ],
        }
        self.save("cells.json", payload)

    def rewrite_prediction(self, index: int, pred: np.ndarray) -> None:
        cell = self.read("cells.json")["cells"][index]
        path = self.root / cell["prediction_file"]
        np.savez(path, log_pred=pred)
        with np.load(self.root / cell["identity_file"]) as archive:
            identity = dict(archive)
        self.update_cell(
            index, prediction_sha256=digest(path), metrics=metrics(pred, identity)
        )

    def test_formal_complete_matrix_and_independent_decision(self) -> None:
        result = verify_module.verify(self.root)
        self.assertEqual(result["status"], "formal_verified")
        self.assertTrue(result["formal_result"])
        self.assertTrue(result["engineering_verified"])
        self.assertTrue(result["reproduction_verified"])
        self.assertEqual(result["cells_verified"], 60)
        self.assertEqual(result["summary"]["decision"], "promising_single_seed_K20")
        self.assertAlmostEqual(result["summary"]["primary_improvement"], 0.25)

    def test_decision_and_verifier_agree_at_the_ten_percent_boundary(self) -> None:
        """Both classifiers must accept an improvement of exactly 10 percent."""
        for factor, expected in (
            (1.8, "promising_single_seed_K20"),
            (1.8001, "no_candidate"),
        ):
            with self.subTest(factor=factor):
                self.make_fixture(treatment_factor=factor)
                summary = verify_module.verify(self.root)["summary"]
                cells = self.read("cells.json")["cells"]
                provisional = decide(
                    [runner.normalize_cell(cell) for cell in cells],
                    engineering_verified=True,
                    reproduction_verified=True,
                )
                self.assertEqual(summary["status"], provisional["status"])
                self.assertEqual(summary["decision"], expected)
                self.assertEqual(provisional["status"], expected)
                self.assertEqual(summary["selected_arm"], provisional["selected_arm"])

    def test_formal_summary_publishes_the_selected_arm_for_the_supervisor(self) -> None:
        """The real verifier summary must expose the field the supervisor compares."""
        result = verify_module.verify(self.root)
        summary = result["summary"]
        self.assertEqual(summary["status"], "promising_single_seed_K20")
        self.assertEqual(summary["selected_arm"], "T_all3")
        (self.root / "status.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "status": "worker_complete",
                    "mode": "formal",
                    "cells": 60,
                    "formal_result": True,
                }
            ),
            encoding="utf-8",
        )
        write_json(self.root / "verification.json", result)
        provisional = {
            "status": summary["status"],
            "selected_arm": summary["selected_arm"],
        }
        write_json(self.root / "decision.provisional.json", provisional)
        self.assertIsNone(supervisor._completion(self.root, "formal"))
        write_json(
            self.root / "decision.provisional.json",
            {**provisional, "selected_arm": None},
        )
        self.assertEqual(
            supervisor._completion(self.root, "formal"), "invalid_decision"
        )

    def _run_real_certification(self) -> None:
        """Mirror the worker's write of the returned certification."""
        before = {"same": "evidence"}
        cells = self.read("cells.json")["cells"]
        with (
            patch(
                "scripts.precision_screen_evidence.source_evidence",
                return_value=before,
            ),
            patch(
                "scripts.precision_screen_evidence.historical_sources",
                return_value=before,
            ),
            patch.object(runner.contract, "recheck"),
        ):
            verified = runner.certify(
                self.root, "formal", cells, {}, {}, before, before
            )
        write_json(self.root / "verification.json", verified)

    def _mark_worker_complete(self) -> None:
        write_json(
            self.root / "status.json",
            {
                "schema": 1,
                "status": "worker_complete",
                "mode": "formal",
                "cells": 60,
                "formal_result": True,
            },
        )

    def test_real_certification_publishes_candidate_only_with_consistent_chain(
        self,
    ) -> None:
        """True certify() -> true summary -> true supervisor publication chain."""
        # Exactly 10 percent improvement: the weakest matrix that may publish.
        self.make_fixture(treatment_factor=1.8)
        self._run_real_certification()
        verified = self.read("verification.json")
        provisional = self.read("decision.provisional.json")
        self.assertEqual(provisional["status"], "promising_single_seed_K20")
        self.assertEqual(provisional["selected_arm"], "T_all3")
        self.assertEqual(verified["summary"]["status"], provisional["status"])
        self.assertEqual(
            verified["summary"]["selected_arm"], provisional["selected_arm"]
        )
        self.assertGreaterEqual(provisional["relative_improvement"], 0.10)
        self.assertFalse((self.root / "decision.json").exists())
        self._mark_worker_complete()
        self.assertIsNone(supervisor._completion(self.root, "formal"))

    def test_real_certification_leaves_unqualified_matrix_unpublished(self) -> None:
        self.make_fixture(treatment_factor=1.9)
        self._run_real_certification()
        provisional = self.read("decision.provisional.json")
        self.assertEqual(provisional["status"], "no_candidate")
        self.assertIsNone(provisional["selected_arm"])
        self.assertIsNone(self.read("verification.json")["summary"]["selected_arm"])
        self._mark_worker_complete()
        self.assertIsNone(supervisor._completion(self.root, "formal"))
        self.assertFalse((self.root / "decision.json").exists())

        self.make_fixture(formal=False)
        result = verify_module.verify(self.root, formal=False)
        self.assertFalse(result["formal_result"])
        self.assertFalse(result["reproduction_verified"])
        self.assertEqual(result["cells_verified"], 12)
        self.assertIsNone(result["summary"])
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)

    def test_bad_sha_nonfinite_prediction_and_prediction_dtype(self) -> None:
        self.update_cell(0, prediction_sha256="0" * 64)
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)
        for pred in (
            np.full(64, np.inf),
            np.ones(64, dtype=np.float32),
            np.ones((64, 1)),
        ):
            with self.subTest(dtype=pred.dtype, shape=pred.shape):
                self.make_fixture()
                cell = self.read("cells.json")["cells"][0]
                path = self.root / cell["prediction_file"]
                np.savez(path, log_pred=pred)
                self.update_cell(0, prediction_sha256=digest(path))
                with self.assertRaises(ValueError):
                    verify_module.verify(self.root)

    def test_matrix_missing_duplicate_target_and_bool_run(self) -> None:
        original = self.read("cells.json")
        for rows in (
            original["cells"][:-1],
            original["cells"] + [original["cells"][0]],
            [{**original["cells"][0], "database": "imdb"}] + original["cells"][1:],
            [{**original["cells"][0], "run": False}] + original["cells"][1:],
        ):
            self.save("cells.json", {**original, "cells": rows})
            with self.assertRaises(ValueError):
                verify_module.verify(self.root)

    def test_head_scope_counts_steps_and_shared_arm_state(self) -> None:
        cases: tuple[dict[str, object], ...] = (
            {"selected_parameter_names": ["encoder.weight"]},
            {"changed_parameter_names": ["encoder.weight"]},
            {"selected_parameter_count": 65},
            {"steps": 1},
            {"state_before": "1" * 64, "state_after": "1" * 64},
            {"changed_parameter_names": []},
        )
        for fields in cases:
            self.make_fixture()
            self.update_cell(1, **fields)
            with self.assertRaises(ValueError):
                verify_module.verify(self.root)
        self.make_fixture()
        self.update_cell(0, changed_parameter_names=["head.4.bias"])
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)

    def test_identity_types_shape_sort_unique_and_exact_pairing(self) -> None:
        for field, value in (
            ("true", np.ones(64, dtype=np.float32)),
            ("n_tables", np.ones(64, dtype=np.int64)),
            ("query_hash", np.full(64, b"x", dtype="S64")),
            ("true", np.ones((64, 1))),
            ("true", np.full(64, np.nan)),
        ):
            self.make_fixture()
            cells = self.read("cells.json")
            path = self.root / cells["cells"][0]["identity_file"]
            with np.load(path) as archive:
                identity = dict(archive)
            np.savez(path, **{**identity, field: value})
            self.save(
                "cells.json",
                {
                    **cells,
                    "cells": [
                        {**c, "identity_sha256": digest(path)}
                        if c["identity_file"] == path.name
                        else c
                        for c in cells["cells"]
                    ],
                },
            )
            with self.assertRaises(ValueError):
                verify_module.verify(self.root)
        self.make_fixture()
        cell = self.read("cells.json")["cells"][0]
        with np.load(self.root / cell["identity_file"]) as archive:
            identity = dict(archive)
        alternate = self.root / "other-identity.npz"
        np.savez(alternate, **{**identity, "true": identity["true"] + 1})
        self.update_cell(
            1, identity_file=alternate.name, identity_sha256=digest(alternate)
        )
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)

    def test_no_symlink_or_escape_and_no_target_open(self) -> None:
        path = self.root / "linked"
        path.symlink_to(self.root, target_is_directory=True)
        self.update_cell(0, identity_file="linked/identity-chembl-0.npz")
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)
        self.make_fixture()
        binding = self.read("input-binding.json")
        self.save(
            "input-binding.json",
            {**binding, "input_sha256": {"/imdb/never-open": "0" * 64}},
        )
        with self.assertRaisesRegex(ValueError, "[Tt]arget"):
            verify_module.verify(self.root)

    def test_reproduction_reloads_raw_anchor_not_status(self) -> None:
        cells = self.read("cells.json")["cells"]
        reference = [
            c for c in cells if c["arm"] == "R_head3" and c["operator"] == "head3"
        ]
        self.assertTrue(
            verify_module.verify_reproduction(self.root, reference)[
                "reproduction_verified"
            ]
        )
        cell = reference[0]
        self.rewrite_prediction(cells.index(cell), np.full(64, 5.0))
        with self.assertRaisesRegex(ValueError, "[Rr]eproduction"):
            verify_module.verify(self.root)

    def test_reproduction_anchor_identity_dtype_sha_and_q95(self) -> None:
        for field, value in (
            ("true", np.full(64, 12.0, dtype=np.float64)),
            ("true", np.ones(64, dtype=np.float32)),
            ("log_pred", np.full(64, np.nan)),
        ):
            self.make_fixture()
            reproduction = self.read("reproduction.json")
            anchor = reproduction["anchors"][0]
            path = Path(anchor["path"])
            with np.load(path) as archive:
                arrays = dict(archive)
            np.savez(path, **{**arrays, field: value})
            anchor = {**anchor, "sha256": digest(path)}
            self.save(
                "reproduction.json",
                {**reproduction, "anchors": [anchor] + reproduction["anchors"][1:]},
            )
            binding = self.read("input-binding.json")
            self.save(
                "input-binding.json",
                {
                    **binding,
                    "input_sha256": {
                        **binding["input_sha256"],
                        str(path): digest(path),
                    },
                },
            )
            with self.assertRaises(ValueError):
                verify_module.verify(self.root)
        self.make_fixture()
        cells = self.read("cells.json")["cells"]
        index = 1
        with np.load(self.root / cells[index]["prediction_file"]) as archive:
            pred = archive["log_pred"]
        # Raw values within log tolerance but a large-log Q95 relative shift >1e-4.
        with patch.object(verify_module.np, "allclose", return_value=True):
            self.rewrite_prediction(index, pred + 0.001)
            with self.assertRaisesRegex(ValueError, "Q95"):
                verify_module.verify(self.root)

    def test_training_contract_and_metadata_binding(self) -> None:
        for field, value in (
            ("sample_plan_digest", "0" * 64),
            ("support_sample_count", 1),
            ("last_step", 3999),
            ("initialization_digest", "0" * 64),
        ):
            self.make_fixture()
            payload = self.read("training-records.json")
            record = {**payload["records"][0], field: value}
            metadata_path = self.root / record["metadata_file"]
            write_json(
                metadata_path,
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("metadata_file", "metadata_sha256")
                },
            )
            record = {**record, "metadata_sha256": digest(metadata_path)}
            self.save(
                "training-records.json",
                {**payload, "records": [record, payload["records"][1]]},
            )
            with self.assertRaises(ValueError):
                verify_module.verify(self.root)
        self.make_fixture()
        (self.root / "old-B-metadata.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)

    def test_metrics_low4_is_union_not_mean_of_two_q95s(self) -> None:
        identity = {
            "true": np.full(40, 10.0),
            "n_tables": np.array([4] * 30 + [5] * 10),
        }
        prediction = np.log(identity["true"]) + np.log(
            np.array([1.0] * 30 + [10.0] * 10)
        )
        result = verify_module._recompute_metrics(prediction, identity)
        self.assertAlmostEqual(result["low4"]["q95"], 10.0)
        average = (
            result["joint"]["<100|4"]["q95"] + result["joint"]["<100|5+"]["q95"]
        ) / 2
        self.assertNotAlmostEqual(result["low4"]["q95"], average)
        self.update_cell(
            0,
            metrics={
                **self.read("cells.json")["cells"][0]["metrics"],
                "low4": {"count": 32, "q50": 3.0, "q95": 5.0, "status": "ok"},
            },
        )
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)

    def test_subgroup_and_negative_transfer_gates_are_and(self) -> None:
        cells = self.read("cells.json")["cells"]
        baseline = verify_module._decision(cells)
        self.assertEqual(baseline["decision"], "promising_single_seed_K20")
        for kind in ("all_q50", "four_q95", "low_r", "low_none", "n", "primary"):
            rows = copy.deepcopy(cells)
            for row in rows:
                if row["arm"] != "T_all3" or row["database"] != "chembl":
                    continue
                if row["operator"] == "head3":
                    if kind == "all_q50":
                        row["metrics"]["all"]["q50"] = 3.0
                    elif kind == "four_q95":
                        row["metrics"]["4+"]["q95"] = 2.21
                    elif kind == "low_r":
                        row["metrics"]["low4"]["q95"] = 2.21
                    elif kind == "n":
                        row["metrics"]["low4"]["count"] = 29
                    elif kind == "primary":
                        row["metrics"]["4+"]["q95"] = 2.4
                elif kind == "low_none":
                    row["metrics"]["low4"]["q95"] = 1.0
            with self.subTest(kind=kind):
                self.assertEqual(
                    verify_module._decision(rows)["decision"], "no_candidate"
                )

    def test_independent_reproduction_rejects_mode_and_invalid_operator(self) -> None:
        cells = self.read("cells.json")["cells"]
        reference = [
            c for c in cells if c["arm"] == "R_head3" and c["operator"] == "head3"
        ]
        binding = self.read("input-binding.json")
        self.save("input-binding.json", {**binding, "mode": "cpu_preflight"})
        with self.assertRaises(ValueError):
            verify_module.verify_reproduction(self.root, reference)
        self.save("input-binding.json", binding)
        invalid = [{**reference[0], "steps": 1}] + reference[1:]
        with self.assertRaises(ValueError):
            verify_module.verify_reproduction(self.root, invalid)

    def test_code_sha_is_rehashed_and_plan_digest_is_frozen(self) -> None:
        binding = self.read("input-binding.json")
        self.save(
            "input-binding.json",
            {**binding, "code_sha256": {"scripts/meta65_scope_verify.py": "0" * 64}},
        )
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)
        self.save("input-binding.json", binding)
        payload = self.read("training-records.json")
        records = [
            {**record, "endpoint_plan_digest": "0" * 64}
            for record in payload["records"]
        ]
        self.save("training-records.json", {**payload, "records": records})
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)

    def test_endpoint_plan_digest_is_required_sha(self) -> None:
        payload = self.read("training-records.json")
        records = [
            {**record, "endpoint_plan_digest": "not-a-sha"}
            for record in payload["records"]
        ]
        self.save("training-records.json", {**payload, "records": records})
        with self.assertRaises(ValueError):
            verify_module.verify(self.root)


if __name__ == "__main__":
    unittest.main()
