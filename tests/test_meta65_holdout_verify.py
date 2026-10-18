import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import meta65_holdout_contract as contract
from scripts import meta65_holdout_report as report
from scripts import meta65_holdout_verify as verifier


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def build_plan() -> dict:
    indices = list(range(130))
    hashes = [f"{index:064x}" for index in indices]
    return {
        database: [
            contract.build_run_plan(database, indices, hashes, seed=2718, run=run)
            for run in range(5)
        ]
        for database in verifier.DATABASES
    }


def build_output(root: Path, *, formal: bool, empty_low4: bool = False) -> Path:
    output = root / ("formal" if formal else "preflight")
    output.mkdir()
    plan = build_plan()
    plan_path = root / "plan.json"
    write_json(plan_path, plan)
    cells = []
    predictions = output / "predictions"
    predictions.mkdir()
    true = np.linspace(500.0, 5000.0, 30, dtype=np.float64)
    if not empty_low4:
        true[:2] = 50.0
    tables = np.full(30, 4, dtype=np.int16)
    for database in verifier.DATABASES:
        for operator in verifier.OPERATORS:
            for run in range(5 if formal else 1):
                query = plan[database][run]["query"][:30]
                path = predictions / f"{operator}-{database}-run{run}.npz"
                pred = np.log(np.maximum(true, 1)) + (0.1 if operator == "none" else 0.05)
                np.savez(path, log_pred=pred, true=true, n_tables=tables,
                         query_hash=np.array([f"h{index}" for index in query], dtype="S64"),
                         query_id=np.array([f"{database}/pretrain/{index}" for index in query],
                                           dtype="S128"))
                metrics = verifier.buckets(pred, true, tables)
                cells.append({
                    "arm": "R_head3", "operator": operator, "database": database, "run": run,
                    "k20": 20, "query_count": len(query),
                    "query_digest": plan[database][run]["query_digest"],
                    "support_digest": plan[database][run]["support_digests"]["20"],
                    "prediction_file": str(path.relative_to(output)),
                    "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "state_before": "a" * 64, "state_after": "a" * 64,
                    "metrics": metrics,
                    "selected_parameter_count": 0 if operator == "none" else 107521,
                    "selected_parameter_names": [] if operator == "none" else ["head.4.weight"],
                    "changed_parameter_names": [] if operator == "none" else ["head.4.weight"],
                    "steps": 0 if operator == "none" else 3,
                })
    write_json(output / ("cells.json" if formal else "microcells.json"), {
        "schema": 1, "experiment": verifier.EXPERIMENT,
        "mode": "formal" if formal else "cpu_preflight", "cells": cells,
    })
    write_json(output / "status.json", {
        "schema": 1, "experiment": verifier.EXPERIMENT,
        "status": "worker_complete", "mode": "formal" if formal else "cpu_preflight",
        "cells": 50 if formal else 0, "micro_cells": 0 if formal else 10,
        "training": True, "target_read": False, "selection_performed": False,
        "formal_result": formal,
    })
    write_json(output / "environment.json", {
        "mode": "formal" if formal else "cpu_preflight", "training": True,
        "target_read": False,
    })
    (output / "R_head3").mkdir()
    write_json(output / "R_head3" / "config.json", {
        "data": {"splits": {"test": list(verifier.TARGETS)}},
    })
    write_json(output / "R_head3" / "metadata.json", {
        "arm": "R_head3",
        "data_signature": "b" * 64,
        "metadata": {
            "train_databases": list(verifier.TRAIN),
            "validation_databases": list(verifier.VALIDATION),
        },
    })
    dummy = root / "dummy.txt"
    dummy.write_text("source", encoding="utf-8")
    input_map = {str(dummy): hashlib.sha256(dummy.read_bytes()).hexdigest()}
    write_json(output / "input-binding.json", {
        "schema": 1, "mode": "formal" if formal else "cpu_preflight",
        "plan_file": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "input_sha256": input_map,
    })
    return output


class HoldoutVerifierTests(unittest.TestCase):
    def test_empty_bucket_is_null_and_not_an_error(self):
        empty = verifier.metric(np.array([], dtype=np.float64))
        self.assertEqual(empty, {
            "count": 0, "status": "empty", "q50": None, "q95": None,
        })
        verifier.compare_metric(empty, empty, "synthetic-low4")

    def test_report_accepts_empty_low4_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = build_output(Path(directory), formal=True, empty_low4=True)
            value = report.build_report(output)
            self.assertIsNone(
                value["groups"]["validation"]["none"]["per_database"]["chembl"]["low4_q95"]
            )

    def test_preflight_verifier(self):
        with tempfile.TemporaryDirectory() as directory:
            output = build_output(Path(directory), formal=False)
            result = verifier.verify(output, formal=False)
            self.assertEqual(result["status"], "preflight_verified")
            self.assertEqual(result["cells_verified"], 10)

    def test_formal_report_never_selects_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = build_output(Path(directory), formal=True)
            value = report.build_report(output)
            write_json(output / "report.json", value)
            verified = verifier.verify(output, formal=True)
            self.assertEqual(value["status"], "baseline_verified")
            self.assertFalse(value["selection_performed"])
            self.assertIsNone(value["selected_arm"])
            self.assertEqual(set(value["groups"]), {"validation", "holdout"})
            self.assertEqual(verified["status"], "formal_verified")


if __name__ == "__main__":
    unittest.main()
