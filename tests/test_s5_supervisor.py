"""Supervisor content gates. S5_REVIEW_ROOT optionally selects a live checkout."""

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("S5_REVIEW_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location(
    "s5_supervisor_under_test", ROOT / "scripts/run_s5_weight.py"
)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
from scripts import s4_anchor_evidence as s4  # noqa: E402
from test_s4_anchor_evidence import rows  # noqa: E402
from test_s5_weight_evidence import record, preflight  # noqa: E402


class SupervisorContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(
            self.stack.enter_context(tempfile.TemporaryDirectory())
        ).resolve()
        self.plan = s.load_plan()
        self.evidence = {**preflight(), "input_sha256": {"fixture": "a" * 64}}
        self.audit = {"passed": True, "nominal_mass_A": 4000, "nominal_mass_B": 4000}
        self.data = {"splits": self.plan["splits"]}
        self.stack.enter_context(
            patch.object(s4, "COVERAGE", dict.fromkeys(s4.DATABASES, (3, 2)))
        )
        self.stack.enter_context(
            patch.object(
                s.verifier, "SOURCE_HASH", s4._digest(self.evidence["sources"])
            )
        )
        for name, value in (
            ("frozen_inputs", self.evidence["input_sha256"]),
            ("code_hashes", self.evidence["code_sha256"]),
            ("data_settings", self.data),
            ("validate_data", None),
            ("historical_sources", {}),
            ("source_evidence", self.evidence["sources"]),
        ):
            self.stack.enter_context(patch.object(s, name, return_value=value))
        self.flow = self.stack.enter_context(
            patch.object(s, "audit_training_flow", return_value=self.audit)
        )
        self.checkpoints = {}
        self.stack.enter_context(
            patch.object(s.anchor, "load_checkpoint", side_effect=self.load_checkpoint)
        )
        self.original_read = s.read_json
        self.baseline = None
        self.stack.enter_context(
            patch.object(s, "read_json", side_effect=self.read_json)
        )
        self.save("protocol.json", self.plan)
        self.save("preflight.json", self.evidence)
        self.save("weight-audit.json", self.audit)
        self.save("verification.json", {"passed": True})
        self.save("smoke/verification.json", {"passed": True})
        for prefix, formal in (("smoke/", False), ("", True)):
            records = []
            for arm in s.ARMS:
                r = copy.deepcopy(record(arm, formal))
                directory = self.root / prefix / arm
                directory.mkdir(parents=True)
                for initial in (False, True):
                    name = "initial_endpoint" if initial else "endpoint"
                    group = rows()
                    if not initial:
                        delta = 2.0 if arm == s.ARMS[0] else 1.0
                        group = {
                            db: [
                                {**row, "log_pred": row["pg_input_log"] + delta}
                                for row in rs
                            ]
                            for db, rs in group.items()
                        }
                    checked = s.verifier.verify_rows(group, 1.0, None, formal=formal)
                    r[name] = checked["endpoint"]
                    r[("initial_" if initial else "") + "rows_verification"] = checked
                    r[("initial_" if initial else "") + "pg_identity_digest"] = checked[
                        "endpoint"
                    ]["pg_identity_digest"]
                    filename = directory / (name + "_rows.jsonl")
                    filename.write_text(
                        "".join(
                            json.dumps({"database": db, **row}) + "\n"
                            for db, rs in group.items()
                            for row in rs
                        )
                    )
                    r[name + "_rows_sha256"] = s.file_hash(filename)
                (directory / "last.pt").write_bytes(b"checkpoint fixture")
                (directory / "validation_manifest.json").write_text("[]")
                r["checkpoint_sha256"] = s.file_hash(directory / "last.pt")
                r["validation_plan_hash"] = s.file_hash(
                    directory / "validation_manifest.json"
                )
                self.checkpoints[str(directory / "last.pt")] = {
                    "metadata": {
                        "training_signature": {
                            k: v
                            for k, v in r["config"].items()
                            if k not in ("steps", "data_config")
                        }
                    },
                    "extra": {
                        "gradient_stats": copy.deepcopy(r["summary"]["gradient_stats"]),
                        "counts": copy.deepcopy(r["counts"]),
                    },
                }
                self.save(prefix + arm + "/result.json", r)
                records.append(r)
            if formal:
                self.stack.enter_context(
                    patch.object(
                        s.verifier,
                        "VALIDATION_HASH",
                        records[0]["validation_plan_hash"],
                    )
                )
                self.baseline = {
                    "runs": [
                        {
                            "arm": "A_anchor1",
                            "endpoint": copy.deepcopy(records[0]["endpoint"]),
                        }
                    ]
                }
            self.save(prefix + "runs.json", {"runs": records})
            self.save(
                prefix + "summary.json",
                {"runs": records, **(s.verifier.choose(records) if formal else {})},
            )
        s.anchor.output_manifest(self.root)

    def save(self, name: str, value: dict) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        s.write_json(path, value)

    def read_json(self, path: Path) -> dict:
        if Path(path) == s.ROOT / s.BASELINE:
            return copy.deepcopy(self.baseline)
        return self.original_read(path)

    def load_checkpoint(self, path: Path, device: str) -> tuple:
        self.assertEqual(device, "cpu")
        return None, copy.deepcopy(self.checkpoints[str(path)])

    def test_valid_content_recomputes_flow_and_real_decision(self) -> None:
        s.verify_content(self.root)
        self.flow.assert_called_once_with(self.data, self.plan["arms"][s.ARMS[0]])
        self.assertEqual(
            self.read_json(self.root / "summary.json")["decision"],
            "promising_single_seed",
        )

    def test_protocol_mass_summary_result_and_decision_errors(self) -> None:
        for filename, field, value in (
            ("protocol.json", "schema", 999),
            ("weight-audit.json", "nominal_mass_B", 3999),
            ("summary.json", "runs", []),
            (s.ARMS[0] + "/result.json", "parameters", 1),
            ("summary.json", "decision", "no_candidate"),
        ):
            with self.subTest(filename=filename, field=field):
                old = self.read_json(self.root / filename)
                self.save(filename, {**old, field: value})
                with self.assertRaises(ValueError):
                    s.verify_content(self.root)
                self.save(filename, old)

    def test_checkpoint_signature_counts_and_gradient_errors(self) -> None:
        key = str(self.root / s.ARMS[0] / "last.pt")
        old = copy.deepcopy(self.checkpoints[key])
        for group, field, value in (
            ("metadata", "training_signature", {}),
            ("extra", "counts", {}),
            ("extra", "gradient_stats", {}),
        ):
            with self.subTest(field=field):
                self.checkpoints[key] = {**old, group: {**old[group], field: value}}
                with self.assertRaises(ValueError):
                    s.verify_content(self.root)
        self.checkpoints[key] = old

    def test_rows_are_recomputed_not_worker_passed(self) -> None:
        path = self.root / s.ARMS[1] / "endpoint_rows.jsonl"
        content = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text(
            "".join(
                json.dumps({**row, "log_pred": row["log_pred"] + 3}) + "\n"
                for row in content
            )
        )
        with self.assertRaises(ValueError):
            s.verify_content(self.root)

    def test_source_and_code_changes_are_rejected(self) -> None:
        for name in ("source_evidence", "code_hashes"):
            with self.subTest(name=name), patch.object(s, name, return_value={}):
                with self.assertRaises(ValueError):
                    s.verify_content(self.root)

    def test_rehashed_false_decision_rejected_by_output_gate(self) -> None:
        summary = self.read_json(self.root / "summary.json")
        self.save("summary.json", {**summary, "decision": "no_candidate"})
        s.anchor.output_manifest(self.root)
        with self.assertRaises(ValueError):
            s.verify_outputs(self.root)

    def test_rehashed_false_endpoint_rejected_by_output_gate(self) -> None:
        records = self.read_json(self.root / "runs.json")["runs"]
        records[1]["endpoint"]["macro_four_plus_q95"] = 99.0
        self.save("runs.json", {"runs": records})
        summary = self.read_json(self.root / "summary.json")
        self.save("summary.json", {**summary, "runs": records})
        self.save(s.ARMS[1] + "/result.json", records[1])
        s.anchor.output_manifest(self.root)
        with self.assertRaises(ValueError):
            s.verify_outputs(self.root)


class TrainingFlowReadTests(unittest.TestCase):
    def test_reads_all_23_pretrain_stores_before_independent_replay(self) -> None:
        import numpy as np
        from unittest.mock import Mock

        plan = s.load_plan()
        data = {"cache_root": "fixture-cache", "splits": plan["splits"]}
        pool = Mock()
        store = Mock(
            rows=[{"query_hash": "one"}, {"query_hash": "two"}],
            offsets=np.asarray([[0, 0, 0], [0, 0, 3], [0, 0, 7]]),
        )
        store.unique_indices.return_value = np.asarray([0, 1])
        pool.get.return_value = store
        result = {
            "passed": True,
            "sample_plan_digest": s.verifier.SAMPLE_DIGEST,
            "episodes": 16000,
            "presentations": 1621915,
        }
        with (
            patch.object(s, "StorePool", return_value=pool) as factory,
            patch.object(s.verifier, "audit_flow", return_value=result) as replay,
        ):
            self.assertEqual(
                s.audit_training_flow(data, plan["arms"][s.ARMS[0]]), result
            )
        factory.assert_called_once_with("fixture-cache", max_open=23)
        self.assertEqual(
            [call.args for call in pool.get.call_args_list],
            [("pretrain", db) for db in data["splits"]["train"]],
        )
        stores, config = replay.call_args.args
        self.assertEqual(set(stores), set(data["splits"]["train"]))
        self.assertEqual(len(stores), 23)
        self.assertEqual(config, plan["arms"][s.ARMS[0]])
        for item in stores.values():
            self.assertEqual(item["query_hashes"], ["one", "two"])
            np.testing.assert_array_equal(item["tables"], [3, 4])
            np.testing.assert_array_equal(item["unique"], [0, 1])


if __name__ == "__main__":
    unittest.main()
