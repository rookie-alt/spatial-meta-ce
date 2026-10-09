"""CPU-only pre-publication verification of retained A/B plus newly evaluated C.

Only independent verify_endpoint helpers are imported. No runner, torch,
checkpoint deserialization, training, remote access, or decision publication.
"""
import argparse
import csv
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import verify_endpoint as h

EXPERIMENT = "meta65-endpoint-completion-20261009"
CONFIG_SHA = "dab251f10cb37dc3001159fe5672a878badbb2ec1c04e03e49cbd5c60d258bd8"
CHECKPOINT_SHA = "58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4"
PLAN_SHA = "ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0"
CODE = (*h.CODE_FILES, "scripts/complete_meta65_endpoint.py",
        "configs/meta65_screen.json", "configs/meta65_server_data.json",
        "reports/meta65-source-screen-20261009/verify_endpoint.py",
        "reports/meta65-source-screen-20261009/verify_completion.py")
STAGE_FILES = ("stage.json", "status.json", "inputs.json", "environment.json",
               "source_before.json", "source_after.json", "endpoint_plan.json",
               "endpoint.csv", "decision.provisional.json",
               f"{h.ARMS[2]}/metadata.json", f"{h.ARMS[2]}/endpoint.csv")
PHASE = "awaiting_independent_verification"


def check_stage(completion, evidence):
    h.require(not any((completion / name).exists() for name in
                      ("decision.json", "decision.invalid.json")), "Decision already published/invalid")
    for name in STAGE_FILES:
        h.require(h.path_checked(completion / name, completion).is_file(), f"Missing stage file: {name}")
    stage, status = (evidence.read(completion / name) for name in ("stage.json", "status.json"))
    h.equal_tree(stage, {"phase": PHASE, "endpoint_cells": 225}, "stage")
    h.require(status.get("status") == "running" and status.get("phase") == PHASE
              and status.get("experiment") == EXPERIMENT and status.get("decision_valid") is False
              and type(status.get("endpoint_cells")) is int and status["endpoint_cells"] == 225
              and not any(k in status for k in ("error", "error_type", "failure_reason")),
              "Stage is not awaiting complete independent verification")
    h.require(h.number(status["elapsed_seconds"], "stage.elapsed_seconds") < 3000,
              "Completion deadline already exceeded")
    return status


def load_combined(path, evidence):
    evidence.bind(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        h.require(tuple(reader.fieldnames or ()) == (*h.CSV_FIELDS, "source_run"),
                  "Unexpected completion CSV schema")
        rows = list(reader)
    for row in rows:
        h.require(set(row) == {*h.CSV_FIELDS, "source_run"}
                  and all(v is not None for v in row.values()), "Malformed completion CSV row")
        for key in ("k", "run", "query_count", "four_plus_count"):
            text = row[key]
            h.require(text.isdigit() and str(int(text)) == text, f"Invalid CSV integer: {key}")
            row[key] = int(text)
        for key in (*h.METRICS, *h.LOG_METRICS, "adaptation_seconds", "prediction_seconds"):
            row[key] = h.number(float(row[key]), key, 1 if key in h.METRICS else 0)
        h.valid_sha(row["prediction_sha256"])
    return rows


def without_source(row):
    return {k: v for k, v in row.items() if k != "source_run"}


def prediction_path(row, retained, completion):
    arm, db, k, run = (row[key] for key in ("arm", "database", "k", "run"))
    h.require(arm in h.ARMS and db in h.VAL and type(k) is int and k in h.SHOTS
              and type(run) is int and run in range(5), "Unexpected prediction cell")
    source = "completion" if arm == h.ARMS[2] else "retained"
    h.require(row.get("source_run") == source, "Wrong source_run for arm")
    expected = f"{arm}/predictions/{db}-k{k}-run{run}.npz"
    h.require(row["prediction_file"] == expected, "Prediction path is not the frozen source cell")
    base = completion if source == "completion" else retained
    return h.path_checked(base / expected, base)


def verify_records(retained, completion, config, sources, plan_digest, evidence):
    records = {}
    for arm in h.ARMS:
        is_c = arm == h.ARMS[2]
        directory = (completion if is_c else retained) / arm
        record = evidence.read(directory / "metadata.json")
        expected_config = {**config["training"], "model": {**config["model"], "width": 64},
                           "loss": "smooth_l1", "adapt_scope": config["arms"][arm]["train_scope"],
                           "inner_steps": config["arms"][arm]["train_steps"], "purpose": h.EXPERIMENT}
        h.equal_tree(record["config"], expected_config, f"{arm}.config")
        counts = {"support_presentations": 2989575 if arm == h.ARMS[0] else 1793745,
                  "query_presentations": 1024000,
                  "backward_calls": 96000 if arm == h.ARMS[0] else 64000, "optimizer_steps": 4000}
        h.equal_tree(record["counts"], counts, f"{arm}.counts")
        for key, value in {"arm": arm, "parameters": 132545, "optimizer_steps": 4000,
                           "last_step": 4000, "sample_plan_episodes": 16000,
                           "support_sample_count": 597915, "endpoint_cells": 75,
                           "endpoint_plan_digest": plan_digest}.items():
            h.equal_tree(record[key], value, f"{arm}.{key}")
        metadata = record["metadata"]
        signature = {k: v for k, v in expected_config.items() if k not in ("steps", "data_config")}
        for key, value in {"training_signature": signature, "train_databases": list(h.TRAIN),
                           "validation_databases": list(h.VAL), "method": "fomaml", "seed": 42,
                           "purpose": h.EXPERIMENT,
                           "manifest_hashes": {db: info["manifest_hash"] for db, info in sources.items()}}.items():
            h.equal_tree(metadata[key], value, f"{arm}.metadata.{key}")
        for key in ("sample_plan_digest", "initialization_digest", "validation_plan_digest"):
            h.valid_sha(record[key])
        evidence.bind(retained / arm / "validation_manifest.json", record["validation_plan_digest"])
        evidence.bind(retained / arm / "last.pt", record["checkpoint_sha256"])
        h.number(record["endpoint_seconds"], f"{arm}.endpoint_seconds")
        if is_c:
            h.equal_tree(record["checkpoint_sha256"], CHECKPOINT_SHA, "C.checkpoint_sha256")
            h.equal_tree(record["new_training_updates"], 0, "C.new_training_updates")
            h.require(record["checkpoint"] == "retained_C_last.pt"
                      and record["initialization_digest_source"] == "authenticated_shared_A_training_record"
                      and record["model_state_unchanged"] is True
                      and not any(k in record for k in ("summary", "optimizer", "training_seconds")),
                      "C must describe retained checkpoint and evaluation only")
            h.number(record["training_seconds_from_checkpoint"], "C.retained_training_seconds")
        else:
            h.require(record["checkpoint"] == "last.pt", "Retained arm must use last checkpoint")
            summary = record["summary"]
            for key, value in {"counts": counts, "last_step": 4000, "parameters": 132545,
                               "purpose": h.EXPERIMENT}.items():
                h.equal_tree(summary[key], value, f"{arm}.summary.{key}")
            h.require(h.number(summary["total_seconds"], "summary.total_seconds")
                      <= h.number(record["training_seconds"], "training_seconds") + h.ATOL,
                      "Retained training duration order mismatch")
        records[arm] = record
    for key in ("initialization_digest", "sample_plan_digest", "validation_plan_digest"):
        h.require(len({record[key] for record in records.values()}) == 1, f"Cross-arm pairing mismatch: {key}")
    return records


def recompute(rows, retained, completion, plans, stores, evidence):
    h.independent_decision([without_source(row) for row in rows])
    recomputed, hashes = [], set()
    for row in rows:
        arm, db, k, run = (row[key] for key in ("arm", "database", "k", "run"))
        plan, store = plans[db][run], stores[db]
        query = plan["query"]
        h.require(row["query_digest"] == plan["query_digest"]
                  and row["support_digest"] == plan["support_digests"][str(k)]
                  and row["query_count"] == len(query) == h.UNIQUE[db] - 100,
                  "CSV query/support identity/count mismatch")
        path = prediction_path(row, retained, completion)
        hashes.add(evidence.bind(path, row["prediction_sha256"]))
        with np.load(path, allow_pickle=False) as archive:
            h.require(len(archive.files) == len(h.NPZ_FIELDS) and set(archive.files) == h.NPZ_FIELDS,
                      "Duplicate/unexpected NPZ arrays")
            arrays = {key: archive[key] for key in h.NPZ_FIELDS}
        h.check_arrays(arrays, store, query)
        mask = arrays["n_tables"] >= 4
        h.require(row["four_plus_count"] == int(mask.sum()) > 0, "Four-plus count mismatch")
        measured = {}
        for prefix, selection in (("all", np.ones(len(query), dtype=bool)), ("four_plus", mask)):
            metrics = h.q_metrics(arrays["log_pred"][selection], arrays["true"][selection])
            measured.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        for key, value in measured.items():
            h.equal_tree(row[key], value, f"{arm}/{db}/{k}/{run}.{key}")
        recomputed.append({**without_source(row), **measured})
    return recomputed, hashes


def verify(root, completion, out):
    started = time.monotonic()
    root = h.path_checked(root)
    completion = h.path_checked(completion, root / "runs")
    out = h.path_checked(out)
    evidence = h.Evidence()
    config_path = root / "configs/meta65_completion.json"
    evidence.bind(config_path, CONFIG_SHA)
    config = evidence.read(config_path)
    retained_root = h.path_checked(config["retained_root"])
    retained = h.path_checked(config["retained_run"], retained_root / "runs")
    h.require(root != retained_root and not root.is_relative_to(retained_root)
              and completion != root / "runs" and completion.is_dir(), "Separate completion root/run required")
    h.require(config["mode"] == "endpoint_completion_only" and config["timeout_seconds"] == 3000
              and config["experiment"] == EXPERIMENT and config["checkpoint_sha256"] == CHECKPOINT_SHA
              and config["endpoint_plan_sha256"] == PLAN_SHA, "Frozen completion contract changed")
    evidence.bind(root / config["protocol_file"], config["protocol_sha256"])
    status = check_stage(completion, evidence)
    h.equal_tree(evidence.read(completion / "inputs.json"), config, "inputs")
    env = evidence.read(completion / "environment.json")
    h.equal_tree(env["retained_inputs_sha256"], config["retained_sha256"], "retained_inputs_sha256")
    for name, expected in config["retained_sha256"].items():
        evidence.bind(h.path_checked(retained_root / name, retained_root), expected)
    old_status = evidence.read(retained / "status.json")
    old_supervisor = evidence.read(retained.with_name(retained.name + ".supervisor.json"))
    h.require(old_status["status"] == old_supervisor["status"] == "failed"
              and old_status["failure_reason"] == "supervisor_timeout" and old_status["decision_valid"] is False
              and type(old_supervisor["exit_code"]) is int and old_supervisor["exit_code"] == 124
              and old_supervisor["timed_out"] is True and old_supervisor["failure_reason"] == "timeout"
              and old_supervisor["decision_valid"] is False and not (retained / "decision.json").exists(),
              "Retained run must remain failed_timeout without decision")
    old_env = evidence.read(retained / "environment.json")
    h.require(set(env["code_sha256"]) == set(CODE) and set(old_env["code_sha256"]) == set(h.CODE_FILES),
              "Code input coverage mismatch")
    for name in CODE:
        evidence.bind(h.path_checked(root / name, root), env["code_sha256"][name])
        if name in h.CODE_FILES:
            h.equal_tree(env["code_sha256"][name], old_env["code_sha256"][name], f"retained code {name}")
    h.equal_tree(env["config_sha256"], CONFIG_SHA, "environment.config_sha256")
    h.equal_tree(env["protocol_sha256"], config["protocol_sha256"], "environment.protocol_sha256")
    h.equal_tree(env["new_training_updates"], 0, "environment.new_training_updates")
    h.equal_tree(env["external_timeout_required_seconds"], 3000, "environment.timeout")
    h.require(env["deterministic_algorithms"] is True
              and env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
              and env["device"] in ("cuda", "cuda:0")
              and isinstance(env["gpu_uuid"], str) and env["gpu_uuid"].startswith("GPU-")
              and env["CUDA_VISIBLE_DEVICES"] == env["gpu_uuid"], "Frozen single-GPU runtime mismatch")
    evidence.bind(root / "configs/meta65_screen.json", h.CONFIG_SHA)
    evidence.bind(root / "configs/meta65_server_data.json", h.DATA_SHA)
    original = evidence.read(root / "configs/meta65_screen.json")
    evidence.bind(root / h.PROTOCOL, h.PROTOCOL_SHA)
    data = evidence.read(root / "configs/meta65_server_data.json")
    h.require(data["splits"]["train"] == list(h.TRAIN) and data["splits"]["validation"] == list(h.VAL),
              "Source-only data split mismatch")
    cache = h.path_checked(data["cache_root"])
    h.require(out.suffix == ".json" and not out.exists() and out.parent.is_dir()
              and not out.is_relative_to(retained_root) and not out.is_relative_to(cache)
              and not out.is_relative_to(Path("/gly/weilin/PRICE-master"))
              and (not out.is_relative_to(root / "runs") or out.parent == completion),
              "Output must be a new JSON outside retained/cache/PRICE inputs")
    sources = h.source_records(root, completion, evidence)
    h.equal_tree(sources, evidence.read(retained / "source_before.json"), "retained source records")
    stores = h.load_validation(cache, sources, evidence)
    evidence.bind(retained / "endpoint_plan.json", PLAN_SHA)
    plans = evidence.read(completion / "endpoint_plan.json")
    h.equal_tree(plans, evidence.read(retained / "endpoint_plan.json"), "retained endpoint plan")
    h.verify_plans(plans, stores)
    records = verify_records(retained, completion, original, sources, h.digest(plans), evidence)
    rows = load_combined(completion / "endpoint.csv", evidence)
    h.independent_decision([without_source(row) for row in rows])
    for arm in h.ARMS:
        base = completion if arm == h.ARMS[2] else retained
        per_arm = h.load_csv(base / arm / "endpoint.csv", evidence)
        h.equal_tree([without_source(row) for row in rows if row["arm"] == arm], per_arm, f"{arm}.CSV")
    recomputed, hashes = recompute(rows, retained, completion, plans, stores, evidence)
    decision = h.independent_decision(recomputed)
    h.equal_tree(evidence.read(completion / "decision.provisional.json"), decision, "provisional decision")
    for arm in h.ARMS:
        seconds = math.fsum(row[key] for row in recomputed if row["arm"] == arm
                            for key in ("adaptation_seconds", "prediction_seconds"))
        h.require(seconds <= records[arm]["endpoint_seconds"] + h.ATOL, f"{arm} endpoint timing mismatch")
    evidence.recheck()
    check_stage(completion, evidence)
    result = {"verification": "passed", "checked225": len(recomputed),
              "scope": "completed_source_endpoint_matrix_after_old_timeout; not retroactive old run success",
              "macros": {arm: {k: v for k, v in decision["arms"][arm].items() if k.startswith("macro_")}
                         for arm in h.ARMS},
              "gates": {arm: {k: decision["arms"][arm][k] for k in
                              ("improvement_gate", "regressions", "eligible")} for arm in h.ARMS[1:]},
              "decision": {k: decision[k] for k in ("status", "selected_arm", "candidates")},
              "verified_counts": {"source_databases": 26, "source_rows": 1300000,
                                  "cache_databases_loaded": list(h.VAL), "prediction_sha256_bound": 225,
                                  "unique_prediction_sha256": len(hashes), "checkpoint_sha256_bound": 3,
                                  "retained_input_sha256_bound": len(config["retained_sha256"]),
                                  "code_sha256_bound": len(CODE), "evidence_files_sha256_bound": len(evidence.hashes)},
              "config_sha256": CONFIG_SHA, "endpoint_plan_digest": h.digest(plans),
              "verification_seconds": time.monotonic() - started,
              "limits": {"csv_rtol": h.RTOL, "csv_atol": h.ATOL, "new_training_updates": 0,
                         "stage_elapsed_seconds_before_verification": status["elapsed_seconds"],
                         "timeout_seconds": 3000, "checkpoint_deserialized": False,
                         "raw_predictions_unclipped": True, "query_rows_filtered": 0,
                         "targets_loaded": False, "parent_supervisor_exit_checked": False,
                         "source26_hashes_are_records_not_full_cache_rehash": True,
                         "training_metadata_and_model_unchanged_are_hash_bound_records": True,
                         "C_initialization_digest_source": "authenticated_shared_A_training_record",
                         "C_initial_parameter_hash_recomputed": False,
                         "not_significance_target_generalization_or_PRICE_superiority": True}}
    with out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def self_test():
    class SyntheticTests(unittest.TestCase):
        def rows(self):
            return [{"arm": arm, "database": db, "k": k, "run": run,
                     **{metric: (10.0 if arm == h.ARMS[0] else 9.0) for metric in h.METRICS},
                     **dict.fromkeys(h.LOG_METRICS, 1.0)}
                    for arm in h.ARMS for db in h.VAL for k in h.SHOTS for run in range(5)]

        def test_source_selection_and_path_rejection(self):
            with tempfile.TemporaryDirectory() as folder:
                retained, completion = Path(folder).resolve() / "old", Path(folder).resolve() / "new"
                for arm in h.ARMS:
                    row = {"arm": arm, "database": "chembl", "k": 5, "run": 0,
                           "prediction_file": f"{arm}/predictions/chembl-k5-run0.npz",
                           "source_run": "completion" if arm == h.ARMS[2] else "retained"}
                    base = completion if arm == h.ARMS[2] else retained
                    self.assertEqual(prediction_path(row, retained, completion), base / row["prediction_file"])
                    with self.assertRaises(ValueError):
                        prediction_path({**row, "source_run": "wrong"}, retained, completion)
                    with self.assertRaises(ValueError):
                        prediction_path({**row, "prediction_file": "../other.npz"}, retained, completion)

        def stage_fixture(self, root):
            for name in STAGE_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            (root / "stage.json").write_text(json.dumps({"phase": PHASE, "endpoint_cells": 225}))
            status = {"status": "running", "phase": PHASE, "experiment": EXPERIMENT,
                      "decision_valid": False, "endpoint_cells": 225, "elapsed_seconds": 1.0}
            (root / "status.json").write_text(json.dumps(status))
            return status

        def test_stage_running_allowed_failure_missing_and_published_rejected(self):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                status = self.stage_fixture(root)
                check_stage(root, h.Evidence())
                (root / "status.json").write_text(json.dumps({**status, "status": "failed"}))
                with self.assertRaises(ValueError):
                    check_stage(root, h.Evidence())
                self.stage_fixture(root)
                (root / "source_after.json").unlink()
                with self.assertRaises(ValueError):
                    check_stage(root, h.Evidence())
                self.stage_fixture(root)
                (root / "decision.json").write_text("{}")
                with self.assertRaises(ValueError):
                    check_stage(root, h.Evidence())

        def test_stage_tampering_rejected(self):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder).resolve()
                self.stage_fixture(root)
                evidence = h.Evidence()
                check_stage(root, evidence)
                (root / "stage.json").write_text(json.dumps({"phase": PHASE, "endpoint_cells": 224}))
                with self.assertRaises(ValueError):
                    check_stage(root, evidence)

        def test_exact_225_and_gate_recomputation(self):
            rows = self.rows()
            decision = h.independent_decision(rows)
            self.assertEqual(decision["selected_arm"], h.ARMS[2])
            self.assertEqual(decision["arms"][h.ARMS[0]]["macro_four_plus_q95"], 10.0)
            for row in rows:
                if row["arm"] != h.ARMS[0] and row["database"] == "telstra" and row["k"] == 5:
                    row["all_q95"] = 11.01
            failed = h.independent_decision(rows)
            self.assertEqual(failed["status"], "no_candidate")
            self.assertTrue(failed["arms"][h.ARMS[2]]["improvement_gate"])
            self.assertFalse(failed["arms"][h.ARMS[2]]["eligible"])
            with self.assertRaises(ValueError):
                h.independent_decision(rows[:-1])
            with self.assertRaises(ValueError):
                h.independent_decision([*rows[:-1], rows[0]])

        def test_combined_csv_and_225_raw_npz_recomputed(self):
            with tempfile.TemporaryDirectory() as folder:
                retained, completion = Path(folder).resolve() / "old", Path(folder).resolve() / "new"
                completion.mkdir()
                stores, plans, rows = {}, {}, []
                for db in h.VAL:
                    stores[db] = {"cards": np.array([[2.0, 1.0], [4.0, 1.0]]),
                                  "tables": np.array([4, 2]),
                                  "rows": [{"id": f"{db}/pretrain/{i}", "query_hash": f"{db}-{i}"}
                                           for i in range(2)]}
                    plans[db] = [{"query": [0, 1], "query_digest": h.digest([db, "query"]),
                                  "support_digests": {str(k): h.digest([db, run, k]) for k in h.SHOTS}}
                                 for run in range(5)]
                for original_row in self.rows():
                    arm, db, k, run = (original_row[key] for key in ("arm", "database", "k", "run"))
                    store = stores[db]
                    arrays = {"log_pred": np.log(store["cards"][:, 0]) + math.log(10 if arm == h.ARMS[0] else 8),
                              "true": store["cards"][:, 0].copy(), "n_tables": store["tables"],
                              "query_hash": np.array([r["query_hash"] for r in store["rows"]]),
                              "query_id": np.array([r["id"] for r in store["rows"]])}
                    row = {**original_row, "source_run": "completion" if arm == h.ARMS[2] else "retained",
                           "prediction_file": f"{arm}/predictions/{db}-k{k}-run{run}.npz",
                           "query_digest": plans[db][run]["query_digest"],
                           "support_digest": plans[db][run]["support_digests"][str(k)],
                           "query_count": 2, "four_plus_count": 1,
                           "adaptation_seconds": 0.0, "prediction_seconds": 0.0}
                    path = prediction_path(row, retained, completion)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, **arrays)
                    row["prediction_sha256"] = h.sha256(path)
                    for prefix, mask in (("all", np.array([True, True])), ("four_plus", np.array([True, False]))):
                        row.update({f"{prefix}_{key}": value for key, value in
                                    h.q_metrics(arrays["log_pred"][mask], arrays["true"][mask]).items()})
                    rows.append(row)
                csv_path = completion / "endpoint.csv"
                with csv_path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=(*h.CSV_FIELDS, "source_run"))
                    writer.writeheader()
                    writer.writerows(rows)
                evidence = h.Evidence()
                loaded = load_combined(csv_path, evidence)
                # Small synthetic denominators apply only inside this isolated test.
                with patch.object(h, "UNIQUE", dict.fromkeys(h.VAL, 102)):
                    checked, _ = recompute(loaded, retained, completion, plans, stores, evidence)
                    self.assertEqual(len(checked), 225)
                    self.assertEqual(h.independent_decision(checked)["selected_arm"], h.ARMS[2])
                    loaded[0]["all_q95"] += 1.0
                    with self.assertRaises(ValueError):
                        recompute(loaded, retained, completion, plans, stores, h.Evidence())
                evidence.recheck()

        def test_wrong_labels_rejected(self):
            store = {"cards": np.array([[2.0, 1.0]]), "tables": np.array([4]),
                     "rows": [{"query_hash": "synthetic", "id": "chembl/pretrain/0"}]}
            arrays = {"log_pred": np.array([1.0]), "true": np.array([2.0]), "n_tables": np.array([4]),
                      "query_hash": np.array(["synthetic"]), "query_id": np.array(["chembl/pretrain/0"])}
            h.check_arrays(arrays, store, [0])
            arrays["true"][0] = np.nextafter(2.0, 3.0)
            with self.assertRaises(ValueError):
                h.check_arrays(arrays, store, [0])

        def test_provisional_decision_tampering_rejected(self):
            decision = h.independent_decision(self.rows())
            with self.assertRaises(ValueError):
                h.equal_tree({**decision, "selected_arm": h.ARMS[1]}, decision, "decision")

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
    return 0 if result.wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--completion", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if any(value is not None for value in (args.root, args.completion, args.out)):
            parser.error("--self-test is synthetic-only; formal paths are forbidden")
        return self_test()
    if any(value is None for value in (args.root, args.completion, args.out)):
        parser.error("--root, --completion and --out are required")
    try:
        result = verify(args.root, args.completion, args.out)
    except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
        print(f"Completion verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: result[key] for key in ("verification", "checked225", "decision")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
