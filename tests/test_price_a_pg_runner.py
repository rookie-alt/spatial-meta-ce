"""Exploratory PG input must not bypass the frozen A training gates."""

import unittest

from scripts import run_price_meta_a as runner


class ExploratoryRunnerTests(unittest.TestCase):
    def test_exploratory_keys_are_single_run_only(self):
        spec = {"execution_mode": "single_run", "exploratory_arm": "A1-PG", "run_key": "A1-PG-seed42"}
        self.assertEqual(runner.execution_keys(spec), ["A1-PG-seed42"])
        with self.assertRaises(ValueError):
            runner.execution_keys({**spec, "execution_mode": "matrix"})
        with self.assertRaises(ValueError):
            runner.execution_keys({**spec, "run_key": "A1-seed42"})
        with self.assertRaises(ValueError):
            runner.execution_keys({**spec, "exploratory_arm": "A1"})

    def test_frozen_matrix_does_not_accept_exploratory_config(self):
        from unittest.mock import patch

        frozen = {"model": {"arch": "price_lite"}, "seed": 42, "fixed_stage": "122-A1"}
        spec = {"execution_mode": "single_run", "exploratory_arm": "A1-PG", "run_key": "A1-PG-seed42"}
        with patch.object(runner, "read_bound_json", side_effect=AssertionError("unbound")):
            with self.assertRaises(ValueError):
                runner.validate_exploratory_spec({**spec, "arms": ["A1-PG"]})
        with self.assertRaises(ValueError):
            runner.exploratory_config("A1-PG-seed43.json", frozen)

    def test_exploratory_requires_live_callback_and_dedicated_certificate(self):
        from unittest.mock import patch

        spec = {
            "execution_mode": "single_run", "exploratory_arm": "A1-PG",
            "run_key": "A1-PG-seed42", "arms": list(runner.ARMS),
            "seeds": list(runner.SEEDS), "total_runs": runner.TOTAL_RUNS,
            "total_cells": runner.TOTAL_CELLS,
            "exploratory_config_dir": "/synthetic/exploratory-configs",
        }
        runner.validate_exploratory_spec(spec)
        with patch.object(runner, "read_p0_certificate", return_value={"bindings": {"a_training_configs": []}}):
            with self.assertRaises(ValueError):
                runner.require_exploratory_certificate(spec, {})

    def test_new_arm_evaluation_emits_unit_progress(self):
        import inspect
        from spatial_meta_ce.source_eval import evaluate_source

        self.assertIn("on_unit", inspect.signature(evaluate_source).parameters)

    def test_pg_config_dir_is_not_the_historical_rebound(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            self.assertRaises(ValueError, runner.validate_exploratory_spec, {
                "execution_mode": "single_run", "exploratory_arm": "A1-PG",
                "run_key": "A1-PG-seed42", "arms": list(runner.ARMS),
                "seeds": list(runner.SEEDS), "total_runs": runner.TOTAL_RUNS,
                "total_cells": runner.TOTAL_CELLS,
                "rebound": str(Path(directory)),
                "exploratory_config_dir": str(Path(directory)),
            })

    def test_pg_configs_are_derived_only_from_frozen_a1(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rebound = root / "rebound"
            rebound.mkdir()
            output = root / "pg-configs"
            manifest = {"p0_gate_passed": False, "formal_training_authorized": False, "files": {}}
            for seed in runner.SEEDS:
                name = f"A1-seed{seed}.json"
                path = rebound / name
                path.write_text('{"seed": %d, "fixed_stage": "122-A1"}' % seed)
                manifest["files"][name] = {"sha256": runner.file_sha256(path)}
            (rebound / "binding_manifest.json").write_text(json.dumps(manifest))
            with patch.object(runner, "exploratory_config", side_effect=lambda name, original: {**original, "fixed_stage": "122-A1-PG"}):
                runner.prepare_exploratory_configs(rebound, output)
            self.assertEqual(len(list(output.glob("A1-PG-seed*.json"))), 3)
            self.assertEqual(len(list(rebound.glob("*.json"))), 4)
            with self.assertRaises(FileExistsError):
                runner.prepare_exploratory_configs(rebound, output)

    def test_certificate_builder_never_signs_or_authorizes(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text('{"schema": 2, "status": "signed", "truth": {"p0_gate_passed": true, "formal_training_authorized": false, "research_training_started": false, "protocol_122_executed": false}, "bindings": {"a_training_configs": [], "frozen_code": {}}}')
            (root / "pg-configs").mkdir()
            for seed in runner.SEEDS:
                (root / "pg-configs" / f"A1-PG-seed{seed}.json").write_text('{}')
            with (
                patch("scripts.verify_p0_certificate.verify", return_value=[]),
                patch.object(runner, "compute_code_sha256", return_value={"scripts/run_price_meta_a.py": "a" * 64}),
            ):
                record = runner.prepare_exploratory_certificate(
                    source, root / "new-code", root / "pg-configs", root / "new-certificate.json"
                )
            self.assertEqual(record["status"], "unsigned")
            self.assertIsNone(record["issued_by"])
            self.assertFalse(record["truth"]["formal_training_authorized"])

    def test_exploratory_eval_requires_33_bound_units(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"databases": {name: {"query_ids": [f"{name}/1"], "query_hashes": ["h"], "query": [0], "support_blocks": [{"support_ids": ["s"] * 20}]} for name in runner.EVAL_DATABASES}}
            protocol = {"arm": "A1-PG", "seed": 42, "kind": "price_source_eval_122_pg_input_exploratory", "completed": True, "target_databases_read": [], "units": [], "source_eval_plan_sha256": "a" * 64}
            plan["plan_sha256"] = "a" * 64
            (root / "predictions.csv").write_text("arm,seed,database,k,run,query_id,query_hash,true_cardinality,log_prediction\n")
            protocol["predictions_sha256"] = runner.file_sha256(root / "predictions.csv")
            with patch.object(runner, "recompute_units", return_value={}):
                with self.assertRaisesRegex(ValueError, "33|complete"):
                    runner.verify_exploratory_evaluation(protocol, root / "predictions.csv", plan, 42)

    def test_exploratory_evaluation_rejects_wrong_population_and_metric(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "predictions.csv"
            rows = ["arm,seed,database,k,run,query_id,query_hash,true_cardinality,log_prediction"]
            units = []
            databases = {}
            for db in runner.EVAL_DATABASES:
                databases[db] = {
                    "query": [0], "query_ids": [f"{db}/0"],
                    "query_hashes": [f"hash-{db}"],
                    "support_blocks": [{"support_ids": [f"{db}/support"] * 20}] * 5,
                }
                for k, runs in ((0, (0,)), (5, range(1, 6)), (20, range(1, 6))):
                    for run in runs:
                        rows.append(f"A1-PG,42,{db},{k},{run},{db}/0,hash-{db},10,0")
                        units.append({"database": db, "k": k, "run": run,
                                      "support_ids": [] if k == 0 else [f"{db}/support"] * k,
                                      "metrics": {"log_q95": 2.302585092994046}})
            path.write_text("\n".join(rows) + "\n")
            plan = {"plan_sha256": "a" * 64, "databases": databases}
            protocol = {"kind": "price_source_eval_122_pg_input_exploratory", "arm": "A1-PG",
                        "seed": 42, "completed": True, "target_databases_read": [],
                        "source_eval_plan_sha256": "a" * 64,
                        "predictions_sha256": runner.file_sha256(path), "units": units}
            self.assertEqual(runner.verify_exploratory_evaluation(protocol, path, plan, 42)["units"], 33)
            damaged = dict(protocol)
            damaged["units"] = [{**units[0], "metrics": {"log_q95": 1}}, *units[1:]]
            with self.assertRaisesRegex(ValueError, "metric"):
                runner.verify_exploratory_evaluation(damaged, path, plan, 42)
            wrong = {**plan, "databases": {**databases, "chembl": {**databases["chembl"], "query_hashes": ["wrong"]}}}
            with self.assertRaisesRegex(ValueError, "population"):
                runner.verify_exploratory_evaluation(protocol, path, wrong, 42)
            invalid = path.read_text().replace("A1-PG,42,chembl", "A1,42,chembl")
            path.write_text(invalid)
            altered_protocol = {**protocol, "predictions_sha256": runner.file_sha256(path)}
            with self.assertRaisesRegex(ValueError, "arm|seed"):
                runner.verify_exploratory_evaluation(altered_protocol, path, plan, 42)

    def test_certificate_verifier_rejects_unsigned_and_tampered_files(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.verify_p0_certificate import verify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "evidence.json"
            payload.write_text('{"safe": true}')
            digest = runner.file_sha256(payload)
            cert = root / "certificate.json"
            bindings = {
                "frozen_code": {"path": str(payload), "sha256": digest},
                "a_training_configs": [{"path": str(payload), "sha256": digest}],
                "training_plans": [{"path": str(payload), "sha256": digest}],
                "cache_manifests": [{"path": str(payload), "sha256": digest, "content_hash": "none"}],
                "source_preflight": {"path": str(payload), "sha256": digest},
                "source_eval_plan": {"path": str(payload), "sha256": digest, "plan_sha256": "a" * 64},
                "data_config_122": {"path": str(payload), "sha256": digest},
            }
            record = {"schema": 2, "protocol": 122, "purpose": "p0-certificate",
                      "status": "unsigned", "issued_at": None,
                      "truth": {"p0_gate_passed": None, "formal_training_authorized": False,
                                  "research_training_started": False, "protocol_122_executed": False},
                      "bindings": bindings}
            cert.write_text(json.dumps(record))
            self.assertTrue(any("status" in x for x in verify(cert)))
            record["status"] = "signed"
            record["issued_by"] = "reviewer"
            record["issued_at"] = "2026-11-17T03:00:00+08:00"
            record["truth"]["p0_gate_passed"] = True
            cert.write_text(json.dumps(record))
            self.assertTrue(any("mismatch" in x for x in verify(cert)))

    def test_supervisor_rejects_unbound_authorization_and_output(self):
        import tempfile
        from pathlib import Path
        from datetime import datetime, timezone

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            budget = {
                "total_seconds": 20.0,
                "phase_seconds": {"preparation": 10.0, "evaluation": 5.0, "certification": 5.0},
                "rss_bytes": 1_000_000, "vram_bytes": 0, "output_bytes": 1_000_000,
                "poll_seconds": 0.5, "cpu_core_seconds": 20,
                "gpu_allocated_seconds": 0, "run_deadline_seconds": 10.0,
                "eval_deadline_seconds": 5.0,
            }
            spec_hash = "b" * 64
            output = root / "new-output"
            authorization = {
                "protocol": 122, "purpose": runner.PURPOSE, "authorized": True,
                "formal_training_authorized": True, "target_databases_read": [],
                "spec_sha256": spec_hash, "budget_sha256": runner.canonical_digest(budget),
                "output_dir": str(output), "granted_by": "test approval",
                "granted_at": datetime.now(timezone.utc).isoformat(),
            }
            runner.validate_authorization(authorization, spec_hash, budget, output)
            for patch in ({"authorized": False}, {"spec_sha256": "c" * 64},
                          {"budget_sha256": "0" * 64},
                          {"output_dir": str(root / "wrong")},
                          {"target_databases_read": ["stats"]}):
                with self.subTest(patch=patch), self.assertRaises(ValueError):
                    runner.validate_authorization({**authorization, **patch}, spec_hash, budget, output)

    def test_exploratory_phase_order_is_monotonic(self):
        import ast
        import inspect

        code = inspect.getsource(runner.worker_main)
        tree = ast.parse(code)
        single = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "spec.get('execution_mode') == 'single_run'")
        first = single.body[0].value
        self.assertEqual(ast.unparse(first.func), "_write_phase")
        self.assertIn("evaluation", ast.unparse(first.args[1]))
        self.assertIn("certification", ast.unparse(first.args[1]))
        branches = [node for node in ast.walk(single) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "spec.get('exploratory_arm') == EXPLORATORY_ARM"]
        self.assertTrue(any("evaluations = evaluate_all" in ast.unparse(node) for node in branches))
        self.assertFalse(any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "_write_phase" and len(node.args) == 2
            and isinstance(node.args[1], ast.Constant) and node.args[1].value == "evaluation"
            for node in ast.walk(single)
        ))

    def test_worker_telemetry_charges_gpu_seconds_and_enforces_budget(self):
        import json as json_module
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts import meta65_joint_supervisor as supervisor

        rows = {100: (1, 100, 100, 42, "S")}
        worker = SimpleNamespace(pid=100)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "memory.json").write_text(
                json_module.dumps({"max_allocated_bytes": 1880 * supervisor.MiB})
            )
            with (
                patch.object(runner.time, "monotonic", side_effect=(1.0, 1.75, 2.0, 2.75)),
                patch.object(supervisor, "_process_rows", return_value=rows),
                patch.object(supervisor, "_gpu_bytes", return_value=0),
                patch.object(runner.BudgetTracker, "_live_cpu_seconds", return_value=0.0),
            ):
                def sample(target):
                    return supervisor.sample_process(
                        target, mode="formal", output=output
                    )
                for limit in (1.0, 0.5):
                    tracker = runner.BudgetTracker(
                        {"cpu_core_seconds": 100, "gpu_allocated_seconds": limit}
                    )
                    if limit == 0.5:
                        with self.assertRaisesRegex(
                            ValueError, "GPU allocated-second budget"
                        ):
                            tracker.sample(worker, sample)
                    else:
                        self.assertEqual(
                            tracker.sample(worker, sample)["vram_bytes"],
                            1880 * supervisor.MiB,
                        )
                    self.assertEqual(tracker.gpu_seconds, 0.75)
                    self.assertAlmostEqual(tracker.gpu_gib_seconds, 0.75 * 1880 / 1024)

    def test_cuda_memory_telemetry_reports_peaks_and_publishes_errors(self):
        import json as json_module
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        cuda = SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: 100,
            memory_reserved=lambda: 200,
            max_memory_allocated=lambda: 300,
            max_memory_reserved=lambda: 400,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            telemetry = runner.CudaMemoryTelemetry(
                output, SimpleNamespace(cuda=cuda), interval=0.01
            )
            self.assertEqual(
                telemetry.sample(),
                {
                    "allocated_bytes": 100,
                    "reserved_bytes": 200,
                    "max_allocated_bytes": 300,
                    "max_reserved_bytes": 400,
                    "interval_seconds": 0.01,
                },
            )
            broken = SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: True,
                    memory_allocated=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                )
            )
            failing = runner.CudaMemoryTelemetry(output, broken, interval=0.01)
            self.assertIn("error", failing.sample())
            written = json_module.loads((output / runner.MEMORY_FILE).read_text())
            self.assertIn("RuntimeError: boom", written["error"])
            with self.assertRaises(ValueError):
                from scripts import meta65_joint_supervisor as supervisor
                with (
                    patch.object(
                        supervisor,
                        "_process_rows",
                        return_value={100: (1, 100, 100, 1, "S")},
                    ),
                    patch.object(supervisor, "_gpu_bytes", return_value=0),
                ):
                    supervisor.sample_process(
                        SimpleNamespace(pid=100), mode="formal", output=output
                    )

    def test_telemetry_heartbeat_and_write_failure_do_not_kill_sampler(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        cuda = SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: 100,
            memory_reserved=lambda: 200,
            max_memory_allocated=lambda: 300,
            max_memory_reserved=lambda: 400,
        )
        torch_module = SimpleNamespace(cuda=cuda)
        with tempfile.TemporaryDirectory() as directory:
            telemetry = runner.CudaMemoryTelemetry(
                Path(directory), torch_module, interval=0.01
            )
            writes = []
            with patch.object(
                runner, "write_json", side_effect=lambda path, value: writes.append(value)
            ):
                telemetry.sample()
                telemetry.sample()
            self.assertEqual(len(writes), 2)
            with patch.object(runner, "write_json", side_effect=OSError("disk full")):
                self.assertEqual(telemetry.sample()["max_allocated_bytes"], 300)

    def test_cuda_run_without_observed_vram_cannot_finalize(self):
        budget = {"cpu_core_seconds": 10, "gpu_allocated_seconds": 10}
        tracker = runner.BudgetTracker(budget, requires_gpu_usage=True)
        with self.assertRaisesRegex(ValueError, "nonzero worker VRAM"):
            tracker.finalize()
        runner.BudgetTracker(budget).finalize()

    def test_parent_sampler_reads_worker_memory_telemetry(self):
        import inspect

        source = inspect.getsource(runner.parent_main)
        self.assertIn("output=worker_output", source)
        self.assertIn("requires_gpu_usage", source)

    def test_exploratory_budget_rejects_training_only_limits(self):
        budget = {
            "total_seconds": 2100.0,
            "phase_seconds": {"preparation": 1800.0, "evaluation": 1.0, "certification": 299.0},
            "rss_bytes": 24 * 1024**3, "vram_bytes": 2 * 1024**3,
            "output_bytes": 256 * 1024**2, "poll_seconds": 0.5,
            "cpu_core_seconds": 2700, "gpu_allocated_seconds": 1800,
            "run_deadline_seconds": 1800.0, "eval_deadline_seconds": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "evaluation budget"):
            runner.validate_exploratory_budget(budget)
        enough = {**budget, "total_seconds": 5700.0,
                  "phase_seconds": {"preparation": 1800.0, "evaluation": 3300.0, "certification": 420.0},
                  "output_bytes": 1024**3, "cpu_core_seconds": 4500,
                  "gpu_allocated_seconds": 3500, "eval_deadline_seconds": 1200.0}
        self.assertEqual(runner.validate_exploratory_budget(enough)["eval_deadline_seconds"], 1200.0)

    def test_real_bound_snapshot_rejects_certificate_and_authorization_tampering(self):
        import json
        import tempfile
        from datetime import datetime, timezone
        from pathlib import Path
        from unittest.mock import patch
        from spatial_meta_ce.data import TRANSFORM_VERSION
        from spatial_meta_ce.manifests import digest
        from spatial_meta_ce.price_capacity import build_capacity
        from spatial_meta_ce.price_protocol import exploratory_model_config, validate_training_preflight

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            rebound = root / "rebound"
            rebound.mkdir()
            code = root / "code"
            (code / "scripts").mkdir(parents=True)
            pg_configs = root / "pg-configs"
            pg_configs.mkdir()
            cache = root / "cache" / "pretrain"
            cache.mkdir(parents=True)

            def save(path, value):
                path.write_text(json.dumps(value, sort_keys=True))
                return runner.file_sha256(path)

            data = {"data_root": str(root / "price"), "cache_root": str(cache.parent),
                    "splits": {"train": list(__import__("spatial_meta_ce.price_capacity", fromlist=["TRAIN_DATABASES"]).TRAIN_DATABASES), "validation": list(runner.EVAL_DATABASES),
                               "test": ["imdb", "stats", "genome", "ergastf1"]}}
            for db in [*data["splits"]["train"], *runner.EVAL_DATABASES]:
                location = cache / db
                location.mkdir()
                counts = [2, 2, 1, 1]
                payload = {"schema": 1, "transform": TRANSFORM_VERSION,
                           "inspection": {"max_counts": counts},
                           "rows": [{"id": f"{db}/pretrain/1", "query_hash": digest(db)}]}
                save(location / "manifest.json", {**payload, "content_hash": digest(payload)})
            capacity = build_capacity(data)
            data_path = rebound / "data_config_122.json"
            data_sha = save(data_path, data)
            source_plan = root / "source_eval_plan.json"
            plan = {"protocol": 122, "analysis_role": "population_primary",
                    "databases": {d: {"excluded_unique": 100} for d in runner.EVAL_DATABASES}}
            plan_sha = runner._plan_digest(plan)
            plan_file_sha = save(source_plan, {**plan, "plan_sha256": plan_sha})
            train_plans = {}
            for seed in runner.SEEDS:
                p = root / f"train-{seed}.json"
                train_plans[str(seed)] = {"path": str(p), "file_sha256": save(p, {"seed": seed}),
                                          "plan_sha256": "c" * 64}
            source_preflight = {"protocol": 122, "status": "data_preflight_passed",
                                "p0_gate_passed": False, "formal_training_authorized": False,
                                "target_databases_read": [], "data_config": str(data_path),
                                "data_config_sha256": data_sha, "capacity": capacity,
                                "coverage": {"full_population_supported": True, "unsupported": [],
                                             "databases": {d: {"manifest_sha256": "a" * 64} for d in runner.EVAL_DATABASES}},
                                "source_eval_plan": {"bound_plan_sha256": plan_sha},
                                "training_plans": train_plans}
            preflight_path = rebound / "preflight.json"
            for db in runner.EVAL_DATABASES:
                location = cache / db
                source_preflight["coverage"]["databases"][db] = {
                    "manifest_sha256": runner.file_sha256(location / "manifest.json"),
                    "content_hash": json.loads((location / "manifest.json").read_text())["content_hash"],
                }
            preflight_sha = save(preflight_path, source_preflight)
            configs = []
            manifest_files = {"preflight.json": {"sha256": preflight_sha},
                              "data_config_122.json": {"sha256": data_sha}}
            for arm in runner.ARMS:
                for seed in runner.SEEDS:
                    name = f"{arm}-seed{seed}.json"
                    p = rebound / name
                    v = {"seed": seed, "fixed_stage": f"122-{arm}",
                         "method": "erm", "meta_objective": "post",
                         "fixed_budget": True, "steps": 4000, "purpose": "research",
                         "meta_batch": 4, "accumulation": 1, "query_size": 64,
                         "support_sizes": [5, 10, 20, 50, 100],
                         "loss": "smooth_l1", "inner_lr": 0.001, "inner_steps": 5,
                         "adapt_scope": "all", "outer_lr": 0.0003,
                         "training_plan": train_plans[str(seed)]["path"],
                         "source_preflight": str(preflight_path),
                         "source_preflight_sha256": preflight_sha,
                         "data_config": str(data_path), "data_config_sha256": data_sha,
                         "model": {"arch": "price_lite", "capacity": capacity}}
                    sha = save(p, v)
                    manifest_files[name] = {"sha256": sha}
                    configs.append({"name": name, "path": str(p), "sha256": sha})
            for seed in runner.SEEDS:
                p = pg_configs / f"A1-PG-seed{seed}.json"
                v = {**json.loads((rebound / f"A1-seed{seed}.json").read_text()),
                     "fixed_stage": "122-A1-PG",
                     "model": exploratory_model_config(capacity)}
                configs.append({"name": p.name, "path": str(p), "sha256": save(p, v)})
            manifest = rebound / "binding_manifest.json"
            manifest_sha = save(manifest, {"p0_gate_passed": False, "formal_training_authorized": False,
                                           "files": manifest_files})
            code_hashes = {"scripts/run_price_meta_a.py": save(code / "scripts/run_price_meta_a.py", {"source": "synthetic"})}
            digest_file = root / "code-digests.json"
            digest_sha = save(digest_file, code_hashes)
            bindings = {
                "frozen_code": {"path": str(digest_file), "sha256": digest_sha,
                                "manifest_digest": runner.canonical_digest(code_hashes),
                                "code_files": [{"name": "scripts/run_price_meta_a.py",
                                                "path": str(code / "scripts/run_price_meta_a.py"), "sha256": code_hashes["scripts/run_price_meta_a.py"]}]},
                "a_training_configs": configs,
                "training_plans": [{"name": f"seed{s}", "path": train_plans[str(s)]["path"],
                                     "sha256": train_plans[str(s)]["file_sha256"]} for s in runner.SEEDS],
                "cache_manifests": [{"name": db, "path": str(cache / db / "manifest.json"),
                                     "sha256": runner.file_sha256(cache / db / "manifest.json"),
                                     "content_hash": json.loads((cache / db / "manifest.json").read_text())["content_hash"]}
                                    for db in [*data["splits"]["train"], *runner.EVAL_DATABASES]],
                "source_preflight": {"path": str(preflight_path), "sha256": preflight_sha},
                "source_eval_plan": {"path": str(source_plan), "sha256": plan_file_sha,
                                     "plan_sha256": plan_sha},
                "data_config_122": {"path": str(data_path), "sha256": data_sha},
            }
            cert_path = root / "p0-certificate.json"
            cert = {"schema": 2, "protocol": 122, "purpose": "p0-certificate",
                    "status": "signed", "issued_by": "synthetic fixture",
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                    "truth": {"p0_gate_passed": True, "formal_training_authorized": False,
                                "research_training_started": False, "protocol_122_executed": False},
                    "bindings": bindings}
            cert_sha = save(cert_path, cert)
            budget = {"total_seconds": 5700.0, "phase_seconds": {"preparation": 1800.0,
                         "evaluation": 3300.0, "certification": 420.0},
                      "rss_bytes": 24 * 1024**3, "vram_bytes": 2 * 1024**3,
                      "output_bytes": 1024**3, "poll_seconds": 0.5, "cpu_core_seconds": 4500,
                      "gpu_allocated_seconds": 3500, "run_deadline_seconds": 1800.0,
                      "eval_deadline_seconds": 1200.0}
            spec_path = root / "spec.json"
            spec = {"protocol": 122, "purpose": runner.PURPOSE, "arms": list(runner.ARMS),
                    "seeds": list(runner.SEEDS), "total_runs": runner.TOTAL_RUNS,
                    "total_cells": runner.TOTAL_CELLS, "device": "cuda",
                    "code": str(code), "code_sha256": code_hashes,
                    "rebound": str(rebound), "binding_manifest_sha256": manifest_sha,
                    "source_eval_plan": {"path": str(source_plan), "file_sha256": plan_file_sha,
                                         "plan_sha256": plan_sha},
                    "p0_certificate": {"path": str(cert_path), "sha256": cert_sha},
                    "budget": budget, "target_databases_read": [],
                    "execution_mode": "single_run", "run_key": "A1-PG-seed42",
                    "exploratory_arm": "A1-PG", "exploratory_config_dir": str(pg_configs)}
            spec_sha = save(spec_path, spec)
            approval_path = root / "authorization.json"
            output = root / "output"
            approval = {"protocol": 122, "purpose": runner.PURPOSE, "authorized": True,
                        "formal_training_authorized": True, "target_databases_read": [],
                        "spec_sha256": spec_sha, "budget_sha256": runner.canonical_digest(budget),
                        "output_dir": str(output), "granted_by": "synthetic fixture",
                        "granted_at": datetime.now(timezone.utc).isoformat()}
            approval_sha = save(approval_path, approval)
            evidence = {"spec": {"path": str(spec_path), "sha256": spec_sha},
                        "authorization": {"path": str(approval_path), "sha256": approval_sha},
                        "output_dir": str(output)}
            with patch.object(runner, "ROOT", code), patch.object(runner, "compute_code_sha256", return_value=code_hashes):
                snapshot = runner.build_execution_snapshot(source_preflight, evidence)
            self.assertTrue(snapshot["exploratory_pg_head_input_authorized"])
            self.assertFalse(source_preflight["p0_gate_passed"])
            runtime = {**json.loads((pg_configs / "A1-PG-seed42.json").read_text()),
                       "source_preflight": str(root / "execution-preflight.json"),
                       "source_preflight_sha256": "0" * 64}
            runtime_path = root / "execution-preflight.json"
            runtime["source_preflight_sha256"] = save(runtime_path, snapshot)
            with patch.object(runner, "ROOT", code), patch.object(runner, "compute_code_sha256", return_value=code_hashes):
                runner.validate_execution_snapshot(snapshot, runtime)
                validate_training_preflight(runtime, data)
            for changed in (
                {**snapshot, "exploratory_pg_head_input_authorized": False},
                {**snapshot, "execution_evidence": {**evidence, "output_dir": str(root / "wrong")}},
            ):
                with patch.object(runner, "ROOT", code), patch.object(runner, "compute_code_sha256", return_value=code_hashes):
                    with self.assertRaises(ValueError):
                        runner.validate_execution_snapshot(changed, runtime)
            approval_path.write_text(approval_path.read_text().replace("synthetic fixture", "altered"))
            with patch.object(runner, "ROOT", code), patch.object(runner, "compute_code_sha256", return_value=code_hashes):
                with self.assertRaises(ValueError):
                    runner.validate_execution_snapshot(snapshot, runtime)

    def test_formal_matrix_and_reference_arm_remain_locked(self):
        self.assertEqual(runner.ARMS, ("A0-ref", "A0", "A1", "A2"))
        self.assertEqual(len(runner.execution_keys({})), 12)
        self.assertEqual(
            runner.execution_keys({"execution_mode": "single_run", "run_key": "A1-seed42"}),
            ["A1-seed42"],
        )


if __name__ == "__main__":
    unittest.main()
