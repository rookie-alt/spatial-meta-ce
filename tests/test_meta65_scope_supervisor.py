"""Synthetic CPU workers exercise the scope contract, not experiment results."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts import meta65_scope_supervisor as s

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"total_seconds": 4, "poll_seconds": 0.02}


@unittest.skipUnless(os.name == "posix", "requires POSIX process ownership")
class ScopeSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name).resolve() / "worker"

    def command(
        self, body: str = "", *, mode: str = "formal", tail: str = ""
    ) -> list[str]:
        cells = 60 if mode == "formal" else 12
        code = (
            "import json,os,signal,subprocess,sys,time\nfrom pathlib import Path\n"
            "out=Path(sys.argv[1]); out.mkdir()\n"
            "assert os.getpid()==os.getpgrp()==os.getsid(0)\n"
            f'status={{"status":"worker_complete","mode":{mode!r},"cells":{cells}}}\n'
            f'verification={{"status":{("formal_verified" if mode == "formal" else "preflight_verified")!r},'
            f'"mode":{mode!r},"cells_verified":{cells},"engineering_verified":True,'
            f'"formal_result":{mode == "formal"},"summary":{{"status":"promising_single_seed_K20"}}}}\n'
            'decision={"status":"promising_single_seed_K20"}\n' + body + "\n"
            '(out/"status.json").write_text(json.dumps(status))\n'
            '(out/"verification.json").write_text(json.dumps(verification))\n'
            '(out/"decision.provisional.json").write_text(json.dumps(decision))\n'
            + tail
            + "\ntime.sleep(0.05)\n"
        )
        return [sys.executable, "-B", "-c", code, str(self.out)]

    def run_worker(
        self,
        body: str = "",
        *,
        mode: str = "formal",
        tail: str = "",
        budget: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return s.supervise(
            self.command(body, mode=mode, tail=tail),
            self.out,
            mode=mode,
            budget_override={**SMALL, **(budget or {})},
        )

    def assert_failed(self, record: dict[str, Any], reason: str) -> None:
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"], reason)
        self.assertFalse((self.out / "decision.json").exists())
        self.assertFalse(record["formal_result"])
        self.assertEqual(json.loads(s.artifact_paths(self.out)[0].read_text()), record)
        if self.out.exists():
            self.assertEqual(
                json.loads((self.out / "status.json").read_text())["status"], "failed"
            )

    def test_formal_cpu_fixture_publishes_only_after_verified_exit(self) -> None:
        record = self.run_worker(
            'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\nprint("fixture")'
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["returncode"], 0)
        self.assertTrue(record["test_only"])
        self.assertFalse(record["formal_result"])
        self.assertEqual(record["pid"], record["pgid"])
        self.assertGreater(record["peak_rss_bytes"], 0)
        self.assertTrue((self.out / "decision.json").is_file())
        self.assertFalse((self.out / "decision.provisional.json").exists())
        self.assertEqual(
            record["outputbytes"],
            s.output_bytes(self.out, s.artifact_paths(self.out)[1]),
        )

    def test_preflight_twelve_cells_never_publishes_candidate(self) -> None:
        for mode in ("cpu_preflight", "gpu_preflight"):
            with self.subTest(mode=mode):
                self.out = self.out.parent / mode
                record = self.run_worker(mode=mode)
                self.assertEqual(record["status"], "preflight_completed")
                self.assertFalse(record["formal_result"])
                self.assertFalse((self.out / "decision.json").exists())
                self.assertTrue((self.out / "decision.provisional.json").exists())

    def test_nonzero_preserves_invalid_decision_and_partial_evidence(self) -> None:
        record = self.run_worker(
            tail='(out/"partial").write_text("keep")\n'
            '(out/"decision.json").write_text(json.dumps(decision))\nsys.exit(7)'
        )
        self.assert_failed(record, "child_exit")
        self.assertEqual(record["returncode"], 7)
        self.assertTrue((self.out / "decision.invalid.json").exists())
        self.assertEqual((self.out / "partial").read_text(), "keep")

    def test_zero_exit_or_file_presence_is_not_completion(self) -> None:
        cases = (
            ('status["status"]="completed"', "", "invalid_worker_status"),
            ('status["mode"]="cpu_preflight"', "", "invalid_worker_status"),
            ('status["cells"]=True', "", "invalid_worker_status"),
            ('status["cells"]=59', "", "invalid_worker_status"),
            ('verification["cells_verified"]=59', "", "invalid_verification"),
            ('verification["engineering_verified"]=1', "", "invalid_verification"),
            ('verification["status"]="preflight_verified"', "", "invalid_verification"),
            ('verification["formal_result"]=False', "", "invalid_verification"),
            ('verification["mode"]="gpu_preflight"', "", "invalid_verification"),
            ("decision={}", "", "invalid_decision"),
            (
                'verification["summary"]["status"]="no_candidate"',
                "",
                "invalid_decision",
            ),
            ('verification.pop("summary")', "", "invalid_decision"),
            ("", '(out/"verification.json").unlink()', "invalid_verification"),
            ("", '(out/"status.json").unlink()', "invalid_worker_status"),
            (
                "",
                '(out/"decision.provisional.json").write_text("bad")',
                "invalid_decision",
            ),
            ("", '(out/"decision.json").write_text("{}")', "premature_decision"),
        )
        for index, (body, tail, reason) in enumerate(cases):
            with self.subTest(index=index):
                self.out = self.out.parent / f"schema-{index}"
                self.assert_failed(self.run_worker(body, tail=tail), reason)

    def test_real_timeout_kills_term_ignoring_worker_with_bounded_cleanup(self) -> None:
        started = time.monotonic()
        record = self.run_worker(
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(30)",
            budget={"total_seconds": 0.35},
        )
        self.assert_failed(record, "total_timeout")
        self.assertIn(signal.SIGKILL, record["signals"])
        self.assertLess(time.monotonic() - started, 10.8)
        self.assertFalse(s.process_is_live(record["pid"]))

    def test_real_rss_and_output_budgets_fail_closed(self) -> None:
        cases = (
            ("rss_bytes", 1, "time.sleep(30)", "rss_limit"),
            (
                "output_bytes",
                4096,
                'print("x"*10000,flush=True)\ntime.sleep(30)',
                "output_limit",
            ),
            (
                "output_bytes",
                4096,
                '(out/"large").write_bytes(b"x"*10000)\ntime.sleep(30)',
                "output_limit",
            ),
        )
        for index, (key, value, body, reason) in enumerate(cases):
            with self.subTest(resource=key):
                self.out = self.out.parent / f"resource-{index}"
                self.assert_failed(self.run_worker(body, budget={key: value}), reason)

    def test_sampler_errors_and_vram_limit_cannot_certify_success(self) -> None:
        with patch.object(s, "_gpu_bytes", side_effect=s.GPUQueryError("unavailable")):
            record = s.supervise(
                self.command("time.sleep(30)"),
                self.out,
                gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                budget_override=SMALL,
            )
        self.assert_failed(record, "sample_error")
        self.out = self.out.parent / "vram"
        with patch.object(s, "_gpu_bytes", return_value=20):
            record = s.supervise(
                self.command("time.sleep(30)"),
                self.out,
                gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
                budget_override={**SMALL, "vram_bytes": 10},
            )
        self.assert_failed(record, "vram_limit")

    def test_budget_is_frozen_and_overrides_only_shrink(self) -> None:
        from scripts import meta65_scope_contract as contract

        limits = contract.FROZEN["limits"]
        budget = s.limits_for("formal")
        self.assertEqual(budget["total_seconds"], limits["total_seconds"])
        self.assertEqual(budget["rss_bytes"], limits["rss_bytes"])
        self.assertEqual(budget["vram_bytes"], limits["vram_bytes"])
        self.assertEqual(budget["output_bytes"], limits["output_bytes"])
        for key, value in budget.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                s.supervise(self.command(), self.out, budget_override={key: value + 1})
        overrides: tuple[dict[str, Any], ...] = (
            {"total_seconds": True},
            {"rss_bytes": -1},
            {"poll_seconds": 0},
            {"total_seconds": float("nan")},
            {"extra": 1},
        )
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                s.supervise(self.command(), self.out, budget_override=override)
        self.assertFalse(self.out.exists())

    def test_cli_has_no_budget_escape_hatch_and_formal_needs_gpu(self) -> None:
        with self.assertRaises(ValueError):
            s.supervise(self.command(), self.out)
        for option in ("--budget-override", "--total-seconds", "--grace"):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                s.main(["--output", str(self.out), option, "20000", "--", "true"])
        with self.assertRaises(ValueError):
            s.supervise([""], self.out, budget_override=SMALL)
        with self.assertRaises(ValueError):
            s.supervise(self.command(), self.out, mode="other", budget_override=SMALL)
        with self.assertRaises(ValueError):
            s.supervise(self.command(), self.out, gpu_uuid="0", budget_override=SMALL)

    def test_live_descendant_after_zero_exit_is_failure_and_not_left_alive(
        self,
    ) -> None:
        body = (
            'child=subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"])\n'
            '(out/"descendant.pid").write_text(str(child.pid))'
        )
        record = self.run_worker(body)
        self.assert_failed(record, "orphan_descendant")
        self.assertFalse(
            s.process_is_live(int((self.out / "descendant.pid").read_text()))
        )

    def test_parent_signal_cleans_escaped_descendant_without_touching_other_group(
        self,
    ) -> None:
        supervisor_code = (
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0,{str(ROOT)!r})\nfrom scripts import meta65_scope_supervisor as s\n"
            f"s.supervise({self.command('time.sleep(30)')!r},Path({str(self.out)!r}),"
            "budget_override={'total_seconds':8,'poll_seconds':0.02})"
        )
        worker = self.command(
            'child=subprocess.Popen([sys.executable,"-B","-c",'
            '"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"],'
            'start_new_session=True)\n(out/"descendant.pid").write_text(str(child.pid))\ntime.sleep(30)'
        )
        supervisor_code = supervisor_code.replace(
            repr(self.command("time.sleep(30)")), repr(worker)
        )
        unrelated = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
            start_new_session=True,
        )
        parent = subprocess.Popen(
            [sys.executable, "-B", "-c", supervisor_code], cwd=ROOT
        )
        descendant: int | None = None
        try:
            deadline = time.monotonic() + 3
            while (
                not (self.out / "descendant.pid").exists()
                and time.monotonic() < deadline
            ):
                time.sleep(0.02)
            self.assertTrue((self.out / "descendant.pid").exists())
            descendant = int((self.out / "descendant.pid").read_text())
            time.sleep(
                0.15
            )  # Allow identity sampling before the child escapes its parent's tree.
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=11)
            record = json.loads(s.artifact_paths(self.out)[0].read_text())
            self.assert_failed(record, "parent_signal")
            self.assertIn(signal.SIGTERM, record["received_signals"])
            self.assertFalse(s.process_is_live(descendant))
            self.assertIsNone(unrelated.poll())
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=2)
            if descendant is not None and s.process_is_live(descendant):
                os.kill(descendant, signal.SIGKILL)
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_gpu_samples_only_owned_pids_on_bound_uuid(self) -> None:
        uuid = "GPU-00000000-0000-0000-0000-000000000000"
        result = subprocess.CompletedProcess(
            [], 0, f"123, {uuid}, 5\n456, GPU-other, 99999\n"
        )
        with patch.object(s.subprocess, "run", return_value=result):
            self.assertEqual(s._gpu_bytes({123}, uuid), 5 * 1024**2)
        for text in ("123, GPU-other, 5", f"123, {uuid}, N/A", "bad"):
            with patch.object(
                s.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, text),
            ):
                with self.assertRaises(s.GPUQueryError):
                    s._gpu_bytes({123}, uuid)

    def test_failed_publication_revokes_valid_decision(self) -> None:
        original = s._write_json

        def fail_after_publication(
            path: Path, value: dict[str, Any], **kwargs: Any
        ) -> None:
            if (
                path == s.artifact_paths(self.out)[0]
                and (self.out / "decision.json").exists()
            ):
                raise OSError("record persistence failed")
            original(path, value, **kwargs)

        with patch.object(s, "_write_json", side_effect=fail_after_publication):
            record = self.run_worker()
        self.assert_failed(record, "supervisor_error")
        self.assertTrue((self.out / "decision.invalid.json").exists())

    def test_final_evidence_write_over_budget_revokes_publication(self) -> None:
        original = s._write_json
        injected = False

        def add_bytes(path: Path, value: dict[str, Any], **kwargs: Any) -> None:
            nonlocal injected
            original(path, value, **kwargs)
            if (
                path == s.artifact_paths(self.out)[0]
                and (self.out / "decision.json").exists()
                and not injected
            ):
                injected = True
                (self.out / "late-output").write_bytes(b"x" * 10000)

        with patch.object(s, "_write_json", side_effect=add_bytes):
            record = self.run_worker(budget={"output_bytes": 8192})
        self.assert_failed(record, "output_limit")
        self.assertTrue((self.out / "decision.invalid.json").exists())

    def test_last_record_write_cannot_leave_an_over_budget_candidate(self) -> None:
        original = s._persist
        count = 0

        def late_write(
            record: dict[str, Any], output: Path, log: Path
        ) -> dict[str, Any]:
            nonlocal count
            if (output / "decision.json").exists():
                count += 1
                if count == 2:
                    (output / "last-output").write_bytes(b"x" * 10000)
            return original(record, output, log)

        with patch.object(s, "_persist", side_effect=late_write):
            record = self.run_worker(budget={"output_bytes": 8192})
        self.assert_failed(record, "output_limit")

    def test_unsafe_worker_file_does_not_prevent_decision_revocation(self) -> None:
        record = self.run_worker(
            tail='(out/"decision.json").write_text(json.dumps(decision))\n'
            '(out/"unsafe").symlink_to(out.parent)\nsys.exit(7)'
        )
        self.assertEqual(record["status"], "failed")
        self.assertFalse((self.out / "decision.json").exists())
        self.assertTrue((self.out / "decision.invalid.json").exists())

    def test_symlinks_and_existing_output_are_rejected(self) -> None:
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_worker()
        self.out.rmdir()
        self.out.symlink_to(self.out.parent, target_is_directory=True)
        with self.assertRaises(s.UnsafeOutputError):
            self.run_worker()


if __name__ == "__main__":
    unittest.main()
