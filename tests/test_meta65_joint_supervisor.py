"""CPU-only synthetic contract tests; no model, cache, SSH, or GPU access."""

import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import meta65_joint_supervisor as s

ROOT = Path(__file__).resolve().parents[1]
SMALL = {
    "total_seconds": 3,
    "phase_seconds": {"preparation": 1, "evaluation": 1, "certification": 1},
    "poll_seconds": 0.01,
}


class _FakeChild:
    """Minimal Popen stand-in for _kill_group permission-error tests."""

    def __init__(self, pid, polls):
        self.pid = pid
        self.returncode = None
        self._polls = list(polls)

    def poll(self):
        return self._polls.pop(0) if self._polls else None

    def wait(self):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


@unittest.skipUnless(os.name == "posix", "requires POSIX owned process groups")
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "worker"

    def tearDown(self):
        self.temp.cleanup()

    def command(self, body="", *, mode="cpu_preflight", cells=0, phase="certification"):
        code = (
            "import json, os, signal, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "out=Path(sys.argv[1])\n"
            'assert not out.exists(), "supervisor must not precreate output"\n'
            "out.mkdir()\n"
            f'(out/"phase.json").write_text(json.dumps({{"phase": {phase!r}}}))\n'
            f'status={{"status":"worker_complete","mode":{mode!r},"cells":{cells!r},'
            '"micro_cells":9}\n' + body + "\n"
            '(out/"status.json").write_text(json.dumps(status))\n'
        )
        return [sys.executable, "-B", "-c", code, str(self.out)]

    def run_worker(self, body="", *, limits=None, sample_fn=None, **kwargs):
        return s.supervise(
            self.command(body, **kwargs),
            self.out,
            mode=kwargs.get("mode", "cpu_preflight"),
            limits={**SMALL, **(limits or {})},
            sample_fn=sample_fn,
            grace=0.08,
            root=self.root,
        )

    def assert_failed(self, record, reason):
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"], reason)
        status = json.loads((self.out / "status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertNotIn("completed", status.values())
        sidecar = json.loads(s.artifact_paths(self.out)[0].read_text())
        self.assertEqual(sidecar, record)

    def test_cpu_success_owns_session_uses_empty_cuda_and_sidecar_log(self):
        body = (
            'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
            'assert os.environ["OMP_NUM_THREADS"] == "1"\n'
            "assert Path.cwd().resolve() == out.parent.resolve()\n"
            "assert os.getpgrp() == os.getpid()\n"
            'print("synthetic-cpu", flush=True)\n'
        )
        with patch.object(
            s, "_gpu_bytes", side_effect=AssertionError("CPU queried GPU")
        ):
            record = self.run_worker(body)
        self.assertEqual(record["status"], "preflight_completed")
        self.assertIsNone(record["failure"])
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(record["pid"], record["pgid"])
        self.assertGreater(record["peak_rss_bytes"], 0)
        self.assertEqual(record["peak_vram_bytes"], 0)
        self.assertEqual(record["phase"], "certification")
        self.assertEqual(
            record["outputbytes"],
            s.output_bytes(self.out, s.artifact_paths(self.out)[1]),
        )
        self.assertIn("synthetic-cpu", s.artifact_paths(self.out)[1].read_text())
        self.assertEqual(
            json.loads((self.out / "status.json").read_text())["status"],
            "preflight_completed",
        )

    def test_formal_mock_sample_requires_exact_135_cells(self):
        record = self.run_worker(
            mode="formal",
            cells=135,
            sample_fn=lambda child: {"rss_bytes": 1, "vram_bytes": 1},
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["peak_vram_bytes"], 1)

    def test_completion_requires_worker_marker_matching_mode_and_cells(self):
        cases = (
            ('status["status"]="completed"', "cpu_preflight", 0),
            ('status["mode"]="formal"', "cpu_preflight", 0),
            ('status.pop("cells")', "cpu_preflight", 0),
            ("", "cpu_preflight", 9),
            ("", "formal", 134),
            ("", "cpu_preflight", False),
        )
        for index, (body, mode, cells) in enumerate(cases):
            with self.subTest(index=index):
                self.out = self.root / f"worker-{index}"
                record = self.run_worker(
                    body,
                    mode=mode,
                    cells=cells,
                    sample_fn=lambda c: {"rss_bytes": 1, "vram_bytes": 0},
                )
                self.assert_failed(record, "invalid_worker_status")

    def test_zero_exit_without_status_is_failure(self):
        record = self.run_worker("sys.exit(0)")
        self.assert_failed(record, "invalid_worker_status")

    def test_nonzero_invalidates_child_completed_and_keeps_partial(self):
        body = (
            '(out/"partial.bin").write_bytes(b"partial")\n'
            '(out/"status.json").write_text(json.dumps({**status,"status":"completed"}))\n'
            "sys.exit(7)"
        )
        record = self.run_worker(body)
        self.assert_failed(record, "child_exit")
        self.assertEqual(record["returncode"], 7)
        self.assertEqual((self.out / "partial.bin").read_bytes(), b"partial")

    def test_real_timeout_reaps_ignoring_term_worker(self):
        record = self.run_worker(
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(20)",
            limits={"total_seconds": 0.25},
        )
        self.assert_failed(record, "total_timeout")
        self.assertIn(int(signal.SIGTERM), record["signals"])
        self.assertIn(int(signal.SIGKILL), record["signals"])
        self.assertEqual(record["returncode"], -signal.SIGKILL)
        with self.assertRaises(ProcessLookupError):
            os.kill(record["pid"], 0)

    def test_kill_group_after_leader_exit_reaps_and_kills_term_ignoring_descendant(
        self,
    ):
        ready = self.root / "descendant-ready"
        descendant_code = (
            "import signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            'Path(sys.argv[1]).write_text("ready"); time.sleep(30)'
        )
        leader_code = (
            "import subprocess,sys; from pathlib import Path; "
            f'child=subprocess.Popen([sys.executable,"-B","-c",{descendant_code!r},'
            f"{str(ready)!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
            "Path(sys.argv[1]).write_text(str(child.pid))"
        )
        unrelated = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
            start_new_session=True,
        )
        leader = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                leader_code,
                str(self.root / "descendant-pid"),
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 2
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            descendant = int((self.root / "descendant-pid").read_text())
            self.assertEqual(os.getpgid(descendant), leader.pid)
            readable, _, _ = select.select([leader.stdout], [], [], 2)
            self.assertTrue(readable)
            self.assertEqual(leader.stdout.read(), b"")  # Exited, not yet reaped.
            sent = []
            with self.assertRaises(ValueError):
                s._kill_group(leader, 0.08, sent, owned_pgid=unrelated.pid)
            self.assertIsNone(unrelated.poll())
            s._kill_group(leader, 0.08, sent, owned_pgid=leader.pid)
            self.assertIn(int(signal.SIGTERM), sent)
            self.assertIn(int(signal.SIGKILL), sent)
            self.assertEqual(
                leader.returncode, 0
            )  # _kill_group reaped the exited leader.
            with self.assertRaises(ProcessLookupError):
                os.kill(leader.pid, 0)
            deadline = time.monotonic() + 2
            while s.process_is_live(descendant) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(s.process_is_live(descendant))
            self.assertIsNone(unrelated.poll())
        finally:
            try:
                os.killpg(leader.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if leader.poll() is None:
                leader.wait(timeout=2)
            leader.stdout.close()
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_kill_group_permission_error_after_exit_is_treated_as_gone(self):
        child = _FakeChild(pid=4242, polls=[None])
        sent = []

        def fake_killpg(pgid, sig):
            if sig == 0:
                raise PermissionError("reused pgid")

        with (
            patch.object(s.os, "killpg", side_effect=fake_killpg),
            patch.object(s, "_process_rows", return_value={}),
        ):
            s._kill_group(child, 0.05, sent, owned_pgid=child.pid)
        self.assertEqual(sent, [int(signal.SIGTERM)])
        self.assertEqual(child.returncode, 0)

    def test_kill_group_permission_error_with_live_foreign_group_fails_closed(self):
        child = _FakeChild(pid=4243, polls=[None])
        sent = []
        foreign = {999: (1, 4243, 0, 0, "R")}

        def fake_killpg(pgid, sig):
            raise PermissionError("reused pgid")

        with (
            patch.object(s.os, "killpg", side_effect=fake_killpg),
            patch.object(s, "_process_rows", return_value=foreign),
            patch.object(s.os, "getsid", return_value=9999),
        ):
            with self.assertRaises(s.GPUQueryError):
                s._kill_group(child, 0.05, sent, owned_pgid=child.pid)
        self.assertEqual(sent, [])

    def test_each_external_phase_timeout_without_worker_deadline(self):
        for phase in ("preparation", "evaluation", "certification"):
            with self.subTest(phase=phase):
                self.out = self.root / phase
                record = self.run_worker(
                    "time.sleep(20)",
                    phase=phase,
                    limits={"phase_seconds": {phase: 0.12}},
                )
                self.assert_failed(record, "phase_timeout")
                self.assertEqual(record["phase"], phase)

    def test_resource_budgets_stop_owned_child(self):
        for resource in ("rss_bytes", "vram_bytes"):
            with self.subTest(resource=resource):
                self.out = self.root / resource

                def sample(child):
                    deadline = time.monotonic() + 1
                    while (
                        not (self.out / "phase.json").exists()
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.002)
                    return {"rss_bytes": 50, "vram_bytes": 50}

                record = self.run_worker(
                    "time.sleep(20)",
                    mode="formal",
                    cells=135,
                    limits={resource: 10},
                    sample_fn=sample,
                )
                self.assert_failed(record, resource.replace("_bytes", "_limit"))

    def test_output_budget_includes_sidecar_log(self):
        record = self.run_worker(
            'print("x"*10000, flush=True)\ntime.sleep(20)',
            limits={"output_bytes": 4096},
        )
        self.assert_failed(record, "output_limit")
        self.assertGreater(record["outputbytes"], 4096)

    def test_output_budget_includes_worker_files(self):
        record = self.run_worker(
            '(out/"large").write_bytes(b"x"*10000)\ntime.sleep(20)',
            limits={"output_bytes": 4096},
        )
        self.assert_failed(record, "output_limit")
        self.assertTrue((self.out / "large").exists())

    def test_status_write_cannot_cross_output_budget_and_leave_completed(self):
        original = s._write_json

        def write_large_final_status(path, value, **kwargs):
            if (
                path == self.out / "status.json"
                and value.get("status") == "preflight_completed"
            ):
                value = {**value, "final_evidence": "x" * 6000}
            original(path, value, **kwargs)

        with patch.object(s, "_write_json", side_effect=write_large_final_status):
            record = self.run_worker(limits={"output_bytes": 4096})
        self.assert_failed(record, "output_limit")
        self.assertLess(
            record["outputbytes"], 4096
        )  # Failed status may discard padding.

    def test_sample_exception_or_zero_after_exit_is_poll_wait_confirmed(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                self.out = self.root / f"race-{raises}"

                def vanished(child, raises=raises):
                    child.wait(timeout=1)
                    if raises:
                        raise ProcessLookupError("Linux /proc race")
                    return {"rss_bytes": 0, "vram_bytes": 0}

                record = self.run_worker(sample_fn=vanished)
                self.assertEqual(record["status"], "preflight_completed")
                self.assertEqual(record["returncode"], 0)

    def test_exit_confirmation_after_deadline_is_not_success(self):
        def late(child):
            child.wait(timeout=1)
            time.sleep(0.18)
            raise ProcessLookupError("exited but observation was too late")

        record = self.run_worker(sample_fn=late, limits={"total_seconds": 0.12})
        self.assert_failed(record, "total_timeout")
        self.assertEqual(record["returncode"], 0)

    def test_successful_sample_after_total_deadline_is_not_success(self):
        def late_success(child):
            child.wait(timeout=1)
            time.sleep(0.18)
            return {"rss_bytes": 0, "vram_bytes": 0}

        record = self.run_worker(sample_fn=late_success, limits={"total_seconds": 0.12})
        self.assert_failed(record, "total_timeout")
        self.assertEqual(record["returncode"], 0)

    def test_new_phase_cannot_erase_expired_previous_phase(self):
        original = s._read_phase
        first_read = True

        def delayed_phase(output, current):
            nonlocal first_read
            if first_read:
                first_read = False
                time.sleep(0.18)
                return "certification"
            return original(output, current)

        with patch.object(s, "_read_phase", side_effect=delayed_phase):
            record = self.run_worker(limits={"phase_seconds": {"preparation": 0.12}})
        self.assert_failed(record, "phase_timeout")

    def test_exited_leader_with_live_descendant_cannot_publish_completed(self):
        descendant_code = (
            "import signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            'Path(sys.argv[1]).write_text("ready"); time.sleep(30)'
        )
        body = (
            f'ready=out/"desc-ready"\n'
            f'desc=subprocess.Popen([sys.executable,"-B","-c",{descendant_code!r},'
            "str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            '(out/"desc-pid").write_text(str(desc.pid))\n'
            "until=time.monotonic()+2\n"
            "while not ready.exists() and time.monotonic()<until: time.sleep(.01)\n"
            "assert ready.exists()\n"
        )
        record = None
        try:
            record = self.run_worker(body)
            self.assert_failed(record, "orphan_descendant")
            descendant = int((self.out / "desc-pid").read_text())
            deadline = time.monotonic() + 2
            while s.process_is_live(descendant) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(s.process_is_live(descendant))
        finally:
            if record is not None:
                try:
                    os.killpg(record["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_nonzero_leader_exit_cleans_owned_descendant_before_reap(self):
        descendant_code = (
            "import signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            'Path(sys.argv[1]).write_text("ready"); time.sleep(30)'
        )
        body = (
            'ready=out/"desc-ready"\n'
            f'desc=subprocess.Popen([sys.executable,"-B","-c",{descendant_code!r},'
            "str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            '(out/"desc-pid").write_text(str(desc.pid))\n'
            "until=time.monotonic()+2\n"
            "while not ready.exists() and time.monotonic()<until: time.sleep(.01)\n"
            "assert ready.exists()\n"
            "raise SystemExit(7)"
        )
        record = None
        unrelated = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time;time.sleep(30)"],
            start_new_session=True,
        )
        try:
            record = self.run_worker(body)
            self.assert_failed(record, "child_exit")
            self.assertEqual(record["returncode"], 7)
            descendant = int((self.out / "desc-pid").read_text())
            deadline = time.monotonic() + 2
            while s.process_is_live(descendant) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(s.process_is_live(descendant))
            self.assertIsNone(unrelated.poll())
        finally:
            if record is not None:
                try:
                    os.killpg(record["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_supervisor_exception_cleans_group_after_leader_exit(self):
        descendant_code = (
            "import signal,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            'Path(sys.argv[1]).write_text("ready"); time.sleep(30)'
        )
        body = (
            'ready=out/"desc-ready"\n'
            f'desc=subprocess.Popen([sys.executable,"-B","-c",{descendant_code!r},'
            "str(ready)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            '(out/"desc-pid").write_text(str(desc.pid))\n'
            "until=time.monotonic()+2\n"
            "while not ready.exists() and time.monotonic()<until: time.sleep(.01)\n"
            "assert ready.exists()\n"
            "raise SystemExit(7)"
        )

        def interrupt_after_child_exits(output, current):
            deadline = time.monotonic() + 2
            while not (output / "desc-ready").exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue((output / "desc-ready").exists())
            time.sleep(0.03)
            raise KeyboardInterrupt("synthetic supervisor exception")

        record = None
        try:
            with patch.object(
                s, "_read_phase", side_effect=interrupt_after_child_exits
            ):
                record = self.run_worker(body)
            self.assert_failed(record, "parent_signal")
            descendant = int((self.out / "desc-pid").read_text())
            deadline = time.monotonic() + 2
            while s.process_is_live(descendant) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(s.process_is_live(descendant))
        finally:
            if record is not None:
                try:
                    os.killpg(record["pgid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_sampling_exception_with_live_worker_is_failure(self):
        def broken(child):
            deadline = time.monotonic() + 1
            while (
                not (self.out / "phase.json").exists() and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            raise RuntimeError("sampler unavailable")

        record = self.run_worker("time.sleep(20)", sample_fn=broken)
        self.assert_failed(record, "sample_error")

    def test_live_worker_unobservable_to_process_table_fails_closed(self):
        def missing_rows():
            deadline = time.monotonic() + 1
            while (
                not (self.out / "phase.json").exists() and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            return {}

        with patch.object(s, "_process_rows", side_effect=missing_rows):
            record = self.run_worker("time.sleep(20)")
        self.assert_failed(record, "sample_error")
        self.assertIn(int(signal.SIGTERM), record["signals"])

    def test_gpu_query_failure_after_worker_exit_is_sample_error(self):
        errors = (
            subprocess.TimeoutExpired("nvidia-smi", 5),
            subprocess.CalledProcessError(1, "nvidia-smi"),
            FileNotFoundError("nvidia-smi"),
        )
        for index, error in enumerate(errors):
            with self.subTest(error=type(error).__name__):
                self.out = self.root / f"gpu-query-{index}"
                calls = []

                def failed_query(*args, _error=error, _calls=calls, **kwargs):
                    _calls.append(kwargs)
                    deadline = time.monotonic() + 1
                    while (
                        not (self.out / "status.json").exists()
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.005)
                    time.sleep(
                        0.04
                    )  # Ensure the leader has exited before handling the error.
                    raise _error

                with (
                    patch.object(
                        s, "_process_rows", return_value={100: (1, 100, 100, 1, "S")}
                    ),
                    patch.object(s, "_owned_pids", return_value={100}),
                    patch.object(s.subprocess, "run", side_effect=failed_query),
                ):
                    start = time.monotonic()
                    record = self.run_worker(mode="formal", cells=135)
                self.assertLess(time.monotonic() - start, 2)
                self.assertTrue(calls)
                self.assertEqual(calls[0].get("timeout"), 5)
                self.assert_failed(record, "sample_error")

    def test_gpu_query_timeout_terminates_live_worker_without_waiting_for_deadline(
        self,
    ):
        def timed_out(*args, **kwargs):
            deadline = time.monotonic() + 1
            while (
                not (self.out / "phase.json").exists() and time.monotonic() < deadline
            ):
                time.sleep(0.005)
            raise subprocess.TimeoutExpired("nvidia-smi", 5)

        with (
            patch.object(s, "_process_rows", return_value={100: (1, 100, 100, 1, "S")}),
            patch.object(s, "_owned_pids", return_value={100}),
            patch.object(s.subprocess, "run", side_effect=timed_out),
        ):
            start = time.monotonic()
            record = self.run_worker(
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(20)",
                mode="formal",
                cells=135,
            )
        self.assertLess(time.monotonic() - start, 2)
        self.assert_failed(record, "sample_error")
        self.assertIn(int(signal.SIGTERM), record["signals"])
        self.assertIn(int(signal.SIGKILL), record["signals"])
        self.assertEqual(record["returncode"], -signal.SIGKILL)

    def test_symlink_artifacts_or_output_are_rejected_before_spawn(self):
        target = self.root / "untouched"
        target.write_text("original")
        for path in (self.out, *s.artifact_paths(self.out)):
            with self.subTest(path=path), patch.object(s.subprocess, "Popen") as spawn:
                path.symlink_to(target)
                with self.assertRaises((ValueError, FileExistsError)):
                    s.supervise(self.command(), self.out)
                spawn.assert_not_called()
                path.unlink()
        self.assertEqual(target.read_text(), "original")

    def test_worker_symlink_is_failed_without_following_it(self):
        target = self.root / "untouched"
        target.write_text("original")
        record = self.run_worker(
            f'(out/"linked").symlink_to({str(target)!r})\ntime.sleep(20)'
        )
        self.assert_failed(record, "unsafe_output")
        self.assertEqual(target.read_text(), "original")

    def test_existing_output_or_evidence_is_never_overwritten(self):
        for path in (self.out, *s.artifact_paths(self.out)):
            with self.subTest(path=path), patch.object(s.subprocess, "Popen") as spawn:
                path.write_text("preserve")
                with self.assertRaises(FileExistsError):
                    s.supervise(self.command(), self.out)
                spawn.assert_not_called()
                self.assertEqual(path.read_text(), "preserve")
                path.unlink()

    def test_invalid_phase_and_phase_regression_fail(self):
        for index, body in enumerate(
            (
                '(out/"phase.json").write_text(json.dumps({"phase":"training"}))\ntime.sleep(20)',
                (
                    'time.sleep(.08)\n(out/"phase.json").write_text(json.dumps({"phase":"preparation"}))\n'
                    "time.sleep(20)"
                ),
            )
        ):
            with self.subTest(index=index):
                self.out = self.root / f"phase-{index}"
                self.assert_failed(self.run_worker(body), "invalid_phase")

    def test_parent_sigterm_and_sigint_cleanup_descendant_not_unrelated(self):
        descendant_code = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
        body = (
            f'desc=subprocess.Popen([sys.executable,"-c",{descendant_code!r}])\n'
            "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            '(out/"pids.json").write_text(json.dumps([os.getpid(),desc.pid]))\n'
            "time.sleep(30)"
        )
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(sig=sig):
                    self.out = self.root / f"signal-{int(sig)}"
                    parent_code = (
                        "from scripts.meta65_joint_supervisor import supervise\n"
                        f"supervise({self.command(body)!r},{str(self.out)!r},"
                        f"limits={SMALL!r},grace=.08)\n"
                    )
                    parent = subprocess.Popen(
                        [sys.executable, "-B", "-c", parent_code],
                        cwd=ROOT,
                        start_new_session=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    try:
                        deadline = time.monotonic() + 3
                        while (
                            not (self.out / "pids.json").exists()
                            and time.monotonic() < deadline
                        ):
                            time.sleep(0.01)
                        self.assertTrue((self.out / "pids.json").exists())
                        pids = json.loads((self.out / "pids.json").read_text())
                        parent.send_signal(sig)
                        _, error = parent.communicate(timeout=3)
                        self.assertEqual(parent.returncode, 0, error)
                        record = json.loads(s.artifact_paths(self.out)[0].read_text())
                        self.assert_failed(record, "parent_signal")
                        self.assertIn(int(sig), record["received_signals"])
                        for pid in pids:
                            self.assertFalse(
                                s.process_is_live(pid), f"owned PID {pid} survived"
                            )
                        self.assertIsNone(unrelated.poll())
                    finally:
                        if parent.poll() is None:
                            parent.kill()
                        parent.communicate(timeout=3)
                        if (self.out / "pids.json").exists():
                            own = json.loads((self.out / "pids.json").read_text())[0]
                            try:
                                os.killpg(own, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)

    def test_gpu_preflight_is_bounded_and_requires_nine_microcells(self):
        budget = s.limits_for("gpu_preflight")
        self.assertEqual(budget["total_seconds"], 900)
        self.assertEqual(
            budget["phase_seconds"],
            {"preparation": 300, "evaluation": 300, "certification": 300},
        )
        self.assertEqual(budget["rss_bytes"], 4 * 1024**3)
        self.assertEqual(budget["vram_bytes"], 8 * 1024**3)
        self.assertEqual(budget["output_bytes"], 1024**3)
        with self.assertRaises(ValueError):
            s.limits_for("gpu_preflight", {"vram_bytes": 8 * 1024**3 + 1})
        record = self.run_worker(
            mode="gpu_preflight",
            cells=0,
            sample_fn=lambda child: {"rss_bytes": 1, "vram_bytes": 1},
        )
        self.assertEqual(record["status"], "preflight_completed")
        self.assertEqual(record["mode"], "gpu_preflight")
        self.out = self.root / "invalid-gpu-micro"
        rejected = self.run_worker(
            mode="gpu_preflight",
            cells=9,
            sample_fn=lambda child: {"rss_bytes": 1, "vram_bytes": 1},
        )
        self.assert_failed(rejected, "invalid_worker_status")


class SamplerTests(unittest.TestCase):
    @unittest.skipUnless(
        sys.platform.startswith("linux") and hasattr(os, "WNOWAIT"),
        "non-reaping waitid is required for formal Linux PID identity",
    )
    def test_peek_exit_preserves_leader_pid_identity_until_wait(self):
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", "pass"], start_new_session=True
        )
        try:
            deadline = time.monotonic() + 2
            while s._peek_exit(child) is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(s._peek_exit(child), 0)
            self.assertIsNone(child.returncode)
            self.assertEqual(child.wait(timeout=1), 0)
        finally:
            if child.returncode is None:
                child.kill()
                child.wait(timeout=1)

    def test_reused_session_identity_cannot_be_signaled(self):
        fake = SimpleNamespace(pid=100, returncode=0, poll=lambda: 0, wait=lambda: 0)
        rows = {101: (1, 100, 999, 1, "S")}
        with (
            patch.object(s, "_process_rows", return_value=rows),
            patch.object(s.os, "killpg") as signal_group,
            self.assertRaises((ValueError, s.GPUQueryError)),
        ):
            s._kill_group(fake, 0.01, [], owned_pgid=100)
        signal_group.assert_not_called()

    def test_missing_proc_root_and_live_worker_without_rows_fail_closed(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(s.GPUQueryError),
        ):
            s._linux_rows(Path(directory) / "missing")
        worker = SimpleNamespace(pid=100, poll=lambda: None)
        with (
            patch.object(s, "_process_rows", return_value={}),
            self.assertRaises(s.GPUQueryError),
        ):
            s.sample_process(worker, mode="formal")

    def test_exited_leader_still_finds_owned_group_descendants(self):
        rows = {101: (1, 100, 100, 42, "S")}
        worker = SimpleNamespace(pid=100, poll=lambda: 0)
        with patch.object(s, "_process_rows", return_value=rows):
            self.assertEqual(
                s.sample_process(worker, mode="cpu_preflight"),
                {"rss_bytes": 42, "vram_bytes": 0},
            )

    def test_rss_sum_includes_group_and_descendant_tree_not_other_processes(self):
        # ppid, pgid, sid, rss_bytes, state
        rows = {
            100: (1, 100, 100, 10, "S"),
            101: (100, 100, 100, 20, "S"),
            102: (101, 102, 102, 30, "S"),
            103: (1, 100, 100, 40, "S"),
            200: (1, 200, 200, 999, "S"),
        }
        with (
            patch.object(s, "_process_rows", return_value=rows),
            patch.object(s, "_gpu_bytes", return_value=0) as gpu,
        ):
            sample = s.sample_process(SimpleNamespace(pid=100), mode="cpu_preflight")
        self.assertEqual(sample, {"rss_bytes": 100, "vram_bytes": 0})
        gpu.assert_not_called()

    def test_gpu_preflight_samples_owned_vram_and_fails_on_query_error(self):
        rows = {100: (1, 100, 100, 42, "S")}
        worker = SimpleNamespace(pid=100)
        with (
            patch.object(s, "_process_rows", return_value=rows),
            patch.object(s, "_gpu_bytes", return_value=123) as gpu,
        ):
            sample = s.sample_process(worker, mode="gpu_preflight")
        self.assertEqual(sample, {"rss_bytes": 42, "vram_bytes": 123})
        gpu.assert_called_once_with({100})
        with (
            patch.object(s, "_process_rows", return_value=rows),
            patch.object(s, "_gpu_bytes", side_effect=s.GPUQueryError("unavailable")),
            self.assertRaises(s.GPUQueryError),
        ):
            s.sample_process(worker, mode="gpu_preflight")

    def test_stale_worker_cuda_telemetry_cannot_be_certified(self):
        rows = {100: (1, 100, 100, 42, "S")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "memory.json"
            path.write_text(json.dumps({"max_allocated_bytes": 1880}))
            stale = time.time() - s.TELEMETRY_STALE_SECONDS - 5
            os.utime(path, (stale, stale))
            with (
                patch.object(s, "_process_rows", return_value=rows),
                patch.object(s, "_gpu_bytes", return_value=0),
                self.assertRaisesRegex(ValueError, "stale"),
            ):
                s.sample_process(
                    SimpleNamespace(pid=100), mode="formal", output=output
                )

    def test_worker_cuda_telemetry_is_billed_when_driver_pid_is_invisible(self):
        rows = {100: (1, 100, 100, 42, "S")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "memory.json").write_text(
                json.dumps({"allocated_bytes": 900, "max_allocated_bytes": 1880})
            )
            with (
                patch.object(s, "_process_rows", return_value=rows),
                patch.object(s, "_gpu_bytes", return_value=0),
            ):
                sample = s.sample_process(
                    SimpleNamespace(pid=100), mode="formal", output=output
                )
        self.assertEqual(sample, {"rss_bytes": 42, "vram_bytes": 1880})

    def test_worker_cuda_telemetry_failure_or_corruption_fails_closed(self):
        rows = {100: (1, 100, 100, 42, "S")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "memory.json"
            with (
                patch.object(s, "_process_rows", return_value=rows),
                patch.object(s, "_gpu_bytes", return_value=0),
            ):
                for payload in ('{"error": "RuntimeError: boom"}', "not json"):
                    with self.subTest(payload=payload):
                        path.write_text(payload)
                        with self.assertRaises(ValueError):
                            s.sample_process(
                                SimpleNamespace(pid=100), mode="formal", output=output
                            )

    def test_formal_gpu_sums_only_owned_pid_rows(self):
        result = SimpleNamespace(stdout="100, 2\n101, 3\n200, 999\n")
        with patch.object(s.subprocess, "run", return_value=result) as run:
            self.assertEqual(s._gpu_bytes({100, 101}), 5 * 1024 * 1024)
        self.assertIn("nvidia-smi", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_malformed_gpu_query_cannot_be_counted_as_zero_vram(self):
        with (
            patch.object(
                s.subprocess, "run", return_value=SimpleNamespace(stdout="not a pid\n")
            ),
            self.assertRaises(s.GPUQueryError),
        ):
            s._gpu_bytes({100})

    def test_linux_proc_disappearance_does_not_poison_remaining_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "101").mkdir()  # disappeared between enumeration and stat read
            (root / "102").mkdir()
            fields = ["S", "100", "100", "100"] + ["0"] * 17 + ["3"]
            (root / "102" / "stat").write_text(
                "102 (name with ) spaces) " + " ".join(fields)
            )
            rows = s._linux_rows(root)
        self.assertEqual(rows[102][:3], (100, 100, 100))
        self.assertEqual(rows[102][3], 3 * os.sysconf("SC_PAGE_SIZE"))
        self.assertNotIn(101, rows)

    def test_frozen_defaults_and_helper_only_accept_reduced_limits(self):
        self.assertEqual(s.limits_for("formal")["total_seconds"], 4800)
        self.assertEqual(
            s.limits_for("formal")["phase_seconds"],
            {"preparation": 300, "evaluation": 3600, "certification": 900},
        )
        self.assertEqual(
            s.limits_for("cpu_preflight")["phase_seconds"],
            {"preparation": 300, "evaluation": 120, "certification": 300},
        )
        self.assertEqual(s.limits_for("cpu_preflight")["total_seconds"], 900)
        self.assertEqual(s.limits_for("cpu_preflight")["vram_bytes"], 0)
        for changes in (
            {"total_seconds": 4801},
            {"poll_seconds": 2},
            {"phase_seconds": {"evaluation": 3601}},
            {"rss_bytes": True},
            {"poll_seconds": float("nan")},
            {"unknown": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                s.limits_for("formal", changes)

    def test_cli_has_no_custom_budget_option(self):
        with self.assertRaises(SystemExit) as caught:
            s.main(
                [
                    "--output",
                    "/tmp/synthetic",
                    "--total-seconds",
                    "999999",
                    "--",
                    "true",
                ]
            )
        self.assertNotEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
