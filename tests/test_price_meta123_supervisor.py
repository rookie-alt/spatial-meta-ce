import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from spatial_meta_ce import price_meta123_supervisor as supervisor


_GPU_UUID = "GPU-0050e82c-8c24-d85d-00d3-b73f12638df0"
_OTHER_GPU_UUID = "GPU-722d57aa-e2e6-8b0a-6c6f-94e99f3031ed"


def _budget(**overrides):
    budget = {
        "max_wall_seconds": 30.0,
        "max_rss_bytes": 4 * 1024**3,
        "max_vram_bytes": 0,
        "max_output_bytes": 64 * 1024**2,
        "poll_seconds": 0.05,
    }
    budget.update(overrides)
    return budget


def _process_table():
    """Portable stand-in for /proc used by the mocked platform tests."""
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=,ppid=,pgid=,rss=,state=", "-ax"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            pid, ppid, pgid, rss = (int(item) for item in parts[:4])
            sid = os.getsid(pid)
        except (ValueError, OSError):
            continue
        rows[pid] = supervisor._ProcRow(ppid, pgid, sid, rss * 1024, parts[4][:1])
    return rows


def _raise_monitoring_error(*_args, **_kwargs):
    raise supervisor.MonitoringError("process table cannot be enumerated")


def _process_is_live(pid):
    """Report whether a pid is still a live (non-zombie) process."""
    try:
        rows = supervisor._read_proc_rows()
    except supervisor.MonitoringError:
        rows = {}
    row = rows.get(int(pid))
    if row is not None:
        return row.state != "Z"
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _kill_quietly(pid):
    """Best-effort cleanup so a failing test leaves no process behind."""
    for sig in (signal.SIGKILL, signal.SIGTERM):
        try:
            os.kill(int(pid), sig)
        except OSError:
            return
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not _process_is_live(pid):
                return
            time.sleep(0.05)


@contextlib.contextmanager
def _mocked_platform(*, rows=None, nvidia=None):
    with mock.patch.object(supervisor, "_is_linux", return_value=True):
        with mock.patch.object(
            supervisor,
            "_read_proc_rows",
            side_effect=rows if rows is not None else _process_table,
        ):
            if nvidia is None:
                yield
            else:
                with mock.patch.object(
                    supervisor,
                    "_run_nvidia_smi",
                    side_effect=nvidia,
                    create=True,
                ):
                    yield


def _argument_after(args, flag):
    """Return the argument that follows a flag in an argv list."""
    for index, item in enumerate(args):
        if item == flag and index + 1 < len(args):
            return args[index + 1]
    return None


class _FakeNvidiaSmi:
    """Scripted nvidia-smi stand-in for the device-total accounting tests."""

    def __init__(
        self,
        *,
        devices=None,
        compute=None,
        indices=None,
        error=None,
        device_states=None,
        compute_states=None,
        fail_from_call=None,
    ):
        self.devices = dict(devices or {})
        self.compute = dict(compute or {})
        self.indices = list(indices or [])
        self.error = error
        self.device_states = {
            key: list(value) for key, value in (device_states or {}).items()
        }
        self.compute_states = {
            key: list(value) for key, value in (compute_states or {}).items()
        }
        self.fail_from_call = fail_from_call
        self.calls = []

    def _next_state(self, uuid):
        states = self.device_states.get(uuid)
        if states:
            if len(states) > 1:
                return states.pop(0)
            return states[0]
        return self.devices.get(uuid)

    def _next_compute(self, uuid):
        states = self.compute_states.get(uuid)
        if states:
            if len(states) > 1:
                return states.pop(0)
            return states[0]
        return self.compute.get(uuid, [])

    def __call__(self, args, **_kwargs):
        self.calls.append(list(args))
        if self.fail_from_call is not None and len(self.calls) >= self.fail_from_call:
            raise supervisor.MonitoringError("nvidia-smi failed")
        if self.error is not None:
            raise self.error
        joined = " ".join(args)
        if "--query-compute-apps=pid,gpu_uuid" in joined:
            rows = []
            sources = dict(self.compute)
            sources.update(self.compute_states)
            for uuid, pids in sources.items():
                if uuid in self.compute_states:
                    pids = self._next_compute(uuid)
                for pid in pids:
                    rows.append(f"{pid}, {uuid}")
            return "".join(f"{row}\n" for row in rows)
        if "--query-gpu=index,uuid" in joined:
            return "".join(f"{index}, {uuid}\n" for index, uuid in self.indices)
        if "--query-gpu=uuid,memory.used,utilization.gpu" in joined:
            uuid = _argument_after(args, "-i")
            value = self._next_state(uuid)
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            used_mib, utilization = value
            return f"{uuid}, {used_mib}, {utilization}\n"
        raise AssertionError(f"unexpected nvidia-smi query: {joined}")


def _idle_nvidia(*, busy_after=None, compute_after=None, **overrides):
    """An nvidia-smi stand-in whose authorized device starts idle.

    ``busy_after`` is the device reading that replaces the idle one once the
    worker has started, which mirrors a real run: the card is empty before
    the launch and carries the workload afterwards.  ``compute_after`` is the
    compute-app list that appears after the idle preflight.
    """
    settings = {
        "devices": {_GPU_UUID: (1, 0)},
        "compute": {},
        "indices": [(6, _GPU_UUID)],
    }
    settings.update(overrides)
    nvidia = _FakeNvidiaSmi(**settings)
    if busy_after is not None:
        nvidia.device_states[_GPU_UUID] = [(1, 0), busy_after]
    if compute_after is not None:
        nvidia.compute_states[_GPU_UUID] = [[], list(compute_after)]
    return nvidia


def _worker_code(body):
    return (
        "import json,os,sys,time\n"
        "from pathlib import Path\n"
        "out=Path(sys.argv[1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n" + body
    )


def _completion(**overrides):
    marker = {
        "status": "completed",
        "protocol": supervisor.COMPLETION_PROTOCOL,
        "binding_sha256": "a" * 64,
    }
    marker.update(overrides)
    return json.dumps(marker)


class BudgetValidationTests(unittest.TestCase):
    def test_budget_requires_exact_field_set(self):
        with self.assertRaises(supervisor.BudgetError):
            supervisor._validate_budget({"max_wall_seconds": 1.0})
        with self.assertRaises(supervisor.BudgetError):
            budget = _budget()
            budget["max_gpu_seconds"] = 1.0
            supervisor._validate_budget(budget)

    def test_budget_rejects_non_positive_values(self):
        for field in (
            "max_wall_seconds",
            "max_rss_bytes",
            "max_output_bytes",
            "poll_seconds",
        ):
            with self.subTest(field=field):
                for value in (0, -1, float("nan"), float("inf")):
                    with self.assertRaises(supervisor.BudgetError):
                        supervisor._validate_budget(_budget(**{field: value}))

    def test_budget_allows_zero_vram_only(self):
        limits = supervisor._validate_budget(_budget(max_vram_bytes=0))
        self.assertEqual(limits["max_vram_bytes"], 0)
        with self.assertRaises(supervisor.BudgetError):
            supervisor._validate_budget(_budget(max_vram_bytes=-1))

    def test_budget_rejects_bool_and_non_numeric(self):
        with self.assertRaises(supervisor.BudgetError):
            supervisor._validate_budget(_budget(max_wall_seconds=True))
        with self.assertRaises(supervisor.BudgetError):
            supervisor._validate_budget(_budget(max_rss_bytes="1024"))
        with self.assertRaises(supervisor.BudgetError):
            supervisor._validate_budget(_budget(max_wall_seconds=0.5, poll_seconds="1"))


class CommandValidationTests(unittest.TestCase):
    def test_command_rejects_shell_string(self):
        with self.assertRaises(ValueError):
            supervisor._validate_command("echo hello")
        with self.assertRaises(ValueError):
            supervisor._validate_command(b"echo hello")

    def test_command_rejects_empty_or_invalid_arguments(self):
        with self.assertRaises(ValueError):
            supervisor._validate_command([])
        with self.assertRaises(ValueError):
            supervisor._validate_command(["python", ""])
        with self.assertRaises(ValueError):
            supervisor._validate_command(["python", 3])
        with self.assertRaises(ValueError):
            supervisor._validate_command(["python", "a\0b"])


class OutputPathTests(unittest.TestCase):
    def test_output_must_not_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                supervisor._validate_output(output)

    def test_output_rejects_symlinked_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            link = root / "link"
            link.symlink_to(real)
            with self.assertRaises(supervisor.UnsafeOutputError):
                supervisor._validate_output(link / "run")

    def test_output_rejects_dotdot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                supervisor._validate_output(root / "a" / ".." / "b")

    def test_output_rejects_relative_path(self):
        with self.assertRaises(ValueError):
            supervisor._validate_output(Path("relative/run"))

    def test_output_outside_boundary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "inside"
            inside.mkdir()
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaises(ValueError):
                supervisor._validate_output(outside / "run", boundary=inside)


class EnvironmentTests(unittest.TestCase):
    def test_env_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            supervisor._build_child_env({"PATH_TUNING": "1"})
        with self.assertRaises(ValueError):
            supervisor._build_child_env({supervisor.SUPERVISED_ENV: "0"})
        with self.assertRaises(ValueError):
            supervisor._build_child_env({supervisor.SUPERVISOR_PID_ENV: "1"})

    def test_env_clears_stale_supervision_fields(self):
        with mock.patch.dict(
            os.environ,
            {
                "META65_SUPERVISED_MODE": "formal",
                "META65_SUPERVISOR_PID": "123",
                "PRICE_META123_SUPERVISED": "0",
                "PRICE_META123_SUPERVISOR_PID": "456",
            },
        ):
            env = supervisor._build_child_env({"OMP_NUM_THREADS": "1"})
        self.assertNotIn("META65_SUPERVISED_MODE", env)
        self.assertNotIn("META65_SUPERVISOR_PID", env)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env[supervisor.SUPERVISED_ENV], "1")
        self.assertEqual(env[supervisor.SUPERVISOR_PID_ENV], str(os.getpid()))


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _run(self, argv, output, budget=None, **kwargs):
        budget = budget if budget is not None else _budget()
        env = kwargs.pop("env", None)
        if budget["max_vram_bytes"] > 0:
            kwargs.setdefault("nvidia", _idle_nvidia())
            env = env if env is not None else {"CUDA_VISIBLE_DEVICES": "6"}
        with _mocked_platform(**kwargs):
            return supervisor.supervise(
                argv,
                output,
                cwd=self.root,
                budget=budget,
                env=env,
                gpu_uuid=_GPU_UUID if budget["max_vram_bytes"] > 0 else None,
            )

    def _completed_worker(self, output, extra=""):
        code = _worker_code(
            f"(out/'completion.json').write_text({_completion()!r})\n" + extra
        )
        return [sys.executable, "-B", "-c", code, str(output)]

    def test_completed_when_exit_zero_and_completion_marker(self):
        output = self.root / "run"
        record = self._run(self._completed_worker(output), output)
        self.assertEqual(record["status"], "completed")
        self.assertIsNone(record["failure"])
        self.assertEqual(record["returncode"], 0)
        self.assertEqual(record["completion"]["status"], "completed")
        self.assertTrue(Path(record["supervisor_json"]).is_file())
        self.assertTrue(Path(record["supervisor_log"]).is_file())

    def test_missing_completion_fails(self):
        output = self.root / "run"
        code = _worker_code("pass\n")
        record = self._run([sys.executable, "-B", "-c", code, str(output)], output)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "missing_completion")

    def test_completion_status_not_completed_fails(self):
        output = self.root / "run"
        code = _worker_code(
            f"(out/'completion.json').write_text({_completion(status='failed')!r})\n"
        )
        record = self._run([sys.executable, "-B", "-c", code, str(output)], output)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "incomplete_completion")

    def test_completion_without_protocol_or_binding_fails(self):
        for payload in (
            json.dumps({"status": "completed"}),
            json.dumps(
                {"status": "completed", "protocol": supervisor.COMPLETION_PROTOCOL}
            ),
            json.dumps(
                {
                    "status": "completed",
                    "protocol": supervisor.COMPLETION_PROTOCOL,
                    "binding_sha": "not-a-hex-digest",
                }
            ),
            json.dumps(
                {
                    "status": "completed",
                    "protocol": 122,
                    "binding_sha": "a" * 64,
                }
            ),
        ):
            with self.subTest(payload=payload):
                output = self.root / ("run-" + str(abs(hash(payload))))
                code = _worker_code(
                    f"(out/'completion.json').write_text({payload!r})\n"
                )
                record = self._run(
                    [sys.executable, "-B", "-c", code, str(output)], output
                )
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["failure"]["type"], "incomplete_completion")

    def test_nonzero_exit_fails(self):
        output = self.root / "run"
        code = _worker_code(
            f"(out/'completion.json').write_text({_completion()!r})\nsys.exit(3)\n"
        )
        record = self._run([sys.executable, "-B", "-c", code, str(output)], output)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "worker_exit")

    def test_wall_limit_terminates_worker(self):
        output = self.root / "run"
        code = _worker_code("time.sleep(60)\n")
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            budget=_budget(max_wall_seconds=0.5),
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "wall_limit")
        self.assertTrue(record["cleanup"]["attempted"])
        self.assertEqual(record["cleanup"]["residual_pids"], [])

    def test_rss_limit_terminates_worker(self):
        output = self.root / "run"
        code = _worker_code("time.sleep(60)\n")
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            budget=_budget(max_rss_bytes=1024),
        )
        self.assertEqual(record["failure"]["type"], "rss_limit")

    def test_vram_limit_terminates_worker(self):
        output = self.root / "run"
        code = _worker_code("time.sleep(60)\n")
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            budget=_budget(max_vram_bytes=1024),
            nvidia=_idle_nvidia(busy_after=(4096, 80)),
        )
        self.assertEqual(record["failure"]["type"], "vram_limit")

    def test_output_limit_terminates_worker(self):
        output = self.root / "run"
        code = _worker_code(
            "(out/'blob.bin').write_bytes(b'x' * 4096)\ntime.sleep(60)\n"
        )
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            budget=_budget(max_output_bytes=1024),
        )
        self.assertEqual(record["failure"]["type"], "output_limit")

    def test_monitoring_error_fails_closed(self):
        output = self.root / "run"
        code = _worker_code("time.sleep(60)\n")
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            rows=_raise_monitoring_error,
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "monitoring_error")

    def test_vram_query_failure_is_not_treated_as_zero(self):
        output = self.root / "run"
        code = _worker_code("time.sleep(60)\n")

        def _explode(*_args, **_kwargs):
            raise supervisor.MonitoringError("nvidia-smi failed")

        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            budget=_budget(max_vram_bytes=1024**3),
            nvidia=_explode,
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "monitoring_error")

    def test_zero_vram_budget_never_queries_the_device(self):
        output = self.root / "run"

        def _explode(_pids):
            raise AssertionError("nvidia-smi must not be queried for a CPU budget")

        code = _worker_code(f"(out/'completion.json').write_text({_completion()!r})\n")
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            budget=_budget(max_vram_bytes=0),
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["peak_vram_bytes"], 0)

    def test_parent_signal_terminates_owned_group(self):
        output = self.root / "run"
        marker = self.root / "started"
        code = (
            "import sys,time\n"
            "from pathlib import Path\n"
            "Path(sys.argv[2]).write_text('x')\n"
            "time.sleep(60)\n"
        )
        argv = [sys.executable, "-B", "-c", code, str(output), str(marker)]

        def _trigger():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not marker.exists():
                time.sleep(0.02)
            if marker.exists():
                os.kill(os.getpid(), signal.SIGTERM)

        previous = signal.signal(signal.SIGTERM, lambda *_args: None)
        self.addCleanup(signal.signal, signal.SIGTERM, previous)
        thread = threading.Thread(target=_trigger, daemon=True)
        thread.start()
        record = self._run(argv, output)
        thread.join(timeout=10)
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "parent_signal")
        self.assertTrue(record["cleanup"]["attempted"])

    def test_supervision_context_is_injected_and_cannot_be_overridden(self):
        output = self.root / "run"
        code = _worker_code(
            "(out/'env.json').write_text(json.dumps(dict(os.environ)))\n"
            f"(out/'completion.json').write_text({_completion()!r})\n"
        )
        argv = [sys.executable, "-B", "-c", code, str(output)]
        with _mocked_platform():
            record = supervisor.supervise(argv, output, cwd=self.root, budget=_budget())
        self.assertEqual(record["status"], "completed")
        environ = json.loads((output / "env.json").read_text(encoding="utf-8"))
        self.assertEqual(environ[supervisor.SUPERVISED_ENV], "1")
        self.assertEqual(environ[supervisor.SUPERVISOR_PID_ENV], str(os.getpid()))
        self.assertEqual([key for key in environ if key.startswith("META65_")], [])
        with _mocked_platform():
            with self.assertRaises(ValueError):
                supervisor.supervise(
                    argv,
                    self.root / "other",
                    cwd=self.root,
                    budget=_budget(),
                    env={supervisor.SUPERVISED_ENV: "0"},
                )

    def test_arguments_are_not_shell_interpreted(self):
        output = self.root / "run"
        marker = "; touch pwned"
        code = _worker_code(
            "(out/'arg.txt').write_text(sys.argv[2])\n"
            f"(out/'completion.json').write_text({_completion()!r})\n"
        )
        argv = [sys.executable, "-B", "-c", code, str(output), marker]
        record = self._run(argv, output)
        self.assertEqual(record["status"], "completed")
        self.assertEqual((output / "arg.txt").read_text(encoding="utf-8"), marker)
        self.assertFalse((self.root / "pwned").exists())

    def test_sidecars_are_never_overwritten(self):
        output = self.root / "run"
        record_path, log_path = supervisor._sidecar_paths(output)
        record_path.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._run(self._completed_worker(output), output)
        self.assertEqual(record_path.read_text(encoding="utf-8"), "keep")
        self.assertFalse(log_path.exists())

    def test_symlinked_sidecar_is_rejected(self):
        output = self.root / "run"
        record_path, _ = supervisor._sidecar_paths(output)
        record_path.symlink_to(self.root / "target")
        with self.assertRaises(supervisor.UnsafeOutputError):
            self._run(self._completed_worker(output), output)

    def test_non_linux_is_rejected(self):
        output = self.root / "run"
        with mock.patch.object(supervisor, "_is_linux", return_value=False):
            with self.assertRaises(supervisor.UnsupportedPlatformError):
                supervisor.supervise(
                    self._completed_worker(output),
                    output,
                    cwd=self.root,
                    budget=_budget(),
                )

    def test_supervisor_does_not_create_the_worker_output(self):
        output = self.root / "run"
        code = _worker_code(f"(out/'completion.json').write_text({_completion()!r})\n")
        record = self._run([sys.executable, "-B", "-c", code, str(output)], output)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(output.is_dir())

    def test_worker_may_own_the_output_directory(self):
        output = self.root / "run"
        code = (
            "import json,sys\n"
            "from pathlib import Path\n"
            "out=Path(sys.argv[1])\n"
            "out.mkdir(parents=True)\n"
            f"(out/'completion.json').write_text({_completion()!r})\n"
        )
        record = self._run([sys.executable, "-B", "-c", code, str(output)], output)
        self.assertEqual(record["status"], "completed")

    def test_worker_that_never_creates_output_fails(self):
        output = self.root / "run"
        code = "import time\ntime.sleep(0.2)\n"
        record = self._run([sys.executable, "-B", "-c", code, str(output)], output)
        self.assertEqual(record["status"], "failed")
        self.assertFalse(output.exists())

    def test_output_may_sit_beside_the_code_root(self):
        """The runtime layout keeps outputs/ next to the code/ root."""
        code_root = self.root / "code"
        code_root.mkdir()
        output = self.root / "outputs" / "run"
        code = _worker_code(f"(out/'completion.json').write_text({_completion()!r})\n")
        with _mocked_platform():
            record = supervisor.supervise(
                [sys.executable, "-B", "-c", code, str(output)],
                output,
                cwd=code_root,
                budget=_budget(),
            )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["cwd"], str(code_root))

    def test_explicit_root_overrides_the_default_boundary(self):
        code_root = self.root / "code"
        code_root.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        output = outside / "run"
        code = _worker_code(f"(out/'completion.json').write_text({_completion()!r})\n")
        with _mocked_platform():
            with self.assertRaises(supervisor.UnsafeOutputError):
                supervisor.supervise(
                    [sys.executable, "-B", "-c", code, str(output)],
                    output,
                    cwd=code_root,
                    budget=_budget(),
                    root=code_root,
                )

    def test_worker_that_cannot_start_is_recorded(self):
        output = self.root / "run"
        with _mocked_platform():
            record = supervisor.supervise(
                [str(self.root / "missing-binary")],
                output,
                cwd=self.root,
                budget=_budget(),
            )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "worker_start")
        self.assertTrue(Path(record["supervisor_json"]).is_file())

    def test_recycled_worker_pid_is_refused(self):
        """A worker pid that changed identity must not be sampled or signalled."""
        rows = {
            10: supervisor._ProcRow(1, 10, 10, 100, "S", "start-10"),
            11: supervisor._ProcRow(10, 11, 11, 200, "S", "start-11"),
        }
        identity = {10: "start-10", 11: "start-11"}
        recycled = {
            10: supervisor._ProcRow(1, 10, 10, 100, "S", "other-10"),
            11: supervisor._ProcRow(10, 11, 11, 200, "S", "start-11"),
        }
        with self.assertRaises(supervisor.PidReuseError):
            supervisor._owned_pids(recycled, 10, identity=identity)
        # A descendant whose pid was recycled is ignored, never counted as owned.
        owned = supervisor._owned_pids(rows, 10, identity=identity)
        self.assertEqual(owned, {10, 11})


class DeviceTotalVramTests(unittest.TestCase):
    """The supervisor must measure the whole authorized card, not a PID set."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _completed_worker(self, output, hold=0.5):
        """A worker that stays alive long enough to be sampled."""
        code = _worker_code(
            "time.sleep(%r)\n" % hold
            + f"(out/'completion.json').write_text({_completion()!r})\n"
        )
        return [sys.executable, "-B", "-c", code, str(output)]

    def _run(self, argv, output, *, nvidia, budget=None, env=None, gpu_uuid=_GPU_UUID):
        def _no_pid_path(_pids):
            raise AssertionError(
                "the pid-based NVML path must not be used for device accounting"
            )

        with mock.patch.object(
            supervisor, "_query_vram_bytes", side_effect=_no_pid_path
        ):
            with _mocked_platform(nvidia=nvidia):
                return supervisor.supervise(
                    argv,
                    output,
                    cwd=self.root,
                    budget=budget if budget is not None else _budget(),
                    env=env,
                    gpu_uuid=gpu_uuid,
                )

    def test_gpu_budget_requires_an_explicit_gpu_uuid(self):
        output = self.root / "run"
        with _mocked_platform(nvidia=_idle_nvidia()):
            with self.assertRaises(ValueError):
                supervisor.supervise(
                    self._completed_worker(output),
                    output,
                    cwd=self.root,
                    budget=_budget(max_vram_bytes=1024**3),
                    env={"CUDA_VISIBLE_DEVICES": "6"},
                    gpu_uuid=None,
                )

    def test_gpu_uuid_must_match_the_selected_device(self):
        """A UUID that is not the selected physical index must be refused."""
        output = self.root / "run"
        nvidia = _idle_nvidia(indices=[(6, _GPU_UUID)])
        with _mocked_platform(nvidia=nvidia):
            with self.assertRaises(ValueError):
                supervisor.supervise(
                    self._completed_worker(output),
                    output,
                    cwd=self.root,
                    budget=_budget(max_vram_bytes=1024**3),
                    env={"CUDA_VISIBLE_DEVICES": "6"},
                    gpu_uuid=_OTHER_GPU_UUID,
                )

    def test_env_must_select_exactly_one_physical_index(self):
        output = self.root / "run"
        nvidia = _idle_nvidia(indices=[(6, _GPU_UUID)])
        with _mocked_platform(nvidia=nvidia):
            with self.assertRaises(ValueError):
                supervisor.supervise(
                    self._completed_worker(output),
                    output,
                    cwd=self.root,
                    budget=_budget(max_vram_bytes=1024**3),
                    env={"CUDA_VISIBLE_DEVICES": ""},
                    gpu_uuid=_GPU_UUID,
                )

    def test_env_may_name_the_authorized_card_by_uuid(self):
        """The caller may pin the card by UUID instead of by physical index."""
        output = self.root / "run"
        nvidia = _idle_nvidia(indices=[(6, _GPU_UUID)])
        with _mocked_platform(nvidia=nvidia):
            record = supervisor.supervise(
                self._completed_worker(output),
                output,
                cwd=self.root,
                budget=_budget(max_vram_bytes=4 * 1024**3),
                env={"CUDA_VISIBLE_DEVICES": _GPU_UUID},
                gpu_uuid=_GPU_UUID,
            )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["gpu_uuid"], _GPU_UUID)
        self.assertEqual(record["gpu_index"], 6)

    def test_worker_is_pinned_to_the_authorized_uuid(self):
        """The child must see the authorized UUID, not a bare index."""
        output = self.root / "run"
        code = _worker_code(
            "(out/'env.json').write_text(json.dumps(dict(os.environ)))\n"
            "time.sleep(0.5)\n"
            f"(out/'completion.json').write_text({_completion()!r})\n"
        )
        nvidia = _idle_nvidia(busy_after=(700, 30))
        with mock.patch.object(
            supervisor,
            "_query_vram_bytes",
            side_effect=AssertionError("the pid-based NVML path must not be used"),
        ):
            with _mocked_platform(nvidia=nvidia):
                record = supervisor.supervise(
                    [sys.executable, "-B", "-c", code, str(output)],
                    output,
                    cwd=self.root,
                    budget=_budget(max_vram_bytes=4 * 1024**3),
                    env={"CUDA_VISIBLE_DEVICES": "6"},
                    gpu_uuid=_GPU_UUID,
                )
        self.assertEqual(record["status"], "completed")
        environ = json.loads((output / "env.json").read_text(encoding="utf-8"))
        self.assertEqual(environ["CUDA_VISIBLE_DEVICES"], _GPU_UUID)

    def test_env_uuid_that_is_not_authorized_is_refused(self):
        output = self.root / "run"
        nvidia = _idle_nvidia(indices=[(6, _GPU_UUID)])
        with _mocked_platform(nvidia=nvidia):
            with self.assertRaises(ValueError):
                supervisor.supervise(
                    self._completed_worker(output),
                    output,
                    cwd=self.root,
                    budget=_budget(max_vram_bytes=4 * 1024**3),
                    env={"CUDA_VISIBLE_DEVICES": _OTHER_GPU_UUID},
                    gpu_uuid=_GPU_UUID,
                )

    def test_busy_card_is_refused_before_the_worker_starts(self):
        """A card with compute apps or memory in use must not host a run."""
        output = self.root / "run"
        busy = _idle_nvidia(
            devices={_GPU_UUID: (4096, 55)},
            compute={_GPU_UUID: [4242]},
        )
        with _mocked_platform(nvidia=busy):
            record = supervisor.supervise(
                self._completed_worker(output),
                output,
                cwd=self.root,
                budget=_budget(max_vram_bytes=1024**3),
                env={"CUDA_VISIBLE_DEVICES": "6"},
                gpu_uuid=_GPU_UUID,
            )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "gpu_not_idle")
        self.assertIsNone(record["pid"])
        self.assertFalse(output.exists())

    def test_device_memory_is_counted_even_without_pid_overlap(self):
        """The D2 failure: the NVML pids are invisible but the card is used."""
        output = self.root / "run"
        nvidia = _idle_nvidia(
            busy_after=(1500, 40),
            compute_after=[82894],
        )
        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=4 * 1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["vram_accounting"], "device_total")
        self.assertEqual(record["gpu_uuid"], _GPU_UUID)
        self.assertEqual(record["peak_vram_bytes"], 1500 * 1024**2)
        self.assertEqual(record["vram_baseline_bytes"], 1 * 1024**2)

    def test_other_cards_do_not_influence_the_measurement(self):
        """Only the authorized UUID may contribute to the accounted bytes."""
        output = self.root / "run"
        nvidia = _idle_nvidia(
            busy_after=(700, 30),
            devices={
                _GPU_UUID: (700, 30),
                _OTHER_GPU_UUID: (12000, 90),
            },
            compute={_OTHER_GPU_UUID: [95566]},
            indices=[(6, _GPU_UUID), (3, _OTHER_GPU_UUID)],
        )
        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["peak_vram_bytes"], 700 * 1024**2)

    def test_cpu_budget_never_queries_the_device(self):
        output = self.root / "run"

        def _explode(*_args, **_kwargs):
            raise AssertionError("nvidia-smi must not be queried for a CPU budget")

        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=_explode,
            budget=_budget(max_vram_bytes=0),
            env={"CUDA_VISIBLE_DEVICES": ""},
            gpu_uuid=None,
        )
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["peak_vram_bytes"], 0)
        self.assertIsNone(record["gpu_uuid"])

    def test_missing_device_row_fails_closed(self):
        """A query that returns no row must stop the run, not report zero."""
        output = self.root / "run"
        nvidia = _idle_nvidia(devices={})
        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "monitoring_error")

    def test_unavailable_device_value_fails_closed(self):
        """An N/A or unparsable device reading must stop the run."""
        output = self.root / "run"
        nvidia = _idle_nvidia(devices={_GPU_UUID: "N/A, N/A"})
        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "monitoring_error")

    def test_uuid_mismatch_in_the_query_fails_closed(self):
        output = self.root / "run"
        nvidia = _idle_nvidia(devices={_OTHER_GPU_UUID: (10, 0)})
        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "monitoring_error")

    def test_query_failure_is_not_treated_as_zero(self):
        output = self.root / "run"

        def _explode(*_args, **_kwargs):
            raise supervisor.MonitoringError("nvidia-smi failed")

        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=_explode,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "monitoring_error")

    def test_device_total_above_the_limit_kills_only_the_owned_group(self):
        """A whole-card overrun must stop the run without killing foreign pids."""
        output = self.root / "run"
        nvidia = _idle_nvidia(
            busy_after=(4096, 80),
            compute_after=[82894, 95566],
        )
        code = _worker_code("time.sleep(60)\n")
        record = self._run(
            [sys.executable, "-B", "-c", code, str(output)],
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["failure"]["type"], "vram_limit")
        self.assertTrue(record["cleanup"]["attempted"])
        self.assertEqual(record["cleanup"]["residual_pids"], [])
        # The foreign NVML pids are never signalled by the supervisor.
        for call in nvidia.calls:
            self.assertNotIn("--query-compute-apps=pid,used_memory", call)

    def test_idle_check_and_sampling_use_the_authorized_uuid(self):
        output = self.root / "run"
        nvidia = _idle_nvidia(busy_after=(12, 5))
        record = self._run(
            self._completed_worker(output),
            output,
            nvidia=nvidia,
            budget=_budget(max_vram_bytes=1024**3),
            env={"CUDA_VISIBLE_DEVICES": "6"},
        )
        self.assertEqual(record["status"], "completed")
        queries = [call for call in nvidia.calls if "-i" in call]
        self.assertTrue(queries)
        for call in queries:
            self.assertEqual(_argument_after(call, "-i"), _GPU_UUID)


class OutputScanTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_vanished_file_is_tolerated(self):
        output = self.root / "run"
        output.mkdir()
        (output / "a.bin").write_bytes(b"x" * 10)
        original = supervisor._safe_stat

        def _flaky(path):
            if path.name == "a.bin":
                raise FileNotFoundError(str(path))
            return original(path)

        with mock.patch.object(supervisor, "_safe_stat", side_effect=_flaky):
            total = supervisor._measure_output_bytes(
                output, self.root / "log", self.root / "rec"
            )
        self.assertEqual(total, 0)

    def test_permission_error_is_not_tolerated(self):
        output = self.root / "run"
        output.mkdir()
        (output / "a.bin").write_bytes(b"x" * 10)

        def _denied(_path):
            raise PermissionError("denied")

        with mock.patch.object(supervisor, "_safe_stat", side_effect=_denied):
            with self.assertRaises(supervisor.MonitoringError):
                supervisor._measure_output_bytes(
                    output, self.root / "log", self.root / "rec"
                )

    def test_symlink_inside_output_is_rejected(self):
        output = self.root / "run"
        output.mkdir()
        (output / "link").symlink_to(self.root / "target")
        with self.assertRaises(supervisor.UnsafeOutputError):
            supervisor._measure_output_bytes(
                output, self.root / "log", self.root / "rec"
            )

    def test_output_bytes_include_log_and_sidecar(self):
        output = self.root / "run"
        output.mkdir()
        (output / "a.bin").write_bytes(b"x" * 10)
        log = self.root / "run.supervisor.log"
        record = self.root / "run.supervisor.json"
        log.write_bytes(b"y" * 7)
        record.write_bytes(b"z" * 3)
        total = supervisor._measure_output_bytes(output, log, record)
        self.assertEqual(total, 20)


class ProcessTableTests(unittest.TestCase):
    def test_uncertified_group_only_signals_owned_leader_before_reaping(self):
        child = mock.Mock(pid=456789)
        cleanup = {
            "attempted": False,
            "signals": [],
            "residual_pids": [],
            "verified": False,
            "error": None,
        }
        sequence = []
        child.wait.side_effect = lambda **_kwargs: sequence.append("reap")
        with (
            mock.patch.object(
                supervisor, "_group_identity_is_certified", return_value=False
            ),
            mock.patch.object(supervisor, "_group_alive", return_value=False),
            mock.patch.object(supervisor, "_group_members", return_value=[]),
            mock.patch.object(supervisor.os, "killpg") as kill_group,
            mock.patch.object(
                supervisor.os,
                "kill",
                side_effect=lambda *_args: sequence.append("signal"),
            ) as kill_leader,
        ):
            supervisor._terminate_owned_group(child, child.pid, 0.01, cleanup)
        kill_group.assert_not_called()
        kill_leader.assert_called_once_with(child.pid, signal.SIGTERM)
        self.assertEqual(sequence, ["signal", "reap"])
        self.assertTrue(cleanup["verified"])

    def test_proc_rows_are_parsed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "42"
            entry.mkdir()
            fields = ["S", "7", "42", "42"]
            fields += ["0"] * 18
            fields[-1] = "2048"
            (entry / "stat").write_text(
                "42 (x) " + " ".join(fields) + "\n", encoding="ascii"
            )
            rows = supervisor._read_proc_rows(root)
        self.assertIn(42, rows)
        self.assertEqual(rows[42].ppid, 7)
        self.assertEqual(rows[42].pgid, 42)
        self.assertEqual(rows[42].sid, 42)
        self.assertEqual(rows[42].state, "S")
        self.assertGreater(rows[42].rss_bytes, 0)

    def test_owned_pids_collect_descendants(self):
        rows = {
            10: supervisor._ProcRow(1, 10, 10, 100, "S"),
            11: supervisor._ProcRow(10, 10, 10, 200, "S"),
            12: supervisor._ProcRow(11, 10, 10, 300, "S"),
            99: supervisor._ProcRow(1, 99, 99, 400, "S"),
        }
        owned = supervisor._owned_pids(rows, 10)
        self.assertEqual(owned, {10, 11, 12})

    def test_owned_pids_reject_reused_process_group(self):
        rows = {
            10: supervisor._ProcRow(1, 10, 10, 100, "S"),
            11: supervisor._ProcRow(1, 10, 77, 200, "S"),
        }
        with self.assertRaises(supervisor.MonitoringError):
            supervisor._owned_pids(rows, 10)

    def test_owned_pids_reject_detached_leader(self):
        rows = {10: supervisor._ProcRow(1, 5, 5, 100, "S")}
        with self.assertRaises(supervisor.MonitoringError):
            supervisor._owned_pids(rows, 10)

    def test_owned_pids_keep_a_descendant_that_left_the_group(self):
        """A descendant that called setsid is still a descendant of the worker."""
        rows = {
            10: supervisor._ProcRow(1, 10, 10, 100, "S"),
            11: supervisor._ProcRow(10, 11, 11, 200, "S"),
        }
        owned = supervisor._owned_pids(rows, 10)
        self.assertEqual(owned, {10, 11})

    def test_owned_pids_reject_a_reused_pid(self):
        """A pid recycled by an unrelated process must not count as owned."""
        rows = {
            10: supervisor._ProcRow(1, 10, 10, 100, "S"),
            11: supervisor._ProcRow(10, 11, 11, 200, "S"),
        }
        identity = {10: "start-10", 11: "start-11"}
        with self.assertRaises(supervisor.MonitoringError):
            supervisor._owned_pids(rows, 10, identity=identity)

    def test_identity_map_detects_a_recycled_pid(self):
        """The same pid with a different start time is a different process."""
        identity = {10: "start-10"}
        self.assertTrue(supervisor._pid_identity_matches(10, "start-10", identity))
        self.assertFalse(supervisor._pid_identity_matches(10, "start-other", identity))
        self.assertFalse(supervisor._pid_identity_matches(10, "start-10", {}))
        self.assertTrue(
            supervisor._pid_identity_matches(10, "start-10", {10: "start-10"})
        )


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux /proc")
class LinuxProcessGroupTests(unittest.TestCase):
    def test_real_proc_rows_expose_group_and_rss(self):
        rows = supervisor._read_proc_rows()
        self.assertIn(os.getpid(), rows)
        own = rows[os.getpid()]
        self.assertEqual(own.pgid, os.getpgrp())
        self.assertGreater(own.rss_bytes, 0)

    def test_real_proc_sampling_counts_descendants(self):
        code = (
            "import subprocess,sys,time\n"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
            "time.sleep(60)\n"
        )
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", code],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 10
            rows = {}
            while time.monotonic() < deadline:
                rows = supervisor._read_proc_rows()
                owned = supervisor._owned_pids(rows, child.pid)
                if len(owned) >= 2:
                    break
                time.sleep(0.05)
            owned = supervisor._owned_pids(rows, child.pid)
            self.assertGreaterEqual(len(owned), 2)
            rss = sum(rows[pid].rss_bytes for pid in owned)
            self.assertGreater(rss, 0)
        finally:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()

    def test_residual_descendant_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            code = (
                "import subprocess,sys\n"
                "from pathlib import Path\n"
                "out=Path(sys.argv[1]); out.mkdir(parents=True)\n"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])\n"
                "(out/'grandchild.pid').write_text(str(child.pid))\n"
            )
            with mock.patch.object(supervisor, "_is_linux", return_value=True):
                record = supervisor.supervise(
                    [sys.executable, "-B", "-c", code, str(output)],
                    output,
                    cwd=root,
                    budget=_budget(),
                )
            grandchild = int((output / "grandchild.pid").read_text(encoding="utf-8"))
            deadline = time.monotonic() + 10
            alive = True
            while time.monotonic() < deadline:
                try:
                    os.kill(grandchild, 0)
                except OSError:
                    alive = False
                    break
                time.sleep(0.05)
            self.assertFalse(alive, "residual owned descendant survived the worker")
            self.assertEqual(record["status"], "failed")

    def test_owned_group_covers_shell_children(self):
        """A worker that spawns through a shell still stays inside the group."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            code = (
                "import subprocess,sys,time\n"
                "from pathlib import Path\n"
                "out=Path(sys.argv[1]); out.mkdir(parents=True)\n"
                "subprocess.Popen(['/bin/sh','-c','sleep 120 & echo $! > "
                + str(output / "shell.pid")
                + "'])\n"
                "time.sleep(60)\n"
            )
            with mock.patch.object(supervisor, "_is_linux", return_value=True):
                record = supervisor.supervise(
                    [sys.executable, "-B", "-c", code, str(output)],
                    output,
                    cwd=root,
                    budget=_budget(max_wall_seconds=1.0),
                )
            self.assertEqual(record["failure"]["type"], "wall_limit")
            self.assertEqual(record["cleanup"]["residual_pids"], [])
            self.assertTrue(record["cleanup"]["verified"])

    def test_non_atomic_output_scan_survives_concurrent_rename(self):
        """A worker that renames files while the supervisor scans must not fail."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            code = (
                "import json,os,sys,time\n"
                "from pathlib import Path\n"
                "out=Path(sys.argv[1]); out.mkdir(parents=True)\n"
                "for index in range(200):\n"
                "    (out/f'a{index}.bin').write_bytes(b'x'*512)\n"
                "    os.replace(out/f'a{index}.bin', out/f'b{index}.bin')\n"
                "time.sleep(60)\n"
            )
            with mock.patch.object(supervisor, "_is_linux", return_value=True):
                record = supervisor.supervise(
                    [sys.executable, "-B", "-c", code, str(output)],
                    output,
                    cwd=root,
                    budget=_budget(max_wall_seconds=1.0),
                )
            self.assertEqual(record["failure"]["type"], "wall_limit")
            self.assertNotEqual(record["failure"]["type"], "monitoring_error")

    def test_setsid_descendant_is_terminated_and_not_reported_clean(self):
        """A descendant that leaves the process group must still be cleaned up."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            code = (
                "import os,sys,time\n"
                "from pathlib import Path\n"
                "out=Path(sys.argv[1]); out.mkdir(parents=True)\n"
                "child=os.fork()\n"
                "if child==0:\n"
                "    os.setsid()\n"
                "    time.sleep(300)\n"
                "    os._exit(0)\n"
                "(out/'escaped.pid').write_text(str(child))\n"
                "time.sleep(300)\n"
            )
            escaped = None
            try:
                with mock.patch.object(supervisor, "_is_linux", return_value=True):
                    record = supervisor.supervise(
                        [sys.executable, "-B", "-c", code, str(output)],
                        output,
                        cwd=root,
                        budget=_budget(max_wall_seconds=1.0),
                    )
                escaped = int((output / "escaped.pid").read_text(encoding="utf-8"))
                self.assertFalse(
                    _process_is_live(escaped),
                    "the escaped descendant survived the cleanup",
                )
            finally:
                if escaped is not None:
                    _kill_quietly(escaped)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["cleanup"]["residual_pids"], [])


if __name__ == "__main__":
    unittest.main()
