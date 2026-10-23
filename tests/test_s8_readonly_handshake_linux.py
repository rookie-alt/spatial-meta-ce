"""Real Linux /proc + pipe helper integration; never execute the data CLI.

The implementation is a byte-identical temporary copy. Its ROOT is derived from
its own __file__, not patched. The fixture ENTRYPOINT is NOT production entrypoint
source authentication, saved-P access, numerical verification or G1 evidence.
"""

import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/verify_s8_readonly_p.py"
CHILD_TIMEOUT = 3
PARENT_TIMEOUT = 5

# Only this temporary parent/child program calls the copied handshake helper.
# --execute authenticates a real parent cmdline; it NEVER dispatches impl.main.
FIXTURE = r"""
import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--execute', action='store_true')
parser.add_argument('--contract', type=Path, required=True)
parser.add_argument('--contract-sha256', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--scenario', required=True)
parser.add_argument('--source-sha256', required=True)
parser.add_argument('--_worker', action='store_true')
parser.add_argument('--_worker-fd', type=int)
parser.add_argument('--_parent-write-fd', type=int)
args = parser.parse_args()

copy_path = Path(__file__).with_name('_handshake_impl.py')
source_sha = hashlib.sha256(copy_path.read_bytes()).hexdigest()
if source_sha != args.source_sha256:
    raise RuntimeError('Fixture implementation is not the byte-bound source copy')
import _handshake_impl as impl

payload = (impl.ROOT / impl.CONTRACT_RELATIVE).read_bytes()
manifest_sha = hashlib.sha256(payload).hexdigest()
contract = json.loads(payload)
output = Path(contract['expected_output_root'])


def numeric_imports():
    return sorted(name for name in sys.modules
                  if name.split('.')[0] in ('numpy', 'torch', 'spatial_meta_ce'))


if args._worker:
    pid = os.getppid()
    command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    child_pipe = os.readlink(f'/proc/self/fd/{args._worker_fd}')
    parent_pipe = os.readlink(f'/proc/{pid}/fd/{args._parent_write_fd}')
    record = {
        'parent_pid': pid, 'child_pid': os.getpid(),
        'parent_cmdline': command,
        'parent_cwd': os.readlink(f'/proc/{pid}/cwd'),
        'child_cwd': os.readlink('/proc/self/cwd'),
        'executable': sys.executable, 'entrypoint': impl.ENTRYPOINT,
        'root': str(impl.ROOT), 'source_sha256': source_sha,
        'child_pipe': child_pipe, 'parent_pipe': parent_pipe,
        'is_fifo': stat.S_ISFIFO(os.fstat(args._worker_fd).st_mode),
        'contract': str(args.contract), 'manifest_sha256': manifest_sha,
        'output': str(args.output),
    }
    try:
        impl.verify_worker_handshake(args, contract, manifest_sha)
    except (ValueError, OSError) as error:
        record.update(accepted=False, error_type=type(error).__name__, error=str(error))
    else:
        record.update(accepted=True, error_type=None, error=None)
    finally:
        # Early rejection can precede the helper's own close(fd).
        try:
            os.close(args._worker_fd)
        except OSError:
            pass
    record['numeric_imports'] = numeric_imports()
    print(json.dumps(record), flush=True)
    raise SystemExit(0 if record['accepted'] else 2)

# The parent keeps the actual writer open while its one owned child checks /proc.
read_fd, write_fd = os.pipe()
fds = {read_fd, write_fd}
parent_write_fd = write_fd
if args.scenario == 'pipe_owner':
    unused_read, parent_write_fd = os.pipe()
    fds.update((unused_read, parent_write_fd))
elif args.scenario == 'non_pipe':
    os.close(read_fd)
    fds.remove(read_fd)
    read_fd = os.open(impl.ROOT / impl.CONTRACT_RELATIVE, os.O_RDONLY)
    fds.add(read_fd)

ticket = {'parent_pid': os.getpid(), 'contract_sha256': manifest_sha,
          'output': str(output), 'package': str(impl.ROOT)}
if args.scenario.startswith('ticket_'):
    field = args.scenario.removeprefix('ticket_')
    ticket[field] = os.getpid() + 1 if field == 'parent_pid' else 'wrong-fixture-binding'
if args.scenario != 'non_pipe':
    os.write(write_fd, json.dumps(ticket).encode())

child_output = impl.ROOT / 'wrong-output' if args.scenario == 'child_output' else output
child_contract = (impl.ROOT / 'other-contract.json' if args.scenario == 'child_contract'
                  else impl.ROOT / impl.CONTRACT_RELATIVE)
command = [sys.executable, '-B', str(impl.ROOT / impl.ENTRYPOINT),
           '--contract', str(child_contract),
           '--contract-sha256', manifest_sha, '--output', str(child_output),
           '--scenario', args.scenario, '--source-sha256', source_sha,
           '--_worker', '--_worker-fd', str(read_fd),
           '--_parent-write-fd', str(parent_write_fd)]
child = None
started = time.monotonic()
try:
    child = subprocess.Popen(command, cwd=impl.ROOT.parent / 'child-cwd',
                             pass_fds=(read_fd,), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    os.close(read_fd)
    fds.remove(read_fd)
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=3)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.kill()
        stdout, stderr = child.communicate(timeout=0.5)
    seconds = time.monotonic() - started
    record = {'child_pid': child.pid, 'child_returncode': child.returncode,
              'child_reaped': child.poll() is not None, 'child_seconds': seconds,
              'timed_out': timed_out, 'stderr': stderr,
              'child': json.loads(stdout) if stdout.strip() else None,
              'ticket_sent': ticket, 'numeric_imports': numeric_imports()}
    print(json.dumps(record), flush=True)
finally:
    if child is not None and child.poll() is None:
        child.kill()
        child.wait(timeout=0.5)
    for fd in fds:
        os.close(fd)
"""


def source_constant(source, name):
    """Read only fixed path literals; never import or modify production globals."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(
        f"Production source must define {name}; no legacy-path fallback"
    )


@unittest.skipUnless(
    sys.platform == "linux", "requires real Linux /proc and inherited pipes"
)
class LinuxReadonlyHandshakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="s8-linux-handshake-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.package = self.base / "package"
        self.package.mkdir()
        (self.base / "child-cwd").mkdir()
        self.source = SOURCE.read_bytes()
        self.source_sha = hashlib.sha256(self.source).hexdigest()
        self.entrypoint = source_constant(self.source, "ENTRYPOINT")
        self.contract_relative = source_constant(self.source, "CONTRACT_RELATIVE")
        entry = self.package / self.entrypoint
        entry.parent.mkdir(parents=True)
        (entry.parent / "__init__.py").write_text("")
        self.copy = entry.with_name("_handshake_impl.py")
        self.copy.write_bytes(self.source)
        entry.write_text(FIXTURE)
        self.output = self.package / "new-proof"
        self.output.mkdir()
        (self.package / "wrong-output").mkdir()
        # Deliberately a minimal helper contract, not a validated production manifest.
        self.contract = self.package / self.contract_relative
        self.contract.parent.mkdir(parents=True)
        self.payload = json.dumps(
            {
                "schema": 1,
                "entrypoint": self.entrypoint,
                "evidence_kind": "synthetic_fixture",
                "expected_package_root": str(self.package),
                "expected_output_root": str(self.output),
            },
            sort_keys=True,
        ).encode()
        self.contract.write_bytes(self.payload)
        self.manifest_sha = hashlib.sha256(self.payload).hexdigest()

    def run_handshake(self, scenario="relative"):
        parent_cwd = self.package
        parent_entry = self.entrypoint
        parent_contract = self.contract_relative
        parent_sha = self.manifest_sha
        parent_output = self.output
        if scenario == "absolute":
            parent_contract = str(self.contract)
        elif scenario == "wrong_parent_cwd":
            parent_cwd = self.base / "wrong-parent-cwd"
            parent_cwd.mkdir()
            parent_entry = str(self.package / self.entrypoint)
            shadow = parent_cwd / self.contract_relative
            shadow.parent.mkdir(parents=True)
            shadow.write_bytes(self.payload)
        elif scenario in ("contract_path", "child_contract"):
            other = self.package / "other-contract.json"
            other.write_bytes(
                self.payload
            )  # Same bytes/SHA do not excuse wrong identity.
            if scenario == "contract_path":
                parent_contract = str(other)
        elif scenario == "sha":
            parent_sha = "0" * 64
        elif scenario == "parent_output":
            parent_output = self.package / "wrong-output"
        elif scenario == "parent_identity":
            fake = self.package / "scripts/fake_parent.py"
            fake.write_text(FIXTURE)
            parent_entry = str(fake)
        if scenario == "stale_proof":
            (self.output / "worker-proof.json").write_text("{}")
        command = [
            sys.executable,
            "-B",
            parent_entry,
            "--execute",
            "--contract",
            parent_contract,
            "--contract-sha256",
            parent_sha,
            "--output",
            str(parent_output),
            "--scenario",
            scenario,
            "--source-sha256",
            self.source_sha,
        ]
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "LD_PRELOAD")
        }
        env.update(
            CUDA_VISIBLE_DEVICES="",
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        parent = subprocess.Popen(
            command,
            cwd=parent_cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = parent.communicate(timeout=PARENT_TIMEOUT)
        finally:
            # Only this fixture's process group, never unrelated processes.
            try:
                os.killpg(parent.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if parent.poll() is None:
                parent.communicate(timeout=1)
        self.assertEqual(parent.returncode, 0, stderr)
        result = json.loads(stdout)
        self.assertFalse(result["timed_out"])
        self.assertTrue(result["child_reaped"])
        self.assertLessEqual(result["child_seconds"], CHILD_TIMEOUT + 0.5)
        # The grandchild was reaped by its own parent, not by this unittest.
        with self.assertRaises(ChildProcessError):
            os.waitpid(result["child_pid"], os.WNOHANG)
        self.assertFalse(Path(f"/proc/{result['child_pid']}").exists())
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["numeric_imports"], [])
        self.assertEqual(self.copy.read_bytes(), self.source)
        self.assertEqual(
            hashlib.sha256(SOURCE.read_bytes()).hexdigest(), self.source_sha
        )
        child = result["child"]
        self.assertIsNotNone(child)
        self.assertEqual(child["source_sha256"], self.source_sha)
        self.assertEqual(child["numeric_imports"], [])
        self.assertEqual(child["root"], str(self.package))
        self.assertEqual(child["parent_pid"], parent.pid)
        self.assertEqual(child["child_pid"], result["child_pid"])
        self.assertEqual(child["child_cwd"], str(self.base / "child-cwd"))
        self.assertEqual(child["parent_cwd"], str(parent_cwd))
        self.assertNotEqual(child["child_cwd"], child["parent_cwd"])
        self.assertEqual(child["executable"], sys.executable)
        self.assertEqual(child["parent_cmdline"][:3], command[:3])
        self.assertEqual(child["parent_cmdline"][1], "-B")
        self.assertIn("--execute", child["parent_cmdline"])
        self.assertNotIn("--_worker", child["parent_cmdline"])
        return result

    def assert_rejected(self, scenario, error):
        result = self.run_handshake(scenario)
        self.assertEqual(result["child_returncode"], 2)
        self.assertFalse(result["child"]["accepted"])
        self.assertEqual(result["child"]["error_type"], "ValueError")
        self.assertRegex(result["child"]["error"], error)
        return result

    def assert_accepted(self, scenario):
        result = self.run_handshake(scenario)
        self.assertEqual(result["child_returncode"], 0)
        child = result["child"]
        self.assertTrue(child["accepted"])
        self.assertIsNone(child["error"])
        self.assertEqual(child["entrypoint"], self.entrypoint)
        self.assertEqual(child["contract"], str(self.contract))
        self.assertEqual(child["manifest_sha256"], self.manifest_sha)
        self.assertEqual(child["output"], str(self.output))
        self.assertTrue(child["is_fifo"])
        self.assertRegex(child["child_pipe"], r"^pipe:\[\d+\]$")
        self.assertEqual(child["child_pipe"], child["parent_pipe"])
        self.assertEqual(
            result["ticket_sent"],
            {
                "parent_pid": child["parent_pid"],
                "contract_sha256": self.manifest_sha,
                "output": str(self.output),
                "package": str(self.package),
            },
        )

    def test_relative_contract_uses_parent_not_child_cwd(self):
        self.assert_accepted("relative")

    def test_absolute_contract_with_different_child_cwd(self):
        self.assert_accepted("absolute")

    def test_wrong_parent_cwd_rejected_with_same_bytes_shadow_contract(self):
        self.assert_rejected(
            "wrong_parent_cwd", "Supervisor arguments do not bind this worker"
        )

    def test_different_contract_path_rejected_even_with_same_sha(self):
        self.assert_rejected(
            "contract_path", "Supervisor arguments do not bind this worker"
        )

    def test_wrong_child_contract_path_rejected(self):
        self.assert_rejected(
            "child_contract", "Worker contract must be the frozen package contract"
        )

    def test_existing_worker_proof_rejected(self):
        self.assert_rejected(
            "stale_proof", "Worker requires fresh supervisor-created output"
        )

    def test_wrong_parent_manifest_sha_rejected(self):
        self.assert_rejected("sha", "Supervisor arguments do not bind this worker")

    def test_wrong_parent_output_rejected(self):
        self.assert_rejected(
            "parent_output", "Supervisor arguments do not bind this worker"
        )

    def test_wrong_child_output_rejected(self):
        self.assert_rejected(
            "child_output", "Worker output must be the frozen new proof directory"
        )

    def test_fake_parent_entrypoint_rejected(self):
        self.assert_rejected(
            "parent_identity", "Worker parent is not the frozen execution supervisor"
        )

    def test_pipe_owned_by_different_parent_fd_rejected(self):
        result = self.assert_rejected(
            "pipe_owner", "Worker pipe is not owned by the verified supervisor"
        )
        self.assertTrue(result["child"]["is_fifo"])
        self.assertNotEqual(
            result["child"]["child_pipe"], result["child"]["parent_pipe"]
        )

    def test_regular_file_cannot_replace_pipe(self):
        result = self.assert_rejected("non_pipe", "Worker handshake requires a pipe")
        self.assertFalse(result["child"]["is_fifo"])

    def test_each_ticket_binding_rejected(self):
        for field in ("parent_pid", "contract_sha256", "output", "package"):
            with self.subTest(field=field):
                self.assert_rejected(
                    f"ticket_{field}", "Inherited worker capability binding mismatch"
                )


if __name__ == "__main__":
    unittest.main()
