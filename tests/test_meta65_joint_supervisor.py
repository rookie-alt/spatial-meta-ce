"""CPU-only synthetic contract tests; no model, cache, SSH, or GPU access."""
import json
import os
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
    'total_seconds': 3,
    'phase_seconds': {'preparation': 1, 'evaluation': 1, 'certification': 1},
    'poll_seconds': 0.01,
}


@unittest.skipUnless(os.name == 'posix', 'requires POSIX owned process groups')
class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'worker'

    def tearDown(self):
        self.temp.cleanup()

    def command(self, body='', *, mode='cpu_preflight', cells=0, phase='certification'):
        code = (
            'import json, os, signal, subprocess, sys, time\n'
            'from pathlib import Path\n'
            'out=Path(sys.argv[1])\n'
            'assert not out.exists(), "supervisor must not precreate output"\n'
            'out.mkdir()\n'
            f'(out/"phase.json").write_text(json.dumps({{"phase": {phase!r}}}))\n'
            f'status={{"status":"worker_complete","mode":{mode!r},"cells":{cells!r},'
            '"micro_cells":9}\n'
            + body + '\n'
            '(out/"status.json").write_text(json.dumps(status))\n'
        )
        return [sys.executable, '-B', '-c', code, str(self.out)]

    def run_worker(self, body='', *, limits=None, sample_fn=None, **kwargs):
        return s.supervise(
            self.command(body, **kwargs), self.out,
            mode=kwargs.get('mode', 'cpu_preflight'),
            limits={**SMALL, **(limits or {})}, sample_fn=sample_fn, grace=0.08,
            root=self.root,
        )

    def assert_failed(self, record, reason):
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure'], reason)
        status = json.loads((self.out / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertNotIn('completed', status.values())
        sidecar = json.loads(s.artifact_paths(self.out)[0].read_text())
        self.assertEqual(sidecar, record)

    def test_cpu_success_owns_session_uses_empty_cuda_and_sidecar_log(self):
        body = (
            'assert os.environ["CUDA_VISIBLE_DEVICES"] == ""\n'
            'assert os.environ["OMP_NUM_THREADS"] == "1"\n'
            'assert Path.cwd().resolve() == out.parent.resolve()\n'
            'assert os.getpgrp() == os.getpid()\n'
            'print("synthetic-cpu", flush=True)\n'
        )
        with patch.object(s, '_gpu_bytes', side_effect=AssertionError('CPU queried GPU')):
            record = self.run_worker(body)
        self.assertEqual(record['status'], 'preflight_completed')
        self.assertIsNone(record['failure'])
        self.assertEqual(record['returncode'], 0)
        self.assertEqual(record['pid'], record['pgid'])
        self.assertGreater(record['peak_rss_bytes'], 0)
        self.assertEqual(record['peak_vram_bytes'], 0)
        self.assertEqual(record['phase'], 'certification')
        self.assertEqual(record['outputbytes'], s.output_bytes(self.out, s.artifact_paths(self.out)[1]))
        self.assertIn('synthetic-cpu', s.artifact_paths(self.out)[1].read_text())
        self.assertEqual(json.loads((self.out / 'status.json').read_text())['status'],
                         'preflight_completed')

    def test_formal_mock_sample_requires_exact_135_cells(self):
        record = self.run_worker(mode='formal', cells=135,
                                 sample_fn=lambda child: {'rss_bytes': 1, 'vram_bytes': 1})
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['peak_vram_bytes'], 1)

    def test_completion_requires_worker_marker_matching_mode_and_cells(self):
        cases = (
            ('status["status"]="completed"', 'cpu_preflight', 0),
            ('status["mode"]="formal"', 'cpu_preflight', 0),
            ('status.pop("cells")', 'cpu_preflight', 0),
            ('', 'cpu_preflight', 9),
            ('', 'formal', 134),
            ('', 'cpu_preflight', False),
        )
        for index, (body, mode, cells) in enumerate(cases):
            with self.subTest(index=index):
                self.out = self.root / f'worker-{index}'
                record = self.run_worker(body, mode=mode, cells=cells,
                                         sample_fn=lambda c: {'rss_bytes': 1, 'vram_bytes': 0})
                self.assert_failed(record, 'invalid_worker_status')

    def test_zero_exit_without_status_is_failure(self):
        record = self.run_worker('sys.exit(0)')
        self.assert_failed(record, 'invalid_worker_status')

    def test_nonzero_invalidates_child_completed_and_keeps_partial(self):
        body = (
            '(out/"partial.bin").write_bytes(b"partial")\n'
            '(out/"status.json").write_text(json.dumps({**status,"status":"completed"}))\n'
            'sys.exit(7)'
        )
        record = self.run_worker(body)
        self.assert_failed(record, 'child_exit')
        self.assertEqual(record['returncode'], 7)
        self.assertEqual((self.out / 'partial.bin').read_bytes(), b'partial')

    def test_real_timeout_reaps_ignoring_term_worker(self):
        record = self.run_worker(
            'signal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(20)',
            limits={'total_seconds': 0.25},
        )
        self.assert_failed(record, 'total_timeout')
        self.assertIn(int(signal.SIGTERM), record['signals'])
        self.assertIn(int(signal.SIGKILL), record['signals'])
        self.assertEqual(record['returncode'], -signal.SIGKILL)
        with self.assertRaises(ProcessLookupError):
            os.kill(record['pid'], 0)

    def test_each_external_phase_timeout_without_worker_deadline(self):
        for phase in ('preparation', 'evaluation', 'certification'):
            with self.subTest(phase=phase):
                self.out = self.root / phase
                record = self.run_worker('time.sleep(20)', phase=phase,
                                         limits={'phase_seconds': {phase: 0.12}})
                self.assert_failed(record, 'phase_timeout')
                self.assertEqual(record['phase'], phase)

    def test_resource_budgets_stop_owned_child(self):
        for resource in ('rss_bytes', 'vram_bytes'):
            with self.subTest(resource=resource):
                self.out = self.root / resource

                def sample(child):
                    deadline = time.monotonic() + 1
                    while not (self.out / 'phase.json').exists() and time.monotonic() < deadline:
                        time.sleep(0.002)
                    return {'rss_bytes': 50, 'vram_bytes': 50}

                record = self.run_worker('time.sleep(20)', mode='formal', cells=135,
                                         limits={resource: 10}, sample_fn=sample)
                self.assert_failed(record, resource.replace('_bytes', '_limit'))

    def test_output_budget_includes_sidecar_log(self):
        record = self.run_worker('print("x"*10000, flush=True)\ntime.sleep(20)',
                                 limits={'output_bytes': 500})
        self.assert_failed(record, 'output_limit')
        self.assertGreater(record['outputbytes'], 500)

    def test_output_budget_includes_worker_files(self):
        record = self.run_worker('(out/"large").write_bytes(b"x"*10000)\ntime.sleep(20)',
                                 limits={'output_bytes': 500})
        self.assert_failed(record, 'output_limit')
        self.assertTrue((self.out / 'large').exists())

    def test_sample_exception_or_zero_after_exit_is_poll_wait_confirmed(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                self.out = self.root / f'race-{raises}'

                def vanished(child, raises=raises):
                    child.wait(timeout=1)
                    if raises:
                        raise ProcessLookupError('Linux /proc race')
                    return {'rss_bytes': 0, 'vram_bytes': 0}

                record = self.run_worker(sample_fn=vanished)
                self.assertEqual(record['status'], 'preflight_completed')
                self.assertEqual(record['returncode'], 0)

    def test_exit_confirmation_after_deadline_is_not_success(self):
        def late(child):
            child.wait(timeout=1)
            time.sleep(0.18)
            raise ProcessLookupError('exited but observation was too late')

        record = self.run_worker(sample_fn=late, limits={'total_seconds': 0.12})
        self.assert_failed(record, 'total_timeout')
        self.assertEqual(record['returncode'], 0)

    def test_sampling_exception_with_live_worker_is_failure(self):
        def broken(child):
            deadline = time.monotonic() + 1
            while not (self.out / 'phase.json').exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            raise RuntimeError('sampler unavailable')

        record = self.run_worker('time.sleep(20)', sample_fn=broken)
        self.assert_failed(record, 'sample_error')

    def test_symlink_artifacts_or_output_are_rejected_before_spawn(self):
        target = self.root / 'untouched'
        target.write_text('original')
        for path in (self.out, *s.artifact_paths(self.out)):
            with self.subTest(path=path), patch.object(s.subprocess, 'Popen') as spawn:
                path.symlink_to(target)
                with self.assertRaises((ValueError, FileExistsError)):
                    s.supervise(self.command(), self.out)
                spawn.assert_not_called()
                path.unlink()
        self.assertEqual(target.read_text(), 'original')

    def test_worker_symlink_is_failed_without_following_it(self):
        target = self.root / 'untouched'
        target.write_text('original')
        record = self.run_worker(f'(out/"linked").symlink_to({str(target)!r})\ntime.sleep(20)')
        self.assert_failed(record, 'unsafe_output')
        self.assertEqual(target.read_text(), 'original')

    def test_existing_output_or_evidence_is_never_overwritten(self):
        for path in (self.out, *s.artifact_paths(self.out)):
            with self.subTest(path=path), patch.object(s.subprocess, 'Popen') as spawn:
                path.write_text('preserve')
                with self.assertRaises(FileExistsError):
                    s.supervise(self.command(), self.out)
                spawn.assert_not_called()
                self.assertEqual(path.read_text(), 'preserve')
                path.unlink()

    def test_invalid_phase_and_phase_regression_fail(self):
        for index, body in enumerate((
            '(out/"phase.json").write_text(json.dumps({"phase":"training"}))\ntime.sleep(20)',
            ('time.sleep(.08)\n(out/"phase.json").write_text(json.dumps({"phase":"preparation"}))\n'
             'time.sleep(20)'),
        )):
            with self.subTest(index=index):
                self.out = self.root / f'phase-{index}'
                self.assert_failed(self.run_worker(body), 'invalid_phase')

    def test_parent_sigterm_and_sigint_cleanup_descendant_not_unrelated(self):
        descendant_code = 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'
        body = (
            f'desc=subprocess.Popen([sys.executable,"-c",{descendant_code!r}])\n'
            'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
            '(out/"pids.json").write_text(json.dumps([os.getpid(),desc.pid]))\n'
            'time.sleep(30)'
        )
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'],
                                     start_new_session=True)
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(sig=sig):
                    self.out = self.root / f'signal-{int(sig)}'
                    parent_code = (
                        'from scripts.meta65_joint_supervisor import supervise\n'
                        f'supervise({self.command(body)!r},{str(self.out)!r},'
                        f'limits={SMALL!r},grace=.08)\n'
                    )
                    parent = subprocess.Popen([sys.executable, '-B', '-c', parent_code], cwd=ROOT,
                                              start_new_session=True, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE, text=True)
                    try:
                        deadline = time.monotonic() + 3
                        while not (self.out / 'pids.json').exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue((self.out / 'pids.json').exists())
                        pids = json.loads((self.out / 'pids.json').read_text())
                        parent.send_signal(sig)
                        _, error = parent.communicate(timeout=3)
                        self.assertEqual(parent.returncode, 0, error)
                        record = json.loads(s.artifact_paths(self.out)[0].read_text())
                        self.assert_failed(record, 'parent_signal')
                        self.assertIn(int(sig), record['received_signals'])
                        for pid in pids:
                            self.assertFalse(s.process_is_live(pid), f'owned PID {pid} survived')
                        self.assertIsNone(unrelated.poll())
                    finally:
                        if parent.poll() is None:
                            parent.kill()
                        parent.communicate(timeout=3)
                        if (self.out / 'pids.json').exists():
                            own = json.loads((self.out / 'pids.json').read_text())[0]
                            try:
                                os.killpg(own, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)


class SamplerTests(unittest.TestCase):
    def test_rss_sum_includes_group_and_descendant_tree_not_other_processes(self):
        # ppid, pgid, sid, rss_bytes, state
        rows = {100: (1, 100, 100, 10, 'S'), 101: (100, 100, 100, 20, 'S'),
                102: (101, 102, 102, 30, 'S'), 103: (1, 100, 100, 40, 'S'),
                200: (1, 200, 200, 999, 'S')}
        with patch.object(s, '_process_rows', return_value=rows), \
                patch.object(s, '_gpu_bytes', return_value=0) as gpu:
            sample = s.sample_process(SimpleNamespace(pid=100), mode='cpu_preflight')
        self.assertEqual(sample, {'rss_bytes': 100, 'vram_bytes': 0})
        gpu.assert_not_called()

    def test_formal_gpu_sums_only_owned_pid_rows(self):
        result = SimpleNamespace(stdout='100, 2\n101, 3\n200, 999\n')
        with patch.object(s.subprocess, 'run', return_value=result) as run:
            self.assertEqual(s._gpu_bytes({100, 101}), 5 * 1024 * 1024)
        self.assertIn('nvidia-smi', run.call_args.args[0])

    def test_linux_proc_disappearance_does_not_poison_remaining_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '101').mkdir()  # disappeared between enumeration and stat read
            (root / '102').mkdir()
            fields = ['S', '100', '100', '100'] + ['0'] * 17 + ['3']
            (root / '102' / 'stat').write_text('102 (name with ) spaces) ' + ' '.join(fields))
            rows = s._linux_rows(root)
        self.assertEqual(rows[102][:3], (100, 100, 100))
        self.assertEqual(rows[102][3], 3 * os.sysconf('SC_PAGE_SIZE'))
        self.assertNotIn(101, rows)

    def test_frozen_defaults_and_helper_only_accept_reduced_limits(self):
        self.assertEqual(s.limits_for('formal')['total_seconds'], 4800)
        self.assertEqual(s.limits_for('formal')['phase_seconds'],
                         {'preparation': 300, 'evaluation': 3600, 'certification': 900})
        self.assertEqual(s.limits_for('cpu_preflight')['phase_seconds'],
                         {'preparation': 300, 'evaluation': 120, 'certification': 300})
        self.assertEqual(s.limits_for('cpu_preflight')['total_seconds'], 900)
        self.assertEqual(s.limits_for('cpu_preflight')['vram_bytes'], 0)
        for changes in ({'total_seconds': 4801}, {'poll_seconds': 2},
                        {'phase_seconds': {'evaluation': 3601}}, {'rss_bytes': True},
                        {'poll_seconds': float('nan')}, {'unknown': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                s.limits_for('formal', changes)

    def test_cli_has_no_custom_budget_option(self):
        with self.assertRaises(SystemExit) as caught:
            s.main(['--output', '/tmp/synthetic', '--total-seconds', '999999', '--', 'true'])
        self.assertNotEqual(caught.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
