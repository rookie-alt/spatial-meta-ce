"""Stdlib-only tests: fake children, never datasets, CUDA or model training."""
import contextlib
import copy
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_meta65_supervised as supervisor


class SupervisedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.out = self.root / 'runs' / 'fake'
        self.data = {'data_root': str(self.root / 'price'),
                     'cache_root': str(self.root / 'cache')}

    def command(self, code):
        return [sys.executable, '-S', '-B', '-c', code]

    def run_child(self, code, timeout=2, grace=0.1):
        return supervisor.supervise(self.command(code), self.out, self.data,
                                    timeout=timeout, grace=grace, root=self.root)

    def record(self):
        return json.loads(supervisor.artifact_paths(self.out)[0].read_text())

    def test_exit_zero_records_completed_without_precreating_runner_output(self):
        code = f"from pathlib import Path; assert not Path({str(self.out)!r}).exists(); print('fake-success')"
        self.assertEqual(self.run_child(code), 0)
        result = self.record()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['returncode'], 0)
        self.assertFalse(result['timed_out'])
        self.assertFalse(self.out.exists())
        self.assertIn('fake-success', supervisor.artifact_paths(self.out)[1].read_text())

    def test_nonzero_is_failed_and_preserves_exit_code(self):
        self.assertEqual(self.run_child("print('fake-failure'); raise SystemExit(7)"), 7)
        result = self.record()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_reason'], 'child_exit')
        self.assertEqual(result['returncode'], 7)
        self.assertNotEqual(result['status'], 'running')

    def test_timeout_terms_owned_group_and_returns_124(self):
        start = time.monotonic()
        self.assertEqual(self.run_child('import time; time.sleep(60)', timeout=0.1), 124)
        result = self.record()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_reason'], 'timeout')
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['exit_code'], 124)
        self.assertLess(time.monotonic() - start, 3)
        self.assertIn(signal.SIGTERM, result['signals_sent'])
        with self.assertRaises(ProcessLookupError):
            os.killpg(result['process_group'], 0)

    def test_ignoring_term_forces_kill_without_killing_unrelated_process(self):
        unrelated = subprocess.Popen(self.command('import time; time.sleep(60)'), start_new_session=True)
        self.addCleanup(lambda: unrelated.poll() is None and unrelated.kill())
        try:
            code = 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
            self.assertEqual(self.run_child(code, timeout=0.2), 124)
            self.assertEqual(self.record()['signals_sent'], [signal.SIGTERM, signal.SIGKILL])
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=2)

    def test_spawned_descendant_is_terminated_with_child_group(self):
        code = ("import subprocess,sys,time; "
                "descendant=subprocess.Popen([sys.executable,'-S','-B','-c','import time; time.sleep(60)']); "
                "time.sleep(60)")
        self.assertEqual(self.run_child(code, timeout=0.2), 124)
        result = self.record()
        self.assertEqual(result['status'], 'failed')
        # No independent subprocess group is launched for descendants.
        self.assertEqual(result['process_group'], result['pid'])

    def test_failed_child_invalidates_decision_and_updates_existing_status(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                out = self.root / 'runs' / ('timeout-artifacts' if timeout else 'exit-artifacts')
                decision = '{"status":"promising_single_seed","selected_arm":"fake"}'
                code = (f"from pathlib import Path; import json,time; p=Path({str(out)!r}); "
                        "p.mkdir(); "
                        f"(p/'decision.json').write_text({decision!r}); "
                        "(p/'status.json').write_text(json.dumps({'status':'running',"
                        "'arms':{'A':'completed','B':'running','C':'planned'},'source':'keep'})); "
                        + ('time.sleep(60)' if timeout else 'raise SystemExit(7)'))
                exit_code = supervisor.supervise(self.command(code), out, self.data,
                                                timeout=0.2 if timeout else 2,
                                                grace=0.1, root=self.root)
                self.assertEqual(exit_code, 124 if timeout else 7)
                record = json.loads(supervisor.artifact_paths(out)[0].read_text())
                self.assertFalse(record['decision_valid'])
                self.assertFalse((out / 'decision.json').exists())
                self.assertEqual((out / 'decision.invalid.json').read_text(), decision)
                status = json.loads((out / 'status.json').read_text())
                self.assertEqual(status['status'], 'failed')
                self.assertEqual(status['source'], 'keep')
                self.assertEqual(status['arms'], {'A':'completed','B':'failed','C':'planned'})
                self.assertEqual(status['failure_reason'], 'supervisor_timeout' if timeout else 'child_exit')

    def test_completed_decision_is_also_invalidated_on_nonzero_exit(self):
        code = (f"from pathlib import Path; import json; p=Path({str(self.out)!r}); p.mkdir(); "
                "(p/'decision.json').write_text('{}'); "
                "(p/'status.json').write_text(json.dumps({'status':'completed','arms':{'A':'completed'}})); "
                "raise SystemExit(9)")
        self.assertEqual(self.run_child(code), 9)
        self.assertFalse(self.record()['decision_valid'])
        self.assertEqual(json.loads((self.out / 'status.json').read_text())['status'], 'failed')
        self.assertEqual(self.record()['child_status_before_failure'], 'completed')
        self.assertTrue((self.out / 'decision.invalid.json').exists())

    def test_spawn_failure_leaves_failed_record_not_running(self):
        with patch.object(supervisor.subprocess, 'Popen', side_effect=OSError('fake launch failure')):
            self.assertEqual(self.run_child('pass'), 1)
        self.assertEqual(self.record()['status'], 'failed')
        self.assertEqual(self.record()['failure_reason'], 'supervisor_error')

    def test_existing_output_and_companions_are_rejected_before_spawn(self):
        for kind in ('output', 'record', 'log', 'temporary_record'):
            with self.subTest(kind=kind):
                out = self.root / 'runs' / kind
                record, log = supervisor.artifact_paths(out)
                path = {'output': out, 'record': record, 'log': log,
                        'temporary_record': record.with_suffix(record.suffix + '.tmp')}[kind]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('must stay unchanged')
                with patch.object(supervisor.subprocess, 'Popen') as spawn:
                    with self.assertRaises(FileExistsError):
                        supervisor.supervise(self.command('pass'), out, self.data,
                                             timeout=1, root=self.root)
                    spawn.assert_not_called()
                self.assertEqual(path.read_text(), 'must stay unchanged')

    def test_symlink_ancestor_and_price_cache_outputs_are_rejected(self):
        (self.root / 'real').mkdir()
        (self.root / 'link').symlink_to(self.root / 'real', target_is_directory=True)
        cases = [self.root / 'link' / 'out', self.root / 'cache' / 'out',
                 self.root / 'price' / 'out', Path('/gly/weilin/PRICE-master/runs/out')]
        for out in cases:
            with self.subTest(out=out), patch.object(supervisor.subprocess, 'Popen') as spawn:
                with self.assertRaises(ValueError):
                    supervisor.supervise(self.command('pass'), out, self.data, timeout=1, root=self.root)
                spawn.assert_not_called()

    def test_symlink_companion_is_rejected(self):
        self.out.parent.mkdir(parents=True)
        target = self.root / 'keep'
        target.write_text('keep')
        supervisor.artifact_paths(self.out)[0].symlink_to(target)
        with patch.object(supervisor.subprocess, 'Popen') as spawn:
            with self.assertRaises(ValueError):
                self.run_child('pass')
            spawn.assert_not_called()
        self.assertEqual(target.read_text(), 'keep')

    def test_start_new_session_and_direct_argument_vector(self):
        popen = subprocess.Popen
        with patch.object(supervisor.subprocess, 'Popen', wraps=popen) as spawn:
            self.assertEqual(self.run_child('pass'), 0)
        self.assertIs(spawn.call_args.kwargs['start_new_session'], True)
        self.assertNotIn('shell', spawn.call_args.kwargs)

    def test_invalid_helper_limits_are_rejected(self):
        for timeout, grace in ((0, 1), (-1, 1), (float('nan'), 1), (1, 5.1), (1, -1)):
            with self.subTest(timeout=timeout, grace=grace), self.assertRaises(ValueError):
                supervisor.supervise(self.command('pass'), self.out, self.data,
                                     timeout=timeout, grace=grace, root=self.root)


class CliTests(unittest.TestCase):
    def invoke(self, *extra):
        return supervisor.main(['--out', '/tmp/not-a-real-run', *extra])

    def test_cli_uses_only_fixed_runner_and_frozen_deadline(self):
        config = copy.deepcopy(supervisor.runner.FROZEN)
        with patch.object(supervisor.runner, 'load_config', return_value=config), \
                patch.object(supervisor.runner, 'validate_data', return_value={}), \
                patch.object(supervisor, 'supervise', return_value=0) as supervise:
            self.assertEqual(self.invoke('--device', 'cuda', '--execute'), 0)
        command = supervise.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, '-B', str(supervisor.ROOT / 'scripts/run_meta65_screen.py')])
        self.assertEqual(command[-3:], ['--device', 'cuda', '--execute'])
        self.assertIn('--config', command)
        self.assertIn('--data-config', command)
        self.assertIn('--out', command)
        self.assertTrue(Path(command[command.index('--out') + 1]).is_absolute())
        self.assertEqual(supervise.call_args.kwargs['timeout'], 7200)

    def test_cli_rejects_short_budget_even_if_loader_is_mocked(self):
        config = copy.deepcopy(supervisor.runner.FROZEN)
        config['timeout_seconds'] = 0.1
        with patch.object(supervisor.runner, 'load_config', return_value=config), \
                patch.object(supervisor, 'supervise') as supervise:
            with self.assertRaises(ValueError):
                self.invoke('--execute')
            supervise.assert_not_called()

    def test_cli_does_not_bypass_runner_config_gate(self):
        with patch.object(supervisor.runner, 'load_config', side_effect=ValueError('protocol mismatch')), \
                patch.object(supervisor, 'supervise') as supervise:
            with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                self.invoke('--execute')
            supervise.assert_not_called()

    def test_no_short_budget_or_arbitrary_child_cli_flags(self):
        for flag in ('--timeout', '--steps', '--command'):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.invoke(flag, '1')


if __name__ == '__main__':
    unittest.main()
