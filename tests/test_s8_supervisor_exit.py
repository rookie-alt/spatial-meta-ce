"""Synthetic exit races; real Linux /proc integration is explicitly gated."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import verify_s8_readonly_p as ro


class ExitRaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output = Path(self.tmp.name).resolve()
        (self.output / 'worker-proof.json').write_text('{}')

    def child(self, code=0, persistent=False, never_reaped=False):
        child = Mock(pid=123456, returncode=None)
        child.poll.return_value = None

        def wait(timeout=None):
            if timeout is None:
                raise AssertionError('all child waits must be bounded')
            if never_reaped or (persistent and child.wait.call_count == 1):
                raise subprocess.TimeoutExpired('fixture', timeout)
            child.returncode = code
            child.poll.return_value = code
            return code

        child.wait.side_effect = wait
        return child

    def run_race(self, error, code=0, persistent=False, **kwargs):
        (self.output / 'supervisor.json').unlink(missing_ok=True)
        child = self.child(code, persistent)
        with patch.object(ro, 'sample_process', side_effect=error), patch.object(ro.os, 'killpg') as kill:
            result = ro.supervise(child, self.output, **kwargs)
        return result, child, kill

    def test_exit_between_counter_failure_and_poll_is_confirmed_and_reaped(self):
        for error in (KeyError('VmRSS'), FileNotFoundError('/proc/fixture/status'),
                      ValueError('missing ps counters')):
            with self.subTest(error=type(error).__name__):
                result, child, kill = self.run_race(error)
                self.assertIsNone(result['failure_reason'])
                self.assertEqual(result['returncode'], 0)
                self.assertTrue(result['exit_confirmed_after_observation_failure'])
                self.assertTrue(any(call.kwargs.get('timeout') for call in child.wait.call_args_list))
                self.assertEqual(json.loads((self.output / 'supervisor.json').read_text()), result)
                kill.assert_called_once()  # owned group cleanup, even after leader exit

    def test_confirmed_nonzero_exit_is_not_sampling_success(self):
        result, _, _ = self.run_race(KeyError('VmRSS'), code=7)
        self.assertEqual(result['failure_reason'], 'child_exit')
        self.assertEqual(result['returncode'], 7)

    def test_confirmed_exit_still_requires_worker_proof(self):
        (self.output / 'worker-proof.json').unlink()
        result, _, _ = self.run_race(KeyError('VmRSS'))
        self.assertEqual(result['failure_reason'], 'missing_worker_proof')

    def test_persistent_live_unobservability_fails_and_cleans_group(self):
        result, _, kill = self.run_race(PermissionError('unobservable live process'),
                                      code=-9, persistent=True)
        self.assertEqual(result['failure_reason'], 'resource_observation_failed')
        self.assertFalse(result['exit_confirmed_after_observation_failure'])
        kill.assert_called_once()

    def test_confirmation_timeout_does_not_extend_wall_budget(self):
        result, _, _ = self.run_race(KeyError('VmRSS'), code=-9, persistent=True,
                                    wall_seconds=0.01)
        self.assertEqual(result['failure_reason'], 'wall_timeout')

    def test_exit_confirmation_finishing_after_deadline_still_fails_wall_budget(self):
        child = self.child()
        with (patch.object(ro, 'sample_process', side_effect=KeyError('VmRSS')),
              patch.object(ro.os, 'killpg'),
              patch.object(ro.time, 'monotonic', side_effect=[0.0, 0.0, 0.0, 1.1])):
            result = ro.supervise(child, self.output, wall_seconds=1.0)
        self.assertEqual(result['returncode'], 0)
        self.assertEqual(result['failure_reason'], 'wall_timeout')

    def test_group_disappearing_during_cleanup_is_not_supervisor_failure(self):
        child = self.child()
        with patch.object(ro, 'sample_process', side_effect=KeyError('VmRSS')), patch.object(
            ro.os, 'killpg', side_effect=ProcessLookupError
        ):
            result = ro.supervise(child, self.output)
        self.assertIsNone(result['failure_reason'])

    def test_cleanup_reap_timeout_keeps_failed_proof_and_restores_signals(self):
        child = self.child(persistent=True, never_reaped=True)
        handlers = {}
        def set_signal(number, handler):
            previous = handlers.get(number, 'original')
            handlers[number] = handler
            return previous
        with (patch.object(ro, 'sample_process', side_effect=PermissionError('live blind worker')),
              patch.object(ro.os, 'killpg'), patch.object(ro.signal, 'signal', side_effect=set_signal)):
            result = ro.supervise(child, self.output)
        self.assertEqual(result['failure_reason'], 'resource_observation_failed')
        self.assertEqual(result['cleanup_failure_reason'], 'child_reap_timeout')
        self.assertFalse(result['child_reaped'])
        self.assertIsNone(result['returncode'])
        self.assertIsNone(result['signal'])
        self.assertTrue(all(call.kwargs.get('timeout') is not None for call in child.wait.call_args_list))
        self.assertTrue((self.output / 'supervisor.json').is_file())
        self.assertTrue(all(handler == 'original' for handler in handlers.values()))

    def test_cleanup_signal_error_is_failed_durable_and_does_not_wait_forever(self):
        child = self.child()
        with (patch.object(ro, 'sample_process', side_effect=KeyError('VmRSS')),
              patch.object(ro.os, 'killpg', side_effect=PermissionError('cannot clean owned group')),
              patch.object(ro.signal, 'signal', return_value='original') as signals):
            result = ro.supervise(child, self.output)
        self.assertEqual(result['failure_reason'], 'process_group_cleanup_failed')
        self.assertTrue(result['child_reaped'])
        self.assertEqual(result['owned_group_cleanup'], 'failed')
        self.assertTrue((self.output / 'supervisor.json').is_file())
        self.assertEqual(signals.call_count, 4)
        self.assertTrue(all(call.kwargs.get('timeout') is not None for call in child.wait.call_args_list))

    def test_proc_zombie_and_missing_window_reject_counter_sample(self):
        with patch.object(ro.sys, 'platform', 'linux'):
            for text in ('Name: fixture\nState: Z (zombie)\n', 'Name: fixture\nState: R (running)\n'):
                with (self.subTest(text=text), patch.object(Path, 'read_text', return_value=text),
                      self.assertRaises((KeyError, ValueError))):
                    ro.sample_process(123456)
            with (patch.object(Path, 'read_text', side_effect=FileNotFoundError),
                  self.assertRaises(FileNotFoundError)):
                ro.sample_process(123456)

    @unittest.skipUnless(sys.platform == 'linux', 'real Linux /proc integration not verified on macOS')
    def test_real_linux_quick_exit_zombie_window(self):
        for _ in range(5):
            (self.output / 'supervisor.json').unlink(missing_ok=True)
            child = subprocess.Popen([sys.executable, '-B', '-c', 'pass'], start_new_session=True)
            result = ro.supervise(child, self.output)
            self.assertEqual(result['returncode'], 0)
            self.assertIsNone(result['failure_reason'])
            self.assertIsNotNone(child.poll())

    @unittest.skipUnless(sys.platform == 'linux', 'real Linux descendant cleanup not verified on macOS')
    def test_real_linux_exited_leader_group_with_live_descendant_is_cleaned(self):
        import os
        import signal
        import time
        code = "import os,time; pid=os.fork(); time.sleep(10) if pid==0 else print(pid,flush=True)"
        child = subprocess.Popen([sys.executable, '-B', '-c', code], stdout=subprocess.PIPE,
                                 text=True, start_new_session=True)
        descendant = int(child.stdout.readline())
        child.stdout.close()
        self.addCleanup(lambda: self.kill_if_present(descendant, signal.SIGKILL))
        child.wait(timeout=2)  # leader exited; descendant still keeps its group alive
        os.kill(descendant, 0)
        result = ro.supervise(child, self.output)
        self.assertIsNone(result['failure_reason'])
        self.assertTrue(result['child_reaped'])
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            path = Path(f'/proc/{descendant}/status')
            if not path.exists() or 'State:\tZ' in path.read_text():
                break
            time.sleep(0.01)
        else:
            self.fail('owned descendant still running after cleanup')

    @staticmethod
    def kill_if_present(pid, signum):
        import os
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass


if __name__ == '__main__':
    unittest.main()
