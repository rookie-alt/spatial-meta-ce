#!/usr/bin/env python3
"""Bounded process-group supervisor for the fixed Meta65 source-only runner.

Formal CLI execution always uses the authenticated config's exact 7200 seconds.
Short budgets exist only in the fake-child unit-test helper, not as CLI options.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_meta65_screen as runner  # noqa: E402


def artifact_paths(output):
    output = Path(output)
    return (output.with_name(output.name + '.supervisor.json'),
            output.with_name(output.name + '.supervisor.log'))


def _stop_group(child, grace, signals):
    """The session leader PID is the only process group we may signal."""
    def send(sig):
        try:
            os.killpg(child.pid, sig)
            signals.append(int(sig))
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also kill any descendants left in the owned group after its leader exits.
    # Do not probe killpg(..., 0): zombie/group observations are platform-specific.
    send(signal.SIGKILL)
    child.wait(timeout=1)


def _invalidate_decision(output, record):
    """Preserve failed-child evidence without leaving the normal decision path."""
    record['decision_valid'] = False
    if not output.exists():
        return
    runner.no_symlinks(output)
    decision = output / 'decision.json'
    if decision.exists() or decision.is_symlink():
        runner.no_symlinks(decision)
        invalid = output / 'decision.invalid.json'
        if invalid.exists() or invalid.is_symlink():
            # Keep both historical files; never overwrite previous invalid evidence.
            invalid = output / f'decision.invalid.{time.time_ns()}.json'
        runner.no_symlinks(invalid)
        decision.rename(invalid)
        record['invalid_decision_file'] = str(invalid)
        record['invalid_decision_sha256'] = runner.file_hash(invalid)
    status_path = output / 'status.json'
    if status_path.exists() or status_path.is_symlink():
        runner.no_symlinks(status_path)
        runner.no_symlinks(status_path.with_suffix(status_path.suffix + '.tmp'))
        status = runner.read_json(status_path)
        record['child_status_before_failure'] = status.get('status')
        if isinstance(status.get('arms'), dict):
            status['arms'] = {arm: 'failed' if state == 'running' else state
                              for arm, state in status['arms'].items()}
        reason = ('supervisor_timeout' if record['timed_out'] else
                  'child_exit' if record.get('failure_reason') == 'child_exit' else 'supervisor_failure')
        status.update(status='failed', failure_reason=reason, decision_valid=False,
                      supervisor_record=str(artifact_paths(output)[0]))
        runner.write_json(status_path, status)


def supervise(command, output, data, *, timeout, grace=5, root=ROOT):
    """Supervise one owned session; helper accepts short fake-child test budgets."""
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0
            or isinstance(grace, bool) or not isinstance(grace, (int, float))
            or not math.isfinite(grace) or not 0 <= grace <= 5):
        raise ValueError('Invalid supervisor timeout or termination grace')
    output = runner.validate_output(output, data, root=root)
    record_path, log_path = artifact_paths(output)
    for path in (record_path, log_path, record_path.with_suffix(record_path.suffix + '.tmp')):
        runner.no_symlinks(path)
        if path.exists():
            raise FileExistsError(f'Supervisor evidence already exists: {path}')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    record = {'schema': 1, 'status': 'running', 'command': command,
              'output': str(output), 'timeout_seconds': timeout, 'grace_seconds': grace,
              'pid': None, 'process_group': None, 'returncode': None,
              'exit_code': None, 'timed_out': False, 'decision_valid': False,
              'signals_sent': []}
    child = None
    handlers = {}

    def interrupted(signum, _frame):
        record['interrupted_signal'] = signum
        raise KeyboardInterrupt(f'Supervisor interrupted by signal {signum}')

    try:
        with log_path.open('x') as log:
            runner.write_json(record_path, record)
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, interrupted)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=root)
            record.update(pid=child.pid, process_group=child.pid)
            runner.write_json(record_path, record)
            try:
                code = child.wait(timeout=max(0, timeout - (time.monotonic() - started)))
                record.update(status='completed' if code == 0 else 'failed', returncode=code,
                              exit_code=code if code >= 0 else 128 - code,
                              decision_valid=code == 0)
                if code:
                    record['failure_reason'] = 'child_exit'
            except subprocess.TimeoutExpired:
                record.update(status='failed', failure_reason='timeout', timed_out=True,
                              exit_code=124)
    except BaseException as error:  # noqa: BLE001 - Always run bounded owned-group cleanup and record failure on cancellation.
        record.update(status='failed', failure_reason='supervisor_error',
                      error_type=type(error).__name__, error=str(error), exit_code=1)
        if 'interrupted_signal' in record:
            record.update(failure_reason='interrupted', exit_code=128 + record['interrupted_signal'])
    finally:
        # Ignore repeated cancellation only during bounded cleanup/final recording.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                _stop_group(child, grace, record['signals_sent'])
                record['returncode'] = child.returncode
        except BaseException as error:  # noqa: BLE001 - Record bounded TERM/KILL/reap failures before finalizing parent evidence.
            record.update(status='failed', decision_valid=False, cleanup_error=str(error))
            if record['exit_code'] in (None, 0):
                record['exit_code'] = 1
        finally:
            try:
                if record['status'] != 'completed':
                    _invalidate_decision(output, record)
            except BaseException as error:  # noqa: BLE001 - Preserve the terminal parent failure record if child evidence repair fails.
                record['evidence_error'] = str(error)
            record['elapsed_seconds'] = time.monotonic() - started
            try:
                runner.write_json(record_path, record)
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
    return record['exit_code']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/meta65_screen.json')
    parser.add_argument('--data-config', type=Path, default=ROOT / 'configs/data.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    config_path = runner.no_symlinks(args.config)
    data_path = runner.no_symlinks(args.data_config)
    config = runner.load_config(config_path)
    if type(config['timeout_seconds']) is not int or config['timeout_seconds'] != 7200:
        raise ValueError('Formal Meta65 supervisor requires exactly 7200 seconds')
    data = runner.validate_data(runner.read_json(data_path))
    command = [sys.executable, '-B', str(ROOT / 'scripts/run_meta65_screen.py'),
               '--config', str(config_path), '--data-config', str(data_path),
               '--out', str(args.out.absolute()), '--device', args.device, '--execute']
    if not args.execute:
        print(json.dumps({'purpose': 'dry-run-not-execution', 'command': command,
                          'timeout_seconds': config['timeout_seconds']}, indent=2))
        return 0
    return supervise(command, args.out, data, timeout=config['timeout_seconds'])


if __name__ == '__main__':
    raise SystemExit(main())
