#!/usr/bin/env python3
"""Complete the frozen C source endpoints without training or altering the old run."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_meta65_screen as original
from scripts.run_meta65_supervised import supervise

OLD_RUN = 'runs/meta65-source-screen-20261009'
PROTOCOL = '实验文档/50-Meta65仅补齐C端点评估协议.md'
FROZEN = {
    'schema': 1, 'experiment': 'meta65-endpoint-completion-20261009',
    'mode': 'endpoint_completion_only', 'timeout_seconds': 3000,
    'retained_root': '/gly/weilin/spatial-meta-ce-meta65-20261009-frozen',
    'retained_run': '/gly/weilin/spatial-meta-ce-meta65-20261009-frozen/' + OLD_RUN,
    'checkpoint_sha256': '58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4',
    'endpoint_plan_sha256': 'ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0',
    'protocol_file': PROTOCOL,
    'protocol_sha256': '08eaa606701fa56ae427e07202b17d10d6debd6dc1d8a878b2a29d73661321e3',
    'retained_sha256': {
        OLD_RUN + '/status.json': '086485752d84301213de18123004bad729ac2b300a4c14ad2c71d6e0a6360a76',
        OLD_RUN + '/environment.json': '43dcac1056de7242c142e6fce98871ac679386c8343c3ce71592b919ef569322',
        OLD_RUN + '/endpoint_plan.json': 'ebeb178de15b134722bd9fbc24ff4b4752f795a311580c1a17cb23c550bfd0f0',
        OLD_RUN + '/source_before.json': '383e8a6c2336aea881a0fbb485900db460385c202bb353745e07b96d2f8bb650',
        OLD_RUN + '/A_all5_head3/metadata.json': 'c0b443fb91235d6db1bb6eb0222b78f916ee7f6c2b057b705eb5ef3ea0dbdd30',
        OLD_RUN + '/B_head3_head3/metadata.json': '73570adc80b163c0944a6db5d775c4e4c6d34317a164642144c62f6e80364756',
        OLD_RUN + '/A_all5_head3/endpoint.csv': '1910b50755b6d5a8831868c8df486d3154df47fafd107531edc24087800533c6',
        OLD_RUN + '/B_head3_head3/endpoint.csv': 'c80a79c6803fb6ff96c8483400c45bf52f5f281f5d6328fa17cc5360fe71ba71',
        OLD_RUN + '/C_last3_last3/last.pt': '58f2f4db6f54cd4578fd6127d5a4aa350dbdc2268a36a057b56f7161cb6097a4',
        OLD_RUN + '/C_last3_last3/validation_manifest.json': '0a110551da2db90ff619d8461cd06a1a4b62eb82341f6c4219eb7f6436e3ddb2',
        OLD_RUN + '/C_last3_last3/history.json': '4f0906b84c2895bfc37b527e1b8f72816fc6732136ee9294c7b865108bb139d8',
        OLD_RUN + '.supervisor.json': 'b5e2028abaffabb97b3a8796d1a9445cba913174296267bd6b8d378f1d74ac71',
        'configs/meta65_screen.json': 'ebd5564cb66a8df03f2c78f293c4cacb493b8af1836932ce7a9cceca6f3d3c36',
        'configs/meta65_server_data.json': 'a6907d602ed92fdfe42abe2828bbbac9369be07de04241c17cf6a3bd61544283',
        'reports/meta65-source-screen-20261009/launch-binding.json': 'c5ee7debe39837fdf3b1b84a00b1f8a1141560691b22ff2653ac33be1bed374d',
    },
}
CODE = (*original.CODE_FILES, 'scripts/complete_meta65_endpoint.py',
        'configs/meta65_screen.json', 'configs/meta65_server_data.json',
        'reports/meta65-source-screen-20261009/verify_endpoint.py',
        'reports/meta65-source-screen-20261009/verify_completion.py')


def validate_contract(config):
    if original.digest(config) != original.digest(FROZEN):
        raise ValueError('Frozen endpoint-completion contract changed')
    return config


def load_contract(path):
    config = validate_contract(original.read_json(original.no_symlinks(path)))
    if original.file_hash(ROOT / PROTOCOL) != config['protocol_sha256']:
        raise ValueError('Endpoint-completion protocol changed')
    return config


def check_retained(root, hashes):
    for name, expected in hashes.items():
        path = original.no_symlinks(root / name)
        if not path.is_relative_to(root.resolve()) or original.file_hash(path) != expected:
            raise ValueError('Retained input SHA changed: ' + name)


def preflight_output(config, output, root=ROOT):
    root = original.no_symlinks(root)
    retained = original.no_symlinks(config['retained_root'])
    if root.is_relative_to(retained) or retained.is_relative_to(root):
        raise ValueError('Completion deployment must be isolated from retained deployment')
    for name in ('configs/meta65_screen.json', 'configs/meta65_server_data.json'):
        if original.file_hash(original.no_symlinks(root / name)) != config['retained_sha256'][name]:
            raise ValueError('New deployment differs from frozen data/model config: ' + name)
    data = original.validate_data(original.read_json(root / 'configs/meta65_server_data.json'), root=root)
    output = original.validate_output(output, data, root=root)
    if output.is_relative_to(retained):
        raise ValueError('Completion output must not enter retained deployment')
    return output, data


def final_source_check(data, historical, before, output, deadline):
    from scripts.precision_screen_evidence import source_evidence
    final = source_evidence(data, historical)
    if final != before:
        raise ValueError('Source cache changed during independent verification')
    original.write_json(output / 'source_after.json', final)
    deadline()


def combine_rows(rows):
    if set(rows) != set(original.ARMS) or any(len(rows[a]) != 75 for a in original.ARMS):
        raise ValueError('Exactly 75 complete endpoint cells per arm required')
    return [{**row, 'source_run': 'completion' if arm == original.ARMS[2] else 'retained'}
            for arm in original.ARMS for row in rows[arm]]


def read_rows(path):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ('k', 'run', 'query_count', 'four_plus_count'):
            row[key] = int(row[key])
        for key in ('all_q50', 'all_q95', 'four_plus_q50', 'four_plus_q95',
                    'all_log_q95', 'four_plus_log_q95', 'adaptation_seconds', 'prediction_seconds'):
            row[key] = float(row[key])
    return rows


def record_failure(output, error):
    decision = output / 'decision.json'
    if decision.exists():
        decision.rename(output / 'decision.invalid.json')
    original.write_json(output / 'status.json', {'status': 'failed',
                        'experiment': FROZEN['experiment'], 'error_type': type(error).__name__,
                        'error': str(error), 'decision_valid': False})


def checkpoint_record(checkpoint, original_config, old_run, reference, plan):
    config = original.arm_config(original_config, original.ARMS[2])
    metadata, extra = checkpoint['metadata'], checkpoint['extra']
    counts = extra['counts']
    signature = {k: v for k, v in config.items() if k not in ('steps', 'data_config')}
    if (checkpoint['next_step'] != 4000 or checkpoint['model_config'] != config['model']
            or metadata['training_signature'] != signature
            or counts != {'support_presentations': 1793745, 'query_presentations': 1024000,
                          'backward_calls': 64000, 'optimizer_steps': 4000}
            or extra['sample_plan_episodes'] != 16000
            or extra['sample_plan_digest'] != reference['sample_plan_digest']
            or metadata['manifest_hashes'] != reference['metadata']['manifest_hashes']):
        raise ValueError('Retained C checkpoint training/source/config mismatch')
    validation_sha = original.file_hash(old_run / original.ARMS[2] / 'validation_manifest.json')
    if validation_sha != reference['validation_plan_digest']:
        raise ValueError('Retained C validation plan mismatch')
    return {'arm': original.ARMS[2], 'config': config, 'metadata': metadata,
            'counts': counts, 'parameters': 132545, 'optimizer_steps': 4000,
            'initialization_digest': reference['initialization_digest'],
            'initialization_digest_source': 'authenticated_shared_A_training_record',
            'sample_plan_digest': extra['sample_plan_digest'], 'sample_plan_episodes': 16000,
            'support_sample_count': 597915, 'validation_plan_digest': validation_sha,
            'endpoint_plan_digest': original.digest(plan), 'last_step': 4000,
            'checkpoint': 'retained_C_last.pt', 'checkpoint_sha256': FROZEN['checkpoint_sha256'],
            'training_seconds_from_checkpoint': extra['total_seconds'],
            'new_training_updates': 0}


def work(config, output, device, config_path):
    from scripts.mapping_smoke import state_hash
    from scripts.precision_screen_evidence import historical_sources, source_evidence
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.data import FeatureStore

    retained_root = original.no_symlinks(config['retained_root'])
    old_run = original.no_symlinks(config['retained_run'])
    output, data = preflight_output(config, output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def deadline():
        original.check_deadline(started, config['timeout_seconds'])

    def status(phase, **details):
        original.write_json(output / 'status.json', {'status': 'running', 'phase': phase,
                            'experiment': config['experiment'], 'elapsed_seconds': time.monotonic() - started,
                            'decision_valid': False, **details})
        original.write_json(output / 'stage.json', {'phase': phase, **details})

    status('preflight')
    original.write_json(output / 'inputs.json', config)
    before = None
    try:
        check_retained(retained_root, config['retained_sha256'])
        old_status = original.read_json(old_run / 'status.json')
        old_supervisor = original.read_json(Path(str(old_run) + '.supervisor.json'))
        if (old_status['status'] != 'failed' or old_supervisor['exit_code'] != 124
                or not old_supervisor['timed_out']):
            raise ValueError('Expected retained failed-timeout run')
        original_config = original.load_config(ROOT / 'configs/meta65_screen.json')
        old_environment = original.read_json(old_run / 'environment.json')
        for name, expected in old_environment['code_sha256'].items():
            if original.file_hash(ROOT / name) != expected:
                raise ValueError('Frozen original code changed: ' + name)
        runtime = original.environment(device)
        runtime.update(code_sha256={p: original.file_hash(ROOT / p) for p in CODE},
                       config_sha256=original.file_hash(config_path), protocol_sha256=config['protocol_sha256'],
                       retained_inputs_sha256=config['retained_sha256'], new_training_updates=0,
                       external_timeout_required_seconds=config['timeout_seconds'])
        original.write_json(output / 'environment.json', runtime)
        historical = historical_sources(ROOT, list(original.TRAIN + original.VALIDATION))
        before = source_evidence(data, historical)
        if before != original.read_json(old_run / 'source_before.json'):
            raise ValueError('Source caches differ from retained training inputs')
        original.write_json(output / 'source_before.json', before)
        plan = original.read_json(old_run / 'endpoint_plan.json')
        original.write_json(output / 'endpoint_plan.json', plan)
        stores, tables = {}, {}
        for db in original.VALIDATION:
            deadline()
            stores[db] = FeatureStore(Path(data['cache_root']) / 'pretrain' / db)
            tables[db] = original.validate_store(stores[db], db)
        model, checkpoint = load_checkpoint(old_run / original.ARMS[2] / 'last.pt', device)
        reference = original.read_json(old_run / original.ARMS[0] / 'metadata.json')
        record = checkpoint_record(checkpoint, original_config, old_run, reference, plan)
        model_before = state_hash(model)
        arm_out = output / original.ARMS[2]
        arm_out.mkdir()
        status('evaluating_C', endpoint_cells=0)
        begun = time.monotonic()
        c_rows = original.evaluate_arm(model, stores, tables, plan, original_config,
                                       original.ARMS[2], arm_out, device, deadline)
        if state_hash(model) != model_before:
            raise ValueError('Evaluation mutated retained meta-initialization')
        for row in c_rows:
            row['prediction_file'] = f"{original.ARMS[2]}/{row['prediction_file']}"
        record.update(endpoint_seconds=time.monotonic() - begun, endpoint_cells=len(c_rows),
                      model_state_unchanged=True)
        original.write_json(arm_out / 'metadata.json', record)
        original.write_endpoint(arm_out / 'endpoint.csv', c_rows)
        rows = {arm: read_rows(old_run / arm / 'endpoint.csv') for arm in original.ARMS[:2]}
        rows[original.ARMS[2]] = c_rows
        combined = combine_rows(rows)
        original.write_endpoint(output / 'endpoint.csv', combined)
        decision = original.decide(combined)
        original.write_json(output / 'decision.provisional.json', decision)
        after = source_evidence(data, historical)
        original.write_json(output / 'source_after.json', after)
        if after != before:
            raise ValueError('Source cache changed during completion')
        check_retained(retained_root, config['retained_sha256'])
        if ({p: original.file_hash(ROOT / p) for p in CODE} != runtime['code_sha256']
                or original.file_hash(config_path) != runtime['config_sha256']
                or original.file_hash(ROOT / PROTOCOL) != config['protocol_sha256']):
            raise ValueError('Completion code/config/protocol changed')
        deadline()
        status('awaiting_independent_verification', endpoint_cells=225)
        command = [sys.executable, '-B', str(ROOT / 'reports/meta65-source-screen-20261009/verify_completion.py'),
                   '--root', str(ROOT), '--completion', str(output), '--out', str(output / 'independent-verification.json')]
        subprocess.run(command, check=True, timeout=min(300, config['timeout_seconds'] - (time.monotonic() - started)))
        deadline()
        verification = original.read_json(output / 'independent-verification.json')
        if verification['verification'] != 'passed' or verification['checked225'] != 225:
            raise ValueError('Independent matrix verification incomplete')
        check_retained(retained_root, config['retained_sha256'])
        if ({p: original.file_hash(ROOT / p) for p in CODE} != runtime['code_sha256']
                or original.file_hash(config_path) != runtime['config_sha256']
                or original.file_hash(ROOT / PROTOCOL) != config['protocol_sha256']):
            raise ValueError('Completion code/config/protocol changed after verification')
        final_source_check(data, historical, before, output, deadline)
        deadline()
        (output / 'decision.provisional.json').rename(output / 'decision.json')
        original.write_json(output / 'status.json', {'status': 'completed', 'experiment': config['experiment'],
                            'elapsed_seconds': time.monotonic() - started, 'decision_valid': True,
                            'endpoint_cells': 225, 'new_training_updates': 0, 'old_formal_status': 'failed_timeout',
                            'independent_verification': 'passed', 'source_before_after_equal': True})
    except BaseException as error:
        # Retain failure evidence before propagating cancellation to the supervisor.
        record_failure(output, error)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/meta65_completion.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/meta65-endpoint-completion-20261009')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = load_contract(args.config)
    if not args.execute:
        print(json.dumps({'purpose': 'dry-run-not-execution', 'mode': config['mode'],
                          'C_cells': 75, 'combined_cells': 225, 'timeout_seconds': 3000,
                          'new_training_updates': 0}, indent=2))
        return 0
    if args.worker:
        work(config, args.out, args.device, args.config)
        return 0
    _, data = preflight_output(config, args.out)
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--config', str(args.config.resolve()),
               '--out', str(args.out.resolve()), '--device', args.device, '--execute', '--worker']
    return supervise(command, args.out, data, timeout=config['timeout_seconds'], root=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
