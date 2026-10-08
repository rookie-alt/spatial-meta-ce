"""Freeze a diagnostic-only saved-P verifier; no gate, threshold or LS changes."""

import gzip
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
ARCHIVE = Path('/private/tmp/s8-g1-readonly-p-diagnostic-package-20261008.tar.gz')
BASE = ROOT / 'reports/s8-g1-readonly-p-handshake-v2-20261008'
OLD_MANIFEST = BASE / 'package-manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def freeze():
    old = json.loads(OLD_MANIFEST.read_text())
    if sha(Path(old['archive'])) != old['archive_sha256']:
        raise ValueError('Prior handshake-v2 archive changed')
    # run_path imports stdlib definitions only; __main__ is not executed.
    module = runpy.run_path(str(ROOT / 'scripts/verify_s8_readonly_p.py'))
    sources = {name: sha(ROOT/name) for name in module['SOURCE_FILES']}
    tests = {'tests/test_s8_readonly_p.py': sha(ROOT/'tests/test_s8_readonly_p.py')}
    contract = json.loads((BASE/'execution-contract.json').read_text())
    contract.update(revision='readonly-P-diagnostic-20261008',
        expected_package_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-diagnostic-code-20261008',
        expected_output_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-diagnostic-proof-20261008',
        prepared_on='2026-10-08', source_sha256=sources,
        diagnostic_only=True,
        diagnostic_purpose='explain_x_small_1216_retained_subspace_failure',
        diagnostic_gate_effect='record_only_no_gate_change',
        prior_failed_A_evidence_sha256={
            'reports/s8-g1-readonly-p-handshake-v2-real-20261008/real-proof/proof.json': sha(ROOT/'reports/s8-g1-readonly-p-handshake-v2-real-20261008/real-proof/proof.json'),
            'reports/s8-g1-readonly-p-handshake-v2-real-20261008/real-proof/x-small_P_zero_reference.json': sha(ROOT/'reports/s8-g1-readonly-p-handshake-v2-real-20261008/real-proof/x-small_P_zero_reference.json'),
            'reports/s8-g1-readonly-p-handshake-v2-real-20261008/independent-record-review.md': sha(ROOT/'reports/s8-g1-readonly-p-handshake-v2-real-20261008/independent-record-review.md')},
        real_P_execution_not_authorized=True,
        not_authorized=['real_P_diagnostic_execution', 'B_smoke', 'formal_training', 'gate_or_threshold_changes'])
    contract['contract_relative_path'] = 'reports/s8-g1-readonly-p-diagnostic-20261008/execution-contract.json'
    module['validate_contract'](contract)
    write('execution-contract.json', contract)
    members = {name: (ROOT/name).read_bytes() for name in (*sources, *tests)}
    members[contract['contract_relative_path']] = (REPORT/'execution-contract.json').read_bytes()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            archive.addfile(info, io.BytesIO(payload))
    with ARCHIVE.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(buffer.getvalue())
    write('package-manifest.json', {
        'status': 'new_diagnostic_package_prepared_not_real_P_executed',
        'revision': 'readonly-P-diagnostic-20261008',
        'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'contract_relative_path': contract['contract_relative_path'],
        'execution_contract_sha256': sha(REPORT/'execution-contract.json'),
        'files_sha256': {name: hashlib.sha256(payload).hexdigest() for name, payload in members.items()},
        'source_sha256': sources, 'test_sha256': tests,
        'prior_handshake_v2_archive_sha256': old['archive_sha256'],
        'new_server_package_if_separately_approved': contract['expected_package_root'],
        'new_server_proof_if_separately_approved': contract['expected_output_root'],
        'gate_or_threshold_changed': False, 'rcond_changed': False, 'LS_changed': False,
        'real_P_read': False, 'B_smoke': False, 'formal_training': False,
        'original_G1_status': 'failed', 'formal_ready': False, 'strict_certificate': 'unknown'})


if __name__ == '__main__':
    freeze()
    print(json.dumps({'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
                      'contract_sha256': sha(REPORT/'execution-contract.json')}, indent=2))
