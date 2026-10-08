"""Freeze a new local handshake-v2 package; never read saved P or old server paths."""

import gzip
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
ARCHIVE = Path('/private/tmp/s8-g1-readonly-p-handshake-v2-package-20261008.tar.gz')
ENTRYPOINT = 'scripts/verify_s8_readonly_p.py'
TESTS = ('tests/test_s8_readonly_p.py', 'tests/test_s8_readonly_handshake_linux.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def freeze():
    old_report = ROOT / 'reports/s8-g1-readonly-p-20261007'
    old_manifest = json.loads((old_report / 'package-manifest.json').read_text())
    if (sha(old_report / 'execution-contract.json') != old_manifest['execution_contract_sha256']
            or sha(Path(old_manifest['archive'])) != old_manifest['archive_sha256']):
        raise ValueError('Original frozen contract/archive changed')
    # Loads only stdlib definitions, not __main__, training or numerical functions.
    module = runpy.run_path(str(ROOT / ENTRYPOINT))
    sources = {name: sha(ROOT / name) for name in module['SOURCE_FILES']}
    contract = json.loads((old_report / 'execution-contract.json').read_text())
    contract.update(expected_package_root=module['REAL_PACKAGE_ROOT'],
                    expected_output_root=module['REAL_OUTPUT_ROOT'], source_sha256=sources,
                    prepared_on='2026-10-08', revision='readonly-P-handshake-v2',
                    contract_relative_path=module['CONTRACT_RELATIVE'],
                    handshake_path_policy='authenticated_parent_cwd_frozen_path_and_SHA_pipe_ticket',
                    execution_authorization='new_real_P_execution_not_authorized',
                    not_authorized=['new_real_P_server_deployment', 'real_P_read', 'smoke', 'training'])
    contract['historical_failed_A_evidence_sha256'] = {
        name: sha(ROOT / name) for name in (
            '实验文档/37-S8G1A单次只读核验握手失败记录.md',
            'reports/s8-g1-readonly-p-execution-20261008/real-proof/proof.json',
            'reports/s8-g1-readonly-p-execution-20261008/independent-record-review.md')}
    module['validate_contract'](contract)
    write('execution-contract.json', contract)
    members = {name: (ROOT / name).read_bytes() for name in (*sources, *TESTS)}
    relative_contract = module['CONTRACT_RELATIVE']
    members[relative_contract] = (REPORT / 'execution-contract.json').read_bytes()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            archive.addfile(info, io.BytesIO(payload))
    with ARCHIVE.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(buffer.getvalue())
    write('package-manifest.json', {
        'status': 'new_local_package_prepared_not_real_P_executed', 'revision': 'readonly-P-handshake-v2',
        'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'contract_relative_path': relative_contract,
        'execution_contract_sha256': sha(REPORT / 'execution-contract.json'),
        'files_sha256': {name: hashlib.sha256(payload).hexdigest() for name, payload in members.items()},
        'source_sha256': sources, 'test_sha256': {name: sha(ROOT / name) for name in TESTS},
        'original_archive_sha256_preserved': old_manifest['archive_sha256'],
        'original_contract_sha256_preserved': old_manifest['execution_contract_sha256'],
        'new_server_package_if_separately_approved': module['REAL_PACKAGE_ROOT'],
        'new_server_proof_if_separately_approved': module['REAL_OUTPUT_ROOT'],
        'future_real_P_deployment_performed': False, 'real_P_read': False,
        'smoke_performed': False, 'formal_training_started': False, 'formal_ready': False,
        'original_G1_status': 'failed', 'strict_certificate': 'unknown'})
    print(json.dumps({'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
                     'contract_sha256': sha(REPORT / 'execution-contract.json')}, indent=2))


if __name__ == '__main__':
    freeze()
