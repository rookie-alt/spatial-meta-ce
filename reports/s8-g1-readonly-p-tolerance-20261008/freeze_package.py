"""Freeze the retained-bound revision as a new isolated execution package.

Reuses the diagnostic package layout (which already carries the retained
diagnostic record) but binds the revised s8_probe and a new isolated
code/proof/contract path. Never reads saved P. Never overwrites old packages.
"""
import gzip
import hashlib
import io
import json
import runpy
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
ARCHIVE = Path('/private/tmp/s8-g1-readonly-p-tolerance-package-20261008.tar.gz')
DIAG = ROOT / 'reports/s8-g1-readonly-p-diagnostic-20261008'
TOL = 'reports/s8-g1-readonly-p-tolerance-20261008'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def freeze():
    old_diag = json.loads((DIAG / 'package-manifest.json').read_text())
    if sha(old_diag['archive']) != old_diag['archive_sha256']:
        raise ValueError('Prior diagnostic archive changed')
    hand = json.loads((ROOT / 'reports/s8-g1-readonly-p-handshake-v2-20261008/package-manifest.json').read_text())
    if sha(hand['archive']) != hand['archive_sha256']:
        raise ValueError('Prior handshake-v2 archive changed')
    module = runpy.run_path(str(ROOT / 'scripts/verify_s8_readonly_p.py'))
    sources = {name: sha(ROOT / name) for name in module['SOURCE_FILES']}
    tests = {'tests/test_s8_readonly_p.py': sha(ROOT / 'tests/test_s8_readonly_p.py'),
             'tests/test_s8_retained_tolerance.py': sha(ROOT / 'tests/test_s8_retained_tolerance.py')}
    contract = json.loads((DIAG / 'execution-contract.json').read_text())
    prior_files = contract['files']
    contract.update(
        revision='readonly-P-tolerance-20261008',
        expected_package_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-tolerance-code-20261008',
        expected_output_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-tolerance-proof-20261008',
        prepared_on='2026-10-08', source_sha256=sources,
        retained_projection_bound='max(engineering_floor, 8*eps*(sigma_1/sigma_cutoff)*scale)',
        retained_projection_scope='engineering_numerical_consistency_only_not_scientific',
        contract_relative_path=f'{TOL}/execution-contract.json',
        prior_packages={'handshake_v2': hand['archive_sha256'], 'diagnostic': old_diag['archive_sha256']},
        input_files_sha_preserved=prior_files,
        real_P_execution_not_authorized=True)
    # validate against current module constants (now tolerance roots).
    module['validate_contract'](contract)
    write('execution-contract.json', contract)
    members = {name: (ROOT / name).read_bytes() for name in (*sources, *tests)}
    members[contract['contract_relative_path']] = (REPORT / 'execution-contract.json').read_bytes()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            archive.addfile(info, io.BytesIO(payload))
    with ARCHIVE.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(buffer.getvalue())
    write('package-manifest.json', {
        'status': 'new_tolerance_package_prepared_not_real_P_executed',
        'revision': 'readonly-P-tolerance-20261008',
        'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'contract_relative_path': contract['contract_relative_path'],
        'execution_contract_sha256': sha(REPORT / 'execution-contract.json'),
        'files_sha256': {name: hashlib.sha256(payload).hexdigest() for name, payload in members.items()},
        'source_sha256': sources, 'test_sha256': tests,
        'input_files_sha_preserved': prior_files,
        'prior_packages': contract['prior_packages'],
        'new_server_package_if_separately_approved': contract['expected_package_root'],
        'new_server_proof_if_separately_approved': contract['expected_output_root'],
        'rcond_changed': False, 'rank_definition_changed': False, 'LS_changed': False,
        'scientific_gate_changed': False, 'strict_certificate': 'unknown',
        'formal_ready': False, 'original_G1_status': 'failed',
        'real_P_read': False, 'B_smoke': False, 'formal_training': False})


if __name__ == '__main__':
    freeze()
    print(json.dumps({'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
                      'contract_sha256': sha(REPORT / 'execution-contract.json')}, indent=2))
