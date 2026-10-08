"""Freeze the real-retained-condition-number revision as a new isolated package.

Binds the revised s8_probe and new isolated code/proof/contract roots. Never
reads saved P. Never overwrites the tolerance/diagnostic/handshake packages.
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
ARCHIVE = Path('/private/tmp/s8-g1-readonly-p-kappa-package-20261008.tar.gz')
TOL = ROOT / 'reports/s8-g1-readonly-p-tolerance-20261008'
DIAG = ROOT / 'reports/s8-g1-readonly-p-diagnostic-20261008'
HAND = ROOT / 'reports/s8-g1-readonly-p-handshake-v2-20261008'
CONTRACT_RELATIVE = 'reports/s8-g1-readonly-p-retained-kappa-20261008/execution-contract.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prior_manifest(name, directory):
    manifest = json.loads((directory / 'package-manifest.json').read_text())
    if sha(manifest['archive']) != manifest['archive_sha256']:
        raise ValueError(f'Prior {name} archive changed')
    return manifest


def freeze():
    tol = prior_manifest('tolerance', TOL)
    diag = prior_manifest('diagnostic', DIAG)
    hand = prior_manifest('handshake_v2', HAND)
    module = runpy.run_path(str(ROOT / 'scripts/verify_s8_readonly_p.py'))
    sources = {name: sha(ROOT / name) for name in module['SOURCE_FILES']}
    tests = {
        'tests/test_s8_readonly_p.py': sha(ROOT / 'tests/test_s8_readonly_p.py'),
        'tests/test_s8_retained_tolerance.py': sha(ROOT / 'tests/test_s8_retained_tolerance.py'),
    }
    contract = json.loads((TOL / 'execution-contract.json').read_text())
    prior_files = contract['files']
    contract.update(
        revision='retained-kappa-real-condition-number-20261008',
        expected_package_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-code-20261008',
        expected_output_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-proof-20261008',
        prepared_on='2026-10-08', source_sha256=sources,
        retained_projection_bound='max(engineering_floor, 8*eps*(sigma_1/sigma_r)*scale)',
        retained_projection_scope='engineering_numerical_consistency_only_not_scientific',
        retained_projection_kappa='real_retained_condition_number_sigma_1_over_sigma_r',
        contract_relative_path=CONTRACT_RELATIVE,
        prior_packages={
            'handshake_v2': hand['archive_sha256'],
            'diagnostic': diag['archive_sha256'],
            'tolerance': tol['archive_sha256'],
        },
        input_files_sha_preserved=prior_files,
        real_P_execution_not_authorized=True)
    module['validate_contract'](contract)
    members = {name: (ROOT / name).read_bytes() for name in (*sources, *tests)}
    members[CONTRACT_RELATIVE] = _contract_bytes(contract)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            archive.addfile(info, io.BytesIO(payload))
    with ARCHIVE.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(buffer.getvalue())
    write('package-manifest.json', {
        'status': 'new_kappa_package_prepared_not_real_P_executed',
        'revision': 'retained-kappa-real-condition-number-20261008',
        'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
        'contract_relative_path': CONTRACT_RELATIVE,
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


def _contract_bytes(contract):
    return (json.dumps(contract, ensure_ascii=False, indent=2) + '\n').encode()


if __name__ == '__main__':
    # Write the contract first so its SHA is stable, then graft it into the archive.
    tol_contract = json.loads((TOL / 'execution-contract.json').read_text())
    module = runpy.run_path(str(ROOT / 'scripts/verify_s8_readonly_p.py'))
    sources = {name: sha(ROOT / name) for name in module['SOURCE_FILES']}
    contract = dict(tol_contract)
    contract.update(
        revision='retained-kappa-real-condition-number-20261008',
        expected_package_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-code-20261008',
        expected_output_root='/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-proof-20261008',
        prepared_on='2026-10-08', source_sha256=sources,
        retained_projection_bound='max(engineering_floor, 8*eps*(sigma_1/sigma_r)*scale)',
        retained_projection_scope='engineering_numerical_consistency_only_not_scientific',
        retained_projection_kappa='real_retained_condition_number_sigma_1_over_sigma_r',
        contract_relative_path=CONTRACT_RELATIVE,
        prior_packages={
            'handshake_v2': prior_manifest('handshake_v2', HAND)['archive_sha256'],
            'diagnostic': prior_manifest('diagnostic', DIAG)['archive_sha256'],
            'tolerance': prior_manifest('tolerance', TOL)['archive_sha256'],
        },
        input_files_sha_preserved=tol_contract['files'],
        real_P_execution_not_authorized=True)
    module['validate_contract'](contract)
    (REPORT / 'execution-contract.json').write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + '\n')
    freeze()
    print(json.dumps({'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
                      'contract_sha256': sha(REPORT / 'execution-contract.json')}, indent=2))
