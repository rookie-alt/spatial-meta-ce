"""Local preparation only: freeze historical SHA bindings, never open saved P."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
ARCHIVE = Path('/private/tmp/s8-g1-readonly-p-package-20261007.tar.gz')
INPUT = '/gly/weilin/spatial-meta-ce/runs/s8-g1-smoke-20261007'
OUTPUT = '/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-proof-20261007'
PACKAGE = '/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-code-20261007'
SOURCES = ['scripts/verify_s8_readonly_p.py', 'spatial_meta_ce/__init__.py',
           'spatial_meta_ce/s8_probe.py', '实验文档/27-S8G1运行器与真实smoke协议.md',
           '实验文档/34-S8G1工程验收v2批准协议.md',
           'reports/s8-g1-readonly-p-20261007/proof.schema.json']
TEST = 'tests/test_s8_readonly_p.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def freeze():
    # Only local JSON/code; four real arrays and checkpoint are never opened.
    history = ROOT / 'reports/s8-g1-smoke-20261007'
    old = ROOT / '实验文档/results/s8-g1-smoke-20261007'
    diagnosis = json.loads((history / 'solver-diagnosis.json').read_text())
    controller = json.loads((history / 'server-controller.json').read_text())
    contract = json.loads((old / 'contract.json').read_text())
    source_sha = {n: sha(ROOT / n) for n in SOURCES}
    shape = {'P-h.npy': [19456, 64], 'P-x.npy': [19456, 774],
             'P-label.npy': [19456], 'P-pg_log.npy': [19456]}
    files = {name: {'sha256': diagnosis['file_sha256'][name], 'shape': dimensions,
                    'dtype': 'float32'} for name, dimensions in shape.items()}
    for name in ('status.json', 'contract.json', 'panels.json', 'preflight.json'):
        if sha(old / name) != controller['archive_sha256'][name]:
            raise ValueError('Local historical archive changed; stop freezing')
        files[name] = {'sha256': controller['archive_sha256'][name]}
    value = {'schema': 1, 'entrypoint': SOURCES[0], 'evidence_kind': 'real_saved_P',
             'purpose': 'readonly_saved_P_resource_engineering_not_G1',
             'input_root': INPUT, 'expected_output_root': OUTPUT,
             'expected_package_root': PACKAGE,
             'expected_runtime': {'numpy': '1.26.1', 'torch': '2.1.0+cu121',
                                  'blas_family': 'openblas64', 'openmp_family': 'libgomp',
                                  'interpreter_prefix': '/gly/weilin/PRICE-master/venv'}, 'files': files,
             'panel': {'train_databases': contract['train_databases'], 'rows_per_database': 1024,
                       'small_rows_per_database': 64,
                       'panel_identity_digest': contract['panel_identity_digest'],
                       'source_binding_sha256': contract['source_binding_sha256'],
                       'original_protocol_sha256': contract['protocol_sha256'],
                       'identity_scope': 'historical_sha_bound_panel_attestation'},
             'normalizer_scope': 'P_only_recomputed_per_panel',
             'numerical_policy': 's8-g1-resource-engineering-v2',
             'budget': {'threads': 4, 'wall_seconds': 900,
                        'address_space_bytes': 12884901888, 'rss_stop_bytes': 12884901888,
                        'cuda_visible_devices': '', 'attempts': 1},
             'source_sha256': source_sha,
             'identity_limitations': ['no_fresh_query_hash_row_mapping_validation',
                                     'no_historical_normalizer_authentication',
                                     'no_source_SQL_or_selectivity_validation'],
             'historical_evidence_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
                 history / 'solver-diagnosis.json', history / 'server-controller.json',
                 history / 'partial-stage-verification.json', history / 'deployment-manifest.json',
                 history / 'thread-backend-precheck.json')},
             'scientific_status': {'original_G1': 'failed', 'formal_ready': False,
                                   'strict_certificate': 'unknown'},
             'not_authorized': ['server_connection', 'deployment', 'real_P_read', 'smoke', 'training']}
    write('execution-contract.json', value)
    members = {name: (ROOT / name).read_bytes() for name in SOURCES + [TEST]}
    for name in ('execution-contract.json', 'proof.schema.json'):
        members[f'reports/{REPORT.name}/{name}'] = (REPORT / name).read_bytes()
    # This is a new local package, not an overwrite of any historical evidence.
    import gzip
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        for name, payload in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            archive.addfile(info, io.BytesIO(payload))
    with ARCHIVE.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
        zipped.write(buffer.getvalue())
    write('package-manifest.json', {'status': 'prepared_not_executed', 'archive': str(ARCHIVE),
          'archive_sha256': sha(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
          'execution_contract_sha256': sha(REPORT / 'execution-contract.json'),
          'files_sha256': {name: hashlib.sha256(payload).hexdigest() for name, payload in members.items()},
          'source_sha256': source_sha, 'test_sha256': {TEST: sha(ROOT / TEST)},
          'server_input_paths_if_approved': [f'{INPUT}/{n}' for n in files],
          'server_package_path_if_approved': PACKAGE, 'server_output_path_if_approved': OUTPUT,
          'real_P_read': False, 'server_access': False, 'deployed': False,
          'real_smoke_executed': False, 'formal_training_started': False, 'formal_ready': False})
    print(json.dumps({'archive': str(ARCHIVE), 'archive_sha256': sha(ARCHIVE),
                     'contract_sha256': sha(REPORT / 'execution-contract.json')}, indent=2))


if __name__ == '__main__':
    freeze()
