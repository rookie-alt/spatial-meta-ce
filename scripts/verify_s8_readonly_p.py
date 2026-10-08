"""Single-attempt saved-P LS engineering check; default CLI is plan-only.

No data reconstruction, model/checkpoint loading, optimizer, smoke, nomination or
training entrypoint. Real execution requires a frozen manifest SHA and Linux.
Historical row identity is an attestation, not fresh source/SQL verification.
"""

import argparse
import contextvars
import ctypes
import hashlib
import io
import json
import math
import os
import resource
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_ACTIVE_OPEN = contextvars.ContextVar('readonly_secure_open_path', default=None)
ENTRYPOINT = 'scripts/verify_s8_readonly_p.py'
REAL_INPUT_ROOT = '/gly/weilin/spatial-meta-ce/runs/s8-g1-smoke-20261007'
REAL_OUTPUT_ROOT = '/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-proof-20261008'
REAL_PACKAGE_ROOT = '/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-code-20261008'
CONTRACT_RELATIVE = 'reports/s8-g1-readonly-p-retained-kappa-20261008/execution-contract.json'
EXPECTED_RUNTIME = {'numpy': '1.26.1', 'torch': '2.1.0+cu121',
                    'blas_family': 'openblas64', 'openmp_family': 'libgomp',
                    'interpreter_prefix': '/gly/weilin/PRICE-master/venv'}
TRAIN = ('airline', 'baseball', 'basketball', 'carcinogenesis', 'ccs', 'consumer',
         'credit', 'financial', 'fnhk', 'hepatitis', 'movielens', 'sakila', 'sap',
         'seznam', 'ssb', 'talkingdata', 'tournament', 'tpc_h', 'tubepricing')
ARRAYS = ('P-h.npy', 'P-x.npy', 'P-label.npy', 'P-pg_log.npy')
METADATA = ('status.json', 'contract.json', 'panels.json', 'preflight.json')
MEMORY = 12 * 1024 ** 3
BUDGET = {'threads': 4, 'wall_seconds': 900, 'address_space_bytes': MEMORY,
          'rss_stop_bytes': MEMORY, 'cuda_visible_devices': '', 'attempts': 1}
POLICY = 's8-g1-resource-engineering-v2'
SOURCE_FILES = (ENTRYPOINT, 'spatial_meta_ce/__init__.py', 'spatial_meta_ce/s8_probe.py',
                '实验文档/27-S8G1运行器与真实smoke协议.md',
                '实验文档/34-S8G1工程验收v2批准协议.md',
                'reports/s8-g1-readonly-p-20261007/proof.schema.json')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def no_symlink(path):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f'Symlink path forbidden: {component}')
    return path


@contextmanager
def secure_parent_fd(path):
    """Walk each directory with O_NOFOLLOW; hold inode capability until I/O ends."""
    path = Path(os.path.abspath(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, 'O_NOFOLLOW', 0)
    token = _ACTIVE_OPEN.set(Path('/'))
    try:
        fd = os.open('/', flags)
    finally:
        _ACTIVE_OPEN.reset(token)
    try:
        cursor = Path('/')
        for name in path.parts[1:-1]:
            cursor = cursor / name
            token = _ACTIVE_OPEN.set(cursor)
            try:
                new_fd = os.open(name, flags, dir_fd=fd)
            finally:
                _ACTIVE_OPEN.reset(token)
            os.close(fd)
            fd = new_fd
        yield fd
    finally:
        os.close(fd)


def capability_open(path, flags, parent_fd, mode=0o600):
    token = _ACTIVE_OPEN.set(path)
    try:
        return os.open(path.name, flags, mode, dir_fd=parent_fd)
    finally:
        _ACTIVE_OPEN.reset(token)


def read_regular(path):
    path = no_symlink(path)
    with secure_parent_fd(path) as parent_fd:
        fd = capability_open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0), parent_fd)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Only regular files may be read')
        return stream.read()


def sha256(path):
    return hashlib.sha256(read_regular(path)).hexdigest()


def is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_contract(contract):
    if not isinstance(contract, dict):
        raise TypeError('Contract must be an object')
    if (contract.get('schema') != 1 or contract.get('entrypoint') != ENTRYPOINT
            or contract.get('purpose') != 'readonly_saved_P_resource_engineering_not_G1'
            or contract.get('evidence_kind') not in ('real_saved_P', 'synthetic_fixture')
            or contract.get('normalizer_scope') != 'P_only_recomputed_per_panel'
            or contract.get('numerical_policy') != POLICY or contract.get('budget') != BUDGET):
        raise ValueError('Non-read-only, unknown policy or unfrozen resource contract')
    root = contract.get('input_root')
    if not isinstance(root, str) or not Path(root).is_absolute() or '..' in Path(root).parts:
        raise ValueError('Input root must be a frozen absolute path')
    files = contract.get('files')
    if not isinstance(files, dict) or set(files) != set(ARRAYS + METADATA):
        raise ValueError('Only four P arrays and four provenance metadata files permitted')
    for name, spec in files.items():
        if not isinstance(spec, dict) or not is_sha(spec.get('sha256')):
            raise ValueError(f'Invalid source SHA: {name}')
        if name in ARRAYS:
            shape = spec.get('shape')
            if (spec.get('dtype') != 'float32' or not isinstance(shape, list)
                    or len(shape) != (2 if name in ARRAYS[:2] else 1)
                    or any(type(n) is not int or n <= 0 for n in shape)):
                raise ValueError('Invalid frozen P shape/dtype')
    panel = contract.get('panel', {})
    if (panel.get('train_databases') != list(TRAIN)
            or panel.get('identity_scope') != 'historical_sha_bound_panel_attestation'
            or any(not is_sha(panel.get(k)) for k in ('panel_identity_digest',
                       'source_binding_sha256', 'original_protocol_sha256'))):
        raise ValueError('Frozen P identity/source scope mismatch')
    full, small = panel.get('rows_per_database'), panel.get('small_rows_per_database')
    if type(full) is not int or type(small) is not int or not 0 < small <= full:
        raise ValueError('Invalid P-only full/small row count')
    rows = len(TRAIN) * full
    if any(spec['shape'][0] != rows for n, spec in files.items() if n in ARRAYS):
        raise ValueError('Inconsistent P row counts')
    sources = contract.get('source_sha256')
    if (not isinstance(sources, dict) or not set(sources) <= set(SOURCE_FILES)
            or any(not is_sha(value) for value in sources.values())):
        raise ValueError('Only the frozen read-only numerical source files may be bound')
    if contract['evidence_kind'] == 'real_saved_P':
        expected = {'P-h.npy': [19456, 64], 'P-x.npy': [19456, 774],
                    'P-label.npy': [19456], 'P-pg_log.npy': [19456]}
        if (root != REAL_INPUT_ROOT or contract.get('expected_output_root') != REAL_OUTPUT_ROOT
                or contract.get('expected_package_root') != REAL_PACKAGE_ROOT
                or contract.get('expected_runtime') != EXPECTED_RUNTIME
                or full != 1024 or small != 64 or set(sources) != set(SOURCE_FILES)
                or any(files[n]['shape'] != shape for n, shape in expected.items())):
            raise ValueError('Real saved-P contract must use the original 19456/1216 panels')
    return contract


def authorize_data_open(path, contract, writing):
    path = Path(os.path.abspath(path))
    allowed = {Path(contract['input_root']) / name for name in ARRAYS + METADATA}
    if writing or path not in allowed:
        raise PermissionError(f'Non-read-only/non-P data access forbidden: {path}')
    return path


def read_bound(contract, name, opened):
    if name not in ARRAYS + METADATA:
        raise ValueError('Unapproved input name')
    path = authorize_data_open(Path(contract['input_root']) / name, contract, False)
    opened.append(str(path))
    payload = read_regular(path)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != contract['files'][name]['sha256']:
        raise ValueError(f'Source SHA mismatch: {name}')
    return payload, actual


def open_output(path):
    path = no_symlink(path)
    with secure_parent_fd(path) as parent_fd:
        fd = capability_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), parent_fd)
    return os.fdopen(fd, 'wb')


def write_json(path, value):
    path = no_symlink(path)
    with secure_parent_fd(path) as parent_fd:
        fd = capability_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), parent_fd)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def create_output(path, contract):
    path, old = no_symlink(path), no_symlink(contract['input_root'])
    protected = (Path('/gly/weilin/PRICE-master'),
                 Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-execution-code-20261007'))
    if any(path == p or p in path.parents for p in protected):
        raise ValueError('Cannot write into PRICE or old execution package')
    if contract['evidence_kind'] == 'real_saved_P' and str(path) != contract['expected_output_root']:
        raise ValueError('Real proof output must match the frozen new directory')
    if path == old or old in path.parents or path in old.parents or path.exists():
        raise ValueError('Proof output must be new, outside and unrelated to the old failed directory')
    with secure_parent_fd(path) as parent_fd:
        os.mkdir(path.name, mode=0o700, dir_fd=parent_fd)
    return path


def initial_proof(contract):
    return {'schema': 1, 'status': 'not_run', 'stage': 'not_run',
            'purpose': contract['purpose'], 'evidence_kind': contract['evidence_kind'],
            'formal_ready': False, 'original_G1_status': 'failed',
            'can_retroactively_pass_original_G1': False, 'real_smoke_executed': False,
            'F0_updates': 0, 'Adam_updates': 0, 'H_read': False, 'target_read': False,
            'checkpoint_read': False, 'GPU_used': False,
            'features': {feature: {panel: {'status': 'not_run', 'engineering_verified': None,
                         'rank_stability': 'not_run', 'strict_certificate': {'status': 'unknown'},
                         'full_linear_optimality_verified': False}
                         for panel in ('small_P_zero_reference', 'full_P_resource')}
                         for feature in ('h', 'x')},
            'input_sha256_before': {}, 'input_sha256_after': {}, 'opened_input_paths': []}


def check_sources(contract):
    for name, expected in contract['source_sha256'].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f'Frozen numerical/entrypoint/protocol source changed: {name}')


def check_metadata(metadata, contract):
    status, old, panels, preflight = (metadata[n] for n in METADATA)
    panel = contract['panel']
    if (status.get('status') != 'failed' or status.get('formal_ready') is not False
            or old.get('schema') != 1 or old.get('mode') != 'smoke'
            or old.get('train_databases') != list(TRAIN)
            or preflight.get('split', {}).get('train') != list(TRAIN)
            or preflight.get('panel_unique') != len(TRAIN) * panel['rows_per_database']
            or any(old.get(k) != panel[k] for k in ('panel_identity_digest', 'source_binding_sha256'))
            or old.get('protocol_sha256') != panel['original_protocol_sha256']
            or digest(preflight.get('input_sha256')) != panel['source_binding_sha256']
            or set(panels) != set(TRAIN)):
        raise ValueError('Historical failed/P identity/source provenance mismatch')
    for indices in panels.values():
        if (not isinstance(indices, list) or len(indices) != panel['rows_per_database']
                or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('Invalid/duplicate frozen P panel positions')
    # Do NOT dereference any preflight input path, query cache or source database.


@contextmanager
def computation_guard():
    previous = sys.getprofile()
    forbidden = {'solve_adam', 'train_f0', 'backward', 'checkpoint_reload',
                 'extract_representations', 'capture', 'step'}
    def guard(frame, event, arg):
        if event == 'call':
            name = frame.f_code.co_name
            module = frame.f_globals.get('__name__', '')
            if name in forbidden or (name == 'load' and module.startswith('torch')):
                raise PermissionError('Non-read-only computation forbidden')
    sys.setprofile(guard)
    try:
        yield
    finally:
        sys.setprofile(previous)


def schema_accepts(value, schema):
    """Validate the frozen proof schema's subset without a server dependency."""
    if 'const' in schema and (type(value) is not type(schema['const']) or value != schema['const']):
        return False
    if 'enum' in schema and value not in schema['enum']:
        return False
    types = schema.get('type', [])
    types = [types] if isinstance(types, str) else types
    kinds = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None)}
    if types and not any(type(value) is kinds.get(kind) for kind in types):
        return False
    if isinstance(value, dict):
        if not set(schema.get('required', [])) <= set(value):
            return False
        properties = schema.get('properties', {})
        if schema.get('additionalProperties') is False and not set(value) <= set(properties):
            return False
        if any(not schema_accepts(value[k], spec) for k, spec in properties.items() if k in value):
            return False
    if isinstance(value, list) and 'items' in schema and any(not schema_accepts(item, schema['items']) for item in value):
        return False
    if 'not' in schema and schema_accepts(value, schema['not']):
        return False
    if any(not schema_accepts(value, part) for part in schema.get('allOf', [])):
        return False
    return not ('if' in schema and schema_accepts(value, schema['if'])
                and not schema_accepts(value, schema.get('then', {})))


def _retained_record_check(cert):
    """Check recorded scalars/formula only; no matrix/spectrum proof is issued.

    The frozen constants are literal, not imported from the producer. Residual
    acceptance uses the independently calculated bound, never a stored tol.
    Actual extrema, scales and residuals cannot be authenticated without the
    original matrix/vectors; those quantities remain explicitly uncertified.
    """
    def number(value):
        if type(value) not in (int, float):
            raise ValueError('retained record requires real JSON numbers')
        try:
            value = float(value)
        except OverflowError as error:
            raise ValueError('retained record numeric overflow') from error
        if not math.isfinite(value) or value < 0:
            raise ValueError('retained record requires finite nonnegative numbers')
        return value

    def equal(actual, expected):
        # Relative-only metadata identity check, NOT a solver tolerance.
        if not math.isfinite(expected) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=0):
            raise ValueError('retained record contradicts frozen formula')

    check = cert.get('retained_subspace_check')
    if not isinstance(check, dict) or check.get('scope') != 'real_retained_condition_number_fp64_reachable_bound':
        raise ValueError('missing or wrong retained check scope')
    rank = cert['observed_ranks']['svd']
    fields = ('tol', 'sigma1', 'sigma_r', 'sigma_cutoff', 'kappa_retained',
              'projection_norm', 'target_scale', 'amplification_scale',
              'engineering_floor', 'reachable_bound', 'accumulation_constant', 'eps')
    for side in ('reported', 'reference'):
        bound = check.get(f'{side}_bound')
        scope = ('max_engineering_floor_and_fp64_reachable_bound' if rank
                 else 'rank_zero_engineering_floor')
        if not isinstance(bound, dict) or bound.get('scope') != scope:
            raise ValueError('missing or wrong retained bound scope')
        values = {key: number(bound.get(key)) for key in fields}
        eps = 2.0**-52
        if values['eps'] != eps or values['accumulation_constant'] != 8.0:
            raise ValueError('retained FP64 constants changed')
        sigma1, sigma_r = values['sigma1'], values['sigma_r']
        equal(values['sigma_cutoff'], 1e-12 * sigma1)
        eta = cert['spectrum_error_bound']
        for actual, expected in zip(cert['rank_cutoff_interval'],
                                    (1e-12 * max(0.0, sigma1 - eta), 1e-12 * (sigma1 + eta))):
            equal(number(actual), expected)
        if rank:
            if not 0 < values['sigma_cutoff'] < sigma_r <= sigma1:
                raise ValueError('retained singular values violate fixed cutoff/rank')
            if rank == 1 and sigma_r != sigma1:
                raise ValueError('rank one must retain leading singular value')
            kappa = sigma1 / sigma_r
            if not 1 <= kappa < 1e12:
                raise ValueError('invalid retained condition number')
        else:
            if any(values[key] != 0 for key in ('sigma1', 'sigma_r', 'sigma_cutoff',
                                                'kappa_retained', 'projection_norm')):
                raise ValueError('rank zero requires zero spectrum/projection')
            kappa = 0.0
        equal(values['kappa_retained'], kappa)
        scale = max(values['target_scale'], values['projection_norm'], eps)
        floor = 1e-12 + 1e-12 * values['projection_norm']
        reachable = 8.0 * eps * kappa * scale
        tolerance = max(floor, reachable)
        for key, expected in (('amplification_scale', scale), ('engineering_floor', floor),
                              ('reachable_bound', reachable), ('tol', tolerance)):
            equal(values[key], expected)
        residual = number(check.get(f'{side}_residual'))
        if residual > tolerance or cert.get(f'{"reference_" if side == "reference" else ""}retained_subspace_pass') is not True:
            raise ValueError('retained residual exceeds independently computed tolerance')
    # Both checks use the same SVD projection and target; only solve residuals differ.
    if check['reported_bound'] != check['reference_bound']:
        raise ValueError('retained sides disagree on shared spectrum/projection/target')


def certificate_complete(certificate, rows=None, columns=None):
    gates = ('metadata_pass', 'normal_residual_pass', 'objective_pass',
             'reference_objective_pass', 'reference_prediction_pass', 'reference_spectrum_pass',
             'svd_backend_pass', 'retained_subspace_pass', 'reference_retained_subspace_pass',
             'observed_rank_agreement', 'spectrum_pass', 'reference_pass')
    try:
        shape = certificate['spectrum_matrix_shape']
        if (type(shape) is not list or len(shape) != 2
                or any(type(n) is not int or n <= 0 for n in shape)):
            return False
        rows, columns = rows or shape[0], columns or shape[1]
        if shape != [rows, columns]:
            return False
        ranks = certificate['observed_ranks']
        if (certificate['engineering_verified'] is not True
                or not all(certificate.get(gate) is True for gate in gates)
                or set(ranks) != {'svd', 'reported_ls', 'reference_ls'}
                or any(type(rank) is not int or not 0 <= rank <= min(rows, columns) for rank in ranks.values())
                or len(set(ranks.values())) != 1):
            return False
        rank, stability = ranks['svd'], certificate['rank_stability']
        if stability not in ('stable', 'unknown'):
            return False
        passed = stability == 'stable'
        for key, expected in (('pass', passed), ('solver_verified', passed),
                 ('rank_limited', rank < columns), ('numerical_rank_limited', rank < columns),
                 ('fullrank_numeric_solver', passed and rank == columns),
                 ('full_linear_optimality', False), ('full_linear_optimality_verified', False)):
            if certificate[key] is not expected:
                return False
        if (certificate['strict_certificate'] != {'status': 'unknown',
                'scopes': ['true_rank', 'rigorous_retained_subspace', 'strict_linear_optimality'],
                'reason': 'no_verified_arithmetic_certificate'}
                or certificate['certification_linalg_calls'] != 2
                or certificate['spectrum_specification'] != 's8-fp64-spectrum-v1.1'
                or certificate['spectrum_dtype'] != 'float64'
                or not isinstance(certificate['spectrum_numpy_version'], str)
                or not certificate['spectrum_numpy_version']):
            return False
        fields = ('spectrum_epsilon', 'spectrum_gamma', 'spectrum_error_bound',
                  'spectrum_comparison_bound', 'svd_scaled_reconstruction_residual',
                  'svd_u_orthogonality_residual', 'svd_v_orthogonality_residual',
                  'svd_u_normalized_orthogonality_residual', 'svd_v_normalized_orthogonality_residual',
                  'svd_normalized_orthogonality_bound', 'normalized_normal_residual',
                  'objective_ls', 'objective_adam', 'reference_objective', 'rank_interval_margin')
        numbers = {k: certificate[k] for k in fields}
        if any(type(v) not in (int, float) or not math.isfinite(v)
               or (v < 0 and k != 'rank_interval_margin') for k, v in numbers.items()):
            return False
        eps, d, k = 2.0 ** -52, max(rows, columns), min(rows, columns)
        close = lambda a, b: math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        gamma, orth = d * eps / (1 - d * eps), (d + k) * eps / (1 - (d + k) * eps)
        if (numbers['spectrum_epsilon'] != eps or not close(numbers['spectrum_gamma'], gamma)
                or not close(numbers['svd_normalized_orthogonality_bound'], orth)
                or not close(numbers['spectrum_comparison_bound'], 2 * numbers['spectrum_error_bound'])
                or numbers['normalized_normal_residual'] > 1e-8
                or numbers['objective_ls'] > numbers['objective_adam'] + 1e-8 * max(1.0, numbers['objective_adam'])
                or not close(numbers['objective_ls'], numbers['reference_objective'])
                or numbers['svd_scaled_reconstruction_residual'] > gamma):
            return False
        for axis in ('u', 'v'):
            raw, normalized = numbers[f'svd_{axis}_orthogonality_residual'], numbers[f'svd_{axis}_normalized_orthogonality_residual']
            if not close(normalized, raw / math.sqrt(k)) or normalized > orth:
                return False
        intervals = certificate['rank_cutoff_intervals']
        if set(intervals) != set(ranks) or certificate['rank_cutoff_interval'] != intervals['svd']:
            return False
        for bounds in intervals.values():
            if (type(bounds) is not list or len(bounds) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                    or not 0 <= bounds[0] <= bounds[1]):
                return False
        _retained_record_check(certificate)
        margin = numbers['rank_interval_margin']
        return not ((stability == 'unknown' and margin > 0)
                    or (passed and margin <= 0 and not (rank == 0 and numbers['spectrum_error_bound'] == margin == 0)))
    except (KeyError, TypeError, AttributeError, ZeroDivisionError, ValueError, OverflowError):
        return False


def selection_sha(rows_per_db, selected_per_db):
    return hashlib.sha256(b''.join(struct.pack('<q', i * rows_per_db + j)
                          for i in range(len(TRAIN)) for j in range(selected_per_db))).hexdigest()


def stage_record_complete(row, contract, feature, panel_name):
    try:
        schema = json.loads(read_regular(ROOT/'reports/s8-g1-readonly-p-20261007/proof.schema.json'))
        fragment = schema['properties']['features']['properties'][feature]['properties'][panel_name]
        if not schema_accepts(row, fragment):
            return False
        full = contract['panel']['rows_per_database']
        selected = full if panel_name == 'full_P_resource' else contract['panel']['small_rows_per_database']
        rows = len(TRAIN) * selected
        columns = contract['files'][f'P-{feature}.npy']['shape'][1] + 1
        for key, expected in (('status', 'verified'), ('engineering_verified', True),
                ('resource_only', True), ('coefficients_discarded', True),
                ('not_used_for_predictions_or_selection', True), ('reference_kind', 'zero_reference_not_Adam'),
                ('claim', 'resource_only_engineering_not_G1' if panel_name == 'full_P_resource' else 'diagnostic_not_original_Adam_probe_gate'),
                ('rows', rows), ('columns', columns), ('matrix_bytes', 8 * rows * columns),
                ('linalg_calls', 3), ('rank', row['certificate']['observed_ranks']['svd']),
                ('full_linear_optimality_verified', False)):
            if type(row[key]) is not type(expected) or row[key] != expected:
                return False
        if (row['strict_certificate'] != row['certificate']['strict_certificate']
                or row['rank_stability'] != row['certificate']['rank_stability']
                or not certificate_complete(row['certificate'], rows, columns)):
            return False
        seconds = [row[k] for k in ('standardization_seconds', 'solver_certification_seconds', 'seconds')]
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in seconds)
                or not math.isclose(sum(seconds[:2]), seconds[2], rel_tol=1e-12, abs_tol=1e-12)):
            return False
        normalizer = row['normalizer_source']
        return not (normalizer['scope'] != 'this_P_panel_only'
                or normalizer['input_sha256'] != contract['files'][f'P-{feature}.npy']['sha256']
                or normalizer['fit_indices_sha256'] != selection_sha(full, selected)
                or normalizer['fit_rows'] != rows or normalizer['feature_columns'] != columns - 1
                or normalizer['dtype'] != 'float64' or normalizer['ddof'] != 0
                or normalizer['panel_identity_digest'] != contract['panel']['panel_identity_digest']
                or set(normalizer['normalizer_sha256']) != {'mean', 'std', 'scale', 'zero_mask', 'tiny_mask'}
                or not all(is_sha(h) for h in normalizer['normalizer_sha256'].values()))
    except (KeyError, TypeError, AttributeError, OSError, ValueError):
        return False


def retained_projection_diagnostic(design, target, ls, reference_theta, probe):
    """Record why retained-projection allclose failed; never change the gate."""
    design = probe._array64(design, 2, 'diagnostic design')
    target = probe._array64(target, 1, 'diagnostic target')
    if design.shape[0] != target.size:
        raise ValueError('diagnostic design rows must match target length')
    # Reuse the exact certification factorization semantics for diagnosis.
    u, singular, _ = probe.np.linalg.svd(design, full_matrices=False)
    singular = probe._array64(singular, 1, 'SVD singular values')
    rank = int(probe.np.count_nonzero(singular > probe.RCOND * singular[0]))
    rank, singular = probe._rank_and_singular(rank, singular, design.shape)
    u = probe._array64(u, 2, 'SVD U')
    projection = u[:, :rank] @ (u[:, :rank].T @ target)
    rtol, atol = 1e-12, 1e-12
    result = {'gate_effect': 'diagnostic_only_no_gate_change', 'shape': list(design.shape),
              'dtype': str(design.dtype), 'rank': rank,
              'cutoff': float(probe.RCOND * singular[0]), 'rtol': rtol, 'atol': atol,
              'numpy': probe.np.__version__, 'projection_kind': 'left_retained_U_projection',
              'formula': 'U_r @ (U_r.T @ target)'}
    for name, theta in (('reported', ls['theta']), ('reference', reference_theta)):
        theta = probe._array64(theta, 1, f'diagnostic {name} theta')
        residual = design @ theta - projection
        threshold = atol + rtol * probe.np.abs(projection)
        abs_residual = probe.np.abs(residual)
        max_index = int(probe.np.argmax(abs_residual))
        coefficients = u[:, :rank].T @ residual
        order = probe.np.argsort(-probe.np.abs(coefficients))[:8]
        directions = []
        for direction in order:
            coordinate = float(coefficients[direction])
            projection_coordinate = float(projection[direction])
            item_threshold = float(atol + rtol * abs(projection_coordinate))
            directions.append({'u_direction_index': int(direction),
                               'singular_value': float(singular[direction]),
                               'residual_coordinate': coordinate,
                               'projection_coordinate': projection_coordinate,
                               'threshold': item_threshold,
                               'ratio': float(abs(coordinate) / item_threshold)})
        result[name] = {
            'residual_l2_norm': float(probe.np.linalg.norm(residual)),
            'residual_linfinity_norm': float(abs_residual.max()),
            'max_abs_residual': float(abs_residual[max_index]),
            'threshold_at_max': float(threshold[max_index]),
            'ratio_to_threshold': float(abs_residual[max_index] / threshold[max_index]),
            'max_row_index': max_index,
            'top_singular_directions': directions,
        }
    return result


def proof_complete(proof, contract, output=None):
    try:
        schema = json.loads(read_regular(ROOT/'reports/s8-g1-readonly-p-20261007/proof.schema.json'))
        if not schema_accepts(proof, schema):
            return False
        if (proof['status'] != 'verified_readonly_P_engineering'
                or proof['formal_ready'] is not False or proof['original_G1_status'] != 'failed'
                or proof['original_inputs_unchanged'] is not True
                or any(proof[k] is not False for k in ('H_read', 'target_read', 'checkpoint_read',
                       'GPU_used', 'real_smoke_executed', 'can_retroactively_pass_original_G1'))
                or any(type(proof[k]) is not int or proof[k] != 0 for k in ('F0_updates', 'Adam_updates'))
                or proof['evidence_kind'] != contract['evidence_kind']
                or proof['panel_identity'] != contract['panel']
                or proof['identity_scope'] != 'historical_sha_bound_panel_attestation_not_fresh_query_semantics'
                or set(proof['features']) != {'h', 'x'}
                or proof['opened_input_paths'] != sorted(str(Path(contract['input_root'])/n) for n in ARRAYS + METADATA)
                or proof['input_sha256_before'] != {n: s['sha256'] for n, s in contract['files'].items()}
                or proof['input_sha256_before'] != proof['input_sha256_after']):
            return False
        for feature in ('h', 'x'):
            for panel_name, row in proof['features'][feature].items():
                if not stage_record_complete(row, contract, feature, panel_name):
                    return False
                if output is not None and digest(json.loads(read_regular(output/f'{feature}-{panel_name}.json'))) != digest(row):
                    return False
        return True
    except (KeyError, TypeError, AttributeError, OSError, ValueError):
        return False


def check_saved_p(contract, output, probe=None):
    validate_contract(contract)
    result = initial_proof(contract)
    start = time.monotonic()
    try:
        result['stage'] = 'source_validation'
        check_sources(contract)
        arrays, metadata = {}, {}
        import numpy as np
        for name in ARRAYS + METADATA:
            payload, actual = read_bound(contract, name, result['opened_input_paths'])
            result['input_sha256_before'][name] = actual
            if name in ARRAYS:
                value = np.load(io.BytesIO(payload), allow_pickle=False)
                spec = contract['files'][name]
                if (not isinstance(value, np.ndarray) or list(value.shape) != spec['shape']
                        or value.dtype != np.dtype('float32') or not np.isfinite(value).all()):
                    raise ValueError(f'P shape/dtype/finite mismatch: {name}')
                arrays[name] = value
            else:
                metadata[name] = json.loads(payload)
        check_metadata(metadata, contract)
        result['panel_identity'] = contract['panel']
        result['identity_scope'] = 'historical_sha_bound_panel_attestation_not_fresh_query_semantics'
        if probe is None:
            sys.path.insert(0, str(ROOT))
            from spatial_meta_ce import s8_probe as probe
        target = probe.residual_targets(arrays['P-label.npy'], arrays['P-pg_log.npy'])
        full, small = (contract['panel'][k] for k in ('rows_per_database', 'small_rows_per_database'))
        small_indices = np.array([i * full + j for i in range(len(TRAIN)) for j in range(small)])
        for feature in ('h', 'x'):
            for panel_name in ('small_P_zero_reference', 'full_P_resource'):
                result['stage'] = f'{feature}_{panel_name}'
                write_json(output/f'{feature}-{panel_name}-started.json',
                           {'stage': result['stage'], 'status': 'started_not_certified'})
                t0 = time.monotonic()
                values = arrays[f'P-{feature}.npy']
                y = target
                if panel_name == 'small_P_zero_reference':
                    values, y = values[small_indices], target[small_indices]
                design, normalizer = probe.standardize_fit(values)
                t1 = time.monotonic()
                ls = probe.solve_ls(design, y)
                zero = {'theta': np.zeros(design.shape[1]), 'objective': float(np.mean(y * y)), 'updates': 0}
                certificate = probe.certify_pair(design, y, zero, ls)
                retained_failed = bool(
                    certificate.get('retained_subspace_pass') is False
                    or certificate.get('reference_retained_subspace_pass') is False)
                row = {}
                linalg_calls = 3
                if retained_failed:
                    # Diagnostic adds one reference lstsq and one SVD factorization.
                    reference_theta = np.linalg.lstsq(design, y, rcond=probe.RCOND)[0]
                    row['retained_subspace_diagnostic'] = retained_projection_diagnostic(
                        design, y, ls, reference_theta, probe)
                    del reference_theta
                    linalg_calls = 5
                t2 = time.monotonic()
                row.update({
                    'status': 'verified' if certificate.get('engineering_verified') is True else 'failed',
                    'engineering_verified': certificate.get('engineering_verified'),
                    'rank_stability': certificate.get('rank_stability'),
                    'strict_certificate': certificate.get('strict_certificate'),
                    'full_linear_optimality_verified': False,
                    'resource_only': True, 'coefficients_discarded': True,
                    'not_used_for_predictions_or_selection': True,
                    'reference_kind': 'zero_reference_not_Adam',
                    'claim': ('diagnostic_not_original_Adam_probe_gate' if panel_name == 'small_P_zero_reference'
                              else 'resource_only_engineering_not_G1'),
                    'rows': len(design), 'columns': design.shape[1], 'matrix_bytes': design.nbytes,
                    'standardization_seconds': t1 - t0, 'solver_certification_seconds': t2 - t1,
                    'seconds': t2 - t0, 'linalg_calls': linalg_calls, 'rank': ls['rank'],
                    'normalizer_source': {'scope': 'this_P_panel_only',
                        'fit_indices_sha256': selection_sha(full, full if panel_name == 'full_P_resource' else small),
                        'fit_rows': len(design), 'feature_columns': values.shape[1], 'dtype': 'float64', 'ddof': 0,
                        'panel_identity_digest': contract['panel']['panel_identity_digest'],
                        'input_sha256': contract['files'][f'P-{feature}.npy']['sha256'],
                        'selection': 'all_P_rows' if panel_name == 'full_P_resource' else 'first_64_per_T_db' if small == 64 else 'synthetic_fixture_prefix',
                        'normalizer_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in normalizer.items()}},
                    'certificate': certificate})
                result['features'][feature][panel_name] = row
                write_json(output / f'{feature}-{panel_name}.json', row)
                del ls, zero, design, normalizer
                if not certificate_complete(certificate):
                    raise ValueError('Numerical engineering gate failed; no fallback/retry')
                for name in ARRAYS + METADATA:
                    _, actual = read_bound(contract, name, result['opened_input_paths'])
                    result['input_sha256_after'][name] = actual
        result['stage'] = 'final_source_validation'
        check_sources(contract)
        result['status'] = 'verified_readonly_P_engineering'
    except Exception as error:  # noqa: BLE001 -- fail-closed evidence boundary, never success
        result['status'] = 'failed'
        result['error'] = f'{type(error).__name__}: {error}'
        for feature, panels in result['features'].items():
            for panel_name, row in panels.items():
                if result['stage'] == f'{feature}_{panel_name}' and row['status'] == 'not_run':
                    row.update(status='failed', stage_started=True, certificate_record_unavailable=True,
                               error=result['error'])
    finally:
        result['seconds'] = time.monotonic() - start
        result['original_inputs_unchanged'] = (result['input_sha256_before'] == result['input_sha256_after']
                                              and len(result['input_sha256_before']) == 8)
        # Byte/read log only; imported runtime libraries are tracked separately.
        result['opened_input_paths'] = sorted(set(result['opened_input_paths']))
        write_json(output / 'worker-proof.json', result)
    return result


def collect_worker_proof(contract, output):
    record_error = None
    if (output/'worker-proof.json').is_file():
        try:
            return json.loads(read_regular(output/'worker-proof.json'))
        except (OSError, ValueError) as error:
            record_error = f'{type(error).__name__}: {error}'
    proof = initial_proof(contract)
    proof.update(status='failed', original_inputs_unchanged=False,
                 stage='parent_observed_worker_exit', partial_records_recovered=False,
                 partial_record_scope='stored_worker_records_only_no_source_finalization')
    if record_error is not None:
        proof['worker_record_error'] = record_error
    for feature in ('h', 'x'):
        for panel_name in ('small_P_zero_reference', 'full_P_resource'):
            path = output/f'{feature}-{panel_name}.json'
            started = output/f'{feature}-{panel_name}-started.json'
            if started.is_file() or path.is_file():
                proof['partial_records_recovered'] = True
                proof['stage'] = f'parent_observed_{feature}_{panel_name}'
                if path.is_file():
                    try:
                        row = json.loads(read_regular(path))
                        if not stage_record_complete(row, contract, feature, panel_name):
                            raise ValueError('partial_stage_record_structure_or_contract_binding_unverified')
                        proof['features'][feature][panel_name] = row
                    except (OSError, ValueError) as error:
                        proof['features'][feature][panel_name].update(status='failed',
                            stage_started=True, certificate_record_unavailable=True,
                            record_unverified=True, observed_record_path=str(path),
                            error=f'{type(error).__name__}: {error}')
                else:
                    proof['features'][feature][panel_name].update(status='failed',
                        stage_started=True, certificate_record_unavailable=True,
                        error='worker_exit_before_stage_certificate')
    return proof


def runtime_threadpools():
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = []
        paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/' in line and any(n in line.lower() for n in ('libopenblas', 'libgomp', 'libmkl', 'libiomp', 'libblis'))}
        for path in sorted(paths):
            if not path.startswith('/gly/weilin/PRICE-master/venv/'):
                raise ValueError('Unexpected numerical runtime origin')
            symbol = 'openblas_get_num_threads64_' if 'libopenblas64_' in path else 'omp_get_max_threads' if 'libgomp' in path else None
            if symbol is None:
                raise ValueError('Numerical runtime has no frozen measurable thread counter')
            function = getattr(ctypes.CDLL(path), symbol)
            function.restype = ctypes.c_int
            pools.append({'filepath': path, 'num_threads': function(), 'counter_symbol': symbol})
    if not pools or any(type(p.get('num_threads')) is not int or not 0 < p['num_threads'] <= 4 for p in pools):
        raise ValueError('Numerical CPU thread budget unverified/exceeded')
    return pools


def install_read_guard(contract, output, contract_path):
    """Python audit guard, not an OS sandbox; reviewed native numerical libs trusted."""
    # Avoid import-time tempfile write probes outside the owned proof directory.
    tempfile.tempdir = str(output)
    inputs = {Path(contract['input_root']) / name for name in ARRAYS + METADATA}
    sources = {ROOT / name for name in SOURCE_FILES} | {contract_path}
    runtime = tuple(Path(p).resolve() for p in (sys.prefix, sys.base_prefix))
    runtime_special = {Path('/proc/self/maps'), Path('/proc/self/status'), Path('/dev/urandom'), Path('/dev/null')}
    observed = {'input_paths': [], 'source_paths': [], 'runtime_open_count': 0}
    def guard(event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.fork', 'os.exec', 'socket.connect', 'socket.bind',
                     'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.truncate',
                     'os.link', 'os.symlink', 'os.utime'):
            raise PermissionError('Read-only worker cannot spawn processes or use the network')
        if event != 'open':
            return
        path, mode, flags = args
        if isinstance(path, int):
            return
        path = _ACTIVE_OPEN.get() or Path(os.path.abspath(os.fsdecode(path)))
        if flags is not None and flags & os.O_DIRECTORY:
            if (path == Path('/') or any(path == p or path in p.parents for p in inputs | sources | {output})
                    or any(p == path or p in path.parents for p in runtime)):
                return
            raise PermissionError('Non-allowlisted directory capability')
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (flags is not None and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing:
            # dill/torch inspect stream types using /dev/null; no persistent data write.
            if path == Path('/dev/null'):
                return
            if path.parent != output or path.suffix not in ('.json', '.log'):
                raise PermissionError('Worker writes only new proof files')
        elif path in inputs:
            observed['input_paths'].append(str(path))
        elif path in sources:
            observed['source_paths'].append(str(path))
        elif path.suffix in ('.npy', '.npz', '.pt', '.pth', '.pkl'):
            raise PermissionError('H/target/checkpoint/non-P array access forbidden')
        elif path in runtime_special or any(p == path or p in path.parents for p in runtime):
            observed['runtime_open_count'] += 1
        else:
            raise PermissionError(f'Non-allowlisted worker read: {path}')
    sys.addaudithook(guard)
    return observed


def sample_process(pid):
    if sys.platform == 'linux':
        text = Path(f'/proc/{pid}/status').read_text()
        fields = {row.split(':', 1)[0]: row.split(':', 1)[1].strip() for row in text.splitlines() if ':' in row}
        return {'rss_bytes': int(fields['VmRSS'].split()[0]) * 1024,
                'address_space_bytes': int(fields['VmSize'].split()[0]) * 1024}
    # Local supervisor fixture tests only; actual --execute rejects non-Linux.
    values = subprocess.check_output(['ps', '-o', 'rss=', '-o', 'vsz=', '-p', str(pid)], text=True).split()
    return {'rss_bytes': int(values[0]) * 1024, 'address_space_bytes': int(values[1]) * 1024}


def supervise(child, output, wall_seconds=900):
    start = time.monotonic()
    def interrupted(signum, _frame):
        raise InterruptedError(f'Supervisor received signal {signum}')
    old_signals = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    record = {'attempts': 1, 'failure_reason': None, 'returncode': None, 'signal': None,
              'sampled_peak_rss_bytes': 0, 'sampled_peak_address_space_bytes': 0,
              'rss_measurement': 'sampled_not_hard_cap', 'poll_seconds': 0.05,
              'RLIMIT_AS_bytes': MEMORY, 'wall_limit_seconds': wall_seconds,
              'exit_confirmation_seconds': 0.15,
              'exit_confirmed_after_observation_failure': False,
              'cleanup_timeout_seconds': 1.0, 'cleanup_failure_reason': None,
              'child_reaped': False}
    try:
        while child.poll() is None:
            if time.monotonic() - start >= wall_seconds:
                record['failure_reason'] = 'wall_timeout'
                break
            try:
                observed = sample_process(child.pid)
            except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError) as error:
                # /proc omits counters for zombies before waitpid reports exit.
                # Confirm/reap the owned child within a fixed grace period AND
                # the remaining wall budget; never forgive a live blind worker.
                remaining = max(0.0, wall_seconds - (time.monotonic() - start))
                allowance = min(record['exit_confirmation_seconds'], remaining)
                record['observation_error'] = f'{type(error).__name__}: {error}'
                try:
                    child.wait(timeout=allowance)
                except subprocess.TimeoutExpired:
                    reason = ('wall_timeout' if remaining <= record['exit_confirmation_seconds']
                              else 'resource_observation_failed')
                    record.update(failure_reason=reason, error=str(error))
                else:
                    record['exit_confirmed_after_observation_failure'] = True
                break
            record['sampled_peak_rss_bytes'] = max(record['sampled_peak_rss_bytes'], observed['rss_bytes'])
            record['sampled_peak_address_space_bytes'] = max(record['sampled_peak_address_space_bytes'], observed['address_space_bytes'])
            if observed['rss_bytes'] > MEMORY:
                record['failure_reason'] = 'rss_stop_threshold'
                break
            if observed['address_space_bytes'] > MEMORY:
                record['failure_reason'] = 'address_space_budget'
                break
            time.sleep(0.05)
    except BaseException as error:  # noqa: BLE001 -- kill owned child even on interruption
        record.update(failure_reason='supervisor_interrupted', error=f'{type(error).__name__}: {error}')
    finally:
        # Clean the owned session even if its leader already exited; otherwise
        # a surviving descendant could outlive the supervision budget.
        try:
            os.killpg(child.pid, signal.SIGKILL)
            record['owned_group_cleanup'] = 'signal_sent'
        except ProcessLookupError:
            record['owned_group_cleanup'] = 'already_absent'
        except BaseException as error:  # noqa: BLE001 -- durable cleanup failure, not success
            record.update(owned_group_cleanup='failed',
                          cleanup_failure_reason='process_group_cleanup_failed',
                          cleanup_error=f'{type(error).__name__}: {error}')
        try:
            # This shutdown allowance never extends the worker wall budget.
            # Even an uninterruptible/zombie wait must not stall proof/handlers.
            child.wait(timeout=record['cleanup_timeout_seconds'])
        except subprocess.TimeoutExpired:
            record['cleanup_failure_reason'] = 'child_reap_timeout'
        except BaseException as error:  # noqa: BLE001 -- restore handlers on interrupted cleanup
            record.update(cleanup_failure_reason='child_reap_failed',
                          cleanup_wait_error=f'{type(error).__name__}: {error}')
        record['child_reaped'] = child.returncode is not None
        record['seconds'] = time.monotonic() - start
        if record['failure_reason'] is None:
            record['failure_reason'] = record['cleanup_failure_reason']
        if record['failure_reason'] is None and record['seconds'] >= wall_seconds:
            record['failure_reason'] = 'wall_timeout'
        record['returncode'] = child.returncode
        record['signal'] = (-child.returncode
                            if child.returncode is not None and child.returncode < 0 else None)
        if record['failure_reason'] is None:
            if child.returncode != 0:
                record['failure_reason'] = 'child_signal' if child.returncode < 0 else 'child_exit'
            elif not (output / 'worker-proof.json').is_file():
                record['failure_reason'] = 'missing_worker_proof'
        # Restore even if reaping failed; that failure must remain in evidence.
        try:
            write_json(output / 'supervisor.json', record)
        finally:
            for signum, handler in old_signals.items():
                signal.signal(signum, handler)
    return record


def check_runtime_binding(contract, numpy_version, torch_version, pools):
    if contract['evidence_kind'] != 'real_saved_P':
        return
    paths = [p.get('filepath', '') for p in pools]
    if (numpy_version != EXPECTED_RUNTIME['numpy'] or torch_version != EXPECTED_RUNTIME['torch']
            or not all(p.startswith(EXPECTED_RUNTIME['interpreter_prefix'] + '/') for p in paths)
            or not any('libopenblas64_' in p for p in paths)
            or not any('libgomp' in p for p in paths)
            or any(not ('libopenblas64_' in p or 'libgomp' in p) for p in paths)):
        raise ValueError('Historical runtime version/family/origin binding mismatch; no saved-P reads')


def verify_worker_handshake(args, contract, manifest_sha):
    if str(no_symlink(args.output)) != contract['expected_output_root']:
        raise ValueError('Worker output must be the frozen new proof directory')
    if not args.output.is_dir() or (args.output/'worker-proof.json').exists():
        raise ValueError('Worker requires fresh supervisor-created output')
    if no_symlink(args.contract) != ROOT / CONTRACT_RELATIVE:
        raise ValueError('Worker contract must be the frozen package contract')
    if args._worker_fd is None or args._parent_write_fd is None:
        raise ValueError('Worker requires inherited supervisor pipe capability')
    pid = os.getppid()
    command = read_regular(Path(f'/proc/{pid}/cmdline')).decode().split('\0')
    parent_cwd = no_symlink(Path(os.readlink(f'/proc/{pid}/cwd')))
    if ('--execute' not in command or '--_worker' in command or len(command) < 3
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or (parent_cwd/command[2]).resolve() != ROOT/ENTRYPOINT):
        raise ValueError('Worker parent is not the frozen execution supervisor')
    for option, expected in (('--contract', str(args.contract.absolute())),
                             ('--contract-sha256', manifest_sha), ('--output', str(args.output))):
        if option not in command:
            raise ValueError('Supervisor arguments do not bind this worker')
        value = command[command.index(option)+1]
        if option in ('--contract', '--output'):
            # Interpret argv relative to the authenticated parent's cwd, never
            # to the child's cwd; lexical normalization must not hide symlinks.
            path = Path(value)
            value = str(no_symlink(path if path.is_absolute() else parent_cwd/path))
        if value != expected:
            raise ValueError('Supervisor arguments do not bind this worker')
    fd = args._worker_fd
    if not stat.S_ISFIFO(os.fstat(fd).st_mode):
        raise ValueError('Worker handshake requires a pipe, not an arbitrary file')
    if os.readlink(f'/proc/self/fd/{fd}') != os.readlink(f'/proc/{pid}/fd/{args._parent_write_fd}'):
        raise ValueError('Worker pipe is not owned by the verified supervisor')
    os.set_blocking(fd, False)
    try:
        ticket = json.loads(os.read(fd, 4096))
    finally:
        os.close(fd)
    if ticket != {'parent_pid': pid, 'contract_sha256': manifest_sha,
                   'output': str(args.output), 'package': str(ROOT)}:
        raise ValueError('Inherited worker capability binding mismatch')


def install_resource_limits():
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
    if resource.getrlimit(resource.RLIMIT_AS) != (MEMORY, MEMORY):
        raise ValueError('RLIMIT_AS not installed')


def worker(contract, output, contract_path):
    start = time.monotonic()
    try:
        install_resource_limits()
        # Audit guard precedes all numerical module imports and saved-P reads.
        observed = install_read_guard(contract, output, contract_path)
        sys.path.insert(0, str(ROOT))
        from spatial_meta_ce import s8_probe
        s8_probe.torch.set_num_threads(4)
        s8_probe.torch.set_num_interop_threads(1)
        pools = runtime_threadpools()
        check_runtime_binding(contract, s8_probe.np.__version__, s8_probe.torch.__version__, pools)
        write_json(output / 'runtime.json', {'threadpools': pools, 'RLIMIT_AS': list(resource.getrlimit(resource.RLIMIT_AS)),
                   'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                   'cpu_affinity': sorted(os.sched_getaffinity(0)),
                   'numpy': s8_probe.np.__version__, 'torch': s8_probe.torch.__version__,
                   'backend_scope': 'historical_version_family_origin_not_old_library_byte_attestation'})
        with computation_guard():
            result = check_saved_p(contract, output, probe=s8_probe)
        check_runtime_binding(contract, s8_probe.np.__version__, s8_probe.torch.__version__, runtime_threadpools())
        check_sources(contract)
        write_json(output / 'access-log.json', observed)
        write_json(output / 'worker-resource.json', {'seconds': time.monotonic() - start,
                   'ru_maxrss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                   'measurement': 'Linux_worker_ru_maxrss_not_matrix_bytes'})
        return 0 if result['status'] == 'verified_readonly_P_engineering' else 1
    except Exception as error:  # noqa: BLE001 -- fail-closed evidence boundary, never success
        write_json(output / 'worker-error.json', {'status': 'failed', 'stage': 'worker_runtime_or_finalize',
                   'error': f'{type(error).__name__}: {error}', 'seconds': time.monotonic() - start})
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--contract-sha256')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--_worker-fd', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--_parent-write-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    expected_contract = ROOT / CONTRACT_RELATIVE
    if Path(os.path.abspath(args.contract)) != expected_contract:
        parser.error('Only the frozen package contract path may be read; no data/checkpoint as contract')
    payload = read_regular(args.contract)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    contract = validate_contract(json.loads(payload))
    if not args.execute and not args._worker:
        print(json.dumps({'status': 'prepared_not_executed', 'saved_P_read': False,
                         'input_paths_if_approved': [str(Path(contract['input_root']) / n) for n in ARRAYS + METADATA],
                         'contract_sha256': manifest_sha, 'budget': BUDGET, 'formal_ready': False}, indent=2))
        return 0
    if (sys.platform != 'linux' or contract['evidence_kind'] != 'real_saved_P'
            or manifest_sha != args.contract_sha256 or args.output is None
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''):
        parser.error('Execution requires Linux, real frozen P contract/SHA, new output and explicit empty CUDA')
    if str(ROOT) != contract.get('expected_package_root'):
        parser.error('Execution only allowed from the frozen new package directory')
    if args._worker:
        check_sources(contract)
        try:
            verify_worker_handshake(args, contract, manifest_sha)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return worker(contract, no_symlink(args.output), args.contract.absolute())
    output = create_output(args.output, contract)
    write_json(output / 'execution-contract.json', contract)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONSTARTUP', 'PYTHONHOME', 'LD_PRELOAD')}
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4',
               S8_READONLY_PARENT_PID=str(os.getpid()))
    try:
        # Parent and child share at most four CPUs; each process has bounded AS.
        install_resource_limits()
        check_sources(contract)
        with open_output(output / 'stdout.log') as stdout, open_output(output / 'stderr.log') as stderr:
            read_fd, write_fd = os.pipe()
            ticket = {'parent_pid': os.getpid(), 'contract_sha256': manifest_sha,
                      'output': str(output), 'package': str(ROOT)}
            os.write(write_fd, json.dumps(ticket).encode())
            child = subprocess.Popen([sys.executable, '-B', str(ROOT / ENTRYPOINT),
                '--contract', str(args.contract.absolute()), '--contract-sha256', manifest_sha,
                '--output', str(output), '--_worker', '--_worker-fd', str(read_fd),
                '--_parent-write-fd', str(write_fd)], env=env, stdout=stdout,
                stderr=stderr, start_new_session=True, cwd=ROOT, pass_fds=(read_fd,))
            os.close(read_fd)
            try:
                record = supervise(child, output)
            finally:
                os.close(write_fd)
        proof = collect_worker_proof(contract, output)
        if record['failure_reason'] is None and not proof_complete(proof, contract, output):
            record['failure_reason'] = 'incomplete_or_conflicting_worker_proof'
        if record['failure_reason'] is None:
            for name in ('runtime.json', 'worker-resource.json', 'access-log.json'):
                if not (output / name).is_file():
                    record['failure_reason'] = 'missing_runtime_evidence'
            if record['failure_reason'] is None:
                runtime = json.loads(read_regular(output / 'runtime.json'))
                usage = json.loads(read_regular(output / 'worker-resource.json'))
                if (usage['ru_maxrss_bytes'] > MEMORY or runtime['RLIMIT_AS'] != [MEMORY, MEMORY]
                        or runtime['CUDA_VISIBLE_DEVICES'] != '' or len(runtime['cpu_affinity']) > 4):
                    record['failure_reason'] = 'final_runtime_budget_failure'
        if record['failure_reason'] is not None or not proof_complete(proof, contract, output):
            proof['status'] = 'failed'
        proof['supervisor'] = record
        proof['execution_contract_sha256'] = manifest_sha
        write_json(output / 'proof.json', proof)
        return 0 if proof['status'] == 'verified_readonly_P_engineering' else 1
    except Exception as error:  # noqa: BLE001 -- fail-closed evidence boundary, never success
        failure = {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'formal_ready': False}
        write_json(output / 'parent-error.json', failure)
        proof = initial_proof(contract)
        proof.update(failure, stage='parent_startup_or_finalize')
        write_json(output / 'proof.json', proof)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
