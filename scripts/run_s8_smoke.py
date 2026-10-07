"""Bounded source-only S8 G1 engineering smoke. No formal training entrypoint."""

import argparse
import ctypes
import hashlib
import io
import json
import math
import os
import resource
import signal
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_s8_preflight as source
from scripts.s2_source_replay import digest
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.s8_contract import (
    HELD_OUT,
    TRAIN,
    VALIDATION,
    extract_representations,
    panel_indices,
)
from spatial_meta_ce.s8_probe import (
    certify_pair,
    residual_targets,
    solve_adam,
    solve_ls,
    standardize_apply,
    standardize_fit,
)

ARMS = ('A_h_adam', 'B_h_ls', 'C_x_adam', 'D_x_ls', 'F0')
MODEL = {'width': 64, 'use_pg': True, 'arch': 'sum', 'layers': 2,
         'heads': 4, 'pg_output_scale': 1.0}
TIMEOUT = 1800
MEMORY = 12 * 1024 ** 3
BATCH_SIZE = 128
THREADS = 4
DATA_ROOT = Path('/gly/weilin/spatial-meta-ce')


def timeout_failure(_signum, _frame):
    raise TimeoutError('S8 G1 wall-clock deadline exceeded')


def runtime_threadpools():
    """Use installed inspector or measured frozen Linux OpenBLAS/OpenMP counters."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        maps = Path('/proc/self/maps').read_text()
        paths = sorted({line.split()[-1] for line in maps.splitlines()
                        if '/' in line and any(name in line.lower() for name in
                        ('libopenblas', 'libgomp', 'libiomp', 'libmkl', 'libblis'))})
        pools = []
        for path in paths:
            if not path.startswith('/gly/weilin/PRICE-master/venv/'):
                raise ValueError('Unexpected BLAS/OpenMP origin in frozen server')
            if 'libopenblas64_' in path:
                symbol, api = 'openblas_get_num_threads64_', 'blas'
            elif 'libgomp' in path:
                symbol, api = 'omp_get_max_threads', 'openmp'
            else:
                raise ValueError('Unrecognized runtime thread backend; cannot certify budget')
            try:
                library = ctypes.CDLL(path)
                function = getattr(library, symbol)
                function.restype, function.argtypes = ctypes.c_int, []
                threads = function()
            except (OSError, AttributeError) as error:
                raise ValueError('Thread counter unavailable; budget unverified') from error
            pools.append({'filepath': path, 'user_api': api, 'num_threads': threads,
                          'counter_symbol': symbol, 'measurement': 'loaded-library-C-counter'})
        if not any(row['user_api'] == 'blas' for row in pools):
            raise ValueError('No authenticated NumPy BLAS thread counter')
        return pools
    return threadpool_info()


def smoke_plan():
    return {'status': 'planned', 'mode': 'smoke', 'purpose': 's8-engineering-smoke-not-research',
            'formal_ready': False, 'device': 'cpu', 'F0_updates': 2, 'F0_episodes': 8,
            'P_unique': 19456, 'probe_fit_unique': 1216, 'probe_adam_updates_each': 2,
            'representation_batch': BATCH_SIZE, 'threads': THREADS,
            'resource_profile': 'full P matrices, fixed-cutoff LS/certificate only; no selection',
            'timeout_seconds': TIMEOUT, 'address_space_bytes': MEMORY,
            'target_data_read': False, 'formal_training': False}


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def event(run_dir, name, **fields):
    with (Path(run_dir) / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'event': name, **fields}, allow_nan=False) + '\n')


def tensor_hash(model):
    sha = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        sha.update(name.encode())
        sha.update(str(value.dtype).encode())
        sha.update(str(value.shape).encode())
        sha.update(value.tobytes())
    return sha.hexdigest()


class SourceStore:
    """Literal source whitelist, authenticated arrays, H gate, and a one-db LRU."""
    def __init__(self, data_root, sources, run_dir, bindings):
        self.root = Path(data_root)
        self.sources = sources
        self.run_dir = Path(run_dir)
        self.bindings = bindings
        self.open = OrderedDict()
        self.nomination = None
        self.h_loaded = set()

    def allow_holdout(self, nomination, expected_sha):
        from spatial_meta_ce.s8_state import require_nomination

        require_nomination(nomination, expected_sha, self.bindings)
        self.nomination = (Path(nomination), expected_sha)

    def get(self, database):
        if database not in (*TRAIN, *HELD_OUT, *VALIDATION):
            raise ValueError('S8 semantic read requires a permitted source database')
        if database in HELD_OUT:
            if self.nomination is None:
                raise ValueError('H semantic access forbidden before nomination seal')
            from spatial_meta_ce.s8_state import require_nomination

            require_nomination(*self.nomination, self.bindings)
            if database not in self.h_loaded:
                event(self.run_dir, 'H_semantic_load', database=database,
                      nomination_sha256=self.nomination[1])
                self.h_loaded.add(database)
        if database not in self.open:
            self.open[database] = self._load(database)
        self.open.move_to_end(database)
        while len(self.open) > 1:
            self.open.popitem(last=False)
        return self.open[database]

    def _load(self, database):
        folder = self.root / 'artifacts/cache/pretrain' / database
        frozen = self.sources[database]
        for name, expected in frozen['file_sha256'].items():
            if source.safe_hash(folder / name) != expected:
                raise ValueError('Source cache changed before semantic read')
        # Gate has already been checked, including for H. Full duplicates now matter.
        cache = source.source_identity_metadata(folder, frozen, labels=True)
        duplicates = source.verify_token_duplicates(folder)
        offsets = source.safe_array(folder / 'offsets.npy')
        groups = [source.safe_array(folder / f'group{i}.npy') for i in range(4)]
        with source.safe_binary(folder / 'manifest.json') as stream:
            rows = json.loads(stream.read().decode('utf-8'))['rows']
        cache.update({'offsets': offsets, 'groups': groups, 'rows': rows,
                      'duplicate_token_pairs_verified': duplicates})
        return cache

    def batch(self, database, positions):
        cache = self.get(database)
        samples = []
        for position in positions:
            if type(position) not in (int, np.int64) or not 0 <= position < len(cache['indices']):
                raise ValueError('Invalid source unique position')
            raw = int(cache['indices'][position])
            samples.append({'id': cache['rows'][raw]['id'],
                            'groups': [values[cache['offsets'][raw, i]:cache['offsets'][raw + 1, i]]
                                       for i, values in enumerate(cache['groups'])],
                            'true': float(cache['true'][position]),
                            'pg': float(cache['pg'][position])})
        return batch_samples(samples, 'cpu')


def train_f0(store, caches, panels, bindings, run_dir):
    from spatial_meta_ce.s8_state import signature

    signature(bindings)
    if bindings['mode'] != 'smoke' or bindings['train_steps'] != 2:
        raise ValueError('This G1 entrypoint cannot execute formal F0 training')
    if set(panels) != set(TRAIN):
        raise ValueError('F0 requires exactly the fixed T panels')
    torch.manual_seed(42)
    model = SetEstimator(**MODEL)
    initial_hash = tensor_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(43)
    previous = hashlib.sha256(b'').hexdigest()
    counts = {'optimizer_steps': 0, 'backward_calls': 0,
              'support_role_presentations': 0, 'query_role_presentations': 0}
    with (Path(run_dir) / 'episodes.jsonl').open('x', encoding='utf-8') as stream:
        for step in range(1, 3):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(4):
                database = str(rng.choice(TRAIN))
                k = int(rng.choice([5, 10, 20, 50, 100]))
                allowed = panels[database]
                if len(allowed) != 1024 or len(set(allowed)) != 1024:
                    raise ValueError('Invalid P-only F0 panel')
                raw_candidates = np.sort(caches[database]['indices'][allowed])
                raw = rng.choice(raw_candidates, size=k + 64, replace=False)
                lookup = {int(value): i for i, value in enumerate(caches[database]['indices'])}
                positions = [lookup[int(i)] for i in raw]
                if not set(positions) <= set(allowed):
                    raise ValueError('Actual F0 sample escaped P')
                hashes = caches[database]['query_hash']
                episode = {'step': step, 'episode': counts['backward_calls'] + 1,
                           'database': database, 'support_positions': positions[:k],
                           'query_positions': positions[k:],
                           'support_hashes': [hashes[i] for i in positions[:k]],
                           'query_hashes': [hashes[i] for i in positions[k:]]}
                previous = digest({'previous': previous, 'episode': {
                    'db': database, 'support': episode['support_hashes'],
                    'query': episode['query_hashes']}})
                stream.write(json.dumps(episode) + '\n')
                batch = store.batch(database, positions)
                prediction = model(batch)
                if counts['backward_calls'] == 0 and not torch.equal(prediction, batch['pg_log']):
                    raise ValueError('Fresh F0 did not start from PG')
                loss = torch.nn.functional.smooth_l1_loss(prediction, batch['label']) / 4
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite F0 smoke loss')
                loss.backward()
                counts['backward_calls'] += 1
                counts['support_role_presentations'] += k
                counts['query_role_presentations'] += 64
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite F0 smoke gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite F0 smoke parameter')
            counts['optimizer_steps'] += 1
    model.eval()
    return model, optimizer, counts, previous, initial_hash


def checkpoint_write(path, model, optimizer, bindings, counts, sample_digest):
    from spatial_meta_ce.s8_state import signature

    payload = {'schema': 1, 'kind': 's8_fresh_F0_smoke', 'bindings': bindings,
               'bindings_sha256': signature(bindings), 'next_step': 2,
               'model_config': MODEL, 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'counts': counts,
               'sample_digest': sample_digest, 'tensor_sha256': tensor_hash(model)}
    with Path(path).open('xb') as stream:
        torch.save(payload, stream)
    return source.safe_hash(path)


def checkpoint_reload(path, expected_sha, bindings):
    from spatial_meta_ce.s8_state import signature

    with source.safe_binary(path) as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('New checkpoint byte hash mismatch; refusing pickle load')
    # Only these freshly produced, caller-hash-bound bytes are eligible for pickle.
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=False)
    if (state['schema'] != 1 or state['kind'] != 's8_fresh_F0_smoke'
            or state['bindings'] != bindings or state['bindings_sha256'] != signature(bindings)
            or state['next_step'] != 2 or state['counts']['optimizer_steps'] != 2
            or state['counts']['backward_calls'] != 8 or state['model_config'] != MODEL):
        raise ValueError('F0 checkpoint signature/endpoint/count mismatch')
    group = state['optimizer']['param_groups']
    if len(group) != 1 or group[0]['lr'] != 3e-4:
        raise ValueError('F0 checkpoint optimizer mismatch')
    for item in state['optimizer']['state'].values():
        if float(item['step']) != 2 or any(
                not torch.isfinite(value).all() for value in item.values() if isinstance(value, torch.Tensor)):
            raise ValueError('F0 checkpoint optimizer endpoint/nonfinite state')
    model = SetEstimator(**MODEL)
    if (set(state['model']) != set(model.state_dict())
            or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                   for value in state['model'].values())):
        raise ValueError('F0 checkpoint must contain finite FP32 weights')
    params = group[0]['params']
    if (len(params) != len(list(model.parameters())) or len(set(params)) != len(params)
            or set(state['optimizer']['state']) != set(params)
            or group[0]['betas'] != (0.9, 0.999) or group[0]['eps'] != 1e-8
            or group[0]['weight_decay'] != 0):
        raise ValueError('F0 checkpoint optimizer states/hyperparameters incomplete')
    for key, parameter in zip(params, model.parameters()):
        item = state['optimizer']['state'][key]
        for moment in ('exp_avg', 'exp_avg_sq'):
            if item[moment].dtype != torch.float32 or item[moment].shape != parameter.shape:
                raise ValueError('F0 checkpoint optimizer moment shape/dtype mismatch')
    model.load_state_dict(state['model'], strict=True)
    model.eval()
    if any(not torch.isfinite(p).all() for p in model.parameters()) or tensor_hash(model) != state['tensor_sha256']:
        raise ValueError('F0 checkpoint tensor fingerprint mismatch')
    return model, state


def quantile(values, percent):
    return float(np.percentile(values, percent))


def metrics_rows(rows):
    grouped = {}
    identities = set()
    for row in rows:
        key = row['database'], row['cohort'], row['query_hash']
        if key in identities or row['database'] not in (*TRAIN, *VALIDATION, *HELD_OUT):
            raise ValueError('Duplicate/invalid smoke endpoint identity')
        identities.add(key)
        true, pg = row['true'], row['pg']
        if (not math.isfinite(true) or true < 0 or not math.isfinite(pg) or pg < 0
                or type(row['n_tables']) is not int or row['n_tables'] < 1
                or set(row['predictions']) != set(ARMS)):
            raise ValueError('Invalid smoke endpoint truth/PG/tables/arms')
        label = math.log(max(true, 1))
        for prediction in row['predictions'].values():
            try:
                q = math.exp(abs(max(prediction, 0) - label))
                square = (prediction - label) ** 2
            except OverflowError as error:
                raise ValueError('Smoke Q/raw overflow') from error
            if not all(math.isfinite(value) for value in (prediction, q, square)):
                raise ValueError('Nonfinite smoke endpoint value')
        grouped.setdefault(key[:2], []).append(row)
    result = []
    for (db, cohort), items in grouped.items():
        for bucket in ('all', 'four_plus'):
            selected = items if bucket == 'all' else [r for r in items if r['n_tables'] >= 4]
            n = len(selected)
            cell = {'database': db, 'cohort': cohort, 'bucket': bucket, 'n': n,
                    'status': 'descriptive' if n >= 200 else 'below_min_cell_n'}
            if n >= 200:
                arms = {}
                for arm in ARMS:
                    raw = np.array([r['predictions'][arm] for r in selected], dtype=np.float64)
                    labels = np.log(np.maximum([r['true'] for r in selected], 1))
                    q = np.exp(np.abs(np.maximum(raw, 0) - labels))
                    error = raw - labels
                    arms[arm] = {'q50': quantile(q, 50), 'q95': quantile(q, 95),
                                 'raw_log_mae': float(np.abs(error).mean()),
                                 'raw_log_mse': float(np.mean(error * error))}
                cell['arms'] = arms
            result.append(cell)
    return result


def capture(model, store, database, positions):
    blocks = {name: [] for name in ('x', 'h', 'raw_log_pred', 'pg_log', 'label')}
    for start in range(0, len(positions), BATCH_SIZE):
        batch = store.batch(database, positions[start:start + BATCH_SIZE])
        result = extract_representations(model, batch)
        with torch.inference_mode():
            rebuilt = batch['pg_log'] + model.head[-1](result['h']).squeeze(-1)
        if torch.max(torch.abs(rebuilt - result['raw_log_pred'])).item() > 1e-4:
            raise ValueError('Frozen forward representation bridge failed')
        result['label'] = batch['label']
        for name, values in blocks.items():
            values.append(result[name].detach().cpu().numpy())
    return {name: np.concatenate(values) for name, values in blocks.items()}


def rank_positions(database, cohort, hashes, limit):
    prefix = 's8-bridge-v1' if cohort == 'V' else 's8-smoke-h-v1'
    return sorted(range(len(hashes)), key=lambda i: (
        hashlib.sha256(f'{prefix}|{database}|{hashes[i]}'.encode()).hexdigest(), hashes[i]))[:limit]


def execute(data_root, run_dir):
    from scripts.s8_independent_verify import verify_smoke
    from spatial_meta_ce.s8_state import seal_nomination, signature

    data_root, run_dir = Path(data_root).absolute(), Path(run_dir).absolute()
    if data_root != DATA_ROOT or '..' in run_dir.parts or '..' in data_root.parts:
        raise ValueError('G1 requires the configured source project root, never PRICE')
    if run_dir.parent != data_root / 'runs' or not run_dir.name.startswith('s8-g1-smoke-'):
        raise ValueError('Smoke output must be a new direct project runs/s8-g1-smoke directory')
    if any(p.is_symlink() for p in (run_dir, *run_dir.parents)):
        raise ValueError('Smoke output path alias forbidden')
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if (np.__version__ != '1.26.1' or torch.__version__ != '2.1.0+cu121'
                or tuple(sys.version_info[:3]) != (3, 11, 4)):
            raise ValueError('Frozen server S8 engineering environment required')
        torch.set_num_threads(THREADS)
        torch.set_num_interop_threads(THREADS)
        threadpools = runtime_threadpools()
        if (not any(row.get('user_api') == 'blas' for row in threadpools)
                or any(not 1 <= row.get('num_threads', 0) <= THREADS for row in threadpools)):
            raise ValueError('Actual NumPy/Torch BLAS/OpenMP threads exceed the fixed budget')
        reference = data_root / 'runs/s6-mean-tail-20261007/preflight.json'
        preflight = source.preflight(data_root, reference)
        with source.safe_binary(reference) as stream:
            sources = json.loads(stream.read().decode('utf-8'))['sources']
        metadata, panels = {}, {}
        for db in TRAIN:
            cache = source.source_identity_metadata(data_root / 'artifacts/cache/pretrain' / db,
                                                    sources[db], labels=True)
            metadata[db] = cache
            panels[db], _ = panel_indices(db, cache['query_hash'])
        bindings = {
            'schema': 1, 'mode': 'smoke', 'purpose': 's8-engineering-smoke-not-research',
            'protocol_sha256': source.safe_hash(ROOT / '实验文档/27-S8G1运行器与真实smoke协议.md'),
            'source_binding_sha256': digest(preflight['input_sha256']),
            'panel_identity_digest': digest({db: [metadata[db]['query_hash'][i] for i in panels[db]] for db in TRAIN}),
            'train_databases': list(TRAIN), 'holdout_databases': list(HELD_OUT),
            'validation_databases': list(VALIDATION), 'model_config': MODEL,
            'seed': 42, 'train_steps': 2, 'meta_batch': 4, 'query_size': 64,
            'support_sizes': [5, 10, 20, 50, 100], 'batch_size': BATCH_SIZE,
            'threads': THREADS, 'probe_fit_per_database': 64, 'probe_adam_steps': 2,
            'timeout_seconds': TIMEOUT, 'address_space_bytes': MEMORY,
            'resourcefixedfields': {'batch_size': BATCH_SIZE, 'threads': THREADS,
                                    'probe_fit_per_database': 64, 'probe_adam_steps': 2,
                                    'timeout_seconds': TIMEOUT, 'address_space_bytes': MEMORY},
        }
        signature(bindings)
        write_json(run_dir / 'contract.json', bindings)
        write_json(run_dir / 'preflight.json', preflight)
        write_json(run_dir / 'panels.json', panels)
        event(run_dir, 'source_authentication_complete')
        store = SourceStore(data_root, sources, run_dir, bindings)
        train_start = time.monotonic()
        model, optimizer, counts, sample_digest, initial_hash = train_f0(
            store, metadata, panels, bindings, run_dir)
        training_seconds = time.monotonic() - train_start
        checkpoint_sha = checkpoint_write(run_dir / 'last.pt', model, optimizer, bindings,
                                          counts, sample_digest)
        restored, _ = checkpoint_reload(run_dir / 'last.pt', checkpoint_sha, bindings)
        bridge_db = TRAIN[0]
        bridge = store.batch(bridge_db, panels[bridge_db][:128])
        with torch.inference_mode():
            original, reloaded = model(bridge), restored(bridge)
        reload_delta = float(torch.max(torch.abs(original - reloaded)))
        if reload_delta > 1e-4:
            raise ValueError('Independent F0 reload bridge failed')
        model = restored
        event(run_dir, 'F0_checkpoint_reloaded', checkpoint_sha256=checkpoint_sha,
              maximum_raw_delta=reload_delta)
        print('S8 smoke: fresh P-only F0/reload complete', flush=True)
        extracted, fit_indices, cursor = {}, [], 0
        extraction_start = time.monotonic()
        for db in TRAIN:
            extracted[db] = capture(model, store, db, panels[db])
            fit_indices.extend(range(cursor, cursor + 64))
            cursor += 1024
        p_features = {name: np.concatenate([extracted[db][name] for db in TRAIN])
                      for name in ('x', 'h', 'label', 'pg_log')}
        target_full = residual_targets(p_features['label'], p_features['pg_log'])
        fit_indices = np.asarray(fit_indices, dtype=np.int64)
        for name, value in p_features.items():
            with (run_dir / f'P-{name}.npy').open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
        fit_target = target_full[fit_indices]
        solvers, normalizers, probe_summary, resource_profile = {}, {}, {}, {}
        for feature, a, b in [('h', 'A_h_adam', 'B_h_ls'), ('x', 'C_x_adam', 'D_x_ls')]:
            design, normalizer = standardize_fit(p_features[feature][fit_indices])
            normalizers[feature] = normalizer
            t = time.monotonic()
            adam, ls = solve_adam(design, fit_target, steps=2), solve_ls(design, fit_target)
            certificate = certify_pair(design, fit_target, adam, ls)
            if not certificate['pass']:
                raise ValueError('Small-matrix smoke probe numerical verification failed')
            solvers[a], solvers[b] = adam['theta'], ls['theta']
            for arm, theta in ((a, adam['theta']), (b, ls['theta'])):
                with (run_dir / f'{arm}-theta.npy').open('xb') as stream:
                    np.save(stream, theta, allow_pickle=False)
            with (run_dir / f'{feature}-normalizer.npz').open('xb') as stream:
                np.savez(stream, **normalizer)
            probe_summary[feature] = {'fit_rows': 1216, 'parameters': design.shape[1],
                                      'adam_updates': 2, 'certificate': certificate,
                                      'seconds': time.monotonic() - t}
            # Resource-only full P target; discard fit coefficients, never select by score.
            full_design, _ = standardize_fit(p_features[feature])
            t = time.monotonic()
            resource_ls = solve_ls(full_design, target_full)
            zero_reference = {'theta': np.zeros(full_design.shape[1]),
                              'objective': float(np.mean(target_full * target_full)), 'updates': 0}
            resource_certificate = certify_pair(full_design, target_full, zero_reference, resource_ls)
            if not resource_certificate['pass']:
                raise ValueError('Resource-profile fixed-cutoff solver numerical failure')
            resource_profile[feature] = {
                'resource_only': True, 'not_used_for_predictions_or_selection': True,
                'rows': len(full_design), 'columns': full_design.shape[1],
                'matrix_bytes': full_design.nbytes, 'seconds': time.monotonic() - t,
                'linalg_calls': 3, 'rank': resource_ls['rank'],
                'full_linear_optimality_verified': False,
            }
            del full_design, resource_ls
        extraction_seconds = time.monotonic() - extraction_start
        write_json(run_dir / 'probe-smoke.json', {'purpose': 'smoke-not-research',
                   'probe_fit_rows': 1216, 'profiles': probe_summary, 'resource_profile': resource_profile})
        print('S8 smoke: P representations/probes/resource profile complete', flush=True)
        endpoints = []

        def add_predictions(db, cohort, positions, features):
            cache = metadata[db]
            predictions = {'F0': features['raw_log_pred'].astype(np.float64)}
            for feature, a, b in [('h', 'A_h_adam', 'B_h_ls'), ('x', 'C_x_adam', 'D_x_ls')]:
                design = standardize_apply(features[feature], normalizers[feature])
                for arm in (a, b):
                    predictions[arm] = features['pg_log'].astype(np.float64) + design @ solvers[arm]
            for j, position in enumerate(positions):
                endpoints.append({'database': db, 'cohort': cohort,
                                  'query_hash': cache['query_hash'][position],
                                  'true': float(cache['true'][position]), 'pg': float(cache['pg'][position]),
                                  'n_tables': int(cache['n_tables'][position]),
                                  'predictions': {arm: float(predictions[arm][j]) for arm in ARMS}})

        for db in TRAIN:
            add_predictions(db, 'P', panels[db], extracted[db])
        for db in VALIDATION:
            cache = store.get(db)
            metadata[db] = {k: cache[k] for k in ('indices', 'query_hash', 'true', 'pg', 'n_tables')}
            selected = rank_positions(db, 'V', cache['query_hash'], 256)
            add_predictions(db, 'V', selected, capture(model, store, db, selected))
        nomination_file = run_dir / 'nomination.json'
        nomination_body = seal_nomination(nomination_file, None, bindings)
        nomination_sha = hashlib.sha256(json.dumps(
            nomination_body, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode()).hexdigest()
        store.allow_holdout(nomination_file, nomination_sha)
        event(run_dir, 'nomination_sealed', file_sha256=nomination_sha)
        for db in HELD_OUT:
            cache = store.get(db)
            metadata[db] = {k: cache[k] for k in ('indices', 'query_hash', 'true', 'pg', 'n_tables')}
            selected = rank_positions(db, 'H', cache['query_hash'], 32)
            add_predictions(db, 'H', selected, capture(model, store, db, selected))
        with (run_dir / 'endpoints.jsonl').open('x', encoding='utf-8') as stream:
            for row in endpoints:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        write_json(run_dir / 'metrics.json', {'purpose': 'smoke-not-research', 'rows': metrics_rows(endpoints)})
        proof = verify_smoke(run_dir, bindings, panels, metadata)
        if not proof['passed']:
            raise ValueError('Independent smoke verification failed')
        for path, expected in preflight['input_sha256'].items():
            if source.safe_hash(path) != expected:
                raise ValueError('Historical source input changed during smoke')
        write_json(run_dir / 'verification.json', proof)
        output_hashes = {p.name: source.safe_hash(p) for p in run_dir.iterdir() if p.is_file()}
        summary = {
            'status': 'completed_smoke', 'purpose': 's8-engineering-smoke-not-research',
            'mode': 'smoke', 'formal_ready': False, 'formal_training': False,
            'F0_smoke_training': True, 'source_smoke_inference': True,
            'target_data_read': False, 'adaptation': False, 'GPU_used': False,
            'device': 'cpu', 'threads': THREADS, 'batch_size': BATCH_SIZE,
            'counts': counts, 'sample_digest': sample_digest, 'initial_tensor_sha256': initial_hash,
            'checkpoint_sha256': checkpoint_sha, 'checkpoint_reload_max_raw_delta': reload_delta,
            'endpoint_rows': len(endpoints), 'P_rows': 19456, 'V_rows': 768, 'H_rows': 128,
            'probe_fit_rows': 1216, 'resource_profile': resource_profile,
            'training_seconds': training_seconds, 'representation_probe_seconds': extraction_seconds,
            'seconds': time.monotonic() - started, 'inputs_unchanged': True,
            'independent_verification_passed': True, 'full_linear_optimality_verified': False,
            'candidate_decision': 'smoke_not_selection', 'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'output_sha256': output_hashes,
            'remaining_gates': ['formal G2-G4 runner/full Q/V/H endpoints and independent verification',
                                'explicit formal resource budget/start authorization'],
            'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                        'threadpools': threadpools},
        }
        write_json(run_dir / 'summary.json', summary)
        write_json(run_dir / 'status.json', {'status': 'completed_smoke', 'purpose': summary['purpose'],
                                           'formal_ready': False, 'summary_sha256': source.safe_hash(run_dir / 'summary.json')})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    except BaseException as error:
        path = run_dir / 'status.json'
        if not path.exists():
            write_json(path, {'status': 'failed', 'purpose': 's8-engineering-smoke-not-research',
                              'formal_ready': False, 'error': str(error), 'seconds': time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(smoke_plan(), indent=2))
        return
    if args.data_root is None or args.output is None:
        parser.error('--execute requires --data-root and --output')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error('CPU-only G1 requires explicitly empty CUDA_VISIBLE_DEVICES')
    signal.signal(signal.SIGALRM, timeout_failure)
    signal.alarm(TIMEOUT)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))
        execute(args.data_root, args.output)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
