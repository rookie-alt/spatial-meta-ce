import contextlib
import gc
import io
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .data import GROUP_DIMS, TRANSFORM_VERSION
from .manifests import digest, file_hash, read_workload

COUNT_KEYS = ('n_join_cols', 'n_fanouts', 'n_tables', 'n_filter_cols')
FIELDS = ('data_features', 'true_cards', 'pg_est_cards', *COUNT_KEYS)


def verify_alignment(rows, data):
    if len(rows) != len(data['true_cards']) or len(rows) != len(data['pg_est_cards']):
        raise ValueError('Workload/features count mismatch; regenerate with provenance')
    for key, field in (('true', 'true_cards'), ('pg', 'pg_est_cards')):
        # Original feature generator stored labels in float32. Check exact
        # float32 equality, then retain workload float64 labels in our cache.
        original = np.asarray([float(v) for v in data[field]], dtype=np.float32)
        expected = np.asarray([r[key] for r in rows], dtype=np.float32)
        if not np.array_equal(original, expected):
            raise ValueError(f'Workload/features {key} sequence mismatch; no positional guessing')


def inspect_features(data):
    if not all(k in data for k in FIELDS):
        raise ValueError('Missing PRICE feature fields')
    n = len(data['true_cards'])
    if n == 0 or any(len(data[k]) != n for k in FIELDS):
        raise ValueError('PRICE fields have inconsistent lengths')
    sizes = np.asarray([[int(data[k][i]) for k in COUNT_KEYS] for i in range(n)], dtype=np.int64)
    if (sizes < 0).any():
        raise ValueError('Negative group count')
    cards = np.asarray([[float(data[k][i]) for k in ('true_cards','pg_est_cards')] for i in range(n)])
    if not np.isfinite(cards).all() or (cards < 0).any():
        raise ValueError('Invalid labels in features')
    total = 0
    for i, parts in enumerate(data['data_features']):
        if len(parts) != 4:
            raise ValueError(f'Invalid feature group count at row {i}')
        for j, (part, dim) in enumerate(zip(parts, GROUP_DIMS)):
            if part is None:
                if sizes[i, j] != 0:
                    raise ValueError('Missing nonempty group')
                continue
            a = np.asarray(part)
            if a.size != sizes[i, j] * dim or not np.isfinite(a).all():
                raise ValueError(f'Invalid features at row {i}, group {j}')
            if j == 1 and (a < 0).any():
                raise ValueError('Negative fanout requires investigation')
            total += a.size
    return {'samples': n, 'values_checked': total, 'max_counts': sizes.max(axis=0).tolist(),
            'true_min': float(cards[:,0].min()), 'true_max': float(cards[:,0].max())}


def write_cache(path, rows, data, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite cache: {path}')
    inspection = inspect_features(data)
    verify_alignment(rows, data)
    n = len(rows)
    sizes = np.asarray([[int(data[k][i]) for k in COUNT_KEYS] for i in range(n)], dtype=np.int64)
    offsets = np.concatenate([np.zeros((1, 4), dtype=np.int64), np.cumsum(sizes, axis=0)])
    staging = path.with_name(path.name + '.building')
    staging.mkdir(parents=True, exist_ok=False)
    for j, dim in enumerate(GROUP_DIMS):
        dest = np.lib.format.open_memmap(staging/f'group{j}.npy', mode='w+', dtype=np.float32,
                                       shape=(int(offsets[-1,j]), dim))
        for i, parts in enumerate(data['data_features']):
            if sizes[i,j]:
                dest[offsets[i,j]:offsets[i+1,j]] = np.asarray(parts[j], dtype=np.float32).reshape(-1, dim)
        dest.flush()
        del dest
    np.save(staging/'offsets.npy', offsets)
    np.save(staging/'cards.npy', np.asarray([[r['true'],r['pg']] for r in rows], dtype=np.float64))
    ids = [{k: v for k, v in r.items() if k not in ('sql', 'true', 'pg')} for r in rows]
    manifest = {'schema': 1, 'transform': TRANSFORM_VERSION, 'provenance': provenance,
                'inspection': inspection, 'rows': ids,
                'unique_queries': len({r['query_hash'] for r in rows})}
    manifest['content_hash'] = digest(manifest)
    (staging/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False))
    os.rename(staging, path)
    return manifest


def lower_except_quotes(sql):
    quote = None
    output = []
    for char in sql:
        if char in "'\"" and (quote is None or quote == char):
            quote = char if quote is None else None
        output.append(char if quote else char.lower())
    return ''.join(output)


def extractor(data_root, database, split):
    from vendor.price_features.features_tool import Sql2Feature
    return Sql2Feature(database, 40, split, data_root=str(data_root))


def regenerate(rows, engine, allow_exclusions=False):
    data = {k: [] for k in FIELDS}
    accepted, failures = [], []
    for i, row in enumerate(rows):
        try:
            if row['true'] <= 0 or row['pg'] <= 0:
                raise ValueError('Legacy PRICE extractor excludes nonpositive true/PG cardinality')
            with contextlib.redirect_stdout(io.StringIO()):
                ret = engine.create_sql_features(lower_except_quotes(row['sql']))
            if ret is None:
                raise ValueError('SQL unsupported by PRICE extractor')
            parts, *counts = ret
            data['data_features'].append(parts)
            data['true_cards'].append(torch.tensor(row['true'], dtype=torch.float32))
            data['pg_est_cards'].append(torch.tensor(row['pg'], dtype=torch.float32))
            for key, value in zip(COUNT_KEYS, counts):
                data[key].append(value)
            accepted.append(row)
        except Exception as exc:
            failures.append({'id': row['id'], 'error': str(exc)})
            if not allow_exclusions:
                raise ValueError(f'Feature generation failed at {row["id"]}: {exc}') from exc
        if (i+1) % 5000 == 0:
            print(json.dumps({'generated': i+1, 'accepted': len(accepted), 'failures': len(failures)}), flush=True)
    if not accepted:
        raise ValueError('No queries could be generated')
    return accepted, data, failures


def prepare_database(data_root, cache_root, database, split, allow_exclusions=False):
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    if cache_root == data_root or data_root in cache_root.parents:
        raise ValueError('Derived caches must be outside the read-only PRICE tree')
    out = cache_root/split/database
    workload = data_root/'datas/workloads'/split/database/'workloads.sql'
    feature = data_root/'setup/features'/split/database/'features40.pkl'
    source = {'workload_sha256': file_hash(workload),
              'feature_sha256': file_hash(feature) if feature.exists() else None,
              'database': database, 'split': split, 'sqlglot': '18.17.0'}
    stats = data_root/'datas/statistics'/split/database
    source['statistics'] = {f.name:file_hash(f) for f in sorted(stats.glob('*.pkl'))}
    source['extractor_sha256'] = file_hash(Path(__file__).resolve().parent.parent/'vendor/price_features/features_tool.py')
    if (out/'manifest.json').exists():
        manifest = json.loads((out/'manifest.json').read_text())
        if manifest['provenance']['source'] != source:
            raise ValueError(f'Source changed for cache {out}; use a new cache root')
        return {'database':database,'split':split,'cached':True,**manifest['inspection'],
                'unique_queries':manifest['unique_queries'],'content_hash':manifest['content_hash']}
    start = time.perf_counter()
    rows = read_workload(workload, database, split)
    engine = None
    regenerated = False
    reason = None
    failures = []
    if feature.exists():
        # PRICE assets supplied by the user are trusted pickle inputs only.
        with feature.open('rb') as f:
            data = pickle.load(f)
        inspect_features(data)
        try:
            verify_alignment(rows, data)
        except ValueError as exc:
            reason = str(exc)
            del data
            gc.collect()
            engine = extractor(data_root, database, split)
            rows, data, failures = regenerate(rows, engine, allow_exclusions)
            regenerated = True
        else:
            engine = extractor(data_root, database, split)
            for i in sorted({0, len(rows)//2, len(rows)-1}):
                with contextlib.redirect_stdout(io.StringIO()):
                    ret = engine.create_sql_features(lower_except_quotes(rows[i]['sql']))
                if ret is None or any(int(a) != int(data[k][i]) for a,k in zip(ret[1:], COUNT_KEYS)):
                    raise ValueError(f'Sampled source alignment failed: {rows[i]["id"]}')
                for a,b in zip(ret[0], data['data_features'][i]):
                    if a is None and b is None:
                        continue
                    if a is None or b is None or not np.allclose(np.asarray(a),np.asarray(b),rtol=1e-5,atol=1e-7):
                        raise ValueError(f'Regenerated feature mismatch: {rows[i]["id"]}')
    else:
        engine = extractor(data_root, database, split)
        rows, data, failures = regenerate(rows, engine, allow_exclusions)
        regenerated = True
        reason = 'missing features'
    provenance = {'source':source,'regenerated':regenerated,'reason':reason,'excluded':failures,
                  'alignment':'full label sequence plus first/middle/last feature spot-check' if not regenerated else 'query ids recorded during extraction'}
    manifest = write_cache(out, rows, data, provenance)
    return {'database':database,'split':split,'cached':False,**manifest['inspection'],
            'unique_queries':manifest['unique_queries'],'excluded':len(failures),
            'regenerated':regenerated,'elapsed_seconds':time.perf_counter()-start,
            'content_hash':manifest['content_hash']}
