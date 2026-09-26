import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .baselines import PriceAdapter, price_log_predictions, price_support
from .checkpoints import load_checkpoint
from .data import StorePool
from .manifests import digest, validate_splits
from .meta import adapt, forward
from .metrics import summarize
from .model import SetEstimator
from .train import seed_all


def support_plan(support_store, test_store, max_k, seed, run):
    excluded = {r['query_hash'] for r in test_store.rows}
    pool = support_store.unique_indices(exclude=excluded)
    if len(pool) < max_k:
        raise ValueError('Insufficient independent support queries')
    rng = np.random.default_rng(np.random.SeedSequence([seed, run]))
    selected = rng.choice(pool, size=max_k, replace=False)
    return selected, {
        'support_pool_unique': len(pool),
        'overlapping_unique_removed': len({r['query_hash'] for r in support_store.rows} & excluded),
    }


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def latency(model, params, batch, device, repeats):
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            forward(model, params, batch)
        sync(device)
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            forward(model, params, batch)
            sync(device)
            times.append((time.perf_counter() - started) * 1000)
    return {
        'batch1_p50_ms': float(np.percentile(times, 50)),
        'batch1_p95_ms': float(np.percentile(times, 95)),
        'device': str(device),
        'includes_feature_extraction': False,
        'repeats': repeats,
    }


def _predict_rows(model, params, store, indices, device, batch_size, baseline):
    predictions = []
    truths = []
    records = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch = store.batch(batch_indices, device)
        with torch.no_grad():
            pred = batch['pg_log'].cpu().numpy() if not model else forward(model, params, batch).cpu().numpy()
        if baseline == 'price':
            pred = price_log_predictions(pred)
        values = batch['true'].cpu().tolist()
        predictions.extend(pred.tolist())
        truths.extend(values)
        records.extend((int(i), float(y), float(p)) for i, y, p in zip(batch_indices, values, pred.tolist()))
    return np.asarray(predictions), np.asarray(truths), records


def _reference_predictions(metadata, db, store, indices):
    """Load immutable zero-shot predictions from a separate run directory."""
    if not metadata:
        return None
    path = metadata.get('pg_prediction_path')
    if not path:
        return None
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if row['database'] == db and int(row['k']) == 0]
    by_id = {}
    for row in rows:
        by_id.setdefault(row['query_id'], float(row['log_prediction']))
    query_ids = [store.rows[int(i)]['id'] for i in indices]
    if any(qid not in by_id for qid in query_ids):
        raise ValueError(f'PG reference does not cover target test rows for {db}')
    return np.asarray([by_id[qid] for qid in query_ids], dtype=np.float64)


def _pairwise_metrics(pred, reference):
    if reference is None:
        return None
    pred, reference = np.asarray(pred), np.asarray(reference)
    if pred.shape != reference.shape:
        raise ValueError('Paired prediction/reference lengths differ')
    reduction = np.abs(reference) - np.abs(pred)
    return {
        'mean_log_q_reduction': float(np.mean(reduction)),
        'median_log_q_reduction': float(np.median(reduction)),
        'fraction_queries_improved': float(np.mean(reduction > 0)),
        'fraction_queries_worsened': float(np.mean(reduction < 0)),
        'paired_delta_q95': float(np.percentile(np.abs(pred), 95) - np.percentile(np.abs(reference), 95)),
    }


def evaluate(config, data_config, output_dir, device='cpu', checkpoint=None, baseline=None,
             test_limit=None, training_seed=42, reference_manifest=None):
    validate_splits(data_config['splits'])
    if any(k < 0 for k in config['shots']) or config['support_runs'] < 1 or config['latency_repetitions'] < 1:
        raise ValueError('Invalid evaluation budgets')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    seed_all(training_seed)
    pool = StorePool(data_config['cache_root'])
    model = None
    metadata = {'method': baseline or 'checkpoint', 'training_seed': training_seed}
    if checkpoint:
        if baseline:
            raise ValueError('Choose checkpoint OR baseline')
        model, state = load_checkpoint(checkpoint, device)
        metadata = state['metadata']
        if metadata.get('adapted_database'):
            raise ValueError('An already-adapted checkpoint is not an independent meta-initialization')
        exposed = set(metadata['train_databases']) | set(metadata['validation_databases'])
        if exposed & set(data_config['splits']['test']):
            raise ValueError('Checkpoint selected/trained on target databases')
        if state['model_config'] != config['model']:
            raise ValueError('Evaluation model config differs from checkpoint')
    elif baseline == 'random':
        model = SetEstimator(**config['model']).to(device)
    elif baseline == 'price':
        maxima = np.ones(4, dtype=np.int64)
        for db in data_config['splits']['test']:
            for split in ('test', 'finetune'):
                maxima = np.maximum(maxima, pool.get(split, db).manifest['inspection']['max_counts'])
        model = PriceAdapter(Path(data_config['data_root']) / 'results/model_params.pth', maxima, device)
        metadata['source_exposure'] = 'Provided PRICE weights trained on all 26 source databases; supplemental reference, not matched 23/3 initialization'
    elif baseline != 'pg':
        raise ValueError('Provide a checkpoint or baseline pg/random/price')
    if model:
        model.eval()
    reference = json.loads(Path(reference_manifest).read_text()) if reference_manifest else None
    manifest = {'schema': 1, 'config': config, 'metadata': metadata, 'databases': {}, 'episodes': []}
    results = []
    with (out / 'predictions.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['database', 'run', 'k', 'query_id', 'query_hash', 'true_cardinality', 'log_prediction'])
        for db in data_config['splits']['test']:
            support_store, test_store = pool.get('finetune', db), pool.get('test', db)
            manifest['databases'][db] = {
                'support_hash': support_store.manifest['content_hash'],
                'test_hash': test_store.manifest['content_hash'],
            }
            indices = np.arange(len(test_store))
            if test_limit:
                indices = indices[:test_limit]
            unique_positions, seen = [], set()
            for position, i in enumerate(indices):
                query_hash = test_store.rows[int(i)]['query_hash']
                if query_hash not in seen:
                    unique_positions.append(position)
                    seen.add(query_hash)
            unique_indices = indices[np.asarray(unique_positions, dtype=np.int64)]
            reference_pred = _reference_predictions(reference, db, test_store, indices)
            unique_reference = _reference_predictions(reference, db, test_store, unique_indices)
            for run in range(config['support_runs']):
                support_indices, overlap = support_plan(
                    support_store, test_store, max(config['shots']), config['support_seed'], run)
                for k in config['shots']:
                    chosen = support_indices[:k]
                    ids = [support_store.rows[int(i)]['id'] for i in chosen]
                    manifest['episodes'].append({'db': db, 'run': run, 'k': k, 'support_ids': ids, **overlap})
                    support = support_store.batch(chosen, device) if k else None
                    if baseline == 'price' and support is not None:
                        support = price_support(support)
                    sync(device)
                    started = time.perf_counter()
                    params = adapt(model, support, config['inner_lr'], config['inner_steps'] if k else 0,
                                   config['adapt_scope'], config['loss'], training=False) if model else None
                    sync(device)
                    adapt_seconds = time.perf_counter() - started
                    pred, truth, records = _predict_rows(model, params, test_store, indices, device,
                                                         config['batch_size'], baseline)
                    for idx, y, p in records:
                        row = test_store.rows[idx]
                        writer.writerow([db, run, k, row['id'], row['query_hash'], y, p])
                    unique_pred, unique_truth, _ = _predict_rows(model, params, test_store, unique_indices,
                                                                  device, config['batch_size'], baseline)
                    paired = _pairwise_metrics(pred, reference_pred) if k > 0 and reference_pred is not None else None
                    paired_unique = _pairwise_metrics(unique_pred, unique_reference) if k > 0 and unique_reference is not None else None
                    timing = latency(model, params, test_store.batch([int(indices[0])], device), device,
                                     config['latency_repetitions']) if model else None
                    record = {
                        'database': db, 'run': run, 'k': k,
                        'metrics': summarize(pred, truth),
                        'unique_query_metrics': summarize(unique_pred, unique_truth),
                        'paired_vs_reference': paired,
                        'paired_unique_vs_reference': paired_unique,
                        'reference_method': reference.get('metadata', {}).get('method') if reference else None,
                        'adapt_seconds': adapt_seconds if model else 0.0,
                        'inference': timing, 'support_ids': ids, **overlap,
                    }
                    results.append(record)
                    print(json.dumps({'database': db, 'run': run, 'k': k,
                                      'q50': record['metrics']['q50'],
                                      'q95': record['metrics']['q95'],
                                      'unique_q95': record['unique_query_metrics']['q95']}), flush=True)
    aggregate = []
    grouped = defaultdict(list)
    for row in results:
        grouped[(row['database'], row['k'])].append(row)
    for (db, k), rows in grouped.items():
        stats = {}
        for metric in ('q50', 'q90', 'q95', 'q99', 'q_mean'):
            for source, field in (('all_rows', 'metrics'), ('unique_query', 'unique_query_metrics')):
                values = [r[field][metric] for r in rows]
                stats[f'{source}_{metric}'] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        paired_rows = [r['paired_unique_vs_reference'] for r in rows if r['paired_unique_vs_reference'] is not None]
        if paired_rows:
            stats['paired_unique'] = {key: float(np.mean([r[key] for r in paired_rows]))
                                      for key in paired_rows[0]}
        aggregate.append({'database': db, 'k': k, 'across_support_runs': stats})
    report = {
        'schema': 2,
        'purpose': 'smoke-not-research' if test_limit else metadata.get('purpose', 'research'),
        'metadata': metadata, 'reference_manifest': str(reference_manifest) if reference_manifest else None,
        'primary_metric': 'unique_query_q95', 'results': results, 'aggregate': aggregate,
        'parameters': sum(p.numel() for p in model.parameters()) if model else 0,
        'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()) if model else 0,
        'manifest_hash': digest(manifest), 'feature_extraction_and_label_acquisition_timed': False,
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
