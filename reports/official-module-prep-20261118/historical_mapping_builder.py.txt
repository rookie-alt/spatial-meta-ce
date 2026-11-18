#!/usr/bin/env python3
import argparse
import copy
import json
import math
import pickle
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlglot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_input_mapping import feature_groups_match, provenance_checks, query_layout, source_databases, write_report
from spatial_meta_ce.cli import data_settings, read_json, safe_output
from spatial_meta_ce.data import FeatureStore
from spatial_meta_ce.manifests import digest, file_hash, read_workload

RELATIONS = ('join_column_tables', 'filter_column_tables', 'fanout_join_columns')
IDENTITY = ('id', 'query_hash', 'exact_hash', 'source_line')
DEFAULT_DATABASES = ('chembl', 'legalacts', 'telstra')


def validate_output_path(path: Path, data: dict) -> Path:
    output = safe_output(path, data)
    cache = Path(data['cache_root']).resolve()
    if output == cache or cache in output.parents:
        raise ValueError('Sidecar outputs must be outside caches')
    if output.exists() or Path(path).is_symlink():
        raise FileExistsError(f'Refusing to overwrite sidecar: {path}')
    return output


def audit_report_passed(report: dict) -> bool:
    totals = report.get('totals', {})
    rows = report.get('workload_rows', 0)
    return (type(rows) is int and rows > 0
            and report.get('alignment_passed') is True
            and report.get('recreate_all_features') is True
            and report.get('inputs_unchanged') is True
            and report.get('provenance_errors') == []
            and report.get('source_feature_rows') == report.get('cache_rows') == rows
            and totals.get('rows_with_errors') == 0
            and all(totals.get(k) == rows for k in (
                'checked_rows', 'recreated_rows', 'mapping_recoverable',
                'sql_mapping_recoverable')))


def load_final_audit(path: Path, data: dict, expected_sha256: str | None = None) -> dict:
    if sqlglot.__version__ != '18.17.0':
        raise ValueError('Sidecar requires sqlglot==18.17.0')
    path = Path(path)
    if expected_sha256 is not None:
        actual_sha256 = file_hash(path / 'summary.json')
        if actual_sha256 != expected_sha256:
            raise ValueError('Trusted audit summary hash mismatch')
    summary = read_json(path / 'summary.json')
    expected = source_databases(data, None)
    reported = summary.get('reported_databases', [])
    reports = summary.get('reports', [])
    names = [r.get('database') for r in reports]
    if (not (path / 'status.txt').read_text().startswith('completed ')
            or summary.get('expected_databases') != expected
            or len(reported) != len(expected) or set(reported) != set(expected)
            or len(names) != len(expected) or set(names) != set(expected)
            or any(summary.get(k) != [] for k in (
                'missing_databases', 'duplicate_databases', 'failed_databases'))
            or not all(audit_report_passed(r) for r in reports)
            or summary.get('checked_rows') != sum(r['workload_rows'] for r in reports)):
        raise ValueError('A completed, fully aligned source audit is required')
    protocols = [read_json(p) for p in sorted(path.glob('group-*/protocol.json'))]
    audited = [db for p in protocols for db in p.get('databases', [])]
    if (len(protocols) != summary.get('groups_completed') or len(audited) != len(expected)
            or set(audited) != set(expected) or not all(
                p.get('script_sha256') == file_hash(ROOT / 'scripts/audit_input_mapping.py')
                and p.get('recreate_all_features') is True and p.get('sqlglot') == '18.17.0'
                for p in protocols)):
        raise ValueError('Audit script/protocol provenance mismatch')
    top_by_db = {r['database']: r for r in reports}
    group_by_db = {}
    for protocol_path in sorted(path.glob('group-*/protocol.json')):
        group = protocol_path.parent
        group_summary_path = group / 'summary.json'
        if not group_summary_path.exists():
            raise ValueError(f'Missing group summary: {group.name}')
        group_summary = read_json(group_summary_path)
        if (group_summary.get('databases') != read_json(group / 'protocol.json').get('databases')
                or group_summary.get('databases_completed') != len(group_summary.get('databases', []))
                or group_summary.get('failed_databases') != []):
            raise ValueError(f'Incomplete group summary: {group.name}')
        for db in group_summary['databases']:
            raw_path = group / f'{db}.json'
            if not raw_path.exists():
                raise ValueError(f'Missing database report: {db}')
            raw = read_json(raw_path)
            if not audit_report_passed(raw):
                raise ValueError(f'Database report is not fully aligned: {db}')
            stripped = {k: v for k, v in raw.items()
                        if k not in ('mapping_examples', 'by_tables', 'by_predicate', 'by_flag')}
            group_report = next((r for r in group_summary.get('reports', [])
                                 if r.get('database') == db), None)
            if group_report != stripped or top_by_db.get(db) != stripped:
                raise ValueError(f'Conflicting audit reports for database: {db}')
            group_by_db[db] = raw
    if set(group_by_db) != set(expected):
        raise ValueError('Group reports do not cover every source database')
    return group_by_db


def validate_row(row: dict, store: FeatureStore, index: int, layout: dict) -> None:
    if any(row.get(k) != store.rows[index].get(k) for k in IDENTITY):
        raise ValueError(f'Cache identity mismatch at row {index}')
    cards = [row.get('true'), row.get('pg')]
    if any(not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0 for v in cards):
        raise ValueError(f'Invalid cardinalities at row {index}')
    if not np.array_equal(np.asarray(cards, dtype=np.float64), store.cards[index]):
        raise ValueError(f'Cache cardinalities mismatch at row {index}')
    counts = np.diff(store.offsets[index:index + 2], axis=0)[0].tolist()
    if not layout['mapping_recoverable'] or layout['counts'] != counts:
        raise ValueError(f'Cache counts or mapping mismatch at row {index}')


def validate_marginals(mapping: dict, shuffled: dict) -> None:
    for key in RELATIONS:
        slices = (slice(None),) if key != 'fanout_join_columns' else (slice(0, None, 2), slice(1, None, 2))
        if any(Counter(mapping[key][s]) != Counter(shuffled[key][s]) for s in slices):
            raise ValueError(f'Shuffle marginal mismatch: {key}')
    metadata_keys = set(mapping) - set(RELATIONS)
    if {k: mapping[k] for k in metadata_keys} != {k: shuffled[k] for k in metadata_keys}:
        raise ValueError('Shuffle modified token metadata instead of associations')


def shuffle_mapping(mapping: dict, seed: int) -> dict:
    rng = random.Random(seed)
    shuffled = copy.deepcopy(mapping)
    for key in RELATIONS[:2]:
        shuffled[key] = rng.sample(mapping[key], len(mapping[key]))
    directions = [rng.sample(mapping['fanout_join_columns'][d::2],
                             len(mapping['fanout_join_columns'][d::2])) for d in (0, 1)]
    shuffled['fanout_join_columns'] = [v for pair in zip(*directions) for v in pair]
    validate_marginals(mapping, shuffled)
    return shuffled


def build_record(row: dict, seed: int = 42, layout: dict | None = None) -> dict:
    mapping = query_layout(row['sql']) if layout is None else layout
    if not mapping['mapping_recoverable']:
        raise ValueError(f'Unrecoverable mapping: {row["id"]}')
    query_seed = int(digest([seed, row['query_hash']]), 16)
    return {**{k: row[k] for k in (*IDENTITY, 'true', 'pg')},
            'source_lines': [row['source_line']], 'mapping': mapping,
            'shuffled_mapping': shuffle_mapping(mapping, query_seed)}


def merge_record(records: dict, record: dict) -> dict:
    key = record['query_hash']
    previous = records.get(key)
    if previous is None:
        return record
    if any(previous[k] != record[k] for k in ('mapping', 'shuffled_mapping', 'true', 'pg')):
        raise ValueError(f'Conflicting mapping or labels for query_hash {key}')
    return {**previous, 'source_lines': [*previous['source_lines'], record['source_line']]}


def build_records(rows: list[dict], seed: int = 42) -> list[dict]:
    records = {}
    for row in rows:
        record = build_record(row, seed)
        records[record['query_hash']] = merge_record(records, record)
    return list(records.values())


def shuffle_counts(record: dict) -> dict:
    mapping, shuffled = record['mapping'], record['shuffled_mapping']
    validate_marginals(mapping, shuffled)
    changed = any(mapping[k] != shuffled[k] for k in RELATIONS)
    eligible = any(len(set(mapping[k])) > 1 for k in RELATIONS[:2]) or any(
        len(set(mapping['fanout_join_columns'][d::2])) > 1 for d in (0, 1))
    weight = len(record['source_lines'])
    return {'rows': weight, 'unique_queries': 1, 'duplicate_rows': weight - 1,
            'changed_rows': weight if changed else 0,
            'changed_unique_queries': int(changed),
            'unchanged_rows': weight if not changed else 0,
            'unchangeable_rows': weight if not eligible else 0,
            'random_unchanged_rows': weight if eligible and not changed else 0}


def summarize_shuffle(records: list[dict]) -> dict:
    totals = Counter()
    flags = Counter()
    for record in records:
        totals.update(shuffle_counts(record))
        flags.update({flag: len(record['source_lines']) for flag in record['mapping'].get('flags', [])})
    return {**dict(totals),
            'changed_fraction': totals['changed_rows'] / totals['rows'] if totals['rows'] else 0,
            'sql_flags': dict(flags)}


def verify_source_values(row: dict, store: FeatureStore, index: int, original: dict) -> None:
    cached = [g[store.offsets[index, j]:store.offsets[index + 1, j]] for j, g in enumerate(store.groups)]
    if not feature_groups_match(original['data_features'][index], cached):
        raise ValueError(f'Current source/cache feature mismatch: {row["id"]}')


def collect_records(rows: list[dict], store: FeatureStore, original: dict, seed: int) -> list[dict]:
    # ponytail: one 50k-row database in memory; stream to SQLite if workloads grow.
    records = {}
    for index, row in enumerate(rows):
        layout = query_layout(row['sql'])
        validate_row(row, store, index, layout)
        verify_source_values(row, store, index, original)
        record = build_record(row, seed, layout)
        records[record['query_hash']] = merge_record(records, record)
    return list(records.values())


def persist_records(output: Path, records: list[dict], rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'mapping.jsonl'
    with path.open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    with path.open() as stream:
        loaded = [json.loads(line) for line in stream]
    if loaded != records:
        raise ValueError('Sidecar JSONL roundtrip mismatch')
    positions = {line: r for r in loaded for line in r['source_lines']}
    if len(positions) != len(rows):
        raise ValueError('Sidecar roundtrip did not cover every source row')
    for row in rows:
        record = positions.get(row['source_line'])
        if record is None or record['query_hash'] != row['query_hash']:
            raise ValueError('Sidecar roundtrip query_hash mismatch')


def input_paths(data: dict, db: str, store: FeatureStore) -> dict:
    source = Path(data['data_root'])
    stats = source / f'datas/statistics/pretrain/{db}'
    return {'workload': source / f'datas/workloads/pretrain/{db}/workloads.sql',
            'source_features': source / f'setup/features/pretrain/{db}/features40.pkl',
            'extractor': ROOT / 'vendor/price_features/features_tool.py',
            **{f'statistics/{p.name}': p for p in sorted(stats.glob('*.pkl'))},
            **{f'cache/{p.name}': p for p in sorted(store.path.glob('*.npy'))},
            'cache/manifest.json': store.path / 'manifest.json'}


def build_database(data: dict, database: str, report: dict, output: Path, seed: int = 42) -> dict:
    source_databases(data, [database])
    output = validate_output_path(output, data)
    if report.get('database') != database or not audit_report_passed(report):
        raise ValueError('A fully aligned database audit is required')
    started = time.perf_counter()
    store = FeatureStore(Path(data['cache_root']) / 'pretrain' / database)
    paths = input_paths(data, database, store)
    snapshots = {k: (p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}
    source = Path(data['data_root'])
    hashes, errors = provenance_checks(store, paths['workload'], paths['source_features'],
                                      source / f'datas/statistics/pretrain/{database}')
    if errors or hashes != report['input_hashes']:
        raise ValueError('Current inputs disagree with final audit provenance')
    input_hashes = {**hashes, **{k: file_hash(p) for k, p in paths.items() if k.startswith('cache/')}}
    rows = read_workload(paths['workload'], database, 'pretrain')
    if len(rows) != len(store) or len(rows) != report['workload_rows']:
        raise ValueError('Workload/cache/audit row count mismatch')
    with paths['source_features'].open('rb') as stream:
        original = pickle.load(stream)
    if len(original['data_features']) != len(rows):
        raise ValueError('Current original features row count mismatch')
    records = collect_records(rows, store, original, seed)
    summary = summarize_shuffle(records)
    if summary['sql_flags'] != report['sql_flags']:
        raise ValueError('Risk flags disagree with final audit')
    persist_records(output, records, rows)
    if any((p.stat().st_size, p.stat().st_mtime_ns) != snapshots[k] for k, p in paths.items()):
        raise ValueError('Inputs changed during CPU preflight')
    manifest = {'schema': 1, 'database': database, 'split': 'pretrain', 'shuffle_seed': seed,
                'created_at': datetime.now().astimezone().isoformat(),
                'cache_content_hash': store.manifest['content_hash'],
                'audit_report_sha256': digest(report), 'input_sha256': input_hashes,
                'builder_sha256': file_hash(Path(__file__)), 'sqlglot': sqlglot.__version__,
                'jsonl_sha256': file_hash(output / 'mapping.jsonl'),
                'interpretation': 'Query-local structure only; predicate semantics not certified',
                'shuffle': 'Independent query-hash RNG; separate join/filter table marginals and even/odd fanout-column marginals',
                'scope': 'CPU source-only preflight; no training, cache writes or target reads'}
    result = {**summary, 'database': database, 'roundtrip_passed': True,
              'inputs_unchanged': True, 'seconds': time.perf_counter() - started}
    write_report(output / 'summary.json', result)
    write_report(output / 'manifest.json', manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description='Build independent source mapping sidecars on CPU')
    parser.add_argument('--config', default='configs/data.json')
    parser.add_argument('--audit', required=True)
    parser.add_argument('--audit-summary-sha256', required=True)
    parser.add_argument('--databases', nargs='+', default=list(DEFAULT_DATABASES))
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data = data_settings(read_json(args.config), args.config)
    databases = source_databases(data, args.databases)
    audit = Path(args.audit).resolve()
    reports = load_final_audit(audit, data, args.audit_summary_sha256)
    output = validate_output_path(Path(args.output), data)
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / 'status.json', {'state': 'running', 'databases': databases})
    try:
        results = []
        for db in databases:
            result = build_database(data, db, reports[db], output / db, args.seed)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        summary = {'state': 'completed', 'databases': databases, 'reports': results,
                   'rows': sum(r['rows'] for r in results), 'shuffle_seed': args.seed,
                   'audit_summary_sha256': file_hash(audit / 'summary.json'),
                   'completed_at': datetime.now().astimezone().isoformat()}
        write_report(output / 'summary.json', summary)
        write_report(output / 'status.json', {'state': 'completed', 'rows': summary['rows']})
    except Exception as exc:
        write_report(output / 'status.json', {'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    main()
