#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_meta_ce.cli import data_settings, read_json, safe_output
from spatial_meta_ce.data import GROUP_DIMS, FeatureStore
from spatial_meta_ce.features import COUNT_KEYS, extractor, lower_except_quotes
from spatial_meta_ce.manifests import file_hash, read_workload

TARGETS = {'imdb', 'stats', 'genome', 'ergastf1'}
COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
PREDICATES = (*COMPARISONS, exp.In, exp.Between, exp.Like, exp.Is)


def source_databases(data: dict, selected: list[str] | None) -> list[str]:
    sources = data['splits']['train'] + data['splits']['validation']
    databases = sources if selected is None else selected
    if len(set(databases)) != len(databases) or not databases:
        raise ValueError('Source databases must be nonempty and unique')
    if set(databases) - set(sources) or set(databases) & TARGETS:
        raise ValueError('Audit accepts source training/validation databases only')
    return list(databases)


def query_layout(sql: str) -> dict:
    tree = sqlglot.parse_one(lower_except_quotes(sql))
    flags = set()
    table_nodes = list(tree.find_all(exp.Table))
    tables = list(dict.fromkeys(t.alias_or_name for t in table_nodes))
    aliases = {t.alias_or_name: t.name for t in table_nodes}
    columns = list(dict.fromkeys(str(c) for c in tree.find_all(exp.Column)))
    if len({t.name for t in table_nodes}) != len(table_nodes):
        flags.add('self_join')
    if len(tables) != len(table_nodes):
        flags.add('duplicate_table_alias')
    if len(list(tree.find_all(exp.Select))) != 1 or list(tree.find_all(exp.Subquery)):
        flags.add('subquery')
    for kind, flag in ((exp.Or, 'or'), (exp.In, 'in'), (exp.Between, 'between'),
                       (exp.Like, 'like'), (exp.Is, 'is'), (exp.NEQ, 'neq'), (exp.Not, 'not')):
        if list(tree.find_all(kind)):
            flags.add(flag)
    if any(j.args.get('on') is not None for j in tree.find_all(exp.Join)):
        flags.add('join_on')
    where = tree.args.get('where')
    if where is None:
        flags.add('missing_where')
    predicates = list(where.find_all(*PREDICATES)) if where else []
    joins = [p for p in predicates if isinstance(p, exp.EQ)
             and isinstance(p.expression, exp.Column)]
    if any(not isinstance(p.this, exp.Column) for p in joins):
        flags.add('malformed_legacy_join')
    if any(isinstance(p, COMPARISONS[1:]) and isinstance(p.this, exp.Column)
           and isinstance(p.expression, exp.Column) for p in predicates):
        flags.add('non_equi_join')
    if any(isinstance(p, COMPARISONS) and not isinstance(p.this, exp.Column)
           and isinstance(p.expression, exp.Column) for p in predicates):
        flags.add('literal_left')
    if any(not c.table for c in tree.find_all(exp.Column)):
        flags.add('unqualified_column')
    if any(c.table and c.table not in tables for c in tree.find_all(exp.Column)):
        flags.add('unknown_column_table')
    endpoints = [(str(p.this), str(p.expression)) for p in joins]
    join_set = {c for pair in endpoints for c in pair}
    join_columns = [c for t in tables for c in columns
                    if c.split('.')[0] == t and c in join_set]
    filter_columns = [c for t in tables for c in columns
                      if c.split('.')[0] == t and c not in join_set]
    filters = [p for p in predicates if p not in joins
               and not (isinstance(p, COMPARISONS)
                        and isinstance(p.this, exp.Column)
                        and isinstance(p.expression, exp.Column))]
    filter_use = Counter(str(c) for p in filters for c in p.find_all(exp.Column))
    if join_set & set(filter_use):
        flags.add('join_filter_overlap')
    if any(n > 1 for n in filter_use.values()):
        flags.add('multiple_filters')
    predicate_columns = {str(c) for p in predicates for c in p.find_all(exp.Column)}
    if set(columns) - predicate_columns:
        flags.add('non_predicate_column')
    adjacency = {t: set() for t in tables}
    for left, right in endpoints:
        a, b = left.split('.')[0], right.split('.')[0]
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    visited, pending = set(), tables[:1]
    while pending:
        node = pending.pop()
        if node not in visited:
            visited.add(node)
            pending.extend(adjacency[node] - visited)
    if len(joins) != len(tables) - 1 or len(visited) != len(tables):
        flags.add('not_join_tree')
    invalid = {'join_on', 'missing_where', 'subquery', 'non_equi_join',
               'unqualified_column', 'unknown_column_table', 'not_join_tree',
               'duplicate_table_alias', 'malformed_legacy_join'}
    index = {c: i for i, c in enumerate(join_columns)}
    return {
        'tables': tables, 'table_names': [aliases[t] for t in tables],
        'join_columns': join_columns, 'filter_columns': filter_columns,
        'join_column_tables': [tables.index(c.split('.')[0]) for c in join_columns],
        'filter_column_tables': [tables.index(c.split('.')[0]) for c in filter_columns],
        'fanout_join_columns': [index.get(c, -1) for pair in endpoints for c in pair],
        'counts': [len(join_columns), len(joins) * 2, len(tables), len(filter_columns)],
        'flags': sorted(flags),
        'predicate_types': sorted({type(p).__name__ for p in predicates}),
        'mapping_recoverable': not bool(flags & invalid),
    }


def feature_groups_match(actual: list | tuple, expected: list | tuple) -> bool:
    if len(actual) != 4 or len(expected) != 4:
        return False
    for dim, a, b in zip(GROUP_DIMS, actual, expected):
        x = np.asarray([] if a is None else a, dtype=np.float32)
        y = np.asarray([] if b is None else b, dtype=np.float32)
        if x.size != y.size or x.size % dim or y.size % dim:
            return False
        if not np.allclose(x.reshape(-1, dim), y.reshape(-1, dim), rtol=1e-5, atol=1e-7):
            return False
    return True


def write_report(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def add_bucket(buckets: dict, name: str, errors: list[str], recoverable: bool,
               sql_recoverable: bool) -> None:
    bucket = buckets.setdefault(name, {'rows': 0, 'mapping_recoverable': 0,
                                      'sql_mapping_recoverable': 0, 'errors': Counter()})
    bucket['rows'] += 1
    bucket['mapping_recoverable'] += int(recoverable)
    bucket['sql_mapping_recoverable'] += int(sql_recoverable)
    bucket['errors'].update(errors)


def provenance_checks(store: FeatureStore, workload: Path, feature: Path,
                      stats: Path) -> tuple[dict, list[str]]:
    paths = {'workload': workload, 'source_features': feature,
             'extractor': ROOT / 'vendor/price_features/features_tool.py',
             **{f'statistics/{p.name}': p for p in sorted(stats.glob('*.pkl'))}}
    hashes = {name: file_hash(path) for name, path in paths.items()}
    source = store.manifest['provenance']['source']
    errors = [name for name, expected in (
        ('workload', source['workload_sha256']),
        ('source_features', source['feature_sha256']),
        ('extractor', source['extractor_sha256']),
    ) if hashes[name] != expected]
    errors += [name for name, value in source['statistics'].items()
               if hashes.get(f'statistics/{name}') != value]
    return hashes, errors


def audit_database(data: dict, database: str, output: Path, recreate: bool) -> dict:
    import pickle

    started = time.perf_counter()
    source = Path(data['data_root'])
    workload = source / 'datas/workloads/pretrain' / database / 'workloads.sql'
    feature = source / 'setup/features/pretrain' / database / 'features40.pkl'
    stats = source / 'datas/statistics/pretrain' / database
    store = FeatureStore(Path(data['cache_root']) / 'pretrain' / database)
    hashes, provenance_errors = provenance_checks(store, workload, feature, stats)
    inputs = [workload, feature, *sorted(stats.glob('*.pkl')),
              *sorted(store.path.glob('*.npy')), store.path / 'manifest.json']
    snapshots = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    rows = read_workload(workload, database, 'pretrain')
    with feature.open('rb') as stream:
        original = pickle.load(stream)
    engine = extractor(source, database, 'pretrain') if recreate else None
    totals, flags = Counter(), Counter()
    by_tables, by_predicate, by_flag = {}, {}, {}
    failures, examples = [], []
    identity_keys = ('id', 'query_hash', 'exact_hash', 'source_line')
    original_rows = len(original['data_features'])
    seen_pg = {}
    for i, row in enumerate(rows):
        errors = []
        previous_pg = seen_pg.setdefault(row['query_hash'], row['pg'])
        if not np.isclose(previous_pg, row['pg'], rtol=1e-6, atol=0):
            errors.append('duplicate_query_pg_conflict')
        try:
            layout = query_layout(row['sql'])
        except Exception as exc:  # noqa: BLE001 - retain every failed row/database in audit reports.
            layout = {'tables': [], 'counts': [], 'flags': ['parse_failed'],
                      'predicate_types': [], 'mapping_recoverable': False}
            errors.append('parse_failed')
            if len(failures) < 10:
                failures.append({'id': row['id'], 'error': f'{type(exc).__name__}: {exc}'[:300]})
        if i < len(store):
            cached_counts = np.diff(store.offsets[i:i + 2], axis=0)[0].tolist()
            cached = [g[store.offsets[i, j]:store.offsets[i + 1, j]]
                      for j, g in enumerate(store.groups)]
            if any(row[k] != store.rows[i].get(k) for k in identity_keys):
                errors.append('cache_identity')
            if not np.array_equal(np.asarray([row['true'], row['pg']], dtype=np.float64), store.cards[i]):
                errors.append('cache_cards')
            if layout['counts'] != cached_counts:
                errors.append('sql_cache_counts')
            if i < original_rows:
                original_counts = [int(original[k][i]) for k in COUNT_KEYS]
                if original_counts != cached_counts:
                    errors.append('source_cache_counts')
                source_cards = [np.float32(original[k][i]) for k in ('true_cards', 'pg_est_cards')]
                if not np.array_equal(source_cards, np.asarray([row['true'], row['pg']], dtype=np.float32)):
                    errors.append('source_workload_cards')
                if not feature_groups_match(original['data_features'][i], cached):
                    errors.append('source_cache_features')
            else:
                errors.append('source_row_missing')
            if recreate:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        result = engine.create_sql_features(lower_except_quotes(row['sql']))
                    if result is None:
                        errors.append('recreate_unsupported')
                    else:
                        if list(result[1:]) != cached_counts:
                            errors.append('recreate_counts')
                        if not feature_groups_match(result[0], cached):
                            errors.append('recreate_cache_features')
                        if i < original_rows and not feature_groups_match(result[0], original['data_features'][i]):
                            errors.append('recreate_source_features')
                        totals['recreated_rows'] += 1
                except Exception as exc:  # noqa: BLE001 - retain every failed row/database in audit reports.
                    errors.append('recreate_failed')
                    if len(failures) < 10:
                        failures.append({'id': row['id'], 'error': f'{type(exc).__name__}: {exc}'[:300]})
        else:
            errors.append('cache_row_missing')
        flags.update(layout['flags'])
        recoverable = layout['mapping_recoverable'] and not errors
        totals.update(errors)
        totals['checked_rows'] += 1
        totals['rows_with_errors'] += int(bool(errors))
        totals['mapping_recoverable'] += int(recoverable)
        totals['sql_mapping_recoverable'] += int(layout['mapping_recoverable'])
        totals['rows_with_risk_flags'] += int(bool(layout['flags']))
        add_bucket(by_tables, str(len(layout['tables'])), errors, recoverable,
                   layout['mapping_recoverable'])
        for name in layout['predicate_types'] or ['none']:
            add_bucket(by_predicate, name, errors, recoverable, layout['mapping_recoverable'])
        for name in layout['flags'] or ['no_risk_flag']:
            add_bucket(by_flag, name, errors, recoverable, layout['mapping_recoverable'])
        if errors and len(failures) < 10:
            failures.append({'id': row['id'], 'errors': errors})
        if recoverable and len(examples) < 2:
            examples.append({'query_hash': row['query_hash'], 'source_line': row['source_line'], **layout})
        if (i + 1) % 5000 == 0:
            progress = {'database': database, 'checked_rows': i + 1, 'rows': len(rows),
                        'seconds': time.perf_counter() - started, 'errors': totals['rows_with_errors']}
            write_report(output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    unchanged = all((p.stat().st_size, p.stat().st_mtime_ns) == snapshots[str(p)] for p in inputs)
    report = {
        'database': database, 'role': 'validation' if database in data['splits']['validation'] else 'train',
        'workload_rows': len(rows), 'source_feature_rows': original_rows, 'cache_rows': len(store),
        'unique_queries': len({r['query_hash'] for r in rows}),
        'source_feature_fields': sorted(original), 'cache_manifest_fields': sorted(store.manifest),
        'cache_row_fields': sorted(store.rows[0]), 'cache_regenerated': store.manifest['provenance']['regenerated'],
        'input_hashes': hashes, 'provenance_errors': provenance_errors,
        'inputs_unchanged': unchanged, 'recreate_all_features': recreate,
        'totals': dict(totals), 'sql_flags': dict(flags), 'by_tables': by_tables,
        'by_predicate': by_predicate, 'by_flag': by_flag,
        'first_failures': failures, 'mapping_examples': examples,
        'seconds': time.perf_counter() - started,
    }
    report['stored_alignment_passed'] = (not provenance_errors and unchanged
                                         and len(rows) == len(store) == original_rows
                                         and totals['rows_with_errors'] == 0)
    report['alignment_passed'] = (report['stored_alignment_passed'] and recreate
                                  and totals['recreated_rows'] == len(rows))
    report['bucket_denominators'] = 'by_tables is exclusive; predicate and risk buckets overlap'
    report['mapping_interpretation'] = 'SQL structure recovery and numerically aligned recovery are separate; neither certifies predicate semantics'
    write_report(output / f'{database}.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description='Source-only read-only input alignment and mapping audit')
    parser.add_argument('--config', default='configs/data.json')
    parser.add_argument('--databases', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--recreate', action='store_true')
    args = parser.parse_args()
    if sqlglot.__version__ != '18.17.0':
        raise ValueError('Audit requires sqlglot==18.17.0')
    data = data_settings(read_json(args.config), args.config)
    databases = source_databases(data, args.databases)
    output = safe_output(args.output, data)
    cache = Path(data['cache_root']).resolve()
    if output == cache or cache in output.parents:
        raise ValueError('Audit outputs must be outside caches')
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now().astimezone().isoformat()
    protocol = {'started_at': started, 'databases': databases,
                'data_splits': data['splits'], 'recreate_all_features': args.recreate,
                'scope': 'pretrain source only; no cache writes, no training, no target evaluation',
                'script_sha256': file_hash(Path(__file__)), 'sqlglot': sqlglot.__version__}
    write_report(output / 'protocol.json', protocol)
    reports = []
    for database in databases:
        try:
            report = audit_database(data, database, output, args.recreate)
        except Exception as exc:  # noqa: BLE001 - retain every failed row/database in audit reports.
            report = {'database': database, 'alignment_passed': False,
                      'fatal_error': f'{type(exc).__name__}: {exc}'}
            write_report(output / f'{database}.json', report)
            print(json.dumps(report), flush=True)
        reports.append(report)
    summary = {**protocol, 'completed_at': datetime.now().astimezone().isoformat(),
               'databases_completed': len(reports),
               'checked_rows': sum(r.get('totals', {}).get('checked_rows', 0) for r in reports),
               'mapping_recoverable': sum(r.get('totals', {}).get('mapping_recoverable', 0) for r in reports),
               'sql_mapping_recoverable': sum(r.get('totals', {}).get('sql_mapping_recoverable', 0) for r in reports),
               'failed_databases': [r['database'] for r in reports if not r['alignment_passed']],
               'reports': [{k: v for k, v in r.items() if k not in ('mapping_examples', 'by_tables', 'by_predicate', 'by_flag')} for r in reports]}
    write_report(output / 'summary.json', summary)
    print(json.dumps({'completed': True, 'checked_rows': summary['checked_rows'],
                      'failed_databases': summary['failed_databases']}), flush=True)
    if summary['failed_databases']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
