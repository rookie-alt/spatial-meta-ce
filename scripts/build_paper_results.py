"""Deterministic paper artifacts from frozen local summaries (standard library only).

Never invoke training, inference, database access, networking or historical exporters.
Input hashes are pinned to the 2026-10-07 drafting baseline; fail closed on change.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = '实验文档/results'
OFFLINE = f'{RESULTS}/offline-synthesis-20261005'
OUT = f'{RESULTS}/paper-draft-20261007'
CHAPTER = '实验文档/25-论文实验章节与图表初稿.md'
DATABASES = ('stats', 'genome', 'ergastf1')
SOURCE_DATABASES = ('chembl', 'legalacts', 'telstra')
SHOTS = (0, 5, 10, 20, 50, 100)
SEEDS = (42, 43, 44)
METRICS = ('q50', 'q90', 'q95', 'q99', 'q_mean')
METHODS = ('FOMAML-set', 'ERM-set', 'FOMAML-sum', 'ERM-sum', 'PRICE', 'PG')
TARGET_FILES = {
    'FOMAML-set': [(s, f'{RESULTS}/frozen3-20260927/s{s}-summary.json') for s in SEEDS],
    'ERM-set': [(s, f'{RESULTS}/erm-s{s}-summary.json') for s in SEEDS],
    'FOMAML-sum': [(s, f'{RESULTS}/sum-target-20260927/s{s}-summary.json') for s in SEEDS],
    'ERM-sum': [(s, f'{RESULTS}/erm-sum-target-20260927/s{s}-summary.json') for s in SEEDS],
    'PRICE': [(None, f'{RESULTS}/price-fewshot-1e-6-20260927/summary.json')],
    'PG': [(None, f'{RESULTS}/pg-summary.json')],
}
STUDIES = {
    'S4': 's4-pg-output-anchor-20261006-pidfix',
    'S5': 's5-complex-weighting-20261007',
    'S6': 's6-mean-tail-20261007',
}
ARMS = {'S4': ('A_anchor1', 'B_anchor0'), 'S5': ('A_uniform', 'B_complex2'),
        'S6': ('A_mean', 'B_mean_tail10')}
S7 = f'{RESULTS}/s7-fit-generalization-20261007'
# Scientific direct inputs pinned at the drafting baseline. AGENTS and live
# 00/23/99 progress documents are deliberately not authenticated inputs.
INPUT_SHA256 = {
    'docs/protocol.md': '4a3068e40b79e0a25e36002ef8ad1380051d655bc2795d32d13f5856a063779b',
    '实验文档/14-离线实验统一对照与论文结论.md': '63c1dbe1ea9f1389155a23b93aacb38d39f4f018918cb8ce41368cb53c6703aa',
    '实验文档/18-S4PG输出锚定单因素训练对照方案.md': 'd63c4f503f55eb87b59896e54f0379768e8829c8d48f2bad0a3f6224b53e0be3',
    '实验文档/19-S5复杂查询损失加权单因素精度提升方案.md': '4ac019d51b384aba7fc616bbdadc536439a33e633596f011f4d78b48011a1d56',
    '实验文档/20-S6平均与上尾混合损失单因素方案.md': '9420a8e98664eb037ea95fa27d1e198241c99e624eeeb854253bd4d387b6c96e',
    '实验文档/21-S7已有预测误差与源侧泛化诊断协议.md': 'da51db373a8dbb3ada71d097f08990c8d31649c4250aca93911298fc9c15aef9',
    '实验文档/22-S7误差定位与泛化诊断结果.md': 'b1ed4214daf7862ab8cfcea6de38e1d99ea8b2b07479074c12e83f09d26f6a01',
    '实验文档/results/offline-synthesis-20261005/summary.json': 'ca342c03c264288ea77731458eeb6e4abf03367a3c1edf1724b446856dea9d87',
    '实验文档/results/offline-synthesis-20261005/target-aggregate.csv': 'ccb2dd3c872a7081c93c1f51e72716a4c737c16b979896ddf87340a6ec4886b2',
    '实验文档/results/offline-synthesis-20261005/per-initialization.csv': '990aafca2b249602358a5a989171b737e156ed3c6b18f4b1319f37062683c93e',
    '实验文档/results/s4-pg-output-anchor-20261006-pidfix/summary.json': 'aafa2bd935deb5fc008982d00aedcbfecbc0a379fde2f96e95c2081e88dda6ee',
    '实验文档/results/s4-pg-output-anchor-20261006-pidfix/metrics.csv': 'ee023e6004df04e5525f19e84ce5c554c7e0182d55a30cf435f3c006c59a7e60',
    '实验文档/results/s4-pg-output-anchor-20261006-pidfix/protocol.json': '20eebcad6a950f43cb6645a52db77476de0960f85ab18d48107a27b884af19da',
    '实验文档/results/s4-pg-output-anchor-20261006-pidfix/verification.json': 'd642d778ac0d979abab1a03fcaac5233c3163b1ec1f2ad2505f8bed9303f5334',
    '实验文档/results/s4-pg-output-anchor-20261006-pidfix/status.json': '1840dbc8784e0ca00153008be8153147a95e052456e50673f54600fe2b36bf42',
    '实验文档/results/s5-complex-weighting-20261007/summary.json': 'f944da91da44faf1216508f3b224c3c31d340615910ef975024bc55beeb09106',
    '实验文档/results/s5-complex-weighting-20261007/metrics.csv': 'fd7793352f5648fc28a303ff715f031ea35193bad81822b00e14a3c89fe33261',
    '实验文档/results/s5-complex-weighting-20261007/protocol.json': '9545c35bb61862844b590cd05d07d0e187a5f62b3d6652f2be1a52b655665df8',
    '实验文档/results/s5-complex-weighting-20261007/verification.json': '95e43b77f02643143424ada3f45096a526a6047e3271bcefccc85797aca233d9',
    '实验文档/results/s5-complex-weighting-20261007/status.json': 'f9fc1f11a15dd32c54783401c1f590b30f3791078730feca21b4aa8bc024d982',
    '实验文档/results/s6-mean-tail-20261007/summary.json': '84587864eb0ab55a29f6465080436ab392dca2507d4b425ca3a1c23d182fe51f',
    '实验文档/results/s6-mean-tail-20261007/metrics.csv': '7381c3d9791fec49808eca82aef7d00c84ed535e0594c3a7fdf2f3aedb1e26be',
    '实验文档/results/s6-mean-tail-20261007/protocol.json': '1034520548f7409dd37908169e32a5f2c4b7ff00aca595a1b94e8e21f7d3baa9',
    '实验文档/results/s6-mean-tail-20261007/verification.json': 'e245a27a32e14b0cca5c67f01223c2b410c7427b33a09f42c50796aaa6ab4661',
    '实验文档/results/s6-mean-tail-20261007/status.json': '703a3beb3f8417b38d4fda0aa685a444c4897a01673cf5711ac6e89ac043c9d7',
    '实验文档/results/s7-fit-generalization-20261007/summary.json': '3fcdd07b49fb5cac304cf1b1ce06c06520395d19d7755ec172ef0c1ca5e8a0c0',
    '实验文档/results/s7-fit-generalization-20261007/overview.json': '211f5a190bc33f31c28e04e9915aab4d8987e7db27a914637ab02cefa0582538',
    '实验文档/results/s7-fit-generalization-20261007/legalacts.json': '0f4908908afe2c5df53c6a30d550d54bd2d7ef6620fe73e0ee3bf6bb0861b04d',
    '实验文档/results/s7-fit-generalization-20261007/generalization.json': 'f0094380e4ba7cff2b6e7e2cf02f9545698981813682ad9c4fe8b2e0e9f9e38a',
    '实验文档/results/s7-fit-generalization-20261007/protocol.json': '1b2b4f540f5cbb527100ff6d734af4a0f6ce8a663aa80d2fb0ffb660b6c1ca94',
    '实验文档/results/s7-fit-generalization-20261007/verification.json': '2c40e863edef7c25e399940f74bf6d4eaf9a112b1aba3a1960b3553c3a42386a',
    '实验文档/results/s7-fit-generalization-20261007/status.json': 'ae2a26d3ecdedc9a4c28fb920122c0a013fe17343e21cfe48eda5a158be61328',
    '实验文档/results/s7-fit-generalization-20261007/sample-plan.json': '0dd4e21b77327f3a4377b2fac00191ba5148feec4684ebad42dc2bcf0b7aa35e',
    '实验文档/results/s7-fit-generalization-20261007/database-matrix.csv': '851f83f4d339b27c843121caa9b9f475c96e5846feef59584870da3cc4a55453',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f'Invalid numeric evidence: {value!r}')
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, f'Invalid integer evidence: {value!r}')
    return value


def close(left, right):
    number(left)
    number(right)
    require(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10),
            f'Aggregate mismatch: {left!r} != {right!r}')


def parse_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'Non-finite JSON: {value}')

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def index_rows(rows, fields, expected):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        require(key not in result, f'Duplicate evidence row: {key}')
        result[key] = row
    require(set(result) == expected, 'Incomplete or unexpected evidence rows')
    return result


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        path = root / relative
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
                'Evidence path must stay inside repository')
        require(path.is_file(), f'Missing evidence: {relative}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f'Source SHA256 mismatch: {relative}')


def read_json(root, relative):
    return parse_json((root / relative).read_text(encoding='utf-8'))


def read_csv(root, relative):
    with (root / relative).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                'Invalid CSV header')
        rows = list(reader)
        require(all(None not in row and None not in row.values() for row in rows),
                'Incomplete or extra CSV columns')
        return rows


def validate_target_summary(summary, method, seed):
    require(summary['purpose'] == 'research' and summary['primary_metric'] == 'unique_query_q95',
            'Wrong target metric or purpose')
    if seed is not None:
        require(summary['metadata']['seed'] == seed, 'Wrong training seed')
    rows = index_rows(summary['results'], ('database', 'run', 'k'),
                      {(d, r, k) for d in DATABASES for r in range(5) for k in SHOTS})
    for (db, repeat, k), row in rows.items():
        integer(row['run'])
        integer(row['k'])
        require(row['reference_method'] == (None if method == 'PG' else 'pg'),
                'Paired reference is not PG')
        support = row['support_ids']
        require(len(support) == k and len(set(support)) == k, 'Invalid support identity')
        for smaller in SHOTS:
            if smaller < k:
                require(support[:smaller] == rows[(db, repeat, smaller)]['support_ids'],
                        'Support is not nested')
        integer(row['unique_query_metrics']['n'], 1)
        integer(row['metrics']['n'], row['unique_query_metrics']['n'])
        integer(row['support_pool_unique'], 100)
        integer(row['overlapping_unique_removed'])
        for metric in METRICS:
            number(row['unique_query_metrics'][metric], 1)
            number(row['metrics'][metric], 1)
    aggregates = index_rows(summary['aggregate'], ('database', 'k'),
                            {(d, k) for d in DATABASES for k in SHOTS})
    for (db, k), aggregate in aggregates.items():
        for metric in METRICS:
            for key, function in (('mean', statistics.mean), ('std', statistics.stdev)):
                for group, label in (('unique_query_metrics', 'unique_query'),
                                     ('metrics', 'all_rows')):
                    values = [rows[(db, r, k)][group][metric] for r in range(5)]
                    close(function(values), aggregate['across_support_runs'][f'{label}_{metric}'][key])
    return rows


def target_tables(root, frozen):
    aggregate_csv = read_csv(root, f'{OFFLINE}/target-aggregate.csv')
    expected = {(m, d, str(k)) for m in METHODS for d in DATABASES for k in SHOTS}
    csv_index = index_rows(aggregate_csv, ('method', 'database', 'k'), expected)
    frozen_index = index_rows(frozen['target_aggregate'], ('method', 'database', 'k'),
                              {(m, d, k) for m in METHODS for d in DATABASES for k in SHOTS})
    seed_csv = read_csv(root, f'{OFFLINE}/per-initialization.csv')
    seed_index = index_rows(seed_csv, ('method', 'training_seed', 'database', 'k'),
                           {(m, '' if s is None else str(s), d, str(k))
                            for m, files in TARGET_FILES.items() for s, _ in files
                            for d in DATABASES for k in SHOTS})
    canonical = None
    init_rows = []
    config = read_json(root, 'configs/data.json')
    for method, files in TARGET_FILES.items():
        for seed, relative in files:
            raw = read_json(root, relative)
            rows = validate_target_summary(raw, method, seed)
            if seed is not None:
                require(raw['metadata']['train_databases'] == config['splits']['train'] and
                        raw['metadata']['validation_databases'] == list(SOURCE_DATABASES),
                        'Training split mismatch')
            close(raw['parameter_bytes'], 4 * raw['parameters'])
            require(frozen['parameters'][method]['parameters'] == raw['parameters'],
                    'Parameter count mismatch')
            if canonical is None:
                canonical = rows
            for key, row in rows.items():
                for field in ('support_ids', 'support_pool_unique', 'overlapping_unique_removed'):
                    require(row[field] == canonical[key][field], f'Target plan mismatch: {field}')
                for group in ('metrics', 'unique_query_metrics'):
                    require(row[group]['n'] == canonical[key][group]['n'], 'Target denominator mismatch')
            for db in DATABASES:
                for k in SHOTS:
                    recorded = seed_index[(method, '' if seed is None else str(seed), db, str(k))]
                    require(recorded['source_file'] == relative and
                            recorded['source_sha256'] == frozen['source_sha256'][relative],
                            'CSV source binding mismatch')
                    require(recorded['support_runs'] == '5' and
                            int(recorded['unique_queries']) == rows[(db, 0, k)]['unique_query_metrics']['n'] and
                            int(recorded['all_rows']) == rows[(db, 0, k)]['metrics']['n'],
                            'CSV denominator mismatch')
                    entry = {'method': method, 'training_seed': seed, 'database': db, 'k': k,
                             'support_runs': 5, 'unique_queries': int(recorded['unique_queries']),
                             'source_file': relative, 'source_sha256': frozen['source_sha256'][relative]}
                    for metric in METRICS:
                        values = [rows[(db, r, k)]['unique_query_metrics'][metric] for r in range(5)]
                        for suffix, function in (('mean', statistics.mean), ('sample_std', statistics.stdev)):
                            name = f'{metric}_support_{suffix}'
                            entry[name] = function(values)
                            close(entry[name], float(recorded[name]))
                    init_rows.append(entry)
    targets = []
    for method in METHODS:
        for db in DATABASES:
            for k in SHOTS:
                selected = [r for r in init_rows if (r['method'], r['database'], r['k']) == (method, db, k)]
                recorded = frozen_index[(method, db, k)]
                tabular = csv_index[(method, db, str(k))]
                training_seeds = list(SEEDS) if method not in ('PG', 'PRICE') else []
                require(recorded['training_seeds'] == training_seeds and
                        parse_json(tabular['training_seeds']) == training_seeds and
                        recorded['training_seed_replicates'] == len(training_seeds) and
                        int(tabular['training_seed_replicates']) == len(training_seeds) and
                        recorded['support_runs_per_initialization'] == 5 and
                        tabular['support_runs_per_initialization'] == '5', 'Invalid replicate semantics')
                entry = {'method': method, 'database': db, 'k': k,
                         'unique_queries': selected[0]['unique_queries'],
                         'training_seed_replicates': len(training_seeds), 'support_runs': 5,
                         'uncertainty': ('none_fixed_pg' if method == 'PG' else
                                         'support_sd_single_checkpoint' if method == 'PRICE' else
                                         'training_seed_sample_sd')}
                for metric in METRICS:
                    values = [r[f'{metric}_support_mean'] for r in selected]
                    mean = statistics.mean(values)
                    sd = statistics.stdev(values) if len(values) == 3 else None
                    close(mean, recorded[f'{metric}_mean'])
                    close(mean, float(tabular[f'{metric}_mean']))
                    sd_field = f'{metric}_between_training_seed_sample_std'
                    if sd is None:
                        require(recorded[sd_field] is None and tabular[sd_field] == '',
                                'PRICE/PG cannot have training seed SD')
                    else:
                        close(sd, recorded[sd_field])
                        close(sd, float(tabular[sd_field]))
                    entry[f'{metric}_mean'] = mean
                    entry[f'{metric}_training_seed_sd'] = sd
                    entry[f'{metric}_support_sd'] = (selected[0][f'{metric}_support_sample_std']
                                                     if method == 'PRICE' else None)
                if method == 'PG':
                    for metric in METRICS:
                        close(entry[f'{metric}_mean'], rows[(db, 0, 0)]['unique_query_metrics'][metric])
                targets.append(entry)
    return targets, init_rows, canonical


def validate_source_summary(summary, study):
    require(summary['purpose'] == 'research-source-only-single-seed', 'Wrong source purpose')
    require(summary['decision'] == 'no_candidate' and summary['candidates'] == [],
            'Unexpected frozen source decision')
    arms = index_rows(summary['runs'], ('arm',), {(a,) for a in ARMS[study]})
    output = []
    for arm in ARMS[study]:
        run = arms[(arm,)]
        require(run['config']['seed'] == 42 and run['optimizer_steps'] == 4000,
                'Source endpoint is not seed42/4000')
        endpoint = run['endpoint']
        require(set(endpoint['per_database']) == set(SOURCE_DATABASES), 'Incomplete source databases')
        for db in SOURCE_DATABASES:
            require(set(endpoint['per_database'][db]) == {'all_tables', 'four_plus_tables'},
                    'Incomplete source buckets')
            for bucket, cell in endpoint['per_database'][db].items():
                output.append({'study': study, 'arm': arm, 'training_seed': 42, 'database': db,
                               'bucket': bucket, 'queries': integer(cell['queries'], 1),
                               'q50': number(cell['q50'], 1), 'q95': number(cell['q95'], 1)})
        require(endpoint['unique_queries'] == sum(endpoint['per_database'][d]['all_tables']['queries']
                                                for d in SOURCE_DATABASES), 'Source denominator mismatch')
        for bucket, name in (('all_tables', 'macro_all_table_q95'),
                             ('four_plus_tables', 'macro_four_plus_q95')):
            close(statistics.mean(endpoint['per_database'][d][bucket]['q95'] for d in SOURCE_DATABASES),
                  endpoint[name])
    a, b = (arms[(arm,)] for arm in ARMS[study])
    for field in ('sample_plan_digest', 'validation_plan_hash', 'initialization_hash'):
        require(a[field] == b[field], f'Source pairing mismatch: {field}')
    for db in SOURCE_DATABASES:
        for bucket in ('all_tables', 'four_plus_tables'):
            require(a['endpoint']['per_database'][db][bucket]['queries'] ==
                    b['endpoint']['per_database'][db][bucket]['queries'], 'Source A/B counts mismatch')
    improvement = 1 - b['endpoint']['macro_four_plus_q95'] / a['endpoint']['macro_four_plus_q95']
    require(math.isclose(improvement, summary['relative_improvement'], abs_tol=1e-10),
            'Source relative improvement mismatch')
    return output


def source_tables(root):
    endpoints, tradeoffs = [], []
    baseline = None
    for study, directory in STUDIES.items():
        summary = read_json(root, f'{RESULTS}/{directory}/summary.json')
        require(read_json(root, f'{RESULTS}/{directory}/verification.json')['passed'] is True and
                read_json(root, f'{RESULTS}/{directory}/status.json')['status'] == 'completed',
                'Source result not certified')
        rows = validate_source_summary(summary, study)
        expected = {(r['arm'], r['database'], r['bucket']) for r in rows}
        csv_rows = read_csv(root, f'{RESULTS}/{directory}/metrics.csv')
        if study == 'S4':
            staged = index_rows(csv_rows, ('arm', 'database', 'bucket', 'stage'),
                                {(*key, stage) for key in expected for stage in ('initial', 'final')})
            for run in summary['runs']:
                for db in SOURCE_DATABASES:
                    for bucket in ('all_tables', 'four_plus_tables'):
                        initial = staged[(run['arm'], db, bucket, 'initial')]
                        recorded = run['initial_endpoint']['per_database'][db][bucket]
                        require(int(initial['queries']) == recorded['queries'], 'S4 initial counts mismatch')
                        for metric in ('q50', 'q95'):
                            close(float(initial[metric]), recorded[metric])
            csv_rows = [r for r in csv_rows if r['stage'] == 'final']
        metrics = index_rows(csv_rows, ('arm', 'database', 'bucket'), expected)
        for row in rows:
            other = metrics[(row['arm'], row['database'], row['bucket'])]
            require(int(other['queries']) == row['queries'], 'Source CSV counts mismatch')
            for metric in ('q50', 'q95'):
                close(row[metric], float(other[metric]))
        endpoints.extend(rows)
        arms = summary['runs']
        base = next(r for r in arms if r['arm'] == ARMS[study][0])['endpoint']
        if baseline is None:
            baseline = base['per_database']
        require(base['per_database'] == baseline, 'S4/S5/S6 A baseline differs')
        for db in (*SOURCE_DATABASES, 'macro'):
            cells = []
            for arm in ARMS[study]:
                chosen = [r for r in rows if r['arm'] == arm and r['bucket'] == 'four_plus_tables'
                          and (db == 'macro' or r['database'] == db)]
                cells.append(statistics.mean(r['q95'] for r in chosen))
            tradeoffs.append({'study': study, 'database': db, 'training_seed': 42,
                              'a_q95': cells[0], 'b_q95': cells[1],
                              'relative_change': cells[1] / cells[0] - 1,
                              'decision': summary['decision'],
                              'aggregation': ('equal_database_arithmetic_macro' if db == 'macro'
                                              else 'unique_query_four_plus_q95')})
    return endpoints, tradeoffs


def load_evidence(root=ROOT):
    root = Path(root)
    verify_hashes(root, INPUT_SHA256)
    frozen = read_json(root, f'{OFFLINE}/summary.json')
    originals = frozen['source_sha256']
    require(len(originals) == 36, 'Expected original 36 source hashes from 14 synthesis')
    verify_hashes(root, originals)
    targets, initializations, canonical = target_tables(root, frozen)
    endpoints, tradeoffs = source_tables(root)
    s7_summary = read_json(root, f'{S7}/summary.json')
    s7_verify = read_json(root, f'{S7}/verification.json')
    require(s7_summary['status'] == 'completed' and s7_verify['passed'] is True and
            s7_summary['s6_decision'] == 'no_candidate' and
            s7_summary['causal_explanation_verified'] is False, 'S7 not certified/descriptive')
    overview = read_json(root, f'{S7}/overview.json')
    verify_hashes(root, {f'{S7}/{name}': sha for name, sha in overview['source_sha256'].items()})
    legalacts = read_json(root, f'{S7}/legalacts.json')
    tails = index_rows(legalacts['tail_groups'], ('group',),
                       {(g,) for g in ('persistent', 'entered', 'left', 'neither')})
    require(sum(integer(g['n']) for g in tails.values()) == legalacts['four_plus']['n'] == 9178,
            'S7 tail denominator mismatch')
    tail_rows = []
    for cell in tails.values():
        require(sum(cell['b_error_direction_counts'].values()) == cell['n'] and
                sum(cell['cardinality_counts'].values()) == cell['n'], 'S7 tail composition mismatch')
        tail_rows.append({'group': cell['group'], 'unique_queries': cell['n'],
                          'b_under': cell['b_error_direction_counts']['under'],
                          'b_over': cell['b_error_direction_counts']['over'],
                          'b_worse_fraction': cell['stats'].get('worse_fraction_b_vs_a'),
                          'a_q95': cell['stats'].get('A', {}).get('q95'),
                          'b_q95': cell['stats'].get('B', {}).get('q95'),
                          'status': cell['stats']['status']})
    plans = []
    for db in DATABASES:
        row = canonical[(db, 0, 0)]
        plans.append({'database': db, 'all_rows': row['metrics']['n'],
                      'unique_queries': row['unique_query_metrics']['n'],
                      'support_pool_unique': row['support_pool_unique'],
                      'overlapping_unique_removed': row['overlapping_unique_removed']})
    hashes = {**INPUT_SHA256, **originals}
    verify_hashes(root, hashes)
    return {'targets': targets, 'initializations': initializations, 'plans': plans,
            'source_endpoints': endpoints, 'source_tradeoffs': tradeoffs,
            'frozen': frozen, 's7_summary': s7_summary, 's7_overview': overview,
            's7_legalacts': legalacts, 's7_tails': tail_rows,
            'original_36_sha256': originals, 'input_sha256': hashes}


def csv_text(rows, fields=None):
    require(bool(rows), 'Empty table')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields or list(rows[0]), lineterminator='\n',
                            extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'


def svg_start(title, description, width=768, height=560):
    return [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'),
            f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="sans-serif" font-size="14" fill="#222">',
            f'<text x="70" y="30" font-size="20">{escape(title)}</text>']


def text(x, y, value, **attrs):
    attributes = ' '.join(f'{k.replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {attributes}>{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ''
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="1.6"{dash}/>')


def target_svg(data, db):
    methods = ('FOMAML-sum', 'ERM-sum', 'PRICE', 'PG')
    colors = ('#0072b2', '#d55e00', '#009e73', '#666666')
    rows = [r for r in data['targets'] if r['database'] == db and r['method'] in methods]
    intervals = [(r['q95_mean'], r['q95_training_seed_sd'] if r['q95_training_seed_sd'] is not None
                  else r['q95_support_sd'] or 0) for r in rows]
    low = max(1, min(m - s for m, s in intervals) * 0.85)
    high = max(m + s for m, s in intervals) * 1.2
    def y(v):
        return 420 - (math.log10(v) - math.log10(low)) / (math.log10(high) - math.log10(low)) * 340

    # Categorical K spacing retains all K without implying a linear support-size distance.
    def x(k):
        return 90 + SHOTS.index(k) * 88
    parts = svg_start(f'{db}: unique-query Q95 (lower is better)',
                      'All K=0,5,10,20,50,100. Categorical K axis; logarithmic Q95 axis. '
                      'Solid caps: sample SD of three training-seed means. Dashed caps: PRICE '
                      'support SD across five repeats of one checkpoint; not seed SD. PG: no SD.')
    for i in range(6):
        v = 10 ** (math.log10(low) + i / 5 * (math.log10(high) - math.log10(low)))
        parts += [line(75, y(v), 550, y(v), '#dddddd'), text(67, y(v) + 4, f'{v:.1f}', text_anchor='end')]
    parts += [text(75, 60, 'Q95 (log scale)'), line(75, 420, 550, 420, '#333333')]
    for k in SHOTS:
        parts.append(text(x(k), 448, k, text_anchor='middle'))
    parts.append(text(310, 477, 'Support size K (categorical spacing)', text_anchor='middle'))
    for method, color in zip(methods, colors):
        selected = [r for r in rows if r['method'] == method]
        coords = ' '.join(f'{x(r["k"]):.2f},{y(r["q95_mean"]):.2f}' for r in selected)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for r in selected:
            xx, mean = x(r['k']), r['q95_mean']
            sd = r['q95_training_seed_sd'] if method != 'PRICE' else r['q95_support_sd']
            if sd is not None:
                require(mean - sd > 0, 'Cannot show nonpositive SD endpoint on log scale')
                dash = method == 'PRICE'
                parts += [line(xx, y(mean - sd), xx, y(mean + sd), color, dash),
                          line(xx - 4, y(mean - sd), xx + 4, y(mean - sd), color, dash),
                          line(xx - 4, y(mean + sd), xx + 4, y(mean + sd), color, dash)]
            parts.append(f'<circle cx="{xx:.2f}" cy="{y(mean):.2f}" r="4" fill="{color}">'
                         f'<title>K={r["k"]}; mean={mean:.6f}; SD={sd}; {r["uncertainty"]}</title></circle>')
        legend_y = 100 + methods.index(method) * 58
        parts += [line(575, legend_y, 599, legend_y, color), text(604, legend_y + 5, method, font_size=12),
                  text(575, legend_y + 25, 'support SD' if method == 'PRICE' else
                       'no SD' if method == 'PG' else 'training seed SD', font_size=11)]
    parts += [text(75, 511, 'Solid caps: mean ± training seed SD (n=3); dashed caps: PRICE support SD (n=5).', font_size=13),
              text(75, 535, 'PG is fixed. SD is descriptive, not a confidence interval or significance test.', font_size=13),
              '</g></svg>']
    return '\n'.join(parts) + '\n'


def source_svg(data):
    parts = svg_start('S4/S5/S6: source-only 4+ table Q95 trade-offs',
                      'Seed42 only, 4000 updates, no error bars. Relative B/A Q95 change for '
                      'chembl/legalacts/telstra and equal-database arithmetic macro, not pooled Q95. '
                      'All three studies: no_candidate. Negative means improvement.', height=600)
    rows = data['source_tradeoffs']
    low = min(-20, *(r['relative_change'] * 100 for r in rows)) - 10
    high = max(20, *(r['relative_change'] * 100 for r in rows)) + 15
    def x(v):
        return 210 + (v - low) / (high - low) * 480

    for v in (-10, 0, 10):
        parts += [line(x(v), 65, x(v), 505, '#666666', v != 0), text(x(v), 532, f'{v:+d}%', text_anchor='middle')]
    colors = ('#0072b2', '#d55e00', '#009e73')
    for i, row in enumerate(rows):
        yy = 83 + i * 35
        val = row['relative_change'] * 100
        color = colors[('S4', 'S5', 'S6').index(row['study'])]
        parts += [text(195, yy + 5, f'{row["study"]} / {row["database"]}', text_anchor='end'),
                  line(x(0), yy, x(val), yy, color),
                  f'<circle cx="{x(val):.2f}" cy="{yy}" r="4" fill="{color}"/>',
                  text(x(val) + 8, yy - 8, f'{val:+.2f}%', font_size=12)]
    parts += [text(75, 562, 'Single seed; no SD. Macro is the mean of database Q95 values, not a pooled quantile.', font_size=13),
              text(75, 585, 'Guides: -10% macro improvement / +10% per-database Q95 limit; Q50 guards not shown.', font_size=13),
              '</g></svg>']
    return '\n'.join(parts) + '\n'


def artifact_payloads(data):
    payloads = {}
    for metric in ('q95', 'q50'):
        fields = ['method', 'database', 'k', 'unique_queries', 'training_seed_replicates', 'support_runs',
                  f'{metric}_mean', f'{metric}_training_seed_sd', f'{metric}_support_sd', 'uncertainty']
        payloads[f'target-{metric}.csv'] = csv_text(data['targets'], fields)
    for name, key in [('target-seed-support.csv', 'initializations'), ('evaluation-plan.csv', 'plans'),
                      ('source-endpoints.csv', 'source_endpoints'), ('source-tradeoffs.csv', 'source_tradeoffs'),
                      ('s7-tail-groups.csv', 's7_tails')]:
        payloads[name] = csv_text(data[key])
    frozen = data['frozen']
    parameters = [{'method': m, **v, 'price_parameter_ratio':
                   frozen['parameters']['PRICE']['parameters'] / v['parameters']}
                  for m, v in frozen['parameters'].items() if m != 'PG']
    payloads['parameters.csv'] = csv_text(parameters)
    payloads['architecture-source.csv'] = csv_text(frozen['architecture_source_selection'])
    payloads['p1-mapping.csv'] = csv_text(frozen['p1']['per_seed'])
    payloads['p0-budget.csv'] = csv_text([
        {'budget': int(k), 'presentations': v['counts']['query_presentations'],
         'optimizer_steps': v['counts']['optimizer_steps']} for k, v in frozen['p0']['budget_summaries'].items()])
    strata = []
    for cell in data['s7_legalacts']['strata']:
        strata.append({'axis': cell['axis'], 'bucket': cell['bucket'], 'n': cell['n'], 'status': cell['status'],
                       'a_q95': cell.get('A', {}).get('q95'), 'b_q95': cell.get('B', {}).get('q95'),
                       'relative_change': cell.get('q95_relative_change_b_vs_a')})
    payloads['s7-legalacts-strata.csv'] = csv_text(strata)
    payloads['s7-generalization-ratios.csv'] = csv_text(data['s7_overview']['within_database_ratios'])
    payloads['s7-generalization-directions.csv'] = csv_text(data['s7_overview']['b_vs_a_direction_counts'])
    for db in DATABASES:
        payloads[f'target-{db}-q95.svg'] = target_svg(data, db)
    payloads['source-s4-s5-s6-tradeoffs.svg'] = source_svg(data)
    payloads['source-bindings.json'] = json_text({
        'schema': 1, 'purpose': 'paper-draft-recorded-evidence-only', 'baseline_commit': '8fe6de4',
        'original_14_source_count': 36, 'original_14_source_sha256': data['original_36_sha256'],
        'input_sha256': data['input_sha256'],
        'aggregation': {'target': '5 support means per training seed, then mean and ddof=1 SD over seeds42/43/44',
                        'price': 'single checkpoint, 5 support repeats; no training seed SD',
                        'source': 'seed42 only, equal-database arithmetic macro; no pooling'},
        'output_sha256': {name: hashlib.sha256(content.encode()).hexdigest()
                          for name, content in sorted(payloads.items())},
        'chapter_sha256': hashlib.sha256(render_chapter(data).encode()).hexdigest(),
        'limitations': ['No new experiment or raw target data certification',
                        'IMDB incomplete; P2 real semantics/selectivity blocked',
                        'No causal or significance claim; no matched PRICE training or end-to-end timing'],
    })
    return payloads


def reject_path_alias(path):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)),
            'Output must not traverse a symlink ancestor or path alias')
    require(not path.exists() or (path.is_file() and path.stat().st_nlink == 1),
            'Output must not overwrite a path alias')


def export(data, out):
    # Validate/render every payload before creating any output, so evidence failures
    # cannot leave a success manifest or partially refreshed scientific table.
    payloads = artifact_payloads(data)
    out = Path(out).absolute()
    require(not any(part.is_symlink() for part in (out, *out.parents)),
            'Output must not traverse a symlink ancestor or path alias')
    out.mkdir(parents=True, exist_ok=True)
    for name in payloads:
        path = out / name
        reject_path_alias(path)
    for name, content in payloads.items():
        (out / name).write_text(content, encoding='utf-8')
    return payloads


def build(root=ROOT, out=None):
    root = Path(root)
    data = load_evidence(root)
    chapter = render_chapter(data)
    payloads = artifact_payloads(data)
    verify_hashes(root, data['input_sha256'])
    if out is None:
        reject_path_alias(root / CHAPTER)
    export(data, root / OUT if out is None else out)
    if out is None:
        (root / CHAPTER).write_text(chapter, encoding='utf-8')
    return {'status': 'success', 'original_14_sources_checked': 36,
            'input_files_checked': len(data['input_sha256']), 'target_cells': len(data['targets']),
            'target_episode_records': 1260, 'source_endpoint_cells': len(data['source_endpoints']),
            'artifacts': len(payloads), 'chapter_written': out is None}


def markdown_table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |',
            '| ' + ' | '.join('---' for _ in headers) + ' |',
            *['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows], '']


def render_chapter(data):
    """Complete experimental chapter, not a fabricated full thesis/template."""
    frozen = data['frozen']
    lookup = {(r['method'], r['database'], r['k']): r for r in data['targets']}
    lines = ['# 25 论文实验章节与图表初稿', '',
             ('> 编排日期：2026-10-07；证据基准：`8fe6de4`。本章仅编排既有正式离线结果，'
             '不包含新训练、推理、适应、目标评估或数据库计数。本文是完整实验章初稿，不是完整毕业论文，'
             '也不假定学校模板、论文总章号或新文献。'), '',
             '## 1. 实验目的与设置', '',
             ('本研究考察空间漂移下的监督式跨数据库少样本基数估计：在源数据库学习共享初始化，'
             '在未参与训练的目标数据库使用少量有真实基数标签的查询进行适应。'
             '实验关注尾部误差、适应的正负迁移以及模型规模，不研究时间漂移、上下文学习或 PostgreSQL 插件。'
             '下文将历史目标评估、源侧受控消融和后验诊断分别呈现，避免将不同端点合并成一个效果结论。'), '',
             '### 1.1 数据划分与适应协议', '',
             ('26 个源数据库中，23 个用于训练，`chembl/legalacts/telstra` 仅用于源验证和方法选择。'
             '目标名单为 `imdb/stats/genome/ergastf1`，不进入训练、超参数、适应策略或 checkpoint 选择。'
             '本章有完整 few-shot 结果的目标仅为后三库，IMDB 的完整支持集尚不可认证，不能写成四库验收完成。'), '',
             ('目标 support 来自独立 finetune workload，test 来自测试 workload；按保守 SQL 规范化身份去重，'
             '支持池排除与 test 重叠的查询。规范化身份不保证穷尽 SQL 语义等价。'
             '每个 repeat 使用嵌套支持集合，全部保留 K=0/5/10/20/50/100，不根据目标误差选择最佳 K。'), '',
             '**表 1 目标评估规模与支持池。** 以下分母来自既有结果包，不是本轮读取原始目标数据后的重新认证。', '']
    lines += markdown_table(['目标库', 'all-rows', 'unique-query', 'support 去重数', '重叠排除数'],
                            [[r['database'], r['all_rows'], r['unique_queries'],
                              r['support_pool_unique'], r['overlapping_unique_removed']] for r in data['plans']])
    chosen = frozen['source_adaptation_selection']['chosen']
    lines += [(f"源验证冻结的适应策略为全部参数更新、{chosen['inner_steps']} 步、学习率 "
              f"`{chosen['inner_lr']}`；PRICE 的学习率按源验证选择为 "
              f"`{frozen['price_lr_source_selection']['inner_lr']}`。FOMAML 以数据库为任务执行一阶元学习；"
              'ERM 是同结构、同源划分的监督初始化，随后采用相同目标适应预算。'
              'PG 是固定估计、不随 K 适应；PRICE 使用已有预训练权重。'
              '不同历史训练使用源验证的 early stopping/best checkpoint；S4–S6 另采用固定 4000 更新的 last.pt。'
              '历史适应评估与源侧零样本端点不能互换。'), '',
              '### 1.2 指标与不确定性', '',
              ('标准 Q-error 定义为 `max(max(pred,1)/max(true,1), max(true,1)/max(pred,1))`。'
              '主指标为 unique-query Q95，越小越好；Q50 为辅助中位数指标。'
              '不混用 all-rows、PRICE shifted `(pred+1)/(true+1)`、pooled Q95 或 mean log-Q95。'), '',
              ('目标模型采用三个独立训练 seed（42/43/44），每个 seed 有五次 support 抽样。'
              '先求每个 seed 的五个 episode Q95 的算术均值，再对三个均值求均值与样本 SD（ddof=1）。'
              '不是合并逐查询预测后重算分位数，也不是 15 个独立训练重复。'
              'PRICE 只有单个提供的 checkpoint，其 support SD 单列；PG 没有适应波动。'
              'SD 仅描述重复波动，不是置信区间，不支持显著性结论。'), '',
              ('源侧结构选择使用适应后的三库 macro mean log-Q95；P0 使用库内表数桶比值及库间的几何平均；'
              'P1 与 S4–S6 使用各源验证库完整 4+ 表群体 Q95 的等权算术 macro。'
              'S4–S6 仅 seed42，没有训练 seed SD，不能与目标跨 seed 表合并。'), '',
              '## 2. 轻量模型规模与源侧结构选择', '',
              ('轻量网络以共享 token MLP 编码连接列直方图、fanout、表和过滤特征，'
              '使用集合池化，输出对 PG 对数估计的残差。sum 是源验证选择的参考结构；'
              '初始残差输出为零，因此从 PG 起点开始，不等于随机基数预测。'), '',
              '**表 2 参数规模。** FP32 纯参数字节不含优化器、checkpoint 元数据和预处理开销。', '']
    price_params = frozen['parameters']['PRICE']['parameters']
    lines += markdown_table(['模型', '参数数', 'FP32 参数字节', 'PRICE / 本模型'],
                            [[m, f"{frozen['parameters'][m]['parameters']:,}",
                              f"{frozen['parameters'][m]['fp32_parameter_bytes']:,}",
                              f"{price_params / frozen['parameters'][m]['parameters']:.2f}×"]
                             for m in ('FOMAML-set', 'FOMAML-sum', 'PRICE')])
    lines += [('当前 sum 模型约为 PRICE 参数数的 1/80.80；107.33×仅对应 set，不用于 sum。'
              '参数缩减不等于训练时间比、checkpoint 体积比或数据库端到端加速。'), '',
              '**表 3 源验证结构选择：macro mean log-Q95（非目标主表 Q95）。**', '']
    lines += markdown_table(['结构', '三 seed 均值', '训练 seed 样本 SD'],
                            [[r['arch'], f"{r['source_macro_log_q95_mean']:.6f}",
                              f"{r['source_macro_log_q95_seed_sample_std']:.6f}"]
                             for r in frozen['architecture_source_selection']])
    lines += [('按历史源验证均值选择 sum，不依据下文目标结果改选结构。'
              '不同结构的参数量及历史 best checkpoint 不同，不能把 set/sum 差异单独归因于表示结构，'
              '有限变体结果也未排除表示层瓶颈。'), '',
              '## 3. 三目标库完整 K 对照', '',
              ('表 4–6 中四个轻量模型为均值 ± **训练 seed SD**；PRICE†单列均值与 **support SD**，'
              '二者不是同一种重复；PG 单列固定值。图 1–3 展示源侧选定的 sum 主线，'
              'set 的全部 K 同样保留在表和 CSV。SVG 使用 Q95 对数轴与等间距分类 K 轴：'
              '后者不表示支持数差值的线性距离。实线帽表示训练 seed SD，虚线帽仅表示 PRICE support SD。'), '']
    for index, db in enumerate(DATABASES, 4):
        lines += [f'### {db}', '', f'**表 {index} {db} 标准 unique-query Q95（越小越好）。**', '']
        table = []
        for k in SHOTS:
            row = [k]
            for method in METHODS:
                cell = lookup[(method, db, k)]
                sd = cell['q95_training_seed_sd']
                row.append(f"{cell['q95_mean']:.2f} ± {sd:.2f}" if sd is not None else f"{cell['q95_mean']:.2f}")
            row.insert(-1, f"{lookup[('PRICE', db, k)]['q95_support_sd']:.2f}")
            table.append(row)
        lines += markdown_table(['K', 'FOMAML-set', 'ERM-set', 'FOMAML-sum', 'ERM-sum',
                                 'PRICE† 均值', 'PRICE support SD', 'PG'], table)
        lines += [f'![图 {index - 3} {db} 完整 K Q95 曲线](results/paper-draft-20261007/target-{db}-q95.svg)', '',
                  (f'**图 {index - 3}。** {db} 完整 K 曲线及描述性 SD；PG 不添加误差条，'
                  'PRICE 不添加训练 seed SD。逐点精确值见 `target-q95.csv`。'), '']
    lines += [('† PRICE 已有权重使用全部 26 个源库，包括本研究三个验证库；'
              '其源数据曝光与轻量模型的 23/3 划分不一致，所以仅为补充参考。'
              '三个目标库所有 K 的轻量模型 Q95 均值均高于 PRICE，本章不声称尾部精度优于 PRICE。'), '',
              '### 3.1 适应收益依赖数据库与架构', '',
              ('Genome 中四种轻量组合适应后的 Q95 均值均低于各自 K=0；'
              'STATS 中所有非零 K 的 Q95 均值均高于各自零样本结果，表明该批结果保留了负迁移。'
              'ergastf1 的 sum 均值变化相对较小，不能将 Genome 的收益推广为普遍单调改善。'), '',
              ('在 set 结构下，Genome 的 FOMAML K=0/5/10 优于 ERM；'
              '但 sum 结构下 Genome 全部非零 K 的 ERM 均值均低于 FOMAML。'
              '例如 K=20 的 FOMAML-sum 为 129.76，ERM-sum 为 100.51。'
              'STATS K=5 的 FOMAML-sum 96.77 低于 ERM-sum 161.68，'
              '但两者都差于自身 K=0；相对其他方法更好不等于适应本身获益。'
              '现有结果不支持跨架构、跨目标的统一元学习优势。'), '',
              ('这些是已记录 Q95 均值的描述性比较。历史 `paired_unique_vs_reference` 的参照是 PG，'
              '不是本模型 K=0，因此不能据此推算“适应造成多少比例查询恶化”，'
              '也不能证明 support 分布、loss 或表示结构是负迁移原因。'), '',
              '### 3.2 中位数与尾部须分别报告', '',
              ('Q50 不能替代尾部 Q95。为减少正文重复，表 7 仅展示 K=0/20/100 切片；'
              '完整六个 K 的所有方法 Q50 仍在 `target-q50.csv` 中保留，切片不是模型选择。'), '',
              '**表 7 Q50 的展示切片。** 轻量方法为训练 seed 均值 ± SD；PRICE 仅均值，support SD 见 CSV。', '']
    q50_rows = []
    for db in DATABASES:
        for k in (0, 20, 100):
            values = [db, k]
            for method in ('FOMAML-sum', 'ERM-sum', 'PRICE', 'PG'):
                cell = lookup[(method, db, k)]
                sd = cell['q50_training_seed_sd']
                values.append(f"{cell['q50_mean']:.2f} ± {sd:.2f}" if sd is not None else f"{cell['q50_mean']:.2f}")
            q50_rows.append(values)
    lines += markdown_table(['库', 'K', 'FOMAML-sum', 'ERM-sum', 'PRICE†', 'PG'], q50_rows)
    lines += [('轻量模型在部分库上有较低中位数误差，但这与 Q95 对 PRICE 的差距可以同时成立。'
              '因此不能用中位数优势表述整体尾部优势。'), '',
              '## 4. 源侧消融与负结果', '',
              ('以下研究均在源侧完成，不使用目标测试来选择方案；端点分别按自身冻结协议计算。'
              '源验证已用于多轮开发，不是新的独立盲测。'), '',
              '### 4.1 P0：增加训练预算未改善冻结尾部指标', '',
              ('P0 为 seed42、ERM-sum 的 4000→20000 更新续训对照。主指标为每库 4+ 表各桶 '
              'long/short Q95 比的几何平均，再对三个库做几何平均；不是 pooled Q95。'), '']
    lines += markdown_table(['更新预算', '实际样本呈现（含重复）', 'optimizer updates'],
                            [[k, f"{v['counts']['query_presentations']:,}", v['counts']['optimizer_steps']]
                             for k, v in frozen['p0']['budget_summaries'].items()])
    lines += markdown_table(['范围', 'long/short 几何比（>1 为恶化）'],
                            [[db, f'{ratio:.6f}'] for db, ratio in frozen['p0']['database_ratios'].items()] +
                            [['三库等权几何汇总', f"{frozen['p0']['primary_geometric_ratio']:.6f}"]])
    lines += [('本配置下更长训练未改善冻结尾部指标；这不是 PRICE 训练量匹配实验，'
              '也不排除其他模型或优化配置受益。次要适应指标不能替代本主端点。'), '',
              '### 4.2 P1：映射信号有局部价值，但未优于 sum', '',
              ('九臂源侧零样本实验使用各库完整 4+ 表 unique-query Q95 的等权算术 macro，'
              '每臂 4000 更新、冻结 last.pt。共同计划为 149,504 unique queries，4+ 主桶为 63,436。'), '',
              '**表 8 映射消融（每行是独立训练 seed 的三库 macro）。**', '']
    lines += markdown_table(['seed', 'sum', 'true mapping', 'shuffled mapping'],
                            [[r['seed'], f"{r['sum']:.4f}", f"{r['true']:.4f}", f"{r['shuffled']:.4f}"]
                             for r in frozen['p1']['per_seed']] +
                            [['三 seed 算术平均', *[f"{frozen['p1']['three_seed_arithmetic_means'][m]:.4f}"
                                                for m in ('sum', 'true', 'shuffled')]]])
    lines += [(f"真实映射跨 seed 平均比打乱低 {frozen['p1']['true_vs_shuffled_relative_reduction'] * 100:.4f}%，"
              f"但比 sum 高 {frozen['p1']['true_vs_sum_relative_increase'] * 100:.4f}%。"
              '百分比为均值之比，不是各 seed 比值平均。true/shuffled 同容量是主要信息对照；'
              'true/sum 的参数量分别为 169,665/132,545，混有容量差异。'
              '同 seed 三臂采样一致，但跨 seed 采样流不同；seed42 的 true 未优于 shuffled，'
              '打乱也可能保留原关联。结果未认证显著性或普遍收益，不能外推为解决目标跨库尾部。'), '',
              '### 4.3 S4/S5/S6：单因素精度改进的取舍', '',
              ('三个对照均为 seed42、sum64、ERM、固定 4000 更新的源验证零样本端点。'
              'A 使用 SmoothL1、PG 输入和固定 PG 输出锚定；S4 仅取消输出锚定但保留 PG 输入，'
              'S5 对 4+ 表查询损失加权 ×2，S6 混合 0.5 均值损失与 0.5 批次上尾损失（上尾 10%）。'
              '各对照 A 的每库端点重现一致，主指标先各库计算完整 4+ Q95，再等权算术平均。'), '',
              '**表 9 源侧主指标。** 正的 B/A−1 表示恶化；只有一个 seed，不填 SD。', '']
    lines += markdown_table(['研究', 'A 4+ Q95 macro', 'B 4+ Q95 macro', 'B/A−1', '判定'],
                            [[r['study'], f"{r['a_q95']:.4f}", f"{r['b_q95']:.4f}",
                              f"{r['relative_change'] * 100:+.2f}%", r['decision']]
                             for r in data['source_tradeoffs'] if r['database'] == 'macro'])
    lines += ['![图 4 源侧 S4/S5/S6 的逐库与 macro 取舍](results/paper-draft-20261007/source-s4-s5-s6-tradeoffs.svg)', '',
              ('**图 4。** 完整 4+ 群体 Q95 的相对变化；macro 是三个库 Q95 的等权平均，'
              '不是合并查询后求分位数。图不绘制不存在的 seed SD，图中参考线不能替代完整 Q50/Q95 门槛。'), '',
              '**表 10 S6 逐库端点，显式保留退化库。**', '']
    lines += markdown_table(['源验证库', 'A Q95', 'B Q95', 'B/A−1'],
                            [[r['database'], f"{r['a_q95']:.4f}", f"{r['b_q95']:.4f}",
                              f"{r['relative_change'] * 100:+.2f}%"] for r in data['source_tradeoffs']
                             if r['study'] == 'S6' and r['database'] != 'macro'])
    lines += [('S6 macro 改善 9.42%，未达到冻结的至少 10%；legalacts 的 4+ Q95 又退化 15.70%，'
              '超过每库 all/4+ Q95、Q50 均不得高于 A 的 1.10 倍的护栏。'
              '因此三项实验正式判定均为 `no_candidate`，不因接近阈值而追认，'
              '不扩 seed43/44 或使用目标结果补选。S4/S5 的失败说明这两种固定干预在当前配置下无合格收益，'
              '并不证明 PG 输入不可替代或复杂度加权在所有情况下无效。'), '',
              '## 5. S7：已有误差与源侧泛化诊断', '',
              ('S7 固定 S6 A/B last.pt，复用三验证库完整已有预测，并对 23 训练库的实际 seen/unseen 各取 '
              '2048 条身份固定样本，共 94,208 条。历史公共采样重放得到 seen 池 858,898、unseen 池 270,816；'
              '重放不是新训练。S7 历史执行确有 CPU 推理，但本次论文导出只读其归档，不重新推理。'
              '固定样本抽取不看标签、PG 或误差；样本精度不冒充完整候选池精度。'), '',
              '### 5.1 legalacts：持续尾部退化，而非仅新入尾', '',
              ('该库完整 4+ 群体为 9,178 条，全部四表。Q95 从 15.6747 到 18.1360，'
              'Q50 从 2.66276 到 2.66325。尾部定义为本臂 Q≥其完整群体 Q95，保留并列，'
              '不是人为裁成恰好 5%。'), '',
              '**表 11 互斥尾部身份。** 小于 200 的组只呈现数量和组成，不推算详细误差分位数。', '']
    labels = {'persistent': '持续尾部', 'entered': '新入 B 尾部', 'left': '退出 A 尾部', 'neither': '两臂均非尾部'}
    lines += markdown_table(['组', 'unique 数', 'B 低估', 'B 高估', 'B 较 A 恶化比例'],
                            [[labels[r['group']], r['unique_queries'], r['b_under'], r['b_over'],
                              '不报（n<200）' if r['b_worse_fraction'] is None
                              else f"{r['b_worse_fraction'] * 100:.2f}%"] for r in data['s7_tails']])
    lines += [('持续尾部 392 条中 73.47% 恶化；其中 true[1000,10000) 有 241 条，'
              'B 低估 306 条。新入/退出尾部各 67 条，不作小组分位数结论。'
              'true[10000,100000) 且 PG 低估的足量桶 n=839，Q95 从 19.276 到 24.444（+26.81%）。'
              '不同分层轴有重叠，Q95 非可加统计量，不能分解“某桶贡献多少退化”；'
              '全部预定义稀疏/空桶见 `s7-legalacts-strata.csv`。'), '']
    whole = data['s7_legalacts']['four_plus']
    lines += markdown_table(['legalacts 完整 4+ 指标', 'A', 'B'],
                            [[name, f"{whole['A'][metric]:.6f}", f"{whole['B'][metric]:.6f}"]
                             for name, metric in [('raw log MAE', 'raw_log_mae'), ('raw log MSE', 'raw_log_mse'),
                                                  ('低估比例', 'under_fraction')]])
    lines += [('raw MAE/MSE 同样恶化，故本库不支持“平均误差改善而牺牲 Q95”的统一解释。'
              '较弱向上修正与 PG 低估桶退化方向一致，仅是相关观察，不是 loss、优化或表示的因果证据。'), '',
              '### 5.2 seen/unseen 没有一致的大幅样本落差', '',
              '**表 12 库内 unseen/seen 样本 Q95 比的中位数。** 先每库求比，再求库间中位数，非 pooled。', '']
    lines += markdown_table(['群体', '臂', '足量库数', '库内比值中位数'],
                            [[r['bucket'], r['arm'], r['database_count'],
                              f"{r['median_within_database_q95_unseen_div_seen']:.4f}"]
                             for r in data['s7_overview']['within_database_ratios']])
    lines += [('20 个足量 4+ 库中，B 在 seen 侧 11 库改善/9 库恶化，在 unseen 侧 13 库改善/7 库恶化。'
              '没有一致的大幅 seen/unseen 落差，且 B 在已见查询上也会退化，'
              '降低了“仅跨库才失败”或普遍记忆式过拟合解释的优先级。'
              '但群体组成和曝光不同，细桶稀疏，unseen 不是随机独立留出；'
              '现有证据不能排除局部过拟合，优化瓶颈与表征瓶颈仍为 unknown。'
              '不改变 S6 的 `no_candidate`，不把后验诊断变为新的模型选择规则。'), '',
              '## 6. 公平性、证据边界与局限', '',
              ('1. **主对照与补充参考不同。** FOMAML/ERM 同结构同源划分，但反向次数、曝光和训练成本未自动匹配；'
              'PRICE 预训练用 26 源库，不能称完全公平的 23/3 初始化。PG 输出锚定、PG 特征及统计预处理成本需披露。'),
              ('2. **不确定性不同。** 三 seed 样本 SD、单 checkpoint support SD 和单 seed 源 macro 分列；'
              '没有新增置信区间或显著性检验，不能凭 SD 重叠与否下结论。源验证反复用于开发，存在选择偏差。'),
              ('3. **完整性有限。** 14 号核验 14 份目标摘要、1260 episode 的内部计划及聚合一致性，'
              '本轮继续检查原 36 来源 SHA256 与离线 CSV；摘要未内嵌完整目标 query_hash 集合/数据内容 hash。'
              '未读原始目标标签和预测，不能称独立复现全部目标数据与 SQL 语义。'),
              ('4. **IMDB 明确未完成。** 完整 few-shot support 缺匹配可认证统计输入。'
              '不使用 `--allow-exclusions` 静默改变支持分布，不从 test 补 support，不伪造标签或特征。'),
              ('5. **P2 明确 blocked / 用户暂缓。** G0、A 结构层及 B 冻结编码核验完成；同版本源表历史来源链未认证；'
              '同版本真实源表、独立 SQL 语义与真实选择率认证尚未完成。编码一致不等于真实过滤计数正确；'
              'PG 估计、join 后标签或审计相关性不替代独立数据库证据。'),
              ('6. **速度结论不外推。** 参数规模是可核对记录；单模型推理耗时不包括特征提取、'
              '标签获取、数据库优化与执行。本章没有完整端到端对照，不声称数据库加速或测得训练成本优势。'),
              ('7. **消融/诊断边界。** P0 单 seed 固定预算、P1 容量混杂、S4–S6 固定干预和 S7 非随机组成'
              '均限制推广；不能以负结果排除全部机制，也不能认证优化、表示或 loss 因果。'), '',
              '## 7. 本章结论', '',
              ('在真实记录的三目标、完整六个 K 对照中，轻量 sum 网络以 132,545 参数获得可运行的跨库适应方案，'
              '参数约为 PRICE 的 1/80.80，但 Q95 仍高于 PRICE 补充参考。'
              '少样本适应对 Genome 有收益，对 STATS 出现负迁移；FOMAML 相对 ERM 的优势依赖结构与 K，'
              '并不构成一致的跨库胜出。'), '',
              ('源侧更长预算、显式映射与三项单因素训练改动未产生足以替换冻结基线的证据。'
              'S6 有局部 macro 改善，却未满足预设阈值和逐库护栏；S7 将下一步关注点转向条件化误差取舍与'
              '跨库迁移，同时保留机制 unknown。合理结论是“轻量化成立、尾部鲁棒性与泛化机制尚未解决”，'
              '而不是优于 PRICE、消除负迁移、显著获益或端到端加速。IMDB 与 P2 缺口继续保留。'), '',
              '## 附录：图表编排、来源绑定与重建', '',
              ('建议正文顺序为设置（表 1）→规模和选择（表 2–3）→每个目标表紧邻其曲线（表 4–6/图 1–3）'
              '→Q50（表 7）→消融和负结果（表 8–10/图 4）→诊断（表 11–12）→局限与结论。'
              '导出 SVG 是矢量图，未指定学校页面尺寸；排版时可以同比缩放，不裁掉失败库或 K。'), '',
              ('全部精确表值见 [精简结果目录](results/paper-draft-20261007/)；'
              '[source-bindings.json](results/paper-draft-20261007/source-bindings.json) 绑定 14 号原 36 来源、'
              '当前直接输入、每个 CSV/SVG 及本章 SHA256。读取记录不访问 raw target、cache、checkpoint 或服务器。'), '',
              ('来源入口：[14 号统一汇总](14-离线实验统一对照与论文结论.md)、'
              '[22 号 S7 结果](22-S7误差定位与泛化诊断结果.md)、'
              '[通用协议](../docs/protocol.md)。冻结协议及机器端点的精确 hash 一并列于来源 JSON。'), '',
              '确定性重建（Python 标准库；只写本任务新增稿件及结果目录，不改历史文件）：', '',
              '```bash', f'cd {ROOT}',
              'PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/bin/python -B scripts/build_paper_results.py', '```', '',
              ('直接输入 hash 若变化，导出器拒绝输出；应先核验历史变更，不自动重新接受新 hash。'
              '运行记录、TDD 与导出验证见 [本轮验证目录](../reports/paper-draft-20261007/)。'), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = build(args.root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Paper export failed: {error}\n')
    print(json_text(result), end='')


if __name__ == '__main__':
    main()
