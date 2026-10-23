"""Reproduce a synthesis of recorded results only; never load data/checkpoints or train.

Run with Python 3.11+: python reports/offline-synthesis-20261005/reproduce.py
All source hashes are checked again after extraction. Requires only the stdlib.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / '实验文档/results'
OUT = RESULTS / 'offline-synthesis-20261005'
DOCUMENT = ROOT / '实验文档/14-离线实验统一对照与论文结论.md'
SEEDS = (42, 43, 44)
DATABASES = ('stats', 'genome', 'ergastf1')
SHOTS = (0, 5, 10, 20, 50, 100)
METRICS = ('q50', 'q90', 'q95', 'q99', 'q_mean')
SOURCE_HASHES = {}


def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    SOURCE_HASHES[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def result(name):
    return read('实验文档/results/' + name)


def close(left, right):
    if not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f'Recorded aggregate mismatch: {left} != {right}')


def average(values):
    return statistics.mean(values)


def deviation(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    if not __debug__:
        raise RuntimeError('Run without -O: evidence validation assertions must remain enabled')
    global SOURCE_HASHES
    SOURCE_HASHES = {}
    config = read('configs/data.json')
    assert len(config['splits']['train']) == 23
    assert config['splits']['validation'] == ['chembl', 'legalacts', 'telstra']
    assert config['splits']['test'] == ['imdb', 'stats', 'genome', 'ergastf1']
    sources = {
        'FOMAML-set': [(s, f'frozen3-20260927/s{s}-summary.json') for s in SEEDS],
        'ERM-set': [(s, f'erm-s{s}-summary.json') for s in SEEDS],
        'FOMAML-sum': [(s, f'sum-target-20260927/s{s}-summary.json') for s in SEEDS],
        'ERM-sum': [(s, f'erm-sum-target-20260927/s{s}-summary.json') for s in SEEDS],
        'PRICE': [(None, 'price-fewshot-1e-6-20260927/summary.json')],
        'PG': [(None, 'pg-summary.json')],
    }
    seed_rows, checks, parameters = [], [], {}
    canonical = None
    all_sources = {}
    expected_keys = {(db, repeat, k) for db in DATABASES for repeat in range(5) for k in SHOTS}
    for method, files in sources.items():
        for seed, name in files:
            summary = result(name)
            assert summary['purpose'] == 'research' and summary['primary_metric'] == 'unique_query_q95'
            rows = {(row['database'], row['run'], row['k']): row for row in summary['results']}
            assert len(rows) == len(summary['results']) and set(rows) == expected_keys
            params = {'parameters': summary['parameters'], 'fp32_parameter_bytes': summary['parameter_bytes']}
            assert params['fp32_parameter_bytes'] == 4 * params['parameters']
            if method in parameters:
                assert params == parameters[method]
            parameters[method] = params
            if seed is not None:
                metadata = summary['metadata']
                assert metadata['seed'] == seed
                assert metadata['train_databases'] == config['splits']['train']
                assert metadata['validation_databases'] == config['splits']['validation']
            if canonical is None:
                canonical = rows
            for key, row in rows.items():
                base = canonical[key]
                assert row['support_ids'] == base['support_ids']
                assert len(row['support_ids']) == key[2]
                assert row['metrics']['n'] == base['metrics']['n']
                assert row['unique_query_metrics']['n'] == base['unique_query_metrics']['n']
                assert row['support_pool_unique'] == base['support_pool_unique']
                assert row['overlapping_unique_removed'] == base['overlapping_unique_removed']
                for metric in METRICS:
                    assert math.isfinite(row['unique_query_metrics'][metric])
                assert row['reference_method'] == (None if method == 'PG' else 'pg')
                for k in SHOTS:
                    smaller = rows[(key[0], key[1], k)]
                    if k < key[2]:
                        assert row['support_ids'][:k] == smaller['support_ids']
            aggregates = {(row['database'], row['k']): row['across_support_runs'] for row in summary['aggregate']}
            assert len(aggregates) == 18
            for db in DATABASES:
                for k in SHOTS:
                    cell = aggregates[(db, k)]
                    selected = [rows[(db, repeat, k)] for repeat in range(5)]
                    for metric in METRICS:
                        values = [row['unique_query_metrics'][metric] for row in selected]
                        close(average(values), cell[f'unique_query_{metric}']['mean'])
                        close(deviation(values), cell[f'unique_query_{metric}']['std'])
                    entry = {'method': method, 'training_seed': seed, 'database': db, 'k': k,
                             'support_runs': 5, 'unique_queries': selected[0]['unique_query_metrics']['n'],
                             'all_rows': selected[0]['metrics']['n'],
                             'source_file': '实验文档/results/' + name,
                             'source_sha256': SOURCE_HASHES['实验文档/results/' + name]}
                    for metric in METRICS:
                        entry[f'{metric}_support_mean'] = cell[f'unique_query_{metric}']['mean']
                        entry[f'{metric}_support_sample_std'] = cell[f'unique_query_{metric}']['std']
                    paired = [row['paired_unique_vs_reference'] for row in selected]
                    entry['fraction_queries_improved_vs_pg'] = (
                        average([row['fraction_queries_improved'] for row in paired]) if all(paired) else None)
                    seed_rows.append(entry)
            all_sources[(method, seed)] = summary
            checks.append({'source': '实验文档/results/' + name, 'support_rows_checked': len(rows),
                           'support_ids_and_query_counts_match': True,
                           'nested_support_verified': True, 'aggregate_recomputed': True,
                           'paired_reference': None if method == 'PG' else 'pg'})
    grouped = {}
    for row in seed_rows:
        grouped.setdefault((row['method'], row['database'], row['k']), []).append(row)
    aggregates = []
    for (method, db, k), rows in grouped.items():
        row = {'method': method, 'database': db, 'k': k,
               'training_seed_replicates': len(rows) if method not in ('PRICE', 'PG') else 0,
               'training_seeds': [r['training_seed'] for r in rows if r['training_seed'] is not None],
               'initialization': 'single supplied checkpoint' if method == 'PRICE' else 'fixed PG labels' if method == 'PG' else 'trained independently for each seed',
               'support_runs_per_initialization': 5, 'unique_queries': rows[0]['unique_queries']}
        for metric in METRICS:
            values = [r[f'{metric}_support_mean'] for r in rows]
            row[f'{metric}_mean'] = average(values)
            row[f'{metric}_between_training_seed_sample_std'] = deviation(values) if len(rows) > 1 else None
        row['fraction_queries_improved_vs_pg_mean'] = (
            average([r['fraction_queries_improved_vs_pg'] for r in rows])
            if all(r['fraction_queries_improved_vs_pg'] is not None for r in rows) else None)
        aggregates.append(row)
    lookup = {(r['method'], r['database'], r['k']): r for r in aggregates}
    def value(method, db, k, metric='q95'):
        return lookup[(method, db, k)][f'{metric}_mean']
    def display(method, db, k, metric='q95'):
        r = lookup[(method, db, k)]
        mean = r[f'{metric}_mean']; sd = r[f'{metric}_between_training_seed_sample_std']
        return f'{mean:.2f} ± {sd:.2f}' if sd is not None else f'{mean:.2f}'

    # Source-only experiments retain their own distinct frozen metrics.
    p0 = result('p0-fixed-budget-20261001/decision.json')
    p0_protocol = result('p0-fixed-budget-20261001/protocol.json')
    by_db = {}
    for row in p0['buckets']:
        close(row['ratio'], row['long_q95'] / row['short_q95'])
        by_db.setdefault(row['database'], []).append(row['ratio'])
    for db, ratios in by_db.items():
        close(statistics.geometric_mean(ratios), p0['database_ratios'][db])
    close(statistics.geometric_mean(p0['database_ratios'].values()), p0['primary_macro_ratio'])
    assert p0['target_evaluation_performed'] is False
    endpoint = result('p1-mapping-ablation-20261004/ablation_endpoint.json')
    runs = result('p1-mapping-ablation-20261004/ablation_runs.json')
    assert runs['seeds'] == list(SEEDS) and runs['arms'] == ['sum', 'true', 'shuffled']
    assert len(runs['runs']) == 9 and len(endpoint['runs']) == 9
    digests = set(); p1_rows = []; means = {}
    for arm in runs['arms']:
        values = []
        for seed in SEEDS:
            row = endpoint['runs'][f'{arm}-seed{seed}']
            cells = row['per_database']
            close(row['macro_four_plus_q95'], average([cells[db]['four_plus_tables']['q95'] for db in config['splits']['validation']]))
            values.append(row['macro_four_plus_q95']); digests.add(row['plan_digest'])
        means[arm] = average(values)
    assert len(digests) == 1
    for seed in SEEDS:
        p1_rows.append({'seed': seed, **{arm: endpoint['runs'][f'{arm}-seed{seed}']['macro_four_plus_q95'] for arm in runs['arms']}})
    sampling_by_seed = {}
    for seed in SEEDS:
        group = [row for row in runs['runs'] if row['seed'] == seed]
        assert len(group) == 3 and len({row['sample_plan_digest'] for row in group}) == 1
        assert len({json.dumps(row['counts'], sort_keys=True) for row in group}) == 1
        maps = [row for row in group if row['arm'] in ('true', 'shuffled')]
        assert maps[0]['initialization_hash'] == maps[1]['initialization_hash']
        sampling_by_seed[str(seed)] = group[0]['sample_plan_digest']
    assert all(row['optimizer_steps'] == 4000 for row in runs['runs'])
    assert len({row['validation_plan_hash'] for row in runs['runs']}) == 1
    search = result('adapt-search-3seed-20260927/result.json')
    price_search = result('price-lr-search-result.json')
    assert search['chosen']['inner_steps'] == 5 and search['chosen']['inner_lr'] == .001 and search['chosen']['scope'] == 'all'
    assert price_search['chosen_inner_lr'] == 1e-6
    assert search['validation_databases'] == price_search['validation_databases'] == config['splits']['validation']
    p2 = result('p2-source-semantics-20261005-03/status.json')
    architecture = []
    for arch in ['set', 'sum', 'attn', 'attn_pair']:
        d = result(f'arch-20260927/select4k-{arch}.json')
        vals = [row['macro_log_q95'] for row in d['results'].values()]
        assert len(vals) == 3
        architecture.append({'arch': arch, 'source_macro_log_q95_mean': average(vals),
                             'source_macro_log_q95_seed_sample_std': deviation(vals)})
    assert min(architecture, key=lambda row: row['source_macro_log_q95_mean'])['arch'] == 'sum'
    run_configs = {}
    for folder in ['frozen3-20260927', 'sum-target-20260927', 'erm-sum-target-20260927']:
        for seed in SEEDS:
            name = f'{folder}/s{seed}-run_config.json'
            d = result(name); assert d['config']['shots'] == list(SHOTS)
            assert d['config']['support_runs'] == 5 and d['config']['support_seed'] == 2026
            assert d['chosen']['inner_steps'] == 5 and d['chosen']['inner_lr'] == .001 and d['chosen']['scope'] == 'all'
            run_configs[name] = {'checkpoint': d['checkpoint'], 'strategy_source': d['strategy_source'], 'chosen': d['chosen']}
    pc = result('price-fewshot-1e-6-20260927/run_config.json')
    assert pc['inner_lr'] == 1e-6 and pc['config']['support_runs'] == 5
    # Recorded descriptions are not treated as measured training costs.
    gaps = [
        'IMDB full few-shot result is unavailable; no exclusion or synthetic replacement.',
        'Random-initialization formal summary is not present in this local result package; no values reconstructed from prose.',
        'Exact target query_hash sets/content hashes are not embedded in summaries; support identities and query counts match, but no new raw prediction or manifest certification performed.',
        'ERM-set standalone run_config is not bundled here; report metadata and matched support plans are available, but do not independently certify the complete historical evaluation command.',
        'PRICE uses supplied 26-source pretrained weights; not matched to the 23-source initialization; PRICE training-seed uncertainty unavailable.',
        'paired_unique_vs_reference in target summaries compares against PG, not each model zero-shot prediction.',
        'No confidence interval or new significance test; SD overlap/non-overlap is not a significance decision.',
        'No original-label re-count, real selectivity certification, end-to-end acceleration or matched training-cost claim.',
    ]
    data = {'schema': 1, 'purpose': 'offline recorded-result synthesis; not a new experiment',
            'date': '2026-10-05', 'methodology': {
                'metric': 'standard unique-query Q-error; not PRICE shifted Q-error',
                'first_level': 'mean and sample SD over 5 support repeats within each initialization',
                'second_level': 'arithmetic mean and sample SD (ddof=1) of the three training-seed means; no pooling of query predictions',
                'price_uncertainty': 'single supplied checkpoint; support SD only in per-initialization CSV; training-seed SD unavailable',
                'pg': 'fixed existing estimates, no adaptation; K does not change values',
                'paired_reference': 'pg, not model zero-shot',
                'execution': 'only local recorded JSON reads; no dataset/cache/checkpoint reads, server access, inference or training'},
            'source_sha256': SOURCE_HASHES, 'validation': checks,
            'target_aggregate': aggregates, 'parameters': parameters,
            'architecture_source_selection': architecture,
            'source_adaptation_selection': {'chosen': search['chosen'], 'validation_databases': search['validation_databases']},
            'price_lr_source_selection': {'inner_lr': price_search['chosen_inner_lr'],
                                          'macro_log_q95': price_search['chosen_macro_log_q95']},
            'p0': {'metric': p0_protocol['primary_metric'], 'seed': 42,
                   'primary_geometric_ratio': p0['primary_macro_ratio'], 'database_ratios': p0['database_ratios'],
                   'budget_summaries': p0['budget_summaries'], 'target_evaluation_performed': False},
            'p1': {'metric': endpoint['metric'], 'per_seed': p1_rows, 'three_seed_arithmetic_means': means,
                   'true_vs_shuffled_relative_reduction': 1 - means['true'] / means['shuffled'],
                   'true_vs_sum_relative_increase': means['true'] / means['sum'] - 1,
                   'plan_digest': next(iter(digests)), 'training_sample_plan_digest_by_seed': sampling_by_seed,
                   'training_pairing_scope': 'same seed across arms; not equality across seeds',
                   'checkpoint_policy': 'frozen last.pt',
                   'capacity_confounded_comparison': 'true vs sum', 'primary_comparison': 'true vs shuffled'},
            'p2': {'overall': p2['overall'], 'G0': p2['G0'], 'A': p2['A'], 'B': p2['B'],
                   'G1': p2['G1'], 'C': p2['C'], 'priority': 'deferred supplementary audit; blocked evidence retained'},
            'historical_run_configs': run_configs, 'gaps': gaps}
    OUT.mkdir(exist_ok=True)
    write_json(OUT / 'summary.json', data)
    for filename, rows in [('per-initialization.csv', seed_rows), ('target-aggregate.csv', aggregates)]:
        with (OUT / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)

    lines = [
        '# 14 离线实验统一对照与论文结论', '',
        '> 整理日期：2026-10-05。仅从本地既有正式结果重新汇总，不是新训练、推理、目标库评估或数据库计数。', '',
        '## 1. 如何阅读这份汇总', '',
        '- 主线使用 PRICE 已有离线查询、标签、统计和特征。训练/验证/目标划分不改，当前参考架构仍为源侧选择的 `sum`。',
        '- **同结构方法主对照是 FOMAML vs ERM；PRICE 是补充参考**：它使用全部 26 个源库预训练权重，而我们的训练库为 23 个，不能把相同目标输入解释为训练条件完全公平。FOMAML/ERM 也不因同结构而自动满足相同反向次数、样本曝光或训练成本。',
        '- 目标库主表统一为标准 **unique-query Q95**，不使用 all-rows 或 `(pred+1)/(true+1)` 的 PRICE shifted 口径。',
        '- 表中 `均值 ± SD`：先对每个训练 seed 的 5 次 support 抽样 Q95 取均值，再对 seed42/43/44 的三个均值求算术均值与样本 SD（ddof=1）；不是合并所有预测后重算 Q95，也不是 15 个独立训练重复。',
        '- PRICE 只有一个提供的 checkpoint，主表不写训练 seed SD。其 5 次 support 抽样 SD 单独保存在 CSV；PG 不适应，随 K 的相同值不代表新训练结果。',
        '- 所有 K 都保留，不按已看过的目标结果挑“最佳 K”。没有新增置信区间或显著性检验，不能用 SD 重叠/不重叠断言显著。', '',
        '### 1.1 本次可核验的共同评估计划', '',
        '| 目标库 | all-rows | unique-query | support 池去重数 | 与 test 重叠并排除的 unique query |',
        '|---|---:|---:|---:|---:|',
    ]
    for db in DATABASES:
        r = canonical[(db, 0, 0)]
        lines.append(f"| {db} | {r['metrics']['n']} | {r['unique_query_metrics']['n']} | {r['support_pool_unique']} | {r['overlapping_unique_removed']} |")
    lines += ['', f'共核对 **{len(checks)} 份目标结果摘要、{sum(c["support_rows_checked"] for c in checks)} 个 episode 记录**：支持 IDs、重复编号、K、嵌套支持关系、测试样本数量和排除数量一致；各摘要中的 Q50/Q90/Q95/Q99/均值及 support SD 均从记录值重算核对。',
              '这是结果包内部一致性验证。摘要未内嵌完整目标 query_hash 集合/数据内容哈希，本轮未读原始目标数据或逐查询预测，不能写成重新认证全部目标标签与独立复现。完整 manifest_hash 含不同模型 metadata，不能要求它们相同或用不同哈希直接判数据不一致。', '',
              '## 2. 目标库标准 unique-query Q95', '']
    order = ['FOMAML-set', 'ERM-set', 'FOMAML-sum', 'ERM-sum', 'PRICE', 'PG']
    for db in DATABASES:
        lines += [f'### {db}', '', '| K | FOMAML-set | ERM-set | FOMAML-sum | ERM-sum | PRICE† | PG |', '|---:|---:|---:|---:|---:|---:|---:|']
        for k in SHOTS:
            lines.append('| ' + ' | '.join([str(k)] + [display(m, db, k) for m in order]) + ' |')
        lines.append('')
    lines += ['† PRICE 为非严格训练条件匹配的补充参考；上述轻量模型在已记录三库所有 K 的 Q95 均值均高于 PRICE，不能主张整体尾部精度超过 PRICE。IMDB 未补齐完整 few-shot 结果，不把“三库”写成“四目标库完整验收”。', '',
              '## 3. Q50：中位数优势与尾部差距必须分开', '',
              '| 库 | K | FOMAML-set | FOMAML-sum | PRICE† | PG |', '|---|---:|---:|---:|---:|---:|']
    for db in DATABASES:
        for k in [0, 20, 100]:
            lines.append('| ' + ' | '.join([db, str(k)] + [display(m, db, k, 'q50') for m in ['FOMAML-set','FOMAML-sum','PRICE','PG']]) + ' |')
    lines += ['', 'Genome 上两种 FOMAML 架构的已记录 Q50 均值在所有 K 均低于 PRICE，但 PG 的 Q50 又更低。不能把“中位数优于 PRICE”扩展成“全面优于传统估计器”或“尾部已解决”。其他两库的中位数表现也不支持跨库普遍优势。', '',
              '## 4. 适应与元学习：可以写什么', '',
              '### 4.1 适应前后（描述性 Q95 均值，不挑最优 K）', '',
              '| 架构/训练 | 库 | K=0 | K=5 | K=20 | K=100 |', '|---|---|---:|---:|---:|---:|']
    for method in order[:4]:
        for db in ['stats','genome']:
            lines.append('| ' + ' | '.join([method,db] + [f'{value(method,db,k):.2f}' for k in [0,5,20,100]]) + ' |')
    lines += ['', '已记录的 Genome 结果显示：四种轻量训练/架构组合均有适应后 Q95 均值降低；STATS 则所有非零 K 的跨 seed Q95 均值均高于本方法 K=0。负迁移在这批对照中未被消除，但不能据此认证原因是 support 分布、架构或训练目标。', '',
              '### 4.2 元学习对照必须限定架构与样本量', '',
              f"- set 架构 Genome K=0：FOMAML **{value('FOMAML-set','genome',0):.2f}** vs ERM **{value('ERM-set','genome',0):.2f}**；K=5/10 的 FOMAML Q95 均值也更低。可写当前配置下的小样本/初始化优势，不写所有架构必然获益。",
              f"- sum 架构 Genome K=20：FOMAML **{value('FOMAML-sum','genome',20):.2f}** vs ERM **{value('ERM-sum','genome',20):.2f}**；K=5/10/20/50/100 的 ERM-sum 均值均更低。跨架构元学习优势不稳健。",
              f"- STATS sum K=5：FOMAML **{value('FOMAML-sum','stats',5):.2f}** vs ERM **{value('ERM-sum','stats',5):.2f}**，但二者仍差于自己的零样本。相对另一训练方法的优势不等于适应改善。", '',
              '### 4.3 必须纠正的配对参照解释', '',
              '**所有目标模型摘要的 `reference_method` 是 `pg`。** `paired_unique_vs_reference` 和聚合后的 `fraction_queries_improved` 是与 PG 比较，不能写成与该模型 K=0 比较。旧文档中以“配对改善比例”支持适应恶化查询占比的语句不可直接搬进论文。',
              '本报告只用同模型不同 K 的已记录 Q95 均值描述适应趋势；没有模型 K>0 与其 K=0 的逐查询配对，就不填“适应造成多少比例查询恶化”，也不填适应因果解释。', '',
              '## 5. 结构选择与轻量化', '',
              '### 5.1 源验证选结构：macro mean log-Q95（不是目标主表指标）', '',
              '| 结构 | 三 seed 源验证均值 | seed 样本 SD |', '|---|---:|---:|']
    for row in architecture:
        lines.append(f"| {row['arch']} | {row['source_macro_log_q95_mean']:.6f} | {row['source_macro_log_q95_seed_sample_std']:.6f} |")
    lines += ['', '按既有源验证均值规则选择 sum，而不是根据本报告目标库表格重新挑模型。历史配置的 early stopping、best checkpoint 与参数量不同；目标库 set/sum 差异不能单独归因于架构。本轮不复算历史 p 值，也不把有限变体负结果写成排除了表示层瓶颈。', '',
              '### 5.2 参数规模（FP32 纯参数，不含优化器/元数据）', '',
              '| 模型 | 参数数 | FP32 参数字节 | PRICE / 本模型参数比 |', '|---|---:|---:|---:|']
    for method in ['FOMAML-set','FOMAML-sum','PRICE']:
        p = parameters[method]
        lines.append(f"| {method} | {p['parameters']:,} | {p['fp32_parameter_bytes']:,} | {parameters['PRICE']['parameters']/p['parameters']:.2f}× |")
    lines += ['| P1 映射模型 | 169,665 | 678,660 | 不沿用 set 的 107×主张 |', '',
              '小模型参数规模有直接记录支撑，但不等于 checkpoint 文件体积、训练成本或端到端加速。不得把“107×”写在当前 sum 模型上；sum 对 PRICE 的参数比例约 80.80×。历史估算的样本曝光差距不作为测得训练成本比。', '',
              '## 6. P0：固定训练预算消融（单 seed、源侧）', '',
              '| 更新预算 | 实际样本呈现（含重复） | optimizer updates |', '|---:|---:|---:|']
    for budget in ['4000','20000']:
        b = p0['budget_summaries'][budget]
        lines.append(f"| {budget} | {b['counts']['query_presentations']:,} | {b['counts']['optimizer_steps']:,} |")
    lines += ['', f"冻结主指标为**每库 4+ 表各桶 long/short Q95 比的几何平均，再对三库几何平均**，不是把全体查询合并算分位数。总比值 **{p0['primary_macro_ratio']:.6f}**；" + '、'.join(f"{db}={ratio:.6f}" for db,ratio in p0['database_ratios'].items()) + '。大于 1 表示长训练尾部更差。',
              '结论限于 seed42、ERM-sum、既有 optimizer/sampling 与 4000→20000 步续训：本条件下增加预算未改善冻结尾部指标。次要适应指标方向可能不同，不代替主指标；不能排除其他规模/配置下更多训练有益。没有目标库评估，也不是 PRICE 训练量匹配试验。', '',
              '## 7. P1：映射消融（九臂、源侧零样本）', '',
              '| seed | sum | true mapping | shuffled mapping |', '|---:|---:|---:|---:|']
    for row in p1_rows:
        lines.append(f"| {row['seed']} | {row['sum']:.4f} | {row['true']:.4f} | {row['shuffled']:.4f} |")
    lines += [f"| 三 seed 算术平均 | {means['sum']:.4f} | {means['true']:.4f} | {means['shuffled']:.4f} |", '',
              f"主指标为 **4+ 表 unique-query Q95 的三库等权算术 macro**；共同完整计划有 149,504 唯一查询，4+ 表主桶 63,436 条。真实映射跨 seed 平均比打乱低 **{100*(1-means['true']/means['shuffled']):.4f}%**，但比 sum 高 **{100*(means['true']/means['sum']-1):.4f}%**。这是均值之比，不是各 seed 比值的平均。",
              '九次 4000 步、冻结 last.pt；评估计划九臂一致，训练采样/呈现计数只要求同 seed 三臂一致，不同 seed 的训练采样 digest 确实不同，不能写成九次训练共享一个采样流。true/shuffled 同 seed 初始化哈希一致。true/shuffled 同容量是主要信息对照；true/sum 参数量 169,665 vs 132,545，不是纯信息消融。seed42 的 true 未胜 shuffled，打乱仍可能保留部分原关联，收益非普遍、未检验显著；此批 ERM/零样本源侧实验不能等同于旧 FOMAML 少样本目标库结果，不能宣称映射模型解决跨库尾部。', '',
              '## 8. P2 与缺口登记', '',
              '- G0 来源核验、A 共享解析器的结构层及 B 冻结编码核验完成；独立数据库语义、同版本真实源表/G1/C 仍 blocked，按用户决定暂缓。编码一致不等于真实选择率准确，不改历史状态 JSON。',
              '- IMDB 完整 few-shot 结果缺失；本地未找到可纳入统一表的 Random 正式摘要。主文不造数、不偷偷排除查询，不因历史文档说做过就当作本轮已核验。',
              '- ERM-set 的完整历史评估 run_config 未包含于本地结果包；已核验摘要 metadata、支持计划及数值，不能扩大成历史命令/目标数据来源全链认证。',
              '- 只有三个训练 seed；support 抽样与 seed 两层不确定性分别保留。结构/训练条件未完全成本匹配，目标库也已用于既有评估，后续改进不能把这些同一目标结果再称为全新未见测试。没有原始表计数认证或新显著性分析。', '',
              '## 9. 建议论文实验叙事（只使用已支持的结论）', '',
              '1. **问题与方法**：研究离线监督式跨数据库少样本估计。以数据库级划分、目标禁入选择、支持/测试去重和无状态适应建立实验边界。',
              '2. **主对照**：先分别报告 set 与 sum 的 FOMAML/ERM；PRICE 作为非严格训练曝光匹配的强参考。尾部未超过 PRICE 的结果明确呈现，不以 Q50 胜出替代主 Q95。',
              '3. **少样本机制与失败边界**：Genome 展现适应后尾部改善；STATS 负迁移仍在。元学习收益随架构与 K 变化，不写“所有库/所有 K 都提高”。',
              '4. **轻量化**：使用实际参数数与纯 FP32 字节；分别写 set 107.33×、sum 80.80×的参数差距。不用不匹配设备/历史训练估算推出公平速度或成本优势。',
              '5. **消融与诊断**：P0 当前预算增长无主指标改善；P1 真实映射对打乱的局部尾部收益未胜 sum；P2 只验证到冻结数值链。负结果限定条件，不宣称已排除根因。',
              '6. **限制**：三目标库覆盖、少 seed、不匹配 PRICE 源曝光、未知真实过滤计数，以及尚未做端到端数据库评估。', '',
              '### 可直接采用的结论段落（草稿）', '',
              '> 本研究在既有离线查询与基数标签上评估轻量级跨数据库估计及少样本适应。轻量模型大幅减少了参数规模，并在部分数据库和样本预算下表现出有利的中位数或适应后尾部误差趋势；但其尾部精度尚未超过所比较的 PRICE 预训练参考，元学习相对 ERM 的收益依赖架构与样本预算，STATS 上仍存在适应负迁移。固定预算与映射消融进一步表明，所测试的长训练和映射方案未能改善当前参考模型的整体尾部表现。上述结果支持有边界的轻量化与适应研究，不构成真实选择率认证、预测误差因果解释或数据库端到端加速的证据。', '',
              '## 10. 可复核产物与本轮验证', '',
              '- [机器可读汇总与输入 SHA256](results/offline-synthesis-20261005/summary.json)：记录来源、口径、分层汇总、P0/P1/P2 和缺口；性质是派生分析，不是新实验。',
              '- [目标汇总 CSV](results/offline-synthesis-20261005/target-aggregate.csv)：所有模型、三库、六个 K、Q50/Q90/Q95/Q99/均值及训练 seed SD。',
              '- [每初始化与 support 不确定性 CSV](results/offline-synthesis-20261005/per-initialization.csv)：不把 PRICE 的 support SD 当作训练 seed SD；配对改善列明确命名 `vs_pg`。',
              '- [复现脚本](../reports/offline-synthesis-20261005/reproduce.py)：只读本地 JSON、写新汇总与本文；仅依赖 Python 标准库，运行不访问服务器、数据缓存或模型 checkpoint。',
              '- [本次分析验证日志](../reports/offline-synthesis-20261005/verification.log)和 [8 项导出回归测试日志](../reports/offline-synthesis-20261005/tests.log)：核验确定性重建，并在临时副本中故意破坏聚合、support 身份、配对参照、训练划分、P0 比值及 P1 同 seed 采样，确认拒绝不一致数据。',
              '- 本轮重算核验摘要统计、跨方法 support 对齐、P0 几何指标与 P1 算术指标，并核对历史结果文件哈希未变；文档链接、git 空白及分析脚本 Ruff 检查另行执行。未重跑模型工程测试，不沿用历史 160 项通过作为本轮分析验证。',
              '- 历史 01–13 文档保留作为阶段记录；其中因果推断、配对参照或训练成本推测不能未经核验直接作为论文结论。本报告用于新的离线主线总结。',
              '- 下一步优先完成论文实验章节与图表编排；若需要新实验，另行冻结源验证假设及预算，不从已看过的目标表中挑模型、K 或 checkpoint。', '',
              '### 本次读取的原始来源（文件内容哈希见机器汇总）', '',
    ]
    for name in SOURCE_HASHES:
        relative = Path(name).relative_to('实验文档') if name.startswith('实验文档/') else None
        link = str(relative) if relative else '../' + name
        lines.append(f'- [{name}]({link})')
    for name, expected in SOURCE_HASHES.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    DOCUMENT.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'target_summary_files': len(checks), 'episode_records_verified': sum(c['support_rows_checked'] for c in checks),
                      'per_initialization_rows': len(seed_rows), 'target_aggregate_rows': len(aggregates),
                      'source_files_hash_verified': len(SOURCE_HASHES), 'support_alignment': 'passed',
                      'p0_geometric_metric': 'passed', 'p1_arithmetic_metric': 'passed',
                      'paired_reference': 'pg', 'source_results_modified': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
