import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, restore_rng, save_checkpoint
from .data import StorePool
from .manifests import digest, episode_indices, validate_splits
from .meta import adapt, forward, loss_value, meta_loss
from .metrics import summarize
from .model import SetEstimator
from .tail_risk import capture_loss, tail_terms


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_best(best_metric, bad_validations, metric, min_delta):
    if not np.isfinite(metric):
        raise ValueError('Validation metric is not finite')
    return (metric, 0, True) if metric < best_metric - min_delta else (best_metric, bad_validations + 1, False)


def validation_plan(pool, databases, config):
    rng = np.random.default_rng(config['validation_seed'])
    episodes = []
    for db in databases:
        store = pool.get('pretrain', db)
        unique = store.unique_indices()
        for k in config['support_sizes']:
            for run in range(config['validation_runs']):
                si, qi = episode_indices(unique, k, config['query_size'], rng)
                episodes.append({'db':db,'k':k,'run':run,'support':si.tolist(),'query':qi.tolist(),
                                 'support_ids':[store.rows[int(i)]['id'] for i in si],
                                 'query_ids':[store.rows[int(i)]['id'] for i in qi],
                                 'support_hashes':[store.rows[int(i)]['query_hash'] for i in si],
                                 'query_hashes':[store.rows[int(i)]['query_hash'] for i in qi]})
    return episodes


def validate(model, pool, episodes, config, device):
    model.eval()
    records = []
    for ep in episodes:
        store = pool.get('pretrain', ep['db'])
        support = store.batch(ep['support'], device)
        query = store.batch(ep['query'], device)
        from .baselines import PriceAdapter, price_log_predictions, price_support
        is_price = isinstance(model, PriceAdapter)
        if is_price:
            support = price_support(support)
        params = adapt(model, support, config['inner_lr'], config['inner_steps'],
                       config['adapt_scope'], config['loss'], training=False)
        with torch.no_grad():
            prediction = forward(model, params, query).cpu().numpy()
        if is_price:
            prediction = price_log_predictions(prediction)
        metrics = summarize(prediction, query['true'].cpu().numpy())
        records.append({'db':ep['db'],'k':ep['k'],'run':ep['run'],'metrics':metrics})
    # Equal database, K and repeat counts: macro mean, not pooled query weighting.
    return float(np.mean([r['metrics']['log_q95'] for r in records])), records


def _legacy_sample_plan(config: dict, databases: list, pools: dict,
                        pool: StorePool, steps: int, generator: dict) -> tuple[str, int]:
    rng = np.random.default_rng(config['seed'] + 1)
    previous = hashlib.sha256(b'').hexdigest()
    for _ in range(steps * config['meta_batch']):
        db = str(rng.choice(databases))
        k = int(rng.choice(config['support_sizes']))
        si, qi = episode_indices(pools[db], k, config['query_size'], rng)
        store = pool.get('pretrain', db)
        episode = {'db': db, 'support': [store.rows[int(i)]['query_hash'] for i in si],
                   'query': [store.rows[int(i)]['query_hash'] for i in qi]}
        previous = digest({'previous': previous, 'episode': episode})
    if rng.bit_generator.state != generator:
        raise ValueError('Legacy sampling RNG cannot be reconstructed')
    return previous, steps * config['meta_batch']


def train(config, data_config, output_dir, device='cpu', resume=None):
    s6 = 'tail_risk_mode' in config
    s5 = 'complex_query_weight' in config
    if s6:
        if config['tail_risk_mode'] not in ('mean', 'mean_tail10'):
            raise ValueError('tail_risk_mode must be mean or mean_tail10')
        if s5:
            raise ValueError('tail_risk_mode forbids complex_query_weight')
        if (config['method'] != 'erm' or config['loss'] != 'smooth_l1'
                or config['accumulation'] != 1 or config['meta_batch'] != 4):
            raise ValueError('tail_risk_mode requires ERM/SmoothL1/accumulation1/meta_batch4')
        if resume:
            raise ValueError('S6 does not authorize resume; train from scratch')
        loss_evidence = {'audit_rows': 0, 'snapshot_rows': 0, 'mode': config['tail_risk_mode']}
    if s5:
        weight = config['complex_query_weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight not in (1, 2):
            raise ValueError('complex_query_weight must be numeric 1 or 2')
        if config['method'] != 'erm' or config['loss'] != 'smooth_l1':
            raise ValueError('complex_query_weight requires method=erm and loss=smooth_l1')
    validate_splits(data_config['splits'])
    for key in ['steps','meta_batch','accumulation','query_size','validate_every','validation_runs','patience']:
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if config['method'] not in ('fomaml','erm') or min(config['support_sizes']) < 1:
        raise ValueError('Invalid training method or support sizes')
    mapped = config.get('model', {}).get('arch') == 'sum_map'
    mapping_config = config.get('mapping')
    if mapped and not mapping_config:
        raise ValueError('sum_map training requires mapping configuration')
    if not mapped and mapping_config:
        raise ValueError('Mapping configuration requires sum_map architecture')
    if mapping_config and mapping_config.get('mode') not in ('mapping', 'shuffled_mapping'):
        raise ValueError('Invalid mapping mode')
    seed_all(config['seed'])
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError('Use a new run directory or explicit --resume')
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = data_config['splits']
    sources = splits['train'] + splits['validation']
    # Opening a mapped store parses its whole sidecar (~3.5s). Keep every source
    # store resident so uniform database sampling cannot thrash the LRU.
    pool = StorePool(data_config['cache_root'], max_open=len(sources),
                     mapping_root=mapping_config['root'] if mapping_config else None,
                     mapping_mode=mapping_config['mode'] if mapping_config else 'mapping')
    hashes = {}
    pools = {}
    mapping_provenance = {}
    for db in sources:
        store = pool.get('pretrain', db)
        hashes[db] = store.manifest['content_hash']
        pools[db] = store.unique_indices()
        if mapping_config:
            mapping_provenance[db] = store.mapping.provenance()
    signature = {k:v for k,v in config.items() if k not in ('steps','data_config')}
    metadata = {'method':config['method'],'seed':config['seed'],
                'train_databases':splits['train'],'validation_databases':splits['validation'],
                'manifest_hashes': hashes,
                'training_signature':signature,
                'purpose':config.get('purpose','research'),
                'selection':'source-validation macro mean log-Q95 after adaptation'}
    if mapping_config:
        metadata = {**metadata, 'mapping_provenance': mapping_provenance}
    rng = np.random.default_rng(config['seed']+1)
    model = SetEstimator(**config['model']).to(device)
    start, best, bad = 0, float('inf'), 0
    history = []
    sample_plan_digest = hashlib.sha256(b'').hexdigest()
    sample_plan_episodes = 0
    counts = {'support_presentations':0,'query_presentations':0,'backward_calls':0,'optimizer_steps':0}
    total_seconds = 0.0
    if s5 or s6:
        gradient_stats = {'count':0,'total_norm':0.0,'max_norm':0.0,'clipped_count':0}
    if resume:
        model, state = load_checkpoint(resume, device)
        if state['metadata'] != metadata:
            raise ValueError('Resume configuration or data manifest changed')
        start, best, bad = state['next_step'], state['best_metric'], state['bad_validations']
        history = state['extra']['history']
        if 'sample_plan_digest' in state['extra']:
            sample_plan_digest = state['extra']['sample_plan_digest']
            sample_plan_episodes = state['extra']['sample_plan_episodes']
        elif not mapped:
            sample_plan_digest, sample_plan_episodes = _legacy_sample_plan(
                config, splits['train'], pools, pool, start, state['extra']['generator'])
        else:
            raise ValueError('Mapped checkpoint is missing its sampling provenance')
        rng.bit_generator.state = state['extra']['generator']
        counts = state['extra']['counts']
        total_seconds = state['extra']['total_seconds']
        if s5:
            if 'gradient_stats' not in state['extra']:
                raise ValueError('Resume checkpoint missing gradient_stats')
            gradient_stats = dict(state['extra']['gradient_stats'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['outer_lr'])
    if resume:
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
    episodes = validation_plan(pool, splits['validation'], config)
    (output_dir/'validation_manifest.json').write_text(json.dumps(episodes, indent=2))
    (output_dir/'config.json').write_text(json.dumps({'training':config,'data':data_config},indent=2))
    started = time.perf_counter()
    last_step = start
    for window_start in range(start, config['steps'], config['accumulation']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        window = min(config['accumulation'], config['steps']-window_start)
        losses = []
        for _ in range(window):
            for _ in range(config['meta_batch']):
                db = str(rng.choice(splits['train']))
                k = int(rng.choice(config['support_sizes']))
                si, qi = episode_indices(pools[db], k, config['query_size'], rng)
                store = pool.get('pretrain', db)
                episode = {'db': db, 'support': [store.rows[int(i)]['query_hash'] for i in si],
                           'query': [store.rows[int(i)]['query_hash'] for i in qi]}
                sample_plan_digest = digest({'previous': sample_plan_digest, 'episode': episode})
                sample_plan_episodes += 1
                if config['method'] == 'fomaml':
                    support, query = store.batch(si,device), store.batch(qi,device)
                    objective = meta_loss(model,support,query,config['inner_lr'],config['inner_steps'],config['adapt_scope'],config['loss'])
                    counts['support_presentations'] += k*config['inner_steps']
                    counts['query_presentations'] += len(qi)
                    counts['backward_calls'] += config['inner_steps']+1
                else:
                    batch = store.batch(np.concatenate([si,qi]),device)
                    if s6:
                        pred = model(batch)
                        terms = tail_terms(pred, batch['label'], config['tail_risk_mode'])
                        objective = (loss_value(pred, batch['label'], config['loss'])
                                     if config['tail_risk_mode'] == 'mean' else terms['objective'])
                        snapshot_rows = capture_loss(
                            output_dir, step=window_start + 1,
                            episode_index=sample_plan_episodes, database=db, k=k,
                            mode=config['tail_risk_mode'], sample_plan_digest=sample_plan_digest,
                            terms=terms, objective=objective, pred=pred, batch=batch,
                            hashes=episode['support'] + episode['query'], steps=config['steps'])
                        loss_evidence = {**loss_evidence,
                                         'audit_rows': loss_evidence['audit_rows'] + 1,
                                         'snapshot_rows': loss_evidence['snapshot_rows'] + snapshot_rows}
                    elif s5 and config['complex_query_weight'] == 2:
                        weights = torch.where(batch['counts'][:,2] >= 4, 2.0, 1.0).to(torch.float32)
                        objective = loss_value(model(batch),batch['label'],config['loss'],weights=weights)
                    else:
                        objective = loss_value(model(batch),batch['label'],config['loss'])
                    counts['query_presentations'] += len(si)+len(qi)
                    counts['backward_calls'] += 1
                losses.append(float(objective.detach()))
                (objective/(window*config['meta_batch'])).backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite outer gradient')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if s5 or s6:
            norm_value = float(norm)
            gradient_stats = {
                'count':gradient_stats['count']+1,
                'total_norm':gradient_stats['total_norm']+norm_value,
                'max_norm':max(gradient_stats['max_norm'],norm_value),
                'clipped_count':gradient_stats['clipped_count']+int(norm_value > 1.0),
            }
        optimizer.step()
        counts['optimizer_steps'] += 1
        last_step = window_start+window
        check = last_step//config['validate_every'] > window_start//config['validate_every'] or last_step == config['steps']
        improved = False
        if check:
            score, records = validate(model,pool,episodes,config,device)
            best,bad,improved = select_best(best,bad,score,config['min_delta'])
            row = {'step':last_step,'train_loss':float(np.mean(losses)),'validation_macro_log_q95':score,
                   'validation':records,'counts':dict(counts)}
            history.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='validation'}),flush=True)
        extra = {'history':history,'generator':rng.bit_generator.state,'counts':counts,
                 'sample_plan_digest': sample_plan_digest,
                 'sample_plan_episodes': sample_plan_episodes,
                 'total_seconds':total_seconds+time.perf_counter()-started}
        if s5 or s6:
            extra = {**extra,'gradient_stats':dict(gradient_stats)}
        if s6:
            extra = {**extra, 'loss_evidence': dict(loss_evidence)}
        save_checkpoint(output_dir/'last.pt',model,optimizer,last_step,best,bad,metadata,extra)
        if improved:
            save_checkpoint(output_dir/'best.pt',model,optimizer,last_step,best,bad,metadata,extra)
        if check and bad >= config['patience']:
            break
    (output_dir/'history.json').write_text(json.dumps(history,indent=2))
    return {'last_step':last_step,'best_validation_macro_log_q95':best,'counts':counts,
            'total_seconds':total_seconds+time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters()),
            'purpose':metadata['purpose'],
            **({'gradient_stats':dict(gradient_stats)} if s5 or s6 else {}),
            **({'loss_evidence':dict(loss_evidence)} if s6 else {})}
