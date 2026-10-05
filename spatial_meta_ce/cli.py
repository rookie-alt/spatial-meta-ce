import argparse
import copy
import datetime
import json
import pickle
from pathlib import Path

import torch

from .data import StorePool
from .features import inspect_features, prepare_database
from .manifests import file_hash, validate_splits

PROJECT = Path(__file__).resolve().parent.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def data_settings(config, config_path):
    if 'splits' in config:
        data=config
    else:
        location=Path(config.get('data_config','configs/data.json'))
        if not location.is_absolute(): location=PROJECT/location
        data=read_json(location)
    data=copy.deepcopy(data)
    cache=Path(data['cache_root'])
    if not cache.is_absolute(): cache=PROJECT/cache
    data['cache_root']=str(cache.resolve())
    data['data_root']=str(Path(data['data_root']).resolve())
    validate_splits(data['splits'])
    return data


def safe_output(path, data):
    p=Path(path).resolve()
    source=Path(data['data_root']).resolve()
    if p==source or source in p.parents:
        raise ValueError('Outputs must be outside the original PRICE tree')
    return p


def audit(data, output):
    root=Path(data['data_root'])
    report={'scope':'all configured feature files; exact SQL duplicates; source feature validity',
            'datasets':[],'missing':[],'exact_target_overlap':{}}
    target_hashes={}
    pairs=[('pretrain',db) for db in data['splits']['train']+data['splits']['validation']]
    pairs += [(split,db) for db in data['splits']['test'] for split in ('finetune','test')]
    for split,db in pairs:
        workload=root/'datas/workloads'/split/db/'workloads.sql'
        feature=root/'setup/features'/split/db/'features40.pkl'
        hashes=set(); count=0
        import hashlib
        with workload.open() as stream:
            for line in stream:
                if not line.strip(): continue
                count+=1
                hashes.add(hashlib.sha256(line.split('||')[0].strip().encode()).hexdigest())
        row={'split':split,'db':db,'workload_rows':count,'distinct_exact_sql':len(hashes),
             'workload_sha256':file_hash(workload),'features_present':feature.exists()}
        if db in data['splits']['test']: target_hashes[(split,db)]=hashes
        if feature.exists():
            with feature.open('rb') as stream: features=pickle.load(stream)
            row['features']=inspect_features(features)
            row['feature_sha256']=file_hash(feature)
            del features
        else:
            report['missing'].append(str(feature))
        report['datasets'].append(row)
        print(json.dumps(row),flush=True)
    for db in data['splits']['test']:
        report['exact_target_overlap'][db]=len(target_hashes[('finetune',db)]&target_hashes[('test',db)])
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2))
    return report


def prepare(data, databases=None, allow_exclusions=False):
    selected=set(databases) if databases else None
    pairs=[('pretrain',db) for db in data['splits']['train']+data['splits']['validation']]
    pairs += [(split,db) for db in data['splits']['test'] for split in ('test','finetune')]
    if selected and selected-{db for _,db in pairs}: raise ValueError('Unknown database')
    for split,db in pairs:
        if selected is None or db in selected:
            print(json.dumps(prepare_database(data['data_root'],data['cache_root'],db,split,allow_exclusions)),flush=True)


def smoke(config,data,output):
    from .evaluate import evaluate
    from .train import train
    tiny=copy.deepcopy(config)
    tiny.update(steps=3,meta_batch=1,accumulation=1,inner_steps=1,support_sizes=[5],query_size=8,
                validate_every=1,validation_runs=1,purpose='smoke-not-research')
    d=copy.deepcopy(data)
    d['splits']['train']=['fnhk']
    d['splits']['validation']=['chembl']
    output.mkdir(parents=True,exist_ok=False)
    report={}
    for method in ('fomaml','erm'):
        tiny['method']=method
        report[method]=train(tiny,d,output/method,'cpu')
    ec=read_json(PROJECT/'configs/evaluation.json')
    ec.update(shots=[0,5],support_runs=1,inner_steps=1,batch_size=8,latency_repetitions=5,model=tiny['model'])
    for method in ('fomaml','erm'):
        report[method+'_evaluation']=evaluate(ec,d,output/(method+'_eval'),'cpu',checkpoint=output/method/'best.pt',test_limit=32)
    for baseline in ('pg','random','price'):
        report[baseline]=evaluate(ec,d,output/(baseline+'_eval'),'cpu',baseline=baseline,test_limit=8 if baseline=='price' else 32)
    (output/'smoke_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('SMOKE_COMPLETE',output,flush=True)


def adapt_one(config,data,output,args):
    from .checkpoints import load_checkpoint, save_checkpoint
    from .evaluate import support_plan
    from .meta import adapt
    if not args.checkpoint or not args.database or args.k<1:
        raise ValueError('adapt requires --checkpoint, --database and positive --k')
    if args.database not in data['splits']['test']:
        raise ValueError('Adaptation target must be a held-out database')
    output.mkdir(parents=True,exist_ok=False)
    model,state=load_checkpoint(args.checkpoint,args.device)
    model.eval()
    if args.database in state['metadata']['train_databases']+state['metadata']['validation_databases']:
        raise ValueError('Target appeared in training or model selection')
    pool=StorePool(data['cache_root'])
    ss=pool.get('finetune',args.database); ts=pool.get('test',args.database)
    indices,overlap=support_plan(ss,ts,args.k,config['support_seed'],0)
    params=adapt(model,ss.batch(indices,args.device),config['inner_lr'],config['inner_steps'],config['adapt_scope'],config['loss'],training=False)
    model.load_state_dict(params,strict=True)
    metadata=dict(state['metadata'],adapted_database=args.database,
                  support_ids=[ss.rows[int(i)]['id'] for i in indices],adaptation=config)
    save_checkpoint(output/'adapted.pt',model,None,0,state['best_metric'],0,metadata)
    (output/'support.json').write_text(json.dumps({'metadata':metadata,**overlap},indent=2))


def main():
    parser=argparse.ArgumentParser(description='Lightweight supervised spatial-drift experiments')
    parser.add_argument('command',choices=['audit','prepare','train','adapt','evaluate','smoke'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--output')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    parser.add_argument('--baseline',choices=['pg','random','price'])
    parser.add_argument('--database')
    parser.add_argument('--databases',nargs='+')
    parser.add_argument('--k',type=int,default=5)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--inner-lr',type=float)
    parser.add_argument('--inner-steps',type=int)
    parser.add_argument('--scope',choices=['all','head'])
    parser.add_argument('--no-pg',action='store_true')
    parser.add_argument('--allow-exclusions',action='store_true')
    args=parser.parse_args()
    if args.threads<1: parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    config=read_json(args.config)
    data=data_settings(config,args.config)
    safe_output(data['cache_root'],data)
    for arg,key in [('seed','seed'),('steps','steps'),('inner_lr','inner_lr'),('inner_steps','inner_steps'),('scope','adapt_scope')]:
        value=getattr(args,arg)
        if value is not None: config[key]=value
    if args.no_pg: config['model']['use_pg']=False
    stamp=datetime.datetime.now(datetime.timezone.utc).astimezone().strftime('%Y%m%d-%H%M%S-%f')
    output=safe_output(args.output or str(PROJECT/'runs'/f'{args.command}-{stamp}'),data)
    if args.command=='audit':
        audit(data,output/'audit.json')
    elif args.command=='prepare':
        prepare(data,args.databases,args.allow_exclusions)
    elif args.command=='train':
        from .train import train
        print(json.dumps(train(config,data,output,args.device,args.resume)),flush=True)
    elif args.command=='evaluate':
        from .evaluate import evaluate
        evaluate(config,data,output,args.device,checkpoint=args.checkpoint,baseline=args.baseline,training_seed=args.seed or 42)
    elif args.command=='adapt':
        adapt_one(config,data,output,args)
    else:
        if args.device!='cpu': parser.error('smoke is CPU-only; formal training is a separate action')
        smoke(config,data,output)


if __name__=='__main__':
    main()
