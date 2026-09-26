#!/usr/bin/env python3
"""Opt-in experiment runner. Without --execute, print the workload only."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--execute',action='store_true')
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--ablations',action='store_true')
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    variants=[('main',True,'all')]
    if args.ablations:
        variants += [('no_pg',False,'all'),('head_only',True,'head')]
    specification={'variants':variants,'methods':['erm','fomaml'],'seeds':[42,43,44],
                   'inner_lr_candidates':[0.001,0.01],'steps':args.steps,'device':args.device,
                   'selection':'minimum source-validation macro log-Q95; never target test'}
    print(json.dumps(specification,indent=2))
    if not args.execute:
        print('Dry run only. Add --execute AFTER reserving a GPU or CPU budget.')
        return
    import torch
    from spatial_meta_ce.cli import read_json,data_settings,safe_output
    from spatial_meta_ce.train import train
    from spatial_meta_ce.checkpoints import load_checkpoint
    from spatial_meta_ce.evaluate import evaluate
    torch.set_num_threads(args.threads)
    output=Path(args.output).resolve()
    dc=data_settings(read_json(ROOT/'configs/data.json'),ROOT/'configs/data.json')
    safe_output(output,dc)
    output.mkdir(parents=True,exist_ok=False)
    (output/'specification.json').write_text(json.dumps(specification,indent=2))
    chosen=[]
    # Complete selection before running any target evaluation.
    for variant,use_pg,scope in variants:
        for method in specification['methods']:
            for seed in specification['seeds']:
                trials=[]
                for lr in specification['inner_lr_candidates']:
                    cfg=read_json(ROOT/'configs'/f'{"meta" if method=="fomaml" else "erm"}.json')
                    cfg.update(seed=seed,steps=args.steps,inner_lr=lr,adapt_scope=scope)
                    cfg['model']['use_pg']=use_pg
                    folder=output/'training'/variant/method/f'seed{seed}-lr{lr}'
                    result=train(cfg,dc,folder,args.device)
                    trials.append((result['best_validation_macro_log_q95'],str(folder/'best.pt'),cfg))
                score,path,cfg=min(trials,key=lambda x:x[0])
                chosen.append({'variant':variant,'method':method,'seed':seed,'score':score,'path':path,'config':cfg})
    # Tune supplementary adaptation baselines on source validation only.
    from spatial_meta_ce.data import StorePool
    from spatial_meta_ce.train import validation_plan, validate, seed_all
    from spatial_meta_ce.baselines import PriceAdapter
    from spatial_meta_ce.model import SetEstimator
    import numpy as np
    vp=StorePool(dc['cache_root'])
    vc=read_json(ROOT/'configs/meta.json')
    episodes=validation_plan(vp,dc['splits']['validation'],vc)
    maxima=np.ones(4,dtype=np.int64)
    for db in dc['splits']['validation']:
        maxima=np.maximum(maxima,vp.get('pretrain',db).manifest['inspection']['max_counts'])
    baseline_choices={}
    for baseline,seeds,rates in [('price',[42],[1e-6,1e-5,1e-4]),('random',specification['seeds'],[0.001,0.01])]:
        for seed in seeds:
            seed_all(seed)
            model=(PriceAdapter(Path(dc['data_root'])/'results/model_params.pth',maxima,args.device)
                   if baseline=='price' else SetEstimator(**vc['model']).to(args.device))
            trials=[]
            for lr in rates:
                candidate=dict(vc,inner_lr=lr)
                score,records=validate(model,vp,episodes,candidate,args.device)
                trials.append({'inner_lr':lr,'score':score,'records':records})
            winner=min(trials,key=lambda t:t['score'])
            baseline_choices[f'{baseline}-{seed}']={'inner_lr':winner['inner_lr'],'trials':trials}
    (output/'frozen_selection.json').write_text(json.dumps({'trained_models':chosen,'baselines':baseline_choices},indent=2,allow_nan=False))
    rows=[]
    ec=read_json(ROOT/'configs/evaluation.json')
    for choice in chosen:
        settings=copy.deepcopy(ec)
        settings.update(inner_lr=choice['config']['inner_lr'],adapt_scope=choice['config']['adapt_scope'],model=choice['config']['model'])
        report=evaluate(settings,dc,output/'evaluation'/choice['variant']/choice['method']/f'seed{choice["seed"]}',args.device,checkpoint=choice['path'])
        rows.append({'variant':choice['variant'],'method':choice['method'],'seed':choice['seed'],'aggregate':report['aggregate']})
    for baseline in ('pg','price','random'):
        for seed in (specification['seeds'] if baseline=='random' else [42]):
            settings=copy.deepcopy(ec)
            if baseline!='pg': settings['inner_lr']=baseline_choices[f'{baseline}-{seed}']['inner_lr']
            evaluate(settings,dc,output/'evaluation'/baseline/f'seed{seed}',args.device,baseline=baseline,training_seed=seed)
    # Keep per-seed distributions separate; don't treat repeats as independent training runs.
    import numpy as np
    groups={}
    for row in rows:
        for dbrow in row['aggregate']:
            for metric,values in dbrow['across_support_runs'].items():
                key=(row['variant'],row['method'],dbrow['database'],dbrow['k'],metric)
                groups.setdefault(key,[]).append(values['mean'])
    summary=[{'variant':v,'method':m,'database':db,'k':k,'metric':metric,
              'training_seed_mean':float(np.mean(vals)),'training_seed_std':float(np.std(vals,ddof=1)),
              'seed_means':vals} for (v,m,db,k,metric),vals in groups.items()]
    (output/'across_training_seeds.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print('Experiment outputs:',output)


if __name__=='__main__':
    main()
