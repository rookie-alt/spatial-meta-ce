"""Read-only fixed-matrix failure diagnosis; no F0 training or smoke restart."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
remote = '''
import hashlib,json,resource,signal,sys
from pathlib import Path
import numpy as np
signal.alarm(300)
resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-execution-code-20261007')
output=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-smoke-20261007')
sys.path.insert(0,str(package))
from spatial_meta_ce.s8_probe import standardize_fit,residual_targets,solve_ls,certify_pair
names=['P-h.npy','P-x.npy','P-label.npy','P-pg_log.npy','last.pt','status.json','contract.json']
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(4*1024*1024),b''):h.update(c)
 return h.hexdigest()
before={n:sha(output/n) for n in names}
selected=np.array([i*1024+j for i in range(19) for j in range(64)])
target=residual_targets(np.load(output/'P-label.npy',allow_pickle=False),np.load(output/'P-pg_log.npy',allow_pickle=False))[selected]
records={}
for feature in ['h','x']:
 z=np.load(output/f'P-{feature}.npy',allow_pickle=False)[selected]
 design,normalizer=standardize_fit(z)
 theta=np.zeros(design.shape[1]);m=np.zeros_like(theta);v=np.zeros_like(theta)
 # Independent fixed 2-step Adam equations; no torch optimizer/backward.
 for step in range(1,3):
  gradient=2*design.T@(design@theta-target)/len(target)
  m=.9*m+.1*gradient;v=.999*v+.001*gradient**2
  theta-=3e-4*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
 adam={'theta':theta,'objective':float(np.mean((design@theta-target)**2)),'updates':2}
 ls=solve_ls(design,target)
 certificate=certify_pair(design,target,adam,ls)
 singular=np.linalg.svd(design,compute_uv=False)
 delta=np.abs(ls['singular_values']-singular)
 tolerance=1e-12*np.abs(singular)
 violating=np.flatnonzero(delta>tolerance)
 records[feature]={'certificate':certificate,'rank':ls['rank'],'columns':design.shape[1],
                   'zero_std_columns':int(np.count_nonzero(normalizer['zero_mask'])),
                   'singular_mismatch_count':len(violating),
                   'singular_max_abs_difference':float(delta.max()),
                   'singular_max_value':float(singular[0]),
                   'mismatch_examples':[{'index':int(i),'lstsq_value':float(ls['singular_values'][i]),
                       'svd_value':float(singular[i]),'abs_difference':float(delta[i]),
                       'atol_zero_relative_limit':float(tolerance[i])} for i in violating[:8]],
                   'J_adam_independent':adam['objective'],'J_LS':ls['objective']}
after={n:sha(output/n) for n in names}
assert before==after
print(json.dumps({'status':'read_only_failure_diagnosis','original_state':'failed',
                 'original_files_unchanged':True,'file_sha256':before,'records':records,
                 'F0_updates_performed':0,'smoke_restarted':False,'H_data_read':False,
                 'GPU_used':False,'formal_ready':False},ensure_ascii=False,indent=2))
'''
result = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'myserver',
     ("timeout 310s env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
      '/gly/weilin/PRICE-master/venv/bin/python -B -')],
    input=remote, text=True, capture_output=True, timeout=330, check=False,
)
(REPORTS / 'solver-diagnosis.log').write_text(result.stderr + result.stdout)
if result.returncode:
    raise SystemExit('Read-only diagnosis failed; original smoke not restarted')
proof=json.loads(result.stdout)
(REPORTS / 'solver-diagnosis.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(proof,ensure_ascii=False,indent=2))
