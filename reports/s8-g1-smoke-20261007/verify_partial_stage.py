"""Read-only authentication of completed F0 stage in a FAILED G1 run."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
manifest = json.loads((REPORTS / 'deployment-manifest.json').read_text())
remote = f'''
import hashlib,json,resource,signal,sys
from pathlib import Path
import numpy as np
signal.alarm(300)
resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,12*1024**3))
package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-execution-code-20261007')
output=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-smoke-20261007')
sys.path.insert(0,str(package))
from scripts.run_s8_preflight import source_identity_metadata,safe_hash
from scripts.s8_independent_verify import TRAIN,_panels,_episodes
expected_code={manifest['file_sha256']!r}
for name,h in expected_code.items():
 if safe_hash(package/name)!=h:raise ValueError('Original execution package changed')
preflight=json.loads((output/'preflight.json').read_text())
for path,h in preflight['input_sha256'].items():
 if safe_hash(path)!=h:raise ValueError('Original source files changed')
reference=json.loads(Path('/gly/weilin/spatial-meta-ce/runs/s6-mean-tail-20261007/preflight.json').read_text())
caches={{db:source_identity_metadata(Path('/gly/weilin/spatial-meta-ce/artifacts/cache/pretrain')/db,reference['sources'][db],labels=False) for db in TRAIN}}
bindings=json.loads((output/'contract.json').read_text())
panels=json.loads((output/'panels.json').read_text())
panel_sha=_panels(bindings,panels,caches)
episodes=[json.loads(s) for s in (output/'episodes.jsonl').read_text().splitlines()]
exposed=_episodes(episodes,panels,caches)
counts={{'optimizer_steps':2,'backward_calls':len(episodes),'support_role_presentations':sum(len(r['support_positions']) for r in episodes),'query_role_presentations':sum(len(r['query_positions']) for r in episodes)}}
events=[json.loads(s) for s in (output/'events.jsonl').read_text().splitlines()]
reload=[r for r in events if r['event']=='F0_checkpoint_reloaded']
assert len(reload)==1 and reload[0]['maximum_raw_delta']==0
assert safe_hash(output/'last.pt')==reload[0]['checkpoint_sha256']
assert not any(r['event'] in ('H_semantic_load','nomination_sealed') for r in events)
status=json.loads((output/'status.json').read_text())
assert status['status']=='failed'
for path,h in preflight['input_sha256'].items():
 if safe_hash(path)!=h:raise ValueError('Sources changed during partial verification')
for name,h in expected_code.items():
 if safe_hash(package/name)!=h:raise ValueError('Code changed during partial verification')
print(json.dumps({{'partial_stage_verification_passed':True,'overall_smoke_status':'failed',
'formal_ready':False,'episodes_verified':len(episodes),'training_unique_exposed':exposed,
'actual_training_counts_from_record':counts,'panel_identity_digest':panel_sha,
'checkpoint_byte_sha256':reload[0]['checkpoint_sha256'],'checkpoint_reload_record_raw_delta':0,
'checkpoint_reload_record_only_not_second_inference':True,'H_events_absent':True,
'inputs_rehashed_before_and_after':len(preflight['input_sha256']),
'execution_code_files_rehashed':len(expected_code),'F0_updates_performed_by_verifier':0,
'new_inference':False,'target_data_read':False,'GPU_used':False,
'claim':'P-only sample identity/provenance and existing checkpoint bytes authenticated; entire G1 NOT passed'}},ensure_ascii=False,indent=2))
'''
run = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'myserver',
     ("timeout 310s env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "
      'OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 '
      '/gly/weilin/PRICE-master/venv/bin/python -B -')],
    input=remote, text=True, capture_output=True, timeout=330, check=False,
)
(REPORTS / 'partial-stage.log').write_text(run.stderr + run.stdout)
if run.returncode:
    raise SystemExit('Partial-stage verification failed; no original task restart')
proof = json.loads(run.stdout)
(REPORTS / 'partial-stage-verification.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(proof,ensure_ascii=False,indent=2))
