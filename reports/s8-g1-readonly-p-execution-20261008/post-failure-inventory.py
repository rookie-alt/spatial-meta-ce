"""Post-failure evidence inventory only; never open old-run inputs."""
import hashlib
import json
from pathlib import Path
import stat

package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-code-20261007')
proof=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-proof-20261007')
expected={'scripts/verify_s8_readonly_p.py': '0f781f3500ea438eebe5c1d4962da0c454be3826a81c85df75412d71f97fcf35', 'spatial_meta_ce/__init__.py': '21e38374a0bde55bcfa0fbe1d6c890cd4b3cefb9d5606f00d10a141683f332bd', 'spatial_meta_ce/s8_probe.py': 'cea49d50eefe814c4ee3ad315d047787f897170d39ad855b18229f796b0cb12a', '实验文档/27-S8G1运行器与真实smoke协议.md': '4dc59409929a09f9bc75b5179205c06a9a46c2dc30740695c14671372de41a0f', '实验文档/34-S8G1工程验收v2批准协议.md': '6b6ef2dddaa79f02673122ae4104000ec94a41f7984737af127759eceb326076', 'reports/s8-g1-readonly-p-20261007/proof.schema.json': '8b375ff815463436736bb55fa32984fbd2f8d3ec7dabb55e455d65c7810665cb', 'tests/test_s8_readonly_p.py': '7113545022afaca7aa76601f6866b208b0b8c88d3714595eba69f9b63292b93a', 'reports/s8-g1-readonly-p-20261007/execution-contract.json': '5d1ea6f4c9cd0f061f3db9ad5d9d7a88485485a88a169c213ddd7a25ed7e8650'}
actual={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in expected}
if actual!=expected: raise ValueError('Deployed frozen code changed')
records={}
for path in sorted(proof.iterdir()):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Unexpected non-regular new proof member')
    if path.suffix not in ('.json','.log') or path.stat().st_size>10*1024*1024:
        raise ValueError('Unexpected new proof member/size')
    records[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
print(json.dumps({'status':'post_failure_new_evidence_and_frozen_code_inventory',
    'deployed_files_sha256':actual,'code_unchanged_after_single_attempt':True,
    'new_proof_files':records,'old_run_inputs_opened_by_inventory':[],
    'no_rerun':True},indent=2))
