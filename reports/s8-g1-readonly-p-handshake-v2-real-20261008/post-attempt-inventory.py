"""Post-attempt SHA inventory only; no old-run input contents or retries."""
import hashlib
import json
from pathlib import Path
import stat

package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-code-handshake-v2-20261008')
proof=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-proof-handshake-v2-20261008')
expected={'scripts/verify_s8_readonly_p.py': '5320ffe926cf7b56ddf2970b7888f79391e18fb62cf50298127e4eb2af8e5471', 'spatial_meta_ce/__init__.py': '21e38374a0bde55bcfa0fbe1d6c890cd4b3cefb9d5606f00d10a141683f332bd', 'spatial_meta_ce/s8_probe.py': 'cea49d50eefe814c4ee3ad315d047787f897170d39ad855b18229f796b0cb12a', '实验文档/27-S8G1运行器与真实smoke协议.md': '4dc59409929a09f9bc75b5179205c06a9a46c2dc30740695c14671372de41a0f', '实验文档/34-S8G1工程验收v2批准协议.md': '6b6ef2dddaa79f02673122ae4104000ec94a41f7984737af127759eceb326076', 'reports/s8-g1-readonly-p-20261007/proof.schema.json': '8b375ff815463436736bb55fa32984fbd2f8d3ec7dabb55e455d65c7810665cb', 'tests/test_s8_readonly_p.py': 'f4b93b28070278814c3328a140317d54d79c9e52e4a199d9f682d204b65d9654', 'tests/test_s8_readonly_handshake_linux.py': 'e91375e111638c245dea0729ee0784426693416eff552ffff38cdf76cfaac100', 'reports/s8-g1-readonly-p-handshake-v2-20261008/execution-contract.json': '35a7ad5fa21374b33a14f13745af8b297d7b857d7cc741eb548a05c900a67b96'}
actual={name:hashlib.sha256((package/name).read_bytes()).hexdigest() for name in expected}
if actual!=expected: raise ValueError('Deployed frozen v2 package changed after single attempt')
records={}
for path in sorted(proof.iterdir()):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Unexpected non-regular new proof member')
    if path.suffix not in ('.json','.log') or path.stat().st_size>20*1024*1024:
        raise ValueError('Unexpected new proof member/size')
    records[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
print(json.dumps({'status':'post_attempt_v2_frozen_code_and_new_proof_inventory',
    'deployed_files_sha256':actual,'code_unchanged_after_single_attempt':True,
    'new_proof_files':records,'old_run_extra_inputs_opened_by_inventory':[],
    'no_rerun':True},indent=2))
