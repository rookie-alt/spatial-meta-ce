"""Post-attempt inventory only; no old-run input contents, no rerun."""
import hashlib,json,stat
from pathlib import Path
package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-diagnostic-code-20261008')
proof=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-diagnostic-proof-20261008')
expected={'scripts/verify_s8_readonly_p.py': '7dc783034d4ce5e48469605296adb253c737b6e866dbdeca34c04591e042bae7', 'spatial_meta_ce/__init__.py': '21e38374a0bde55bcfa0fbe1d6c890cd4b3cefb9d5606f00d10a141683f332bd', 'spatial_meta_ce/s8_probe.py': 'cea49d50eefe814c4ee3ad315d047787f897170d39ad855b18229f796b0cb12a', '实验文档/27-S8G1运行器与真实smoke协议.md': '4dc59409929a09f9bc75b5179205c06a9a46c2dc30740695c14671372de41a0f', '实验文档/34-S8G1工程验收v2批准协议.md': '6b6ef2dddaa79f02673122ae4104000ec94a41f7984737af127759eceb326076', 'reports/s8-g1-readonly-p-20261007/proof.schema.json': '8b375ff815463436736bb55fa32984fbd2f8d3ec7dabb55e455d65c7810665cb', 'tests/test_s8_readonly_p.py': '9730b3c009955441b1942bb52464d3cebc047a0753fbf259c068acca8aee9707', 'reports/s8-g1-readonly-p-diagnostic-20261008/execution-contract.json': '5b350dc533697adbe1bad226542c56b2cdd4c8a8737fcc0c2456f6d7a6bfc1c7'}
actual={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in expected}
if actual!=expected: raise ValueError('Deployed diagnostic package changed after single attempt')
rec={}
for p in sorted(proof.iterdir()):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size<20*1024*1024
    rec[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
print(json.dumps({'status':'post_attempt_diagnostic_inventory','deployed_files_sha256':actual,
  'code_unchanged_after_attempt':True,'new_proof_files':rec,'old_run_extra_inputs_opened':[],'no_rerun':True},indent=2))
