"""Post-attempt inventory only; no old-run contents, no rerun."""
import hashlib,json
from pathlib import Path
package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-tolerance-code-20261008')
proof=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-tolerance-proof-20261008')
expected={'scripts/verify_s8_readonly_p.py': '85e0ccda9719040276d44993d87f29b8b1c643d818662450daef759b7074c5f3', 'spatial_meta_ce/__init__.py': '21e38374a0bde55bcfa0fbe1d6c890cd4b3cefb9d5606f00d10a141683f332bd', 'spatial_meta_ce/s8_probe.py': '36d4eb6c9a61b9280e3c29705ee23e436bbb4ac81ffde6f30e02415c7f251281', '实验文档/27-S8G1运行器与真实smoke协议.md': '4dc59409929a09f9bc75b5179205c06a9a46c2dc30740695c14671372de41a0f', '实验文档/34-S8G1工程验收v2批准协议.md': '6b6ef2dddaa79f02673122ae4104000ec94a41f7984737af127759eceb326076', 'reports/s8-g1-readonly-p-20261007/proof.schema.json': '8b375ff815463436736bb55fa32984fbd2f8d3ec7dabb55e455d65c7810665cb', 'tests/test_s8_readonly_p.py': '9730b3c009955441b1942bb52464d3cebc047a0753fbf259c068acca8aee9707', 'tests/test_s8_retained_tolerance.py': '74c4eaee07340516a3f444088139233cd182b653c79e8ba5f819f5731b21107a', 'reports/s8-g1-readonly-p-tolerance-20261008/execution-contract.json': 'b85c41f0b04fb394c391f6143008e4d4e5bca907b7c6a1fb35d9c03105f6290e'}
actual={n:hashlib.sha256((package/n).read_bytes()).hexdigest() for n in expected}
if actual!=expected: raise ValueError('deployed tolerance package changed after attempt')
rec={}
for p in sorted(proof.iterdir()):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size<20*1024*1024
    rec[p.name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
print(json.dumps({'status':'post_attempt_tolerance_inventory','deployed_files_sha256':actual,
 'code_unchanged_after_attempt':True,'new_proof_files':rec,'old_run_extra_inputs_opened':[],'no_rerun':True},indent=2))
