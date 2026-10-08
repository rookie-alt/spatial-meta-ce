"""Bounded kappa-package deployment/runtime preflight; no old-run reads."""
import hashlib, json, os, resource, sys, time
from pathlib import Path
start=time.monotonic()
package=Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-kappa-code-20261008')
expected={'scripts/verify_s8_readonly_p.py': 'bba63497b9da593827f1f20c60a7766aacc30d470561b9334280396d929f61bf', 'spatial_meta_ce/__init__.py': '21e38374a0bde55bcfa0fbe1d6c890cd4b3cefb9d5606f00d10a141683f332bd', 'spatial_meta_ce/s8_probe.py': '7c7847ce1122f0cf265541781812f4ed8cff2f671c9dc42c8e2abda71e48cfad', '实验文档/27-S8G1运行器与真实smoke协议.md': '4dc59409929a09f9bc75b5179205c06a9a46c2dc30740695c14671372de41a0f', '实验文档/34-S8G1工程验收v2批准协议.md': '6b6ef2dddaa79f02673122ae4104000ec94a41f7984737af127759eceb326076', 'reports/s8-g1-readonly-p-20261007/proof.schema.json': '8b375ff815463436736bb55fa32984fbd2f8d3ec7dabb55e455d65c7810665cb', 'tests/test_s8_readonly_p.py': '9730b3c009955441b1942bb52464d3cebc047a0753fbf259c068acca8aee9707', 'tests/test_s8_retained_tolerance.py': 'eff6c5ff2f6741cb65ae7044e50192a11b78e8da4154b0b03f4ee0666b06ef84', 'reports/s8-g1-readonly-p-retained-kappa-20261008/execution-contract.json': '54f9c24c47062b1a73e5f2be4b651fb143760a1ce689ae75bd27e63567ddd99f'}
if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise ValueError('require Linux + empty CUDA')
if Path(sys.prefix)!=Path('/gly/weilin/PRICE-master/venv'):
    raise ValueError('unexpected interpreter prefix')
sys.path.insert(0,str(package))
from scripts import verify_s8_readonly_p as ro
ro.install_resource_limits()
from spatial_meta_ce import s8_probe
s8_probe.torch.set_num_threads(4); s8_probe.torch.set_num_interop_threads(1)
pools=ro.runtime_threadpools()
actual={n:ro.sha256(package/n) for n in expected}
if actual!=expected: raise ValueError('deployed SHA mismatch')
contract=ro.validate_contract(json.loads(ro.read_regular(package/'reports/s8-g1-readonly-p-retained-kappa-20261008/execution-contract.json')))
ro.check_runtime_binding(contract, s8_probe.np.__version__, s8_probe.torch.__version__, pools)
print(json.dumps({'status':'kappa_runtime_preflight_passed','source_test_contract_SHA_match':True,
 'deployed_files_sha256':actual,'sys_platform':sys.platform,'sys_prefix':sys.prefix,'python_version':sys.version,
 'numpy':s8_probe.np.__version__,'torch':s8_probe.torch.__version__,
 'torch_threads':s8_probe.torch.get_num_threads(),'torch_interop_threads':s8_probe.torch.get_num_interop_threads(),
 'threadpools':pools,'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],'RLIMIT_AS_bytes':list(resource.getrlimit(resource.RLIMIT_AS)),
 'seconds':time.monotonic()-start,'old_run_files_opened':[],'real_P_read':False},indent=2))
