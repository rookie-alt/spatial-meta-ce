"""One bounded deployment/runtime preflight; no old-run data reads or training."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

start = time.monotonic()
package = Path('/gly/weilin/spatial-meta-ce/runs/s8-g1-readonly-p-code-20261007')
expected = {'scripts/verify_s8_readonly_p.py': '0f781f3500ea438eebe5c1d4962da0c454be3826a81c85df75412d71f97fcf35', 'spatial_meta_ce/__init__.py': '21e38374a0bde55bcfa0fbe1d6c890cd4b3cefb9d5606f00d10a141683f332bd', 'spatial_meta_ce/s8_probe.py': 'cea49d50eefe814c4ee3ad315d047787f897170d39ad855b18229f796b0cb12a', '实验文档/27-S8G1运行器与真实smoke协议.md': '4dc59409929a09f9bc75b5179205c06a9a46c2dc30740695c14671372de41a0f', '实验文档/34-S8G1工程验收v2批准协议.md': '6b6ef2dddaa79f02673122ae4104000ec94a41f7984737af127759eceb326076', 'reports/s8-g1-readonly-p-20261007/proof.schema.json': '8b375ff815463436736bb55fa32984fbd2f8d3ec7dabb55e455d65c7810665cb', 'tests/test_s8_readonly_p.py': '7113545022afaca7aa76601f6866b208b0b8c88d3714595eba69f9b63292b93a', 'reports/s8-g1-readonly-p-20261007/execution-contract.json': '5d1ea6f4c9cd0f061f3db9ad5d9d7a88485485a88a169c213ddd7a25ed7e8650'}
if sys.platform != 'linux' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise ValueError('Preflight requires Linux and explicit empty CUDA')
if Path(sys.prefix) != Path('/gly/weilin/PRICE-master/venv'):
    raise ValueError('Unexpected existing interpreter prefix')
sys.path.insert(0, str(package))
from scripts import verify_s8_readonly_p as ro
ro.install_resource_limits()
tempfile.tempdir = str(package)
actual = {name: ro.sha256(package/name) for name in expected}
if actual != expected:
    raise ValueError('Deployed file SHA mismatch; no checker run')
contract = ro.validate_contract(json.loads(ro.read_regular(package/'reports/s8-g1-readonly-p-20261007/execution-contract.json')))
from spatial_meta_ce import s8_probe
s8_probe.torch.set_num_threads(4)
s8_probe.torch.set_num_interop_threads(1)
pools = ro.runtime_threadpools()
ro.check_runtime_binding(contract, s8_probe.np.__version__, s8_probe.torch.__version__, pools)
if ro.runtime_threadpools() != pools:
    raise ValueError('Runtime thread counters changed')
print(json.dumps({'status':'deployment_runtime_preflight_passed',
    'source_and_test_and_contract_SHA_match':True,'deployed_files_sha256':actual,
    'sys_platform':sys.platform,'sys_executable':sys.executable,'sys_prefix':sys.prefix,
    'python_version':sys.version,'numpy':s8_probe.np.__version__,'torch':s8_probe.torch.__version__,
    'torch_threads':s8_probe.torch.get_num_threads(),
    'torch_interop_threads':s8_probe.torch.get_num_interop_threads(),
    'threadpools':pools,'cpu_affinity':sorted(os.sched_getaffinity(0)),
    'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
    'RLIMIT_AS_bytes':list(resource.getrlimit(resource.RLIMIT_AS)),
    'ru_maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'seconds':time.monotonic()-start,'old_run_files_opened':[],
    'training_or_smoke':False,'real_P_read':False},indent=2))
