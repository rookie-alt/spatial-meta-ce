"""Transport a bounded, read-only S8 metadata preflight; no deployment or training."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
REFERENCE = ROOT / '实验文档/results/s6-mean-tail-20261007/preflight.json'
reference_hash = hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
old = json.loads((ROOT / '实验文档/results/s7-fit-generalization-20261007/protocol.json').read_text())
replay_hash = old['code_sha256']['scripts/s2_source_replay.py']
code = {name: (ROOT / name).read_bytes() for name in (
    'spatial_meta_ce/s8_contract.py', 'scripts/run_s8_preflight.py')}
encoded = {name: base64.b64encode(value).decode() for name, value in code.items()}
remote = f'''
import base64, hashlib, importlib.util, json, os, resource, signal, sys, time, types
from pathlib import Path
signal.alarm(300)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3, 8 * 1024 ** 3))
root = Path('/gly/weilin/spatial-meta-ce')
reference = root / 'runs/s6-mean-tail-20261007/preflight.json'
replay_path = root / 'runs/s7-execution-code-20261007/scripts/s2_source_replay.py'
for path, expected in [(reference, {reference_hash!r}), (replay_path, {replay_hash!r})]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Preflight input source authentication failed')
script_parent = types.ModuleType('scripts')
script_parent.__path__ = []
sys.modules['scripts'] = script_parent
spec = importlib.util.spec_from_file_location('scripts.s2_source_replay', replay_path)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
sys.modules['scripts.s2_source_replay'] = replay
script_parent.s2_source_replay = replay
codes = {{name: base64.b64decode(value) for name, value in {encoded!r}.items()}}
contract = types.ModuleType('spatial_meta_ce.s8_contract')
exec(compile(codes['spatial_meta_ce/s8_contract.py'], 's8_contract.py', 'exec'), contract.__dict__)
sys.modules['spatial_meta_ce.s8_contract'] = contract
namespace = {{'__name__': 's8_preflight_memory', '__file__': '/gly/weilin/spatial-meta-ce/scripts/run_s8_preflight.py'}}
exec(compile(codes['scripts/run_s8_preflight.py'], 'run_s8_preflight.py', 'exec'), namespace)
start = time.monotonic()
result = namespace['preflight'](root, reference)
result.update({{'seconds': time.monotonic() - start,
               'code_sha256': {{name: hashlib.sha256(value).hexdigest() for name, value in codes.items()}},
               'replay_sha256': {replay_hash!r}, 'timeout_seconds': 300,
               'address_space_bytes': 8 * 1024 ** 3, 'server_files_written': False,
               'source_label_values_read_for_H': False,
               'source_token_values_read_for_H': False,
               'peak_rss_kib_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}})
for path, expected in [(reference, {reference_hash!r}), (replay_path, {replay_hash!r})]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Preflight reference changed during operation')
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
'''
result = subprocess.run([
    'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'myserver',
    ("timeout 310s env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "
     'OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
     '/gly/weilin/PRICE-master/venv/bin/python -B -'),
], input=remote, text=True, capture_output=True, timeout=330, check=False)
(REPORTS / 'source-preflight.log').write_text(result.stderr + result.stdout)
if result.returncode:
    raise SystemExit(f'S8 metadata preflight failed ({result.returncode}); retained log, no retry')
proof = json.loads(result.stdout)
expected = {name: hashlib.sha256(value).hexdigest() for name, value in code.items()}
if proof['code_sha256'] != expected or proof['state'] != 'verified_metadata':
    raise ValueError('S8 metadata proof/code mismatch')
if proof['formal_ready'] or proof['training'] or proof['inference'] or proof['target_data_read']:
    raise ValueError('S8 preflight scope changed')
(REPORTS / 'source-preflight.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: proof[k] for k in (
    'state', 'formal_ready', 'totals', 'P_unique', 'Q_unique', 'seconds',
    'peak_rss_kib_linux', 'inputs_unchanged', 'H_identity_counts_sha_only')}, indent=2))
