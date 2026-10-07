"""Deploy an isolated exact package, supervise one CPU G1 smoke, archive small proofs."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
manifest = json.loads((REPORTS / 'deployment-manifest.json').read_text())
package = Path(manifest['local_package'])
files = {name: base64.b64encode((package / name).read_bytes()).decode()
         for name in manifest['file_sha256']}
for name, expected in manifest['file_sha256'].items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
        raise ValueError('Package no longer matches reviewed local files')
remote = f'''
import base64, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
root = Path('/gly/weilin/spatial-meta-ce')
package = root / 'runs/s8-g1-execution-code-20261007'
output = root / 'runs/s8-g1-smoke-20261007'
if package.exists() or output.exists():
    raise ValueError('Existing G1 directory; no overwrite or automatic restart')
expected = {manifest['file_sha256']!r}
contents = {files!r}
package.mkdir(exist_ok=False)
for name, text in contents.items():
    path = package / name
    if not path.is_relative_to(package) or '..' in Path(name).parts:
        raise ValueError('Package path escape')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(base64.b64decode(text))
for name, sha in expected.items():
    if hashlib.sha256((package / name).read_bytes()).hexdigest() != sha:
        raise ValueError('Uploaded package hash mismatch')
environment = dict(os.environ)
environment.update({{'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}})
started = time.monotonic()
log = root / 'runs/s8-g1-smoke-20261007.log'
with log.open('x') as stream:
    run = subprocess.run(['timeout', '1810s', '/gly/weilin/PRICE-master/venv/bin/python', '-B',
                          str(package / 'scripts/run_s8_smoke.py'), '--execute', '--data-root', str(root),
                          '--output', str(output)], cwd=package, env=environment,
                         stdout=stream, stderr=subprocess.STDOUT, timeout=1825, check=False)
code_unchanged = all(hashlib.sha256((package / n).read_bytes()).hexdigest() == h for n,h in expected.items())
result = {{'returncode': run.returncode, 'seconds': time.monotonic()-started,
          'code_unchanged': code_unchanged, 'server_package': str(package),
          'server_output': str(output), 'GPU_used': False,
          'training_scope': 'two-step-source-engineering-smoke-not-research',
          'log_tail': log.read_text()[-16000:], 'archives': {{}}}}
small = ('contract.json','preflight.json','panels.json','nomination.json',
         'probe-smoke.json','verification.json','summary.json','status.json','metrics.json',
         'events.jsonl','episodes.jsonl')
for name in small:
    p = output / name
    if p.exists():
        raw = p.read_bytes()
        result['archives'][name] = {{'sha256': hashlib.sha256(raw).hexdigest(),
                                    'base64': base64.b64encode(raw).decode()}}
print(json.dumps(result, ensure_ascii=False))
'''
result = subprocess.run(
    ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'myserver',
     'PYTHONDONTWRITEBYTECODE=1 /gly/weilin/PRICE-master/venv/bin/python -B -'],
    input=remote, text=True, capture_output=True, timeout=1900, check=False,
)
(REPORTS / 'server-controller.stderr.log').write_text(result.stderr)
if result.returncode:
    (REPORTS / 'server-controller.stdout.log').write_text(result.stdout)
    raise SystemExit(f'Controller failed ({result.returncode}); inspect original process, no restart')
proof = json.loads(result.stdout)
(REPORTS / 'execution.log').write_text(proof.pop('log_tail'))
archives = proof.pop('archives')
output = ROOT / '实验文档/results/s8-g1-smoke-20261007'
output.mkdir(exist_ok=False)
for name, item in archives.items():
    raw = base64.b64decode(item['base64'])
    if hashlib.sha256(raw).hexdigest() != item['sha256']:
        raise ValueError('Smoke archive byte hash mismatch')
    with (output / name).open('xb') as stream:
        stream.write(raw)
proof['archive_sha256'] = {name: row['sha256'] for name, row in archives.items()}
proof['large_matrices_or_checkpoints_downloaded'] = False
(REPORTS / 'server-controller.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(proof, ensure_ascii=False, indent=2))
if proof['returncode'] or not proof['code_unchanged']:
    raise SystemExit('G1 smoke failed; archived available evidence, no automatic retry')
