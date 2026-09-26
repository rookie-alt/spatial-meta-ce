#!/usr/bin/env python3
"""Deploy only managed source files. Use SSH agent/key or an external SSH wrapper.
Never stores credentials; refuses to overwrite remote manual changes or artifacts.
"""
import argparse
import base64
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

REMOTE = r'''
import sys,json,base64,hashlib
from pathlib import Path
payload=json.load(sys.stdin)
root=Path(payload['root'])
marker=root/'.deployment-manifest.json'
if root.exists() and not marker.exists():
    raise RuntimeError('Existing unmanaged target: refusing deployment')
old=json.loads(marker.read_text()) if marker.exists() else {}
for name in payload['files']:
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('Invalid path')
    target=root/rel
    if target.exists():
        current=hashlib.sha256(target.read_bytes()).hexdigest()
        if name not in old or current!=old[name]:
            raise RuntimeError('Remote manual changes: '+name)
root.mkdir(parents=True,exist_ok=True)
for name,b64 in payload['files'].items():
    data=base64.b64decode(b64)
    target=root/name
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    old[name]=hashlib.sha256(data).hexdigest()
marker.write_text(json.dumps(old,indent=2))
print('Deployed',len(payload['files']),'managed source files. No artifacts deleted.')
'''


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ssh',required=True,help='SSH command ending in user@host; use agent auth, not inline passwords')
    p.add_argument('--remote-root',default='/gly/weilin/spatial-meta-ce')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    files={}
    for f in root.rglob('*'):
        if not f.is_file() or f.is_symlink(): continue
        rel=f.relative_to(root)
        if set(rel.parts)&{'__pycache__','runs','artifacts','.git'} or f.name.startswith('.'): continue
        if rel.suffix not in ('.py','.json','.md','.sh','.txt','.toml'): continue
        files[str(rel)]=base64.b64encode(f.read_bytes()).decode()
    payload={'root':args.remote_root,'files':files}
    subprocess.run(shlex.split(args.ssh)+['python3 -c '+shlex.quote(REMOTE)],input=json.dumps(payload),text=True,check=True)


if __name__=='__main__':
    main()
