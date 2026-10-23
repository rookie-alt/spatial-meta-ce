"""Bounded read-only S7 certificate recheck; no deployment/training/inference."""

import base64
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).resolve().parent
S7 = ROOT / "实验文档/results/s7-fit-generalization-20261007"
code = (ROOT / "scripts/s7_independent_verify.py").read_bytes()
protocol = json.loads((S7 / "protocol.json").read_text())
status = json.loads((S7 / "status.json").read_text())
hashes = json.loads(
    (
        ROOT / "reports/s7-fit-generalization-20261007/server-archive-sha256.json"
    ).read_text()
)
server_root = "/gly/weilin/spatial-meta-ce"
output = server_root + "/runs/s7-fit-generalization-20261007"
old_code = server_root + "/runs/s7-execution-code-20261007"
bindings = dict(protocol["input_sha256"])
bindings.update({old_code + "/" + k: v for k, v in protocol["code_sha256"].items()})
bindings.update({output + "/" + k: v for k, v in hashes.items()})
bindings.update({output + "/" + k: v for k, v in status["output_sha256"].items()})
remote = f"""
import base64, hashlib, importlib.util, json, resource, signal, time
from pathlib import Path
signal.alarm(300)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 ** 3, 8 * 1024 ** 3))
started = time.monotonic()
bindings = {bindings!r}
def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()
def authenticate():
    for path, expected in bindings.items():
        if file_hash(path) != expected:
            raise ValueError("Historical input changed: " + path)
authenticate()
code = base64.b64decode({base64.b64encode(code).decode()!r})
namespace = {{"__name__": "s7_recheck_verifier"}}
exec(compile(code, "s7_recheck_verifier.py", "exec"), namespace)
spec = importlib.util.spec_from_file_location("authenticated_replay", {old_code!r} + "/scripts/s2_source_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
root = Path({server_root!r})
s6 = root / "runs/s6-mean-tail-20261007"
s7 = Path({output!r})
frozen = json.loads((s6 / "summary.json").read_text())
preflight = json.loads((s6 / "preflight.json").read_text())
caches = {{db: replay.read_cache(root, db, preflight["sources"][db])
          for db in (*replay.TRAIN, *replay.VALIDATION)}}
sampled = replay.replay(caches, list(replay.TRAIN), frozen["runs"][0]["config"], prefixes=(4000,))
replay.certify(sampled)
proof = namespace["verify"](s7, root, s6, caches, sampled["prefixes"][4000]["databases"])
authenticate()
proof.update({{"recheck_only": True, "training": False, "inference": False,
              "adaptation": False, "target_data_read": False, "device": "cpu",
              "historical_files_rehashed_before_and_after": len(bindings),
              "inputs_unchanged": True, "historical_code_unchanged": True,
              "verifier_sha256": hashlib.sha256(code).hexdigest(),
              "seconds": time.monotonic() - started,
              "sample_digest": sampled["digest"],
              "memory_limit_bytes": 8 * 1024 ** 3, "timeout_seconds": 300}})
print(json.dumps(proof, ensure_ascii=False, indent=2))
"""
(REPORTS / "server-recheck-inputs.json").write_text(
    json.dumps(bindings, ensure_ascii=False, indent=2) + "\n"
)
result = subprocess.run(
    [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "myserver",
        (
            "timeout 310s env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            "/gly/weilin/PRICE-master/venv/bin/python -B -"
        ),
    ],
    input=remote,
    text=True,
    capture_output=True,
    timeout=330,
    check=False,
)
(REPORTS / "server-recheck.log").write_text(result.stderr + result.stdout)
if result.returncode:
    raise SystemExit(f"Read-only recheck failed ({result.returncode}); no retry")
proof = json.loads(result.stdout)
if not proof["passed"] or proof["verifier_sha256"] != hashlib.sha256(code).hexdigest():
    raise ValueError("Server recheck proof mismatch")
(REPORTS / "server-recheck.json").write_text(
    json.dumps(proof, ensure_ascii=False, indent=2) + "\n"
)
print(result.stdout)
