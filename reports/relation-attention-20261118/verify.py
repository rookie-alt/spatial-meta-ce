"""Capture authorized local checks and exit codes; no cache or server access."""

import json
import hashlib
import os
import re
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ENV = {
    **os.environ,
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
PYTHON = "/opt/anaconda3/bin/python3"
CHANGED = [
    "spatial_meta_ce/price_model.py",
    "spatial_meta_ce/price_relations.py",
    "spatial_meta_ce/model.py",
    "spatial_meta_ce/checkpoints.py",
    "tests/test_price_relation_attention.py",
    "reports/relation-attention-20261118/verify.py",
]
COMMANDS = {
    "focused": [
        PYTHON,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_price_relation_attention.py",
        "-v",
    ],
    "regression": [
        PYTHON,
        "-B",
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_price_model.py",
        "-v",
    ],
    "full": [PYTHON, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
    "ruff": ["ruff", "check", "scripts", "tests", "spatial_meta_ce"],
    "ruff_changed": ["ruff", "check", "--select", "F,E", "--ignore", "E501", *CHANGED],
    "diff": ["git", "diff", "--check"],
    "protected": [
        "git",
        "diff",
        "--exit-code",
        "HEAD",
        "--",
        "vendor",
        "spatial_meta_ce/mappings.py",
        "spatial_meta_ce/train.py",
        "spatial_meta_ce/price_protocol.py",
        "实验文档/130-双分支模块实现计划.md",
        "实验文档/99-进度与下一步.md",
    ],
}


def record_evidence():
    """Measure synthetic fixtures and bind report facts to source/log bytes."""
    sys.path.insert(0, str(ROOT))
    import torch

    from spatial_meta_ce.manifests import digest
    from spatial_meta_ce.model import create_model
    from spatial_meta_ce.price_relations import RELATION_NAMES

    fixture = runpy.run_path(str(ROOT / "tests/test_price_relation_attention.py"))
    config = {
        "arch": "price_lite",
        "capacity": fixture["synthetic_capacity"](),
        "pg_head_input": True,
    }
    measured = {}
    canonical_config = None
    for name, extra in (
        ("A1-PG", {}),
        ("relation", {"relation_attention": True}),
        ("relation_lora", {"relation_attention": True, "lora": True}),
    ):
        torch.manual_seed(42)
        model = create_model({**config, **extra})
        measured[name] = {
            "model_type": model.model_type,
            "parameters": sum(p.numel() for p in model.parameters()),
            "relation_bias_parameters": sum(
                p.numel()
                for n, p in model.named_parameters()
                if n.endswith("relation_bias")
            ),
            "lora_parameters": sum(
                p.numel()
                for n, p in model.named_parameters()
                if n.endswith(("lora_A", "lora_B"))
            ),
            "canonical_config_sha256": digest(model.config),
        }
        if name == "relation":
            canonical_config = model.config
    (OUT / "synthetic-model-config.json").write_text(
        json.dumps(canonical_config, indent=2) + "\n"
    )
    checks = {}
    for path in sorted(OUT.glob("*-checks.json")):
        for check in json.loads(path.read_text())["results"]:
            text = (OUT / check["log"]).read_text()
            summary = re.search(r"Ran (\d+) tests in ([\d.]+)s", text)
            checks[f"{path.name}:{check['check']}"] = {
                **check,
                "tests": int(summary[1]) if summary else None,
                "seconds": float(summary[2]) if summary else None,
                "log_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
    evidence = {
        "purpose": "synthetic-smoke-not-research",
        "baseline_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "worktree": str(ROOT),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "parameters": measured,
        "relation_classes": dict(enumerate(RELATION_NAMES)),
        "checks": checks,
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in CHANGED
        },
        "source_mapping_certification": "pending",
        "runtime_source_binding": "pending-no-train-or-fixed_model_scope-changes",
        "independent_review": "pending-parent-read-only-review",
        "target_sidecars_produced": [],
        "committed": False,
        "pushed": False,
    }
    (OUT / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"evidence": "evidence.json", "parameters": measured}), flush=True)


def main():
    stage = sys.argv[1]
    if stage == "record":
        record_evidence()
        return
    names = sys.argv[2:] or ["focused"]
    results = []
    for name in names:
        command = COMMANDS[name]
        path = OUT / f"{stage}-{name}.log"
        with path.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=ENV,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results.append(
            {
                "check": name,
                "command": command,
                "returncode": result.returncode,
                "log": path.name,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    (OUT / f"{stage}-checks.json").write_text(
        json.dumps(
            {
                "environment": {
                    k: ENV[k]
                    for k in (
                        "PYTHONDONTWRITEBYTECODE",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
