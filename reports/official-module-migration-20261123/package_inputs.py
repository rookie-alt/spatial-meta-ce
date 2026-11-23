"""Package certified module26 inputs without modifying any source file."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import resource
import stat
import tarfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path("/gly/weilin/module26-prep-20261118")
ARCHIVE = "module26-official26-4-inputs.tar.gz"
CODE_COMMIT = "9e40dd571c225269d7f781ca5d019c9fcdf66556"


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def regular_file(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected absolute regular path: {path}")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError(f"Symlink input is forbidden: {path}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Input is not a regular file: {path}")
    return path


def read_bound(binding):
    path = regular_file(binding["path"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError(f"Input binding changed: {path}")
    return json.loads(raw)


def inventory():
    ready_path = BASE / "training-ready-v2/readiness.json"
    ready = json.loads(regular_file(ready_path).read_text())
    if ready.get("prepared") is not True or ready.get("authorized") is not False:
        raise ValueError("Expected the original unsigned preparation record")
    bindings = ready["common_bindings"]
    report = read_bound(bindings["mapping_report"])
    if (
        report.get("state") != "completed"
        or report.get("failed_databases") != []
        or report.get("unsupported_count") != 0
        or len(report["databases"]["pretrain"]) != 26
        or set(report["databases"]["test"]) != {"imdb", "stats", "genome", "ergastf1"}
    ):
        raise ValueError("Expected complete official26/test4 input certification")
    entries = {}

    def add(source, checksum, category, archive_path=None):
        path = regular_file(source)
        if not path.is_relative_to("/gly/weilin"):
            raise ValueError(f"Input escapes research tree: {path}")
        target = archive_path or "rootfs/" + str(path).lstrip("/")
        record = {
            "archive_path": target,
            "source_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": checksum,
            "category": category,
        }
        if target in entries and entries[target] != record:
            raise ValueError(f"Conflicting duplicate archive input: {target}")
        entries[target] = record

    add(
        bindings["mapping_report"]["path"],
        bindings["mapping_report"]["sha256"],
        "official_mapping",
    )
    for split, databases in report["databases"].items():
        for db, entry in databases.items():
            folder = Path(report["mapping_root"]) / split / db
            manifest_path = folder / "manifest.json"
            manifest = read_bound(
                {"path": str(manifest_path), "sha256": entry["manifest_sha256"]}
            )
            add(manifest_path, entry["manifest_sha256"], "official_mapping")
            add(folder / "mapping.jsonl", entry["jsonl_sha256"], "official_mapping")
            if manifest["input_sha256"] != entry["input_sha256"]:
                raise ValueError(f"Report/manifest input mismatch: {split}/{db}")
            for name, raw in manifest["input_paths"].items():
                if name == "workload":
                    category = "workloads"
                elif name == "source_features":
                    category = "features"
                elif name.startswith("statistics/"):
                    category = "statistics"
                elif name.startswith("cache/"):
                    category = "cache"
                elif name == "extractor":
                    category = "producer_evidence"
                else:
                    raise ValueError(f"Unknown certified input category: {name}")
                add(raw, manifest["input_sha256"][name], category)
            imported = entry.get("source_sidecar")
            if imported:
                for kind in ("manifest", "jsonl"):
                    add(
                        imported[f"{kind}_path"],
                        imported[f"{kind}_sha256"],
                        "original_mapping",
                    )
                proof = imported["builder_compatibility"]
                for kind in ("historical", "current"):
                    add(
                        proof[f"{kind}_source"],
                        proof[f"{kind}_sha256"],
                        "producer_evidence",
                    )
    for name in ("data_config", "capacity", "split_manifest"):
        add(
            bindings[name]["path"], bindings[name]["sha256"], "sealed_plans_and_configs"
        )
    for record in bindings["training_plans"].values():
        add(record["path"], record["sha256"], "sealed_plans_and_configs")
    add(
        ready_path,
        sha256_file(ready_path),
        "reference",
        "references/training-ready-v2/readiness.json",
    )
    for record in ready["runs"]:
        for name, filename in (
            ("spec", "training_spec.json"),
            ("authorization_template", "authorization_template.json"),
        ):
            binding = record[name]
            value = read_bound(binding)
            if (
                name == "authorization_template"
                and value.get("authorized") is not False
            ):
                raise ValueError("Signed authorization is not a migration input")
            add(
                binding["path"],
                binding["sha256"],
                "reference",
                f"references/training-ready-v2/{record['key']}/{filename}",
            )
    budget_path = BASE / "evidence/training-budget-draft.json"
    if json.loads(regular_file(budget_path).read_text()) != ready["budget_per_run"]:
        raise ValueError("Reference resource budget changed")
    add(budget_path, sha256_file(budget_path), "budget")
    for name, checksum in bindings["code_sha256"].items():
        if sha256_file(regular_file(Path(bindings["code_dir"]) / name)) != checksum:
            raise ValueError(f"Training code drift: {name}")
    return sorted(entries.values(), key=lambda item: item["archive_path"]), bindings[
        "code_sha256"
    ]


def write_new(path, raw):
    with Path(path).open("xb") as stream:
        stream.write(raw)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def member_info(name, size):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o644, 0
    info.uid = info.gid = 0
    return info


def build(output, entries, code_sha256):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if len({item["archive_path"] for item in entries}) != len(entries):
        raise ValueError("Repeated archive member")
    for item in entries:
        regular_file(item["source_path"])
        member_info(item["archive_path"], item["bytes"])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema": 1,
        "purpose": "official-module-26-4-input-migration",
        "code_commit": CODE_COMMIT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_run": False,
        "model_predictions_run": False,
        "old_signed_authorizations_included": False,
        "layout": "rootfs/gly/weilin preserves certified absolute input paths; references are non-executable preparation records",
        "code_sha256": code_sha256,
        "files": entries,
        "file_count": len(entries),
        "uncompressed_bytes": sum(item["bytes"] for item in entries),
    }
    raw_manifest = json_bytes(manifest)
    raw_sums = "".join(
        f"{item['sha256']}  {item['archive_path']}\n" for item in entries
    ).encode()
    write_new(output / "file-manifest.json", raw_manifest)
    write_new(output / "payload.sha256", raw_sums)
    write_new(
        output / "required-code-sha256.json", json_bytes({"code_sha256": code_sha256})
    )
    partial = output / (ARCHIVE + ".partial")
    with partial.open("xb") as target:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=target, compresslevel=1, mtime=0
        ) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT
            ) as archive:
                for number, item in enumerate(entries, 1):
                    path = regular_file(item["source_path"])
                    with path.open("rb") as source:
                        before = os.fstat(source.fileno())
                        if before.st_size != item["bytes"]:
                            raise ValueError(f"Input size changed: {path}")
                        reader = HashReader(source)
                        archive.addfile(
                            member_info(item["archive_path"], before.st_size), reader
                        )
                        after = os.fstat(source.fileno())
                        if reader.digest.hexdigest() != item["sha256"] or (
                            before.st_ino,
                            before.st_size,
                            before.st_mtime_ns,
                            before.st_ctime_ns,
                        ) != (
                            after.st_ino,
                            after.st_size,
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                        ):
                            raise ValueError(
                                f"Certified input changed while packaging: {path}"
                            )
                    if number % 100 == 0:
                        print(
                            json.dumps(
                                {"packed_files": number, "total_files": len(entries)}
                            ),
                            flush=True,
                        )
                for name, raw in (
                    ("meta/file-manifest.json", raw_manifest),
                    ("meta/payload.sha256", raw_sums),
                ):
                    archive.addfile(member_info(name, len(raw)), io.BytesIO(raw))
        target.flush()
        os.fsync(target.fileno())
    archive_path = output / ARCHIVE
    partial.rename(archive_path)
    categories = Counter()
    for item in entries:
        categories[item["category"]] += item["bytes"]
    result = {
        "state": "packaged_not_yet_readback_verified",
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "file_count": len(entries),
        "uncompressed_bytes": manifest["uncompressed_bytes"],
        "category_bytes": dict(categories),
        "all_packaged_bytes_matched_certified_sha256": True,
        "seconds": time.monotonic() - started,
        "training_run": False,
    }
    write_new(output / "package-result.json", json_bytes(result))
    sums = []
    for name in (
        ARCHIVE,
        "file-manifest.json",
        "payload.sha256",
        "required-code-sha256.json",
        "package-result.json",
    ):
        sums.append(f"{sha256_file(output / name)}  {name}\n")
    write_new(output / "SHA256SUMS", "".join(sums).encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    entries, code = inventory()
    print(
        json.dumps(
            {
                "inventory_files": len(entries),
                "bytes": sum(item["bytes"] for item in entries),
            }
        ),
        flush=True,
    )
    print(json.dumps(build(args.output, entries, code), indent=2), flush=True)


if __name__ == "__main__":
    main()
