"""Read back every migration member and reject missing, changed or unsafe data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
import time
from datetime import datetime
from pathlib import Path, PurePosixPath


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    directory = Path(directory)
    started = time.monotonic()
    sums = (directory / "SHA256SUMS").read_text().splitlines()
    names = set()
    for row in sums:
        checksum, name = row.split("  ", 1)
        if name in names or Path(name).name != name:
            raise ValueError("Invalid or repeated package checksum entry")
        names.add(name)
        if sha256_file(directory / name) != checksum:
            raise ValueError(f"Package checksum mismatch: {name}")
    required = {
        "module26-official26-4-inputs.tar.gz",
        "file-manifest.json",
        "payload.sha256",
        "required-code-sha256.json",
        "package-result.json",
    }
    if names != required:
        raise ValueError("Incomplete package checksum inventory")
    raw_manifest = (directory / "file-manifest.json").read_bytes()
    raw_sums = (directory / "payload.sha256").read_bytes()
    manifest = json.loads(raw_manifest)
    expected = {item["archive_path"]: item for item in manifest["files"]}
    if (
        len(expected) != len(manifest["files"])
        or len(expected) != manifest["file_count"]
    ):
        raise ValueError("Repeated or incomplete payload manifest")
    expected_sums = "".join(
        f"{item['sha256']}  {item['archive_path']}\n" for item in manifest["files"]
    ).encode()
    if expected_sums != raw_sums:
        raise ValueError("Payload checksums differ from manifest")
    meta = {"meta/file-manifest.json": raw_manifest, "meta/payload.sha256": raw_sums}
    seen, total = set(), 0
    archive_path = directory / "module26-official26-4-inputs.tar.gz"
    with gzip.open(archive_path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            for member in archive:
                name = member.name
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or name in seen
                    or not member.isfile()
                ):
                    raise ValueError(f"Unsafe or repeated archive member: {name}")
                seen.add(name)
                if name not in expected and name not in meta:
                    raise ValueError(f"Unexpected member: {name}")
                stream = archive.extractfile(member)
                if name in meta:
                    if stream.read() != meta[name]:
                        raise ValueError(f"Embedded metadata mismatch: {name}")
                    continue
                record = expected[name]
                if member.size != record["bytes"]:
                    raise ValueError(f"Payload length mismatch: {name}")
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    digest.update(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Payload checksum mismatch: {name}")
                total += member.size
                if len(seen) % 100 == 0:
                    print(json.dumps({"verified_members": len(seen)}), flush=True)
        for block in iter(lambda: compressed.read(1024**2), b""):
            if any(block):
                raise ValueError("Nonzero data follows the tar end marker")
    if seen != set(expected) | set(meta) or total != manifest["uncompressed_bytes"]:
        raise ValueError("Missing payload members or byte count mismatch")
    return {
        "state": "verified",
        "verified_at": datetime.now().astimezone().isoformat(),
        "archive_sha256": sha256_file(archive_path),
        "verified_payload_files": len(expected),
        "verified_payload_bytes": total,
        "embedded_metadata_members": len(meta),
        "seconds": time.monotonic() - started,
        "training_run": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--record")
    args = parser.parse_args()
    result = verify(args.directory)
    if args.record:
        with Path(args.record).open("x") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
