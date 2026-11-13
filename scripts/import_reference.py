#!/usr/bin/env python3
"""Import a private asset bundle after digest + member-path validation. Never unpickles weights."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]


def validate_member(member):
    path = PurePosixPath(member.name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.parts[0] != "reference"
    ):
        raise ValueError("Unsafe asset member path: " + member.name)
    if not (member.isfile() or member.isdir()):
        raise ValueError("Links/devices are not allowed: " + member.name)
    return path


def import_bundle(archive, runtime, manifest):
    digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    if digest != manifest["archive_sha256"]:
        raise ValueError("Bundle SHA256 mismatch; refusing extraction")
    runtime = Path(runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    target = runtime / "reference"
    if target.exists():
        raise ValueError(
            "Reference directory already exists; keep it or choose a fresh FBI_DOOR_RUNTIME"
        )
    with tempfile.TemporaryDirectory(prefix=".reference-import-", dir=runtime) as temp:
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
            if sum(m.size for m in members) > 256 * 1024 * 1024:
                raise ValueError("Asset bundle exceeds unpacked-size limit")
            for member in members:
                validate_member(member)
            # All member paths/types were checked; no links may redirect extraction.
            for member in members:
                destination = Path(temp) / member.name
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with stream.extractfile(member) as src, destination.open(
                        "wb"
                    ) as dst:
                        shutil.copyfileobj(src, dst)
        for name, expected in manifest["required_files_sha256"].items():
            path = Path(temp) / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError("Required asset checksum mismatch: " + name)
        shutil.move(str(Path(temp) / "reference"), str(target))
    return {
        "imported": str(target),
        "archive_sha256": digest,
        "hardware_approved": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument(
        "--runtime", default=os.environ.get("FBI_DOOR_RUNTIME", str(REPO / ".runtime"))
    )
    args = parser.parse_args()
    with (REPO / "provenance/reference_bundle.json").open() as stream:
        manifest = json.load(stream)
    print(json.dumps(import_bundle(args.archive, args.runtime, manifest), indent=2))


if __name__ == "__main__":
    main()
