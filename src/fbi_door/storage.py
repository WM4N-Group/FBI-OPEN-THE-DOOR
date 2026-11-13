"""Atomic artifacts and one-way, failure-tolerant runtime replication.
Local buffering is not durable shared storage until replication succeeds.
"""

import errno
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(destination, data):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix="." + destination.name + "-", suffix=".tmp", dir=destination.parent
    )
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, destination)
        fsync_directory(destination.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def atomic_json(destination, value):
    atomic_bytes(destination, (json.dumps(value, indent=2) + "\n").encode())


def valid_checkpoint_bytes(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if (
            not {"data", "policy.pth", "policy.optimizer.pth"}.issubset(
                archive.namelist()
            )
            or archive.testzip() is not None
        ):
            raise ValueError("Incomplete checkpoint archive")


def atomic_save_model(model, destination):
    destination = Path(destination)
    if destination.suffix != ".zip":
        destination = destination.with_name(destination.name + ".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix="." + destination.stem + "-", suffix=".tmp.zip", dir=destination.parent
    )
    os.close(fd)
    temp = Path(name)
    try:
        model.save(str(temp))
        valid_checkpoint_bytes(temp.read_bytes())
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, destination)
        fsync_directory(destination.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return destination


class RuntimeMirror:
    """Mirror only explicitly selected subtrees; never delete source or destination."""

    def __init__(self, local_root, shared_root, relative_roots):
        self.local = Path(local_root).resolve()
        self.shared = Path(shared_root).resolve()
        if (
            self.local == self.shared
            or self.local in self.shared.parents
            or self.shared in self.local.parents
        ):
            raise ValueError("Mirror roots must be distinct and non-nested")
        self.roots = []
        for name in relative_roots:
            p = Path(name)
            if p.is_absolute() or ".." in p.parts or not p.parts:
                raise ValueError("Unsafe relative subtree")
            self.roots.append(p)
        self.shared_device = self.shared.stat().st_dev
        self.synced = {}

    def sync_once(self):
        result = {"copied": 0, "pending": 0, "errors": []}
        candidates = []
        for sub in self.roots:
            for source in (self.local / sub).rglob("*"):
                rel = source.relative_to(self.local)
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or any(part.startswith(".") for part in rel.parts)
                    or source.name.endswith(".tmp")
                ):
                    continue
                stat = source.stat()
                signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
                if self.synced.get(str(rel)) != signature:
                    candidates.append((source, rel, signature))
        candidates.sort(key=lambda item: (item[0].suffix != ".zip", -item[2][2]))
        result["pending"] = len(candidates)
        if self.shared.stat().st_dev != self.shared_device:
            result["errors"].append("Shared mount device changed; refusing replication")
            return result
        for source, rel, signature in candidates:
            try:
                data = source.read_bytes()
                if source.suffix == ".zip":
                    valid_checkpoint_bytes(data)
                elif source.suffix == ".json":
                    json.loads(data)
                if (
                    source.stat().st_ino,
                    source.stat().st_size,
                    source.stat().st_mtime_ns,
                ) != signature:
                    continue
                target = self.shared / rel
                atomic_bytes(target, data)
                if (
                    source.suffix == ".zip"
                    and hashlib.sha256(target.read_bytes()).digest()
                    != hashlib.sha256(data).digest()
                ):
                    raise OSError(errno.EIO, "Checkpoint readback mismatch")
                self.synced[str(rel)] = signature
                result["copied"] += 1
                result["pending"] -= 1
            except (ValueError, zipfile.BadZipFile):
                continue  # Incomplete active metadata/archive: retry a later snapshot.
            except OSError as exc:
                result["errors"].append(
                    {"file": str(rel), "errno": exc.errno, "error": str(exc)}
                )
                break  # Avoid hammering unavailable/full shared storage. Keep all local files.
        return result
