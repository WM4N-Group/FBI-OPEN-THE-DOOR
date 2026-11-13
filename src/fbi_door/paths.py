"""All generated/private artifacts live outside tracked source."""

import os
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()
RUNTIME_ROOT = (
    Path(os.environ.get("FBI_DOOR_RUNTIME", str(REPO_ROOT / ".runtime")))
    .expanduser()
    .resolve()
)


def ensure_runtime():
    for name in ("assets", "reports", "reports/freehinge_change_20261113", "runs"):
        (RUNTIME_ROOT / name).mkdir(parents=True, exist_ok=True)


def artifact_hashes(names):
    """Hash source from the installed package and generated assets from runtime."""
    import hashlib

    source = Path(__file__).resolve().parent / "sim"
    return {
        name: hashlib.sha256(
            ((source if name.endswith(".py") else RUNTIME_ROOT) / name).read_bytes()
        ).hexdigest()
        for name in names
    }
