#!/usr/bin/env python3
"""Run shared-storage replication independently from training; failures stay local."""

import argparse
import json
import time
from pathlib import Path
from fbi_door.storage import RuntimeMirror, atomic_json


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--local-root", required=True)
    pa.add_argument("--shared-root", required=True)
    pa.add_argument("--subtree", action="append", required=True)
    pa.add_argument("--interval", type=float, default=30)
    pa.add_argument("--once", action="store_true")
    a = pa.parse_args()
    if a.interval < 1:
        pa.error("interval must be >= 1 second")
    mirror = RuntimeMirror(a.local_root, a.shared_root, a.subtree)
    while True:
        try:
            result = mirror.sync_once()
        except OSError as exc:
            result = {"copied": 0, "pending": None, "errors": [str(exc)]}
        result["observed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        result["all_current_artifacts_replicated"] = (
            not result["errors"] and result["pending"] == 0
        )
        atomic_json(Path(a.local_root) / "mirror_status.json", result)
        try:
            atomic_json(Path(a.shared_root) / "buffered_mirror_status.json", result)
        except OSError:
            pass
        if result["copied"] or result["errors"]:
            print(json.dumps(result), flush=True)
        if a.once:
            return 0 if not result["errors"] else 1
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
