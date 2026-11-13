#!/usr/bin/env python3
"""Run inherited offline tests only with their real dependencies; never mock a pass."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]


def main():
    missing = []
    for name in (
        "numpy",
        "can",
        "rclpy",
        "trajectory_msgs",
        "sensor_msgs",
        "std_msgs",
        "std_srvs",
    ):
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(name)
    if not (
        REPO / "hardware/third_party/piper_sdk_runtime/piper_sdk/__init__.py"
    ).is_file():
        missing.append("pinned hardware/third_party/piper_sdk_runtime (0.6.2)")
    if missing:
        print(
            json.dumps(
                {
                    "status": "not_run_missing_dependencies",
                    "missing": missing,
                    "hardware_validated": False,
                },
                indent=2,
            )
        )
        return 77
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(REPO / "hardware/arm") + os.pathsep + env.get("PYTHONPATH", "")
    )
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(REPO / "hardware/arm/tests"),
            "-v",
        ],
        env=env,
        cwd=REPO,
    )


if __name__ == "__main__":
    raise SystemExit(main())
