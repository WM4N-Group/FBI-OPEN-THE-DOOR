#!/usr/bin/env python3
"""Detached training worker. Only executes the two explicit research modules."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--spec", required=True)
    args = pa.parse_args()
    specfile = Path(args.spec).resolve()
    spec = json.loads(specfile.read_text())
    if spec["module"] not in ("fbi_door.sim.train", "fbi_door.sim.dual_perception"):
        raise ValueError("Unsupported experiment module")
    statusfile = specfile.with_suffix(".status.json")

    def status(state, **details):
        value = {
            "name": spec["name"],
            "state": state,
            "worker_pid": os.getpid(),
            "gpu": spec["gpu"],
            "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "exit_code_is_not_task_success": True,
            **details,
        }
        temp = statusfile.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(statusfile)

    dependency = spec.get("dataset_dependency")
    if dependency:
        status("waiting_for_completed_dataset")
        deadline = time.monotonic() + 7200
        while not (
            Path(dependency).is_file()
            and Path(dependency).with_name("dataset.json").is_file()
        ):
            if time.monotonic() > deadline:
                status("failed_dependency_timeout")
                return 1
            time.sleep(10)
    env = dict(
        os.environ,
        FBI_DOOR_RUNTIME=spec["runtime"],
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(spec["gpu"]),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    command = [spec["python"], "-m", spec["module"]] + spec["arguments"]
    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
    status("running", training_pid=child.pid)
    code = child.wait()
    primary = Path(spec["runtime"]) / "runs" / spec["run_name"] / "status.json"
    result = json.loads(primary.read_text()) if primary.is_file() else None
    state = "process_completed" if code == 0 else "failed_process"
    if result and result.get("state") == "stopped_gate_budget_exhausted":
        state = "finished_gate_budget_exhausted"
    status(state, training_pid=child.pid, exit_code=code, curriculum_status=result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
