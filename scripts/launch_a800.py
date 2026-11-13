#!/usr/bin/env python3
"""Launch four residual-PPO seeds + paired-camera training on two GPUs.
No hardware interfaces, process termination, CUDA_VISIBLE_DEVICES remapping, or gate bypass.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from fbi_door.experiments import curriculum_plan, maximum_requested_steps


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--runtime", required=True)
    pa.add_argument("--batch-name", required=True)
    pa.add_argument("--envs-per-run", type=int, default=6)
    pa.add_argument("--stage-scale", type=float, default=2.0)
    pa.add_argument("--gate-retries", type=int, default=2)
    pa.add_argument("--camera-scenes", type=int, default=8192)
    pa.add_argument("--camera-epochs", type=int, default=160)
    pa.add_argument("--camera-batch-scenes", type=int, default=128)
    pa.add_argument("--dry-run", action="store_true")
    a = pa.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", a.batch_name):
        pa.error("Invalid batch name")
    cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    if a.envs_per_run < 1 or 4 * a.envs_per_run + 2 > cpus:
        pa.error("Reserve at least two visible CPU threads for support work")
    if a.camera_scenes < 1152 or a.camera_epochs < 1 or a.camera_batch_scenes < 1:
        pa.error("Invalid paired-camera budget")
    runtime = Path(a.runtime).expanduser().resolve()
    plan = curriculum_plan(a.stage_scale, "grasp", a.gate_retries)
    jobs = []
    for seed, gpu, lr, entropy in [
        (42, 0, 3e-4, 0.002),
        (43, 1, 3e-4, 0.002),
        (44, 0, 1.5e-4, 0.004),
        (45, 1, 3e-4, 0.004),
    ]:
        name = f"{a.batch_name}_ppo_s{seed}_g{gpu}"
        argv = [
            "--run-name",
            name,
            "--seed",
            str(seed),
            "--envs",
            str(a.envs_per_run),
            "--device",
            f"cuda:{gpu}",
            "--lr",
            str(lr),
            "--entropy",
            str(entropy),
            "--stage-scale",
            str(a.stage_scale),
            "--gate-retries",
            str(a.gate_retries),
        ]
        jobs.append(
            {
                "name": name,
                "run_name": name,
                "gpu": gpu,
                "module": "fbi_door.sim.train",
                "arguments": argv,
            }
        )
    data_name = f"{a.batch_name}_camera_s46_g0"
    for seed, gpu in [(46, 0), (47, 1)]:
        name = f"{a.batch_name}_camera_s{seed}_g{gpu}"
        argv = [
            "--output",
            name,
            "--seed",
            str(seed),
            "--device",
            f"cuda:{gpu}",
            "--epochs",
            str(a.camera_epochs),
            "--batch-scenes",
            str(a.camera_batch_scenes),
        ]
        spec = {
            "name": name,
            "run_name": name,
            "gpu": gpu,
            "module": "fbi_door.sim.dual_perception",
            "arguments": argv,
        }
        if gpu == 0:
            argv += ["--scenes", str(a.camera_scenes)]
        else:
            data = str(runtime / "runs" / data_name / "dual_rgbd.npz")
            argv += ["--skip-data", "--datafile", data]
            spec["dataset_dependency"] = data
        jobs.append(spec)
    for job in jobs:
        job.update(python=sys.executable, runtime=str(runtime))
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "visible_cpus": cpus,
        "parallel_physics_envs": 4 * a.envs_per_run,
        "ppo_run_count": 4,
        "fresh_high_level_initialization": True,
        "old_checkpoint_recovered": False,
        "original_leg_policy_restored_frozen": True,
        "base_requested_ppo_steps_all_seeds": 4
        * maximum_requested_steps(curriculum_plan(a.stage_scale)),
        "maximum_requested_ppo_steps_with_retries": 4 * maximum_requested_steps(plan),
        "camera_dataset_scenes": a.camera_scenes,
        "camera_dataset_images": 2 * a.camera_scenes,
        "hardware_motion": False,
        "gpu_note": "PPO uses CUDA, but MuJoCo/IK are CPU-bound. Paired RGB training also uses both GPUs; no promise of saturation.",
        "jobs": jobs,
    }
    if a.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if not (runtime / "manifest.json").is_file():
        pa.error("Build and validate simulation assets first")
    batch = runtime / "batches" / a.batch_name
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2))
    worker = Path(__file__).with_name("run_experiment_job.py")
    for spec in jobs:
        specfile = batch / (spec["name"] + ".json")
        specfile.write_text(json.dumps(spec, indent=2))
        log = batch / (spec["name"] + ".log")
        with log.open("wb") as stream:
            child = subprocess.Popen(
                [sys.executable, str(worker), "--spec", str(specfile)],
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (batch / (spec["name"] + ".pid")).write_text(str(child.pid) + "\n")
        print(
            "STARTED",
            spec["name"],
            "worker_pid",
            child.pid,
            "gpu",
            spec["gpu"],
            flush=True,
        )
    print("BATCH", batch, flush=True)


if __name__ == "__main__":
    main()
