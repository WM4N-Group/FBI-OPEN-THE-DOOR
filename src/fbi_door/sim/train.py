"""Grasp-v3 residual training. Stage selection remains scripted, NOT learned HRL manager."""

from ..paths import artifact_hashes


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, time
import hashlib
from pathlib import Path
from ..experiments import (
    curriculum_plan,
    maximum_requested_steps,
    with_initial_stage_steps,
)
from ..storage import atomic_save_model, atomic_json
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from .door_env import DoorEnv
from ..paths import RUNTIME_ROOT as ROOT


def make_env(mode, stage, seed, logdir):
    def factory():
        e = DoorEnv(perception=mode, curriculum=stage)
        e.reset(seed=seed)
        return Monitor(
            e,
            str(logdir / f"env_{seed}.csv"),
            info_keywords=(
                "full_success",
                "stage_success",
                "phase",
                "latch_unlocked",
                "grasp_ever_valid",
                "max_door_angle_rad",
                "teacher_privileged",
            ),
        )

    return factory


def evaluate(model, mode, stage, n=32):
    e = DoorEnv(perception=mode, curriculum=stage)
    results = []
    for i in range(n):
        o, _ = e.reset(seed=880000 + i)
        for step in range(550):
            a, _ = model.predict(o, deterministic=True)
            o, _, term, trunc, info = e.step(a)
            if term or trunc:
                break
        if stage == "grasp":
            passed = info["stage_success"] and info["grasp_ever_valid"]
        elif stage == "press":
            passed = (
                info["stage_success"]
                and info["latch_unlocked"]
                and info["grasp_ever_valid"]
            )
        elif stage == "push":
            passed = (
                info["stage_success"]
                and info["max_door_angle_rad"] > 1.1
                and info["grasp_ever_valid"]
            )
        else:
            passed = info["full_success"] and info["grasp_ever_valid"]
        row = {
            "seed": 880000 + i,
            "passed": bool(passed),
            "seconds": (step + 1) * 0.1,
            **info,
        }
        results.append(row)
        print("EVAL_EPISODE", json.dumps(row), flush=True)
    e.close()
    return {
        "passed": sum(x["passed"] for x in results),
        "total": n,
        "results": results,
        "scope": "initial-pose curriculum gate only; not randomized hardware acceptance",
    }


class AtomicCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="policy"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            destination = (
                self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            atomic_save_model(self.model, destination)
            print("CHECKPOINT_ATOMIC", str(destination), flush=True)
        return True


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--envs", type=int, default=8)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--run-name", required=True)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--entropy", type=float, default=0.002)
    pa.add_argument("--log-std-init", type=float, default=-1.8)
    pa.add_argument("--stage-scale", type=float, default=1.0)
    pa.add_argument("--gate-retries", type=int, default=0)
    pa.add_argument(
        "--start-stage", choices=["grasp", "press", "push", "full"], default="grasp"
    )
    pa.add_argument(
        "--resume-zip",
        help="Trusted, compatible PPO checkpoint; never an untrusted downloaded pickle",
    )
    pa.add_argument(
        "--initial-stage-steps",
        type=int,
        help="Remaining first-stage budget when resuming; does not bypass validation",
    )
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--grasp-only", action="store_true")
    a = pa.parse_args()
    if a.envs < 1:
        pa.error("--envs must be positive")
    plan = with_initial_stage_steps(
        curriculum_plan(a.stage_scale, a.start_stage, a.gate_retries),
        a.initial_stage_steps,
    )
    if a.initial_stage_steps is not None and not a.resume_zip:
        pa.error("--initial-stage-steps requires a trusted resume checkpoint")
    if a.grasp_only:
        if a.start_stage != "grasp":
            pa.error("--grasp-only requires --start-stage grasp")
        plan = plan[:1]
    if a.smoke:
        plan = [{"mode": "teacher", "stage": a.start_stage, "attempt_steps": [512]}]
    resume = None if a.resume_zip is None else Path(a.resume_zip).expanduser().resolve()
    if resume is not None and not resume.is_file():
        pa.error("resume checkpoint does not exist")
    if a.start_stage != "grasp" and resume is None:
        pa.error(
            "Later stages require a compatible checkpoint, not an untrained policy"
        )
    torch.set_num_threads(1)
    out = ROOT / "runs" / a.run_name
    out.mkdir(parents=True, exist_ok=False)
    cfg = {
        "plan": plan,
        "maximum_requested_steps": maximum_requested_steps(plan),
        "seed": a.seed,
        "device": a.device,
        "parallel_envs": a.envs,
        "lr": a.lr,
        "entropy": a.entropy,
        "log_std_init": a.log_std_init,
        "resume_checkpoint": None if resume is None else str(resume),
        "initial_stage_steps_override": a.initial_stage_steps,
        "checkpoint_save": "local atomic zip + independently mirrored durable shared storage when configured",
        "resume_sha256": None
        if resume is None
        else hashlib.sha256(resume.read_bytes()).hexdigest(),
        "old_high_level_checkpoint_used": resume is not None,
        "initialization": "fresh high-level policy; original leg policy restored/frozen"
        if resume is None
        else "trusted compatible high-level checkpoint",
        "manager": "scripted stages, not learned HRL manager",
        "leg_policy": "frozen original 12D leg policy",
        "high_level_action": "10D residual: base vx/vy/wz, TCP xyz and rotvec, gripper closure adjustment",
        "teacher_privileged": True,
        "visual_closed_loop_enabled": False,
        "hardware_approved": False,
        "gate": {
            "required": 30,
            "episodes": 32,
            "role": "reused validation seeds, NOT independent final-test evidence",
            "max_extra_attempts": a.gate_retries,
        },
        "task_manifest": json.loads((ROOT / "manifest.json").read_text()),
        "source_sha256": artifact_hashes(
            [
                "core.py",
                "door_env.py",
                "build_scene.py",
                "train.py",
                "assets/door_scene.xml",
                "assets/robot.xml",
                "manifest.json",
            ]
        ),
    }
    atomic_json(out / "config.json", cfg)

    def status(record):
        record = dict(record, updated=time.strftime("%Y-%m-%d %H:%M:%S %z"))
        atomic_json(out / "status.json", record)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    model = None
    requested_total = 0
    for phase in plan:
        mode, stage = phase["mode"], phase["stage"]
        run = out / (mode + "_" + stage)
        run.mkdir()
        gate_passed = False
        for attempt, steps in enumerate(phase["attempt_steps"]):
            attempt_dir = run / ("attempt_%d" % attempt)
            attempt_dir.mkdir()
            vec = SubprocVecEnv(
                [
                    make_env(
                        mode, stage, a.seed * 100 + i + attempt * 10000, attempt_dir
                    )
                    for i in range(a.envs)
                ],
                start_method="spawn",
            )
            if model is None and resume is not None:
                model = PPO.load(str(resume), env=vec, device=a.device)
                model.tensorboard_log = str(out / "tensorboard")
                model.learning_rate = a.lr
                model.lr_schedule = lambda _: a.lr
                model.ent_coef = a.entropy
                print(
                    "RESUMED",
                    resume.name,
                    "recorded_steps",
                    model.num_timesteps,
                    flush=True,
                )
            elif model is None:
                model = PPO(
                    "MlpPolicy",
                    vec,
                    device=a.device,
                    n_steps=128,
                    batch_size=256,
                    n_epochs=5,
                    learning_rate=a.lr,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=a.entropy,
                    policy_kwargs={
                        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                        "log_std_init": a.log_std_init,
                    },
                    verbose=1,
                    seed=a.seed,
                    tensorboard_log=str(out / "tensorboard"),
                )
            else:
                model.set_env(vec)
            record = {
                "state": "training",
                "mode": mode,
                "stage": stage,
                "attempt": attempt,
                "requested_attempt_steps": steps,
                "requested_steps_completed_this_launch": requested_total,
                "resumed": resume is not None,
                "device": a.device,
                "physics_version": cfg["task_manifest"]["task_version"],
            }
            status(record)
            print("START_STAGE", json.dumps(record), flush=True)
            before = [p.detach().clone() for p in model.policy.parameters()]
            try:
                model.learn(
                    total_timesteps=steps,
                    reset_num_timesteps=False,
                    callback=AtomicCheckpointCallback(
                        save_freq=max(10000 // a.envs, 1),
                        save_path=str(attempt_dir / "checkpoints"),
                        name_prefix="policy",
                    ),
                    tb_log_name=mode + "_" + stage,
                )
                change = sum(
                    float((p - b).abs().sum())
                    for p, b in zip(model.policy.parameters(), before)
                )
                assert change > 0
                atomic_save_model(model, attempt_dir / "policy_final")
                atomic_save_model(model, run / "policy_latest")
            finally:
                vec.close()
            requested_total += steps
            if a.smoke:
                record.update(
                    state="gradient_smoke_passed",
                    parameter_l1_change=change,
                    recorded_model_steps=model.num_timesteps,
                )
                status(record)
                return
            record["state"] = "evaluating"
            status(record)
            ev = evaluate(model, mode, stage)
            atomic_json(attempt_dir / "validation.json", ev)
            print(
                "EVAL_STAGE",
                mode,
                stage,
                "attempt",
                attempt,
                ev["passed"],
                ev["total"],
                flush=True,
            )
            if ev["passed"] >= 30:
                gate_passed = True
                break
            record.update(
                state="gate_not_met_retry_pending"
                if attempt + 1 < len(phase["attempt_steps"])
                else "stopped_gate_budget_exhausted",
                passed=ev["passed"],
                total=ev["total"],
                reason="No gate bypass. Extra training is bounded; all attempt checkpoints retained.",
            )
            status(record)
        if not gate_passed:
            return
    status(
        {
            "state": "finished_candidate_curriculum",
            "requested_steps_completed_this_launch": requested_total,
            "recorded_model_steps": model.num_timesteps,
            "hardware_approved": False,
            "requires": "independent visual full-task contact and robustness acceptance",
        }
    )


if __name__ == "__main__":
    main()
