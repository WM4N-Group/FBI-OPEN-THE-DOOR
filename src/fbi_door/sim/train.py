"""Grasp-v3 residual training. Stage selection remains scripted, NOT learned HRL manager."""

from ..paths import artifact_hashes


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse, json, time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
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


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--envs", type=int, default=8)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--run-name", required=True)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--entropy", type=float, default=0.002)
    pa.add_argument("--smoke", action="store_true")
    pa.add_argument("--grasp-only", action="store_true")
    a = pa.parse_args()
    torch.set_num_threads(1)
    out = ROOT / "runs" / a.run_name
    out.mkdir(parents=True, exist_ok=False)
    schedule = [
        ("teacher", "grasp", 150000),
        ("teacher", "press", 300000),
        ("teacher", "push", 500000),
        ("teacher", "full", 800000),
    ]
    if a.grasp_only:
        schedule = schedule[:1]
    if a.smoke:
        schedule = [("teacher", "grasp", 512)]
    cfg = {
        "schedule": schedule,
        "seed": a.seed,
        "device": a.device,
        "parallel_envs": a.envs,
        "lr": a.lr,
        "entropy": a.entropy,
        "manager": "scripted stages, not learned HRL manager",
        "leg_policy": "frozen original 12D leg policy",
        "high_level_action": "10D residual: base vx/vy/wz, TCP xyz and rotvec, gripper closure adjustment",
        "teacher_privileged": True,
        "visual_closed_loop_enabled": False,
        "hardware_approved": False,
        "old_high_level_checkpoint_used": False,
        "reason": "new grasp observation and phase contract; old 60k models are retained in parent directory",
        "task_manifest": json.loads((ROOT / "manifest.json").read_text()),
        "source_sha256": artifact_hashes(
            [
                "core.py",
                "door_env.py",
                "build_scene.py",
                "assets/door_scene.xml",
                "assets/robot.xml",
                "manifest.json",
            ]
        ),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    model = None
    total = 0
    for mode, stage, steps in schedule:
        run = out / f"{mode}_{stage}"
        run.mkdir(exist_ok=True)
        vec = SubprocVecEnv(
            [make_env(mode, stage, a.seed * 100 + i, run) for i in range(a.envs)],
            start_method="spawn",
        )
        if model is None:
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
                    "log_std_init": -1.8,
                },
                verbose=1,
                seed=a.seed,
                tensorboard_log=str(out / "tensorboard"),
            )
        else:
            model.set_env(vec)
        status = {
            "state": "training",
            "mode": mode,
            "stage": stage,
            "requested_stage_steps": steps,
            "previous_stage_steps": total,
            "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "physics_version": "door-grasp-wrist-v4-clean",
        }
        (out / "status.json").write_text(json.dumps(status, indent=2))
        print("START_STAGE", json.dumps(status), flush=True)
        before = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=CheckpointCallback(
                save_freq=max(10000 // a.envs, 1),
                save_path=str(run / "checkpoints"),
                name_prefix="policy",
            ),
            tb_log_name=f"{mode}_{stage}",
        )
        change = sum(
            float((p - b).abs().sum())
            for p, b in zip(model.policy.parameters(), before)
        )
        assert change > 0
        model.save(run / "policy_final")
        total += steps
        vec.close()
        if a.smoke:
            status.update(state="gradient_smoke_passed", parameter_l1_change=change)
            (out / "status.json").write_text(json.dumps(status, indent=2))
            break
        status["state"] = "evaluating"
        (out / "status.json").write_text(json.dumps(status, indent=2))
        ev = evaluate(model, mode, stage)
        (run / "heldout.json").write_text(json.dumps(ev, indent=2))
        print("EVAL_STAGE", mode, stage, ev["passed"], ev["total"], flush=True)
        if ev["passed"] < 30:
            status.update(
                state="stopped_gate_not_met",
                passed=ev["passed"],
                total=ev["total"],
                reason="Preserve checkpoint. Inspect grasp/physics before increasing training.",
            )
            (out / "status.json").write_text(json.dumps(status, indent=2))
            break
    else:
        (out / "status.json").write_text(
            json.dumps(
                {
                    "state": "finished_candidate_curriculum",
                    "finished_requested_steps": total,
                    "hardware_approved": False,
                    "requires": "independent visual full-task contact and robustness acceptance",
                },
                indent=2,
            )
        )
