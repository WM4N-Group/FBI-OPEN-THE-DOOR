"""Bounded multi-stage residual PPO for box pushing. Rule manager, not learned HRL."""

import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from .evaluate_box import evaluate, environment_class
from ..paths import RUNTIME_ROOT as ROOT
from ..storage import atomic_json, atomic_save_model
from ..box_task import LEG_CHECKPOINT_SHA256


class Progress(BaseCallback):
    def __init__(self, out, status, deadline):
        super().__init__()
        self.out = out
        self.deadline = deadline
        self.timed_out = False
        self.status = status
        self.next_save = 25000
        self.successes = []
        self.failures = {}

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.successes.append(bool(info["is_success"]))
                reason = info["failure"] or "success"
                self.failures[reason] = self.failures.get(reason, 0) + 1
        if self.num_timesteps >= self.next_save:
            atomic_save_model(self.model, self.out / "policy_latest.zip")
            self.next_save = self.num_timesteps + 25000
        self.timed_out = time.monotonic() >= self.deadline
        return not self.timed_out

    def _on_rollout_end(self):
        atomic_json(
            self.out / "progress.json",
            {
                **self.status,
                "state": "running",
                "steps": self.num_timesteps,
                "ppo_updates": self.model._n_updates,
                "last100_success_rate": float(np.mean(self.successes[-100:]))
                if self.successes
                else None,
                "episodes": len(self.successes),
                "outcomes": self.failures,
                "updated_unix": time.time(),
                "train_metrics": {
                    k: float(v)
                    for k, v in self.model.logger.name_to_value.items()
                    if k.startswith("train/")
                    and isinstance(v, (int, float, np.floating))
                },
            },
        )


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument(
        "--controller", choices=["legacy", "clearance_v2"], default="clearance_v2"
    )
    pa.add_argument("--run-name", required=True)
    pa.add_argument("--envs", type=int, default=6)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--seed", type=int, default=100)
    pa.add_argument("--nominal-steps", type=int, default=200000)
    pa.add_argument("--perturbed-steps", type=int, default=800000)
    pa.add_argument("--gate-retry-steps", type=int, default=200000)
    pa.add_argument(
        "--resume-zip",
        type=Path,
        help="Trusted compatible box PPO only; never load untrusted pickle",
    )
    pa.add_argument(
        "--start-stage", choices=["nominal", "initial_pose"], default="nominal"
    )
    pa.add_argument("--smoke", action="store_true")
    a = pa.parse_args()
    if (
        a.envs < 1
        or min(a.nominal_steps, a.perturbed_steps) < 1
        or a.gate_retry_steps < 0
    ):
        pa.error("Positive environment/budget counts required")
    if not a.run_name or Path(a.run_name).name != a.run_name:
        pa.error("run-name must be a single directory name")
    if a.resume_zip and not a.resume_zip.is_file():
        pa.error("resume checkpoint does not exist")
    if a.start_stage != "nominal" and not a.resume_zip:
        pa.error("Later stage requires a compatible box checkpoint")
    env_class = environment_class(a.controller)
    torch.set_num_threads(1)
    out = ROOT / "runs" / a.run_name
    out.mkdir(parents=True, exist_ok=False)
    policy = ROOT / "reference/models/policies/arm_moving_clock/model_7999.pt"
    if hashlib.sha256(policy.read_bytes()).hexdigest() != LEG_CHECKPOINT_SHA256:
        raise ValueError("Frozen leg policy mismatch")
    source = Path(__file__).parent
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "train_box.py",
            "box_env.py",
            "box_physics.py",
            "core.py",
            "build_box_scene.py",
            "evaluate_box.py",
            "box_collision.py",
            "box_clearance_env.py",
        )
    }
    hashes["box_clearance.py"] = hashlib.sha256(
        (source.parent / "box_clearance.py").read_bytes()
    ).hexdigest()
    hashes["box_task.py"] = hashlib.sha256(
        (source.parent / "box_task.py").read_bytes()
    ).hexdigest()
    for name in (
        "assets/box_scene.xml",
        "assets/box_robot.xml",
        "box_manifest.json",
        "reference/models/policies/arm_moving_clock/exported/policy.onnx",
        "reference/third_party/locomotion_policy/deploy/locomotion_policy_wrapper.py",
        "reference/third_party/locomotion_policy/deploy/config.py",
    ):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    plan = [("nominal", 0.0, a.nominal_steps), ("initial_pose", 1.0, a.perturbed_steps)]
    if a.start_stage == "initial_pose":
        plan = plan[1:]
    if a.smoke:
        plan = [("smoke", 0.0, 512)]
    cfg = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "plan": plan,
        "resume_sha256": hashlib.sha256(a.resume_zip.read_bytes()).hexdigest()
        if a.resume_zip
        else None,
        "source_and_asset_sha256": hashes,
        "frozen_leg_sha256": LEG_CHECKPOINT_SHA256,
        "high_level_action": "6D residual base vx/vy/wz + pre-contact TCP xyz; arm residual disabled during push",
        "arm_lock": "latch commanded joint targets; bounded PD, no weld",
        "manager": "rules, NOT learned HRL",
        "perception": "privileged box geometry, NOT vision",
        "maximum_budget_before_vector_rollout_rounding": sum(
            x[2] + (0 if a.smoke else a.gate_retry_steps) for x in plan
        ),
        "scope": "simulation; fixed 2kg box, friction .35, only initial pose randomization",
        "hardware_enabled": False,
        "wall_clock_timeout_hours": 18,
    }
    atomic_json(out / "config.json", cfg)
    state = {"state": "starting", "run_name": a.run_name, "seed": a.seed, "steps": 0}
    atomic_json(out / "status.json", state)
    model = None
    start = time.monotonic()
    gate_failed = False
    try:
        for stage, randomization, budget in plan:
            env = SubprocVecEnv(
                [
                    lambda r=randomization: Monitor(env_class(randomization=r))
                    for _ in range(a.envs)
                ]
            )
            env.seed(a.seed)
            try:
                if model is None and a.resume_zip:
                    model = PPO.load(
                        a.resume_zip,
                        env=env,
                        device=a.device,
                        seed=a.seed,
                        tensorboard_log=str(out / "tb"),
                    )
                    atomic_save_model(model, out / "policy_initial.zip")
                elif model is None:
                    model = PPO(
                        "MlpPolicy",
                        env,
                        learning_rate=2e-4,
                        n_steps=128 if a.smoke else 512,
                        batch_size=128 if a.smoke else 512,
                        n_epochs=5,
                        gamma=0.995,
                        gae_lambda=0.95,
                        ent_coef=0.001,
                        policy_kwargs=dict(
                            net_arch=dict(pi=[128, 128], vf=[128, 128]),
                            log_std_init=-2.2,
                        ),
                        seed=a.seed,
                        device=a.device,
                        verbose=1,
                        tensorboard_log=str(out / "tb"),
                    )
                    # Start at the physically verified zero-residual teacher, not random high-level motion.
                    with torch.no_grad():
                        model.policy.action_net.weight.zero_()
                        model.policy.action_net.bias.zero_()
                    atomic_save_model(model, out / "policy_initial.zip")
                else:
                    model.set_env(env)
                for attempt, steps in enumerate(
                    [budget]
                    + (
                        [a.gate_retry_steps]
                        if a.gate_retry_steps and not a.smoke
                        else []
                    )
                ):
                    state.update(state="running", stage=stage, attempt=attempt)
                    atomic_json(out / "status.json", state)
                    callback = Progress(out, state, start + 18 * 3600)
                    model.learn(
                        total_timesteps=steps,
                        callback=callback,
                        reset_num_timesteps=False,
                        tb_log_name=stage,
                    )
                    atomic_save_model(model, out / "policy_latest.zip")
                    atomic_save_model(
                        model, out / f"policy_{stage}_attempt{attempt}.zip"
                    )
                    state["steps"] = model.num_timesteps
                    if callback.timed_out:
                        state.update(state="stopped_wall_clock_budget")
                        atomic_json(out / "status.json", state)
                        return
                    if a.smoke:
                        # Actual gradient and save/reload checks, not a task-success gate.
                        restored = PPO.load(out / "policy_latest.zip", device=a.device)
                        test_obs = env.reset()
                        np.testing.assert_allclose(
                            model.predict(test_obs, deterministic=True)[0],
                            restored.predict(test_obs, deterministic=True)[0],
                            atol=1e-6,
                        )
                        initial = PPO.load(out / "policy_initial.zip", device="cpu")
                        changed = any(
                            not torch.equal(
                                v.detach().cpu(), initial.policy.state_dict()[k].cpu()
                            )
                            for k, v in model.policy.state_dict().items()
                        )
                        if not changed or model._n_updates < 1:
                            raise RuntimeError("No gradient update detected")
                        state.update(
                            state="smoke_gradient_save_reload_passed",
                            ppo_updates=model._n_updates,
                        )
                        atomic_json(out / "status.json", state)
                        break
                    gate = evaluate(
                        model,
                        count=8,
                        seed=810000 + int(randomization) * 1000,
                        randomization=randomization,
                        controller=a.controller,
                    )
                    atomic_json(out / f"gate_{stage}_attempt{attempt}.json", gate)
                    passed = gate["passed"] >= 7
                    print("BOX_GATE", stage, attempt, gate["passed"], "/ 8", flush=True)
                    if passed:
                        break
                    if attempt == 1 or not a.gate_retry_steps:
                        gate_failed = True
                if gate_failed:
                    state.update(state="stopped_gate_budget_exhausted")
                    atomic_json(out / "status.json", state)
                    break
            finally:
                env.close()
            if time.monotonic() - start > 18 * 3600:
                state.update(state="stopped_wall_clock_budget")
                atomic_json(out / "status.json", state)
                return
        if not a.smoke and not gate_failed:
            final = evaluate(
                model, count=32, seed=910000, randomization=1.0, controller=a.controller
            )
            atomic_json(out / "final_evaluation.json", final)
            state.update(
                state="completed_budget",
                final_passed=final["passed"],
                final_total=32,
                validation_scope=final["scope"],
            )
            atomic_json(out / "status.json", state)
    except Exception as exc:
        state.update(state="failed", error=repr(exc))
        atomic_json(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
