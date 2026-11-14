"""Independent GPU PPO runs; full 12D leg actor is trainable, arm is not an action."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import torch
from .contracts import LEG_SHA256, OBS_DIM, SCHEMA
from .gpu_env import LegSkillEnv
from .policy import ActorCritic


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def save(path, state):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=("climb", "down"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--steps", type=int, default=200_000_000)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--difficulty", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--checkpoint-every", type=int, default=1_000_000)
    p.add_argument("--eval-every", type=int, default=10_000_000)
    p.add_argument("--resume", type=Path)
    p.add_argument("--resume-sha256")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if (
        a.envs < 1
        or a.horizon < 2
        or a.steps < 1
        or a.batch_size < 2
        or a.epochs < 1
        or not 0 <= a.difficulty <= 1
    ):
        p.error("Invalid training dimensions/difficulty")
    if a.resume and not a.resume_sha256:
        p.error("Resume requires explicit checkpoint SHA256")
    return a


@torch.no_grad()
def evaluate(model, runtime, task, seed, count=32):
    env = LegSkillEnv(runtime, task, count, seed, split="heldout", difficulty=1.0)
    active = torch.ones(count, device="cuda", dtype=torch.bool)
    results = []
    for _ in range(1002):
        _, _, done, info = env.step(model.actor(env.obs), auto_reset=True)
        ended = active & done
        for i in ended.nonzero().flatten().tolist():
            results.append(
                {
                    "index": i,
                    "success": bool(info["success"][i]),
                    "failure": bool(info["failure"][i]),
                    "timeout": bool(info["timeout"][i]),
                    "duration": float(info["duration"][i]),
                    "arm_error_rad": float(info["arm_error"][i]),
                    "forbidden_peak_n": float(info["forbidden_peak"][i]),
                    "under_seen": bool(info["under_seen"][i]),
                    "domain": env.domains[i],
                }
            )
        active &= ~done
        if not bool(active.any()):
            break
    return {
        "split": "heldout",
        "count": count,
        "successes": sum(x["success"] for x in results),
        "completed": len(results),
        "episodes": results,
        "scope": "privileged geometry, locked-arm simulation; no hardware or skill-chain validation",
    }


def main():
    a = arguments()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(
        schema=SCHEMA,
        source_sha256=sources,
        original_leg_sha256=LEG_SHA256,
        observation_dim=OBS_DIM,
        action_dim=12,
        trainable="ALL actor weights, task input columns, critic and log_std; not a 6D high-level residual",
        arm="original home target for all six joints, bounded PD+bias; no weld or arm policy",
        perception="privileged geometry teacher",
        domain_schedule="20% full difficulty,80% fixed easy/mid mixture; no claimed adaptive curriculum",
        success_scope="independent skills; not a learned manager",
        hardware_enabled=False,
        gpu_name=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        training_steps_are_aggregate_env_steps=True,
    )
    atomic_json(a.output / "config.json", config)
    original = a.runtime / "reference/models/policies/arm_moving_clock/model_7999.pt"
    model = ActorCritic()
    model.initialize(original)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    start_steps = 0
    if a.resume:
        if hashlib.sha256(a.resume.read_bytes()).hexdigest() != a.resume_sha256:
            raise ValueError("Resume checkpoint mismatch")
        state = torch.load(a.resume, map_location="cuda", weights_only=True)
        if (
            state["task"] != a.task
            or state["schema"] != SCHEMA
            or state["original_leg_sha256"] != LEG_SHA256
        ):
            raise ValueError("Incompatible skill checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_steps = state["steps"]
    stop = {"requested": False}

    def request_stop(*_):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state = {
        "state": "initializing_gpu_physics",
        "task": a.task,
        "steps": start_steps,
        "budget": a.steps,
        "seed": a.seed,
    }
    atomic_json(a.output / "status.json", state)
    start = time.monotonic()
    env = None
    steps = start_steps

    def checkpoint(name):
        path = a.output / name
        save(
            path,
            {
                "schema": SCHEMA,
                "task": a.task,
                "steps": steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "original_leg_sha256": LEG_SHA256,
                "source_sha256": sources,
                "seed": a.seed,
                "obs_dim": OBS_DIM,
            },
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        env = LegSkillEnv(a.runtime, a.task, a.envs, a.seed, difficulty=a.difficulty)
        env.save_domains(a.output / "domains.json")
        initial_sha = checkpoint("policy_initial.pt")
        initial_actor = {
            k: v.detach().clone() for k, v in model.actor.state_dict().items()
        }
        shape = (a.horizon, a.envs)
        obsbuf = torch.empty((*shape, OBS_DIM), device="cuda")
        actions = torch.empty((*shape, 12), device="cuda")
        logprobs = torch.empty(shape, device="cuda")
        rewards = torch.empty(shape, device="cuda")
        values = torch.empty(shape, device="cuda")
        dones = torch.empty(shape, device="cuda")
        advantages = torch.empty(shape, device="cuda")
        gamma, lam = 0.99, 0.95
        next_save = steps + a.checkpoint_every
        next_eval = steps + a.eval_every
        update = 0
        training_start = time.monotonic()
        obs = env.obs
        stats = {
            "episodes": 0,
            "successes": 0,
            "failures": 0,
            "timeouts": 0,
            "return_sum": 0.0,
            "duration_sum": 0.0,
            "arm_error_max": 0.0,
        }
        while steps < a.steps and not stop["requested"]:
            collect_start = time.monotonic()
            for t in range(a.horizon):
                obsbuf[t] = obs
                with torch.no_grad():
                    action, logprob, value = model.act(obs)
                    obs, reward, done, info = env.step(action)
                    # Correct time-limit bootstrap; reset obs never bootstraps terminal states.
                    reward = (
                        reward
                        + gamma
                        * model.value(info["terminal_obs"])
                        * info["timeout"].float()
                    )
                actions[t], logprobs[t], values[t] = action, logprob, value
                rewards[t], dones[t] = reward, done.float()
                # Accumulate on GPU; synchronize only once per PPO update.
                if t == 0:
                    counters = torch.zeros(19, device="cuda")
                counters[:4] += torch.stack(
                    (
                        done.sum(),
                        info["success"].sum(),
                        info["failure"].sum(),
                        info["timeout"].sum(),
                    )
                )
                counters[4] += (info["return"] * done).sum()
                counters[5] += (info["duration"] * done).sum()
                counters[6] = torch.maximum(counters[6], info["arm_error"].max())
                counters[7] += ((info["physics_overflow"] & 511) != 0).sum()
                counters[8] += ((info["physics_overflow"] & 1536) != 0).sum()
                counters[9:16] += info["failure_reasons"].sum(0)
                counters[16] = torch.maximum(counters[16], info["base_x"].max())
                counters[17] = torch.maximum(counters[17], info["dwell"].max())
                counters[18] += info["under_seen"].sum()
            c = counters.cpu().tolist()
            if c[7] or c[9]:
                raise RuntimeError(
                    "Physics capacity/CCD overflow or nonfinite state; abort before optimizer update"
                )
            with torch.no_grad():
                nextvalue = model.value(obs)
                gae = torch.zeros(a.envs, device="cuda")
                for t in reversed(range(a.horizon)):
                    nv = nextvalue if t == a.horizon - 1 else values[t + 1]
                    alive = 1 - dones[t]
                    delta = rewards[t] + gamma * nv * alive - values[t]
                    gae = delta + gamma * lam * alive * gae
                    advantages[t] = gae
                returns = advantages + values
            flatobs = obsbuf.flatten(0, 1)
            flatac = actions.flatten(0, 1)
            flatlog = logprobs.flatten()
            flatadv = advantages.flatten()
            flatret = returns.flatten()
            flatadv = (flatadv - flatadv.mean()) / (flatadv.std() + 1e-8)
            rollout_seconds = time.monotonic() - collect_start
            batch = a.envs * a.horizon
            losses = []
            for _ in range(a.epochs):
                order = torch.randperm(batch, device="cuda")
                for begin in range(0, batch, a.batch_size):
                    ids = order[begin : begin + a.batch_size]
                    dist = model.distribution(flatobs[ids])
                    log = dist.log_prob(flatac[ids]).sum(-1)
                    ratio = (log - flatlog[ids]).exp()
                    surrogate = torch.minimum(
                        ratio * flatadv[ids], ratio.clamp(0.8, 1.2) * flatadv[ids]
                    )
                    value_loss = (
                        0.5 * (model.value(flatobs[ids]) - flatret[ids]).square().mean()
                    )
                    loss = (
                        -surrogate.mean()
                        + value_loss
                        - 0.003 * dist.entropy().sum(-1).mean()
                    )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not bool(torch.isfinite(grad)):
                        raise FloatingPointError("Nonfinite gradient")
                    optimizer.step()
                    losses.append(loss.detach())
            steps += batch
            update += 1
            for k, v in zip(
                (
                    "episodes",
                    "successes",
                    "failures",
                    "timeouts",
                    "return_sum",
                    "duration_sum",
                ),
                c[:6],
            ):
                stats[k] += v
            stats["arm_error_max"] = max(stats["arm_error_max"], c[6])
            changed = max(
                float((v - initial_actor[k]).abs().max())
                for k, v in model.actor.state_dict().items()
            )
            recent_rate = c[1] / max(c[0], 1)
            state = {
                "state": "training",
                "task": a.task,
                "seed": a.seed,
                "steps": steps,
                "budget": a.steps,
                "update": update,
                "elapsed_s": time.monotonic() - start,
                "aggregate_sps": (steps - start_steps)
                / max(time.monotonic() - training_start, 0.01),
                "rollout_s": rollout_seconds,
                "update_loss": float(torch.stack(losses).mean()),
                "recent_success_rate": recent_rate,
                "recent_episodes": c[0],
                "solver_limit_env_step_fraction": c[8] / batch,
                "recent_failure_reasons_nonexclusive": dict(
                    zip(
                        (
                            "nonfinite",
                            "fall",
                            "arm_drift",
                            "forbidden_contact",
                            "self_contact",
                            "box_state",
                            "bounds",
                        ),
                        c[9:16],
                    )
                ),
                "recent_base_x_max": c[16],
                "recent_stable_dwell_max_s": c[17],
                "under_seen_env_step_fraction": c[18] / batch,
                "cumulative": stats,
                "actor_parameter_max_change": changed,
                "arm_target_max_change_rad": float(
                    (env.target[:, 12:18] - env.home[19:25]).abs().max()
                ),
                "initial_checkpoint_sha256": initial_sha,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            atomic_json(a.output / "status.json", state)
            with (a.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(state, allow_nan=False) + "\n")
            print(json.dumps(state, allow_nan=False), flush=True)
            if update == 1 or steps >= next_save:
                state["latest_checkpoint_sha256"] = checkpoint("policy_latest.pt")
                next_save = steps + a.checkpoint_every
                atomic_json(a.output / "status.json", state)
            if not a.smoke and steps >= next_eval:
                # Fixed heldout seed reused for monitoring: not an independent final test.
                report = evaluate(model, a.runtime, a.task, a.seed + 1000, 16)
                report["steps"] = steps
                atomic_json(a.output / ("monitor_%012d.json" % steps), report)
                next_eval = steps + a.eval_every
            if a.smoke and update >= 2:
                break
        digest = checkpoint("policy_latest.pt")
        state.update(
            state="stopped_on_request"
            if stop["requested"]
            else "smoke_complete"
            if a.smoke
            else "training_budget_complete",
            steps=steps,
            latest_checkpoint_sha256=digest,
            elapsed_s=time.monotonic() - start,
        )
        # Completion means budget done, not success qualification.
        atomic_json(a.output / "status.json", state)
        if not a.smoke and not stop["requested"]:
            report = evaluate(model, a.runtime, a.task, a.seed + 9000, 64)
            atomic_json(a.output / "final_heldout.json", report)
    except BaseException as exc:
        state.update(
            state="failed",
            error=repr(exc),
            steps=steps,
            elapsed_s=time.monotonic() - start,
        )
        atomic_json(a.output / "status.json", state)
        raise


if __name__ == "__main__":
    main()
