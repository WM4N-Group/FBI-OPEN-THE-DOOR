"""Gated v3 PPO. Never calls a partial-goal reset a full task success."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import torch
from .curriculum_contracts import SCHEMA, OBS_DIM, STAGES, LIMITS, next_stage
from .curriculum_env import CurriculumEnv
from .curriculum_policy import ResidualActorCritic
from .contracts import LEG_SHA256
from .train import atomic_json, save


@torch.no_grad()
def evaluate(policy, env_args, stage, seed, count=32, amplitude=1.0):
    env = CurriculumEnv(
        **env_args, count=count, seed=seed, stage=stage, split="heldout"
    )
    env.residual_amplitude = amplitude
    active = torch.ones(count, device="cuda", dtype=torch.bool)
    results = []
    for _ in range(int(LIMITS[env.task]["timeout_s"] / 0.02) + 5):
        _, _, done, info = env.step(policy.actor(env.obs))
        ended = active & done
        for i in ended.nonzero().flatten().tolist():
            row = {
                k: bool(info[k][i])
                for k in (
                    "success",
                    "failure",
                    "timeout",
                    "task_success",
                    "style_clean",
                    "crouch_achieved",
                    "under_seen",
                    "phase_violation",
                )
            }
            row.update(
                index=i,
                duration_s=float(info["duration"][i]),
                phase=int(info["phase"][i]),
                domain=env.domains[i],
            )
            for k in (
                "max_stride_m",
                "max_foreaft_m",
                "max_lateral_m",
                "max_speed_m_s",
                "max_foot_force_n",
                "arm_error",
                "self_peak",
                "dwell",
                "style_dwell",
            ):
                row[k] = float(info[k][i])
            results.append(row)
        active &= ~done
        if not bool(active.any()):
            break
    report = dict(
        schema=SCHEMA,
        stage=stage,
        partial_goal=stage != "full",
        geometry_difficulty=env.curriculum_difficulty,
        successes=sum(e["success"] for e in results),
        count=count,
        completed=len(results),
        task_successes=sum(e["task_success"] for e in results),
        physical_failures=sum(e["failure"] for e in results),
        timeouts=sum(e["timeout"] for e in results),
        episodes=results,
        scope="partial goals are NOT climb/under-rail completion; full starts upright on floor, style/physical conditions required; privileged simulation only",
        controller="frozen v1 actor + explicit static-load/explicit crawl gait and retargeting + learned bounded joint residual",
        residual_amplitude=amplitude,
    )
    del env
    return report


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=("climb", "down"), required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-sha256", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--envs", type=int, default=2048)
    p.add_argument("--steps", type=int, default=150000000)
    p.add_argument("--partial-stage-cap", type=int, default=12000000)
    p.add_argument("--min-stage-steps", type=int, default=1000000)
    p.add_argument("--gate-every", type=int, default=2000000)
    p.add_argument("--gate-count", type=int, default=32)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--full-no-success-cap", type=int, default=30000000)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if (
        min(
            a.envs,
            a.steps,
            a.horizon,
            a.epochs,
            a.batch_size,
            a.partial_stage_cap,
            a.gate_count,
            a.gate_every,
        )
        < 1
    ):
        p.error("Positive dimensions/budgets required")
    return a


def main():
    a = args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    policy = ResidualActorCritic().cuda()
    optimizer = torch.optim.Adam(policy.parameters(), lr=a.lr)
    env_args = dict(
        runtime=a.runtime,
        task=a.task,
        reference=a.reference,
        reference_sha=a.reference_sha256,
    )
    stages = STAGES[a.task]
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(
        schema=SCHEMA,
        source_sha256=sources,
        observation_dim=OBS_DIM,
        action_dim=12,
        original_leg_sha256=LEG_SHA256,
        stages=stages,
        controller="NEW zero-initialized 12D joint residual; v1 actor frozen, static-load posture and hand-designed low crawl + compressed pretrained gait are explicit nonlearned baselines",
        stage_gate=">=75% completed heldout stage goals, min-stage-steps, then advance; partial phase capped, no blind budget increase",
        full_eval="upright floor starts; no on-box/low-reset completion counted as full success",
        limits=LIMITS[a.task],
        perception="privileged geometry",
        hardware_enabled=False,
        gamma=0.995,
        gae_lambda=0.95,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu=torch.cuda.get_device_name(0),
    )
    atomic_json(a.output / "config.json", config)
    stopped = {"requested": False}

    def stop(*_):
        stopped["requested"] = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    start = time.monotonic()
    steps = 0
    updates = 0
    stage = stages[0]
    stage_start = 0
    next_gate = a.gate_every
    next_save = 0
    bestfull = -1
    full_difficulty = 0.4
    env = None
    status = dict(
        state="initializing",
        task=a.task,
        seed=a.seed,
        stage=stage,
        steps=0,
        budget=a.steps,
    )
    atomic_json(a.output / "status.json", status)

    def checkpoint(name):
        path = a.output / name
        save(
            path,
            dict(
                schema=SCHEMA,
                task=a.task,
                steps=steps,
                stage=stage,
                model=policy.state_dict(),
                optimizer=optimizer.state_dict(),
                source_sha256=sources,
                reference_sha256=a.reference_sha256,
                obs_dim=OBS_DIM,
                seed=a.seed,
                original_leg_sha256=LEG_SHA256,
            ),
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        env = CurriculumEnv(**env_args, count=a.envs, seed=a.seed, stage=stage)
        env.save_domains(a.output / ("domains_" + stage + ".json"))
        # Freeze and fingerprint the teacher separately from the trainable residual.
        teacher_before = {
            k: v.detach().cpu().clone() for k, v in env.teacher.state_dict().items()
        }
        initial = checkpoint("policy_initial.pt")
        dims = (a.horizon, a.envs)
        obsbuf = torch.empty((*dims, OBS_DIM), device="cuda")
        actions = torch.empty((*dims, 12), device="cuda")
        logs = torch.empty(dims, device="cuda")
        values = torch.empty(dims, device="cuda")
        rewards = torch.empty(dims, device="cuda")
        dones = torch.empty(dims, device="cuda")
        adv = torch.empty(dims, device="cuda")
        obs = env.obs
        while steps < a.steps and not stopped["requested"]:
            stage_steps = steps - stage_start
            amplitude = min(1.0, 0.15 + stage_steps / 3000000.0)
            env.residual_amplitude = amplitude
            counters = torch.zeros(16, device="cuda")
            for t in range(a.horizon):
                obsbuf[t] = obs
                with torch.no_grad():
                    action, log, value = policy.act(obs)
                    obs, reward, done, info = env.step(action)
                    reward = (
                        reward
                        + 0.995
                        * policy.value(info["terminal_obs"])
                        * info["timeout"].float()
                    )
                    counters[:5] += torch.stack(
                        (
                            done.sum(),
                            info["success"].sum(),
                            info["failure"].sum(),
                            info["timeout"].sum(),
                            info["task_milestone"].sum(),
                        )
                    )
                    counters[5] += (info["return"] * done).sum()
                    counters[6] += ((info["physics_overflow"] & 511) != 0).sum()
                    counters[7] += ((info["physics_overflow"] & 1536) != 0).sum()
                    counters[8] = torch.maximum(counters[8], info["max_stride_m"].max())
                    counters[9] = torch.maximum(
                        counters[9], info["max_speed_m_s"].max()
                    )
                    counters[10] += info["crouch_achieved"].sum()
                    counters[11] += info["under_seen"].sum()
                    counters[12] += info["style_clean"].sum()
                    counters[13] = torch.maximum(counters[13], info["arm_error"].max())
                    counters[14] += (~torch.isfinite(env.q).all(-1)).sum()
                    counters[15] = torch.maximum(counters[15], info["dwell"].max())
                actions[t] = action
                logs[t] = log
                values[t] = value
                rewards[t] = reward
                dones[t] = done.float()
            c = counters.cpu().tolist()
            if c[6] or c[14]:
                raise RuntimeError(
                    "Physics capacity/CCD overflow or nonfinite state: abort BEFORE optimizer"
                )
            with torch.no_grad():
                nv = policy.value(obs)
                gae = torch.zeros(a.envs, device="cuda")
                for t in reversed(range(a.horizon)):
                    following = nv if t == a.horizon - 1 else values[t + 1]
                    alive = 1 - dones[t]
                    delta = rewards[t] + 0.995 * following * alive - values[t]
                    gae = delta + 0.995 * 0.95 * alive * gae
                    adv[t] = gae
                ret = adv + values
            x = obsbuf.flatten(0, 1)
            ac = actions.flatten(0, 1)
            oldlog = logs.flatten()
            advantages = adv.flatten()
            returns = ret.flatten()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            losses = []
            batch = a.horizon * a.envs
            for _ in range(a.epochs):
                order = torch.randperm(batch, device="cuda")
                for offset in range(0, batch, a.batch_size):
                    ids = order[offset : offset + a.batch_size]
                    dist = policy.distribution(x[ids])
                    newlog = dist.log_prob(ac[ids]).sum(-1)
                    ratio = (newlog - oldlog[ids]).exp()
                    pg = -torch.minimum(
                        ratio * advantages[ids], ratio.clamp(0.8, 1.2) * advantages[ids]
                    ).mean()
                    vl = 0.5 * (policy.value(x[ids]) - returns[ids]).square().mean()
                    # A small residual prior protects the frozen skill; it is not a teacher success label.
                    prior = 0.002 * dist.mean.square().mean()
                    loss = pg + vl - 0.001 * dist.entropy().sum(-1).mean() + prior
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    if not bool(torch.isfinite(grad)):
                        raise FloatingPointError("Nonfinite gradient")
                    optimizer.step()
                    losses.append(loss.detach())
            steps += batch
            updates += 1
            status = dict(
                state="training",
                task=a.task,
                seed=a.seed,
                stage=stage,
                partial_goal=stage != "full",
                steps=steps,
                budget=a.steps,
                stage_steps=steps - stage_start,
                geometry_difficulty=env.curriculum_difficulty,
                update=updates,
                elapsed_s=time.monotonic() - start,
                aggregate_sps=steps / (time.monotonic() - start),
                recent_episodes=c[0],
                recent_stage_successes=c[1],
                recent_physical_failures=c[2],
                recent_timeouts=c[3],
                recent_task_successes=c[4],
                mean_completed_return=c[5] / max(c[0], 1),
                solver_limit_env_step_fraction=c[7] / batch,
                max_stride_m=c[8],
                max_speed_m_s=c[9],
                crouch_fraction=c[10] / batch,
                under_rail_fraction=c[11] / batch,
                style_clean_fraction=c[12] / batch,
                actual_arm_error_max_rad=c[13],
                stable_dwell_max_s=c[15],
                residual_amplitude=amplitude,
                actor_output_weight_max_abs=float(policy.actor[-1].weight.abs().max()),
                arm_target_max_change_rad=float(
                    (env.target[:, 12:18] - env.home[19:25]).abs().max()
                ),
                loss=float(torch.stack(losses).mean()),
                initial_checkpoint_sha256=initial,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            )
            atomic_json(a.output / "status.json", status)
            with (a.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
            if steps >= next_save:
                digest = checkpoint("policy_latest.pt")
                status["latest_checkpoint_sha256"] = digest
                atomic_json(a.output / "status.json", status)
                next_save = steps + 1000000
            if a.smoke and updates >= 2:
                break
            if not a.smoke and steps >= next_gate:
                for k, v in env.teacher.state_dict().items():
                    if not torch.equal(v.detach().cpu(), teacher_before[k]):
                        raise RuntimeError("Frozen teacher changed")
                report = evaluate(
                    policy, env_args, stage, a.seed + 1000, a.gate_count, amplitude
                )
                report.update(
                    steps=steps,
                    checkpoint_sha256=checkpoint("policy_latest.pt"),
                    monitor_reused=True,
                )
                atomic_json(a.output / ("gate_%s_%012d.json" % (stage, steps)), report)
                if stage == "full":
                    if report["successes"] > bestfull:
                        bestfull = report["successes"]
                        report["best_checkpoint_sha256"] = checkpoint(
                            "policy_best_monitor.pt"
                        )
                        atomic_json(a.output / "best_monitor.json", report)
                elif (
                    steps - stage_start >= a.min_stage_steps
                    and report["completed"] == a.gate_count
                ):
                    following = next_stage(
                        a.task, stage, report["successes"], report["count"]
                    )
                    if following != stage:
                        checkpoint("policy_passed_" + stage + ".pt")
                        del env
                        stage = following
                        stage_start = steps
                        env = CurriculumEnv(
                            **env_args,
                            count=a.envs,
                            seed=a.seed,
                            stage=stage,
                            difficulty=full_difficulty if stage == "full" else 1.0,
                        )
                        env.save_domains(a.output / ("domains_" + stage + ".json"))
                        obs = env.obs
                        # Old value estimates refer to a different reset-goal distribution.
                        for layer in policy.critic:
                            if isinstance(layer, torch.nn.Linear):
                                layer.reset_parameters()
                        optimizer = torch.optim.Adam(policy.parameters(), lr=a.lr)
                # Only the TRAINING full geometry is ramped. Heldout full is always difficulty=1.
                if stage == "full":
                    desired_difficulty = min(
                        1.0, 0.4 + 0.2 * ((steps - stage_start) // 10000000)
                    )
                    if desired_difficulty > full_difficulty:
                        full_difficulty = desired_difficulty
                        del env
                        env = CurriculumEnv(
                            **env_args,
                            count=a.envs,
                            seed=a.seed,
                            stage=stage,
                            difficulty=full_difficulty,
                        )
                        env.save_domains(
                            a.output / ("domains_full_d%.1f.json" % full_difficulty)
                        )
                        obs = env.obs
                next_gate = steps + a.gate_every
                if (
                    stage == "full"
                    and bestfull <= 0
                    and steps - stage_start >= a.full_no_success_cap
                ):
                    status.update(
                        state="curriculum_stalled",
                        stage=stage,
                        error="No complete style-qualified full heldout success before watchdog cap; preserve checkpoint and diagnose",
                    )
                    break
                if stage != "full" and steps - stage_start >= a.partial_stage_cap:
                    status.update(
                        state="stage_gate_failed",
                        stage=stage,
                        error="Partial stage did not meet heldout gate before cap; saved instead of spending full budget",
                    )
                    break
        digest = checkpoint("policy_latest.pt")
        status.update(
            state=(
                "stopped_on_request"
                if stopped["requested"]
                else "smoke_complete"
                if a.smoke
                else status.get("state")
                if status.get("state") in ("stage_gate_failed", "curriculum_stalled")
                else "training_budget_complete"
            ),
            steps=steps,
            stage=stage,
            latest_checkpoint_sha256=digest,
            elapsed_s=time.monotonic() - start,
        )
        atomic_json(a.output / "status.json", status)
        if not a.smoke and not stopped["requested"]:
            # Stage final report and full upright-floor report are separate, even on gate failure.
            report = evaluate(policy, env_args, stage, a.seed + 9000, 64, 1.0)
            atomic_json(a.output / "final_stage_heldout.json", report)
            full = evaluate(policy, env_args, "full", a.seed + 19000, 64, 1.0)
            atomic_json(a.output / "final_full_heldout.json", full)
    except BaseException as exc:
        status.update(
            state="failed",
            error=repr(exc),
            steps=steps,
            stage=stage,
            elapsed_s=time.monotonic() - start,
        )
        atomic_json(a.output / "status.json", status)
        raise


if __name__ == "__main__":
    main()
