"""v2 mechanics/phase/target-slew preflight, not successful skill evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import warp as wp
from .style_env import SmallStepEnv
from .style_policy import StyleActorCritic
from .policy import mlp
from .style_contracts import STYLE_OBS_DIM, STYLE_SCHEMA


def validate(runtime, task, warm_start, warm_sha256):
    runtime = Path(runtime)
    env = SmallStepEnv(runtime, task, 16, 123, split="heldout", difficulty=1.0)
    model = StyleActorCritic()
    provenance = model.warm_start(warm_start, warm_sha256, task)
    model.cuda()
    state = torch.load(warm_start, map_location="cpu", weights_only=True)
    reference = mlp(provenance["source_input_dim"], 12)
    reference.load_state_dict(
        {k[6:]: v for k, v in state["model"].items() if k.startswith("actor.")}
    )
    reference.cuda()
    torch.manual_seed(123)
    x = torch.randn((64, STYLE_OBS_DIM), device="cuda")
    with torch.no_grad():
        parity = float(
            (model.actor(x) - reference(x[:, : provenance["source_input_dim"]]))
            .abs()
            .max()
        )
    assert parity < 2e-5, parity
    # Exact same fixed PD targets on CPU/GPU; never policy reset/teleport as a success aid.
    m = mujoco.MjModel.from_xml_path(str(runtime / "assets/scene.xml"))
    d = mujoco.MjData(m)
    for name in (
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_aabb",
        "geom_rbound",
        "geom_friction",
        "body_mass",
        "body_inertia",
    ):
        getattr(m, name)[:] = (
            wp.to_torch(getattr(env.wm, name))[0]
            .cpu()
            .numpy()
            .reshape(getattr(m, name).shape)
        )
    mujoco.mj_setConst(m, d)
    d.qpos[:] = env.q[0].cpu().numpy()
    d.qvel[:] = env.v[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    target = env.target[0].cpu().numpy().copy()
    errors = []
    flags = []
    for _ in range(5):
        env.audit.zero_()
        env.overflow.zero_()
        wp.capture_launch(env.graph)
        wp.synchronize()
        for _ in range(10):
            d.ctrl[:12] = np.clip(
                float(env.motor[0])
                * (
                    20 * float(env.kp[0]) * (target[:12] - d.qpos[7:19])
                    - 1.5 * float(env.kd[0]) * d.qvel[6:18]
                ),
                m.actuator_ctrlrange[:12, 0],
                m.actuator_ctrlrange[:12, 1],
            )
            d.ctrl[12:18] = np.clip(
                50 * (target[12:18] - d.qpos[19:25])
                - 4 * d.qvel[18:24]
                + d.qfrc_bias[18:24],
                m.actuator_ctrlrange[12:18, 0],
                m.actuator_ctrlrange[12:18, 1],
            )
            d.ctrl[18] = np.clip(800 * (0.006 - d.qpos[25]) - 5 * d.qvel[24], -20, 20)
            mujoco.mj_step(m, d)
        errors.append(float(np.abs(d.qpos - env.q[0].cpu().numpy()).max()))
        flags.append(int(env.overflow[0]))
    assert max(errors) < 0.002 and not any(f & 511 for f in flags), (errors, flags)
    contact_detected = bool((env.foot_current > 0.5).any())
    assert contact_detected, "No actual foot-ground contact detected"
    env.reset(torch.ones(16, device="cuda", dtype=torch.bool))
    env.obs = env.observe()
    with torch.no_grad():
        _, _, _, info = env.step(
            torch.full((16, 12), 6.0, device="cuda"), auto_reset=False
        )
    rate = float(info["max_target_rate_rad_s"].max())
    assert rate <= env.limits["target_rate_rad_s"] + 1e-4, rate
    arm_delta = float((env.target[:, 12:18] - env.home[19:25]).abs().max())
    assert arm_delta == 0.0
    # Explicit negative-test reset: inject premature forward position, not a training rollout.
    phase_negative = None
    if task == "down":
        env.reset(torch.ones(16, device="cuda", dtype=torch.bool))
        env.obs = env.observe()
        env.q[:, 0] += 0.22
        wp.to_torch(env.wd.qacc_warmstart).zero_()
        env.v.zero_()
        with torch.no_grad():
            _, _, _, bad = env.step(
                torch.zeros((16, 12), device="cuda"), auto_reset=False
            )
        phase_negative = bool(
            bad["phase_violation"].all()
            and bad["failure"].all()
            and not bad["success"].any()
        )
        assert phase_negative
    # Preserve original forbidden-contact detector: explicit reset into obstacle, must fail.
    env.reset(torch.ones(16, device="cuda", dtype=torch.bool))
    env.obs = env.observe()
    if task == "down":
        env.q[:, 0] = env.center_x
        env.q[:, 2] = env.normal_base
    else:
        env.q[:, 0] = env.q[:, 27] - 0.15
        env.q[:, 2] = 0.12
    env.v.zero_()
    wp.to_torch(env.wd.qacc_warmstart).zero_()
    with torch.no_grad():
        _, _, _, bad = env.step(torch.zeros((16, 12), device="cuda"), auto_reset=False)
    collision = float(env.audit[:, 4].max())
    assert (
        collision > 2.0
        and bool(bad["failure"].any())
        and not bool(bad["success"].any())
    ), collision
    return dict(
        schema=STYLE_SCHEMA,
        task=task,
        observation_dim=STYLE_OBS_DIM,
        warm_start=provenance,
        actor_transfer_max_error=parity,
        cpu_gpu_qpos_max_errors_rad_or_m=errors,
        physics_flags=flags,
        foot_ground_contact_detected=contact_detected,
        negative_collision_detected=True,
        negative_premature_crawl_detected=phase_negative,
        arm_target_max_change_rad=arm_delta,
        observed_target_rate_rad_s=rate,
        target_rate_limit_rad_s=env.limits["target_rate_rad_s"],
        scope="engineering preflight only; forced poses are explicit negative-test resets, never successful episodes; no hardware",
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--task", choices=("climb", "down"), required=True)
    p.add_argument("--warm-start", type=Path, required=True)
    p.add_argument("--warm-start-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = validate(a.runtime, a.task, a.warm_start, a.warm_start_sha256)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
