"""Independent physics/reference/gate preflight; partial goals != full success."""

import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
import warp as wp
from .curriculum_env import CurriculumEnv
from .curriculum_policy import ResidualActorCritic
from .curriculum_contracts import SCHEMA, OBS_DIM, LIMITS
from .gpu_env import LegSkillEnv


def validate(runtime, task, reference, sha):
    torch.set_num_threads(1)
    cfg = dict(
        runtime=runtime,
        task=task,
        count=16,
        seed=123,
        reference=reference,
        reference_sha=sha,
        stage="full",
        split="heldout",
    )
    env = CurriculumEnv(**cfg)
    before = {k: v.detach().cpu().clone() for k, v in env.teacher.state_dict().items()}
    residual = ResidualActorCritic().cuda()
    with torch.no_grad():
        assert float(residual.actor(env.obs).abs().max()) == 0
    assert env.obs.shape == (16, OBS_DIM) and bool((env.q[:, 2] > 0.25).all())
    # First ascent action/PD target is exactly v1, not just actor weight parity.
    baseline_error = None
    if task == "climb":
        target, action = env.reference_control()
        expected = LegSkillEnv.leg_targets(env, action)
        actual = torch.maximum(torch.minimum(target, env.upper), env.lower)
        baseline_error = float((actual - expected).abs().max())
        assert baseline_error < 1e-6, baseline_error
    # Independent C MuJoCo integration of the same fixed controls; original torque caps.
    m = mujoco.MjModel.from_xml_path(str(Path(runtime) / "assets/scene.xml"))
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
        source = wp.to_torch(getattr(env.wm, name))
        j = 0
        getattr(m, name)[:] = source[j].cpu().numpy().reshape(getattr(m, name).shape)
    mujoco.mj_setConst(m, d)
    d.qpos[:] = env.q[0].cpu().numpy()
    d.qvel[:] = env.v[0].cpu().numpy()
    mujoco.mj_forward(m, d)
    targets = env.target[0].cpu().numpy().copy()
    errors = []
    for _ in range(5):
        env.audit.zero_()
        env.overflow.zero_()
        wp.capture_launch(env.graph)
        wp.synchronize()
        for _ in range(10):
            d.ctrl[:12] = np.clip(
                float(env.motor[0])
                * (
                    20 * float(env.kp[0]) * (targets[:12] - d.qpos[7:19])
                    - 1.5 * float(env.kd[0]) * d.qvel[6:18]
                ),
                m.actuator_ctrlrange[:12, 0],
                m.actuator_ctrlrange[:12, 1],
            )
            d.ctrl[12:18] = np.clip(
                50 * (targets[12:18] - d.qpos[19:25])
                - 4 * d.qvel[18:24]
                + d.qfrc_bias[18:24],
                m.actuator_ctrlrange[12:18, 0],
                m.actuator_ctrlrange[12:18, 1],
            )
            d.ctrl[18] = np.clip(800 * (0.006 - d.qpos[25]) - 5 * d.qvel[24], -20, 20)
            mujoco.mj_step(m, d)
        errors.append(float(np.abs(d.qpos - env.q[0].cpu().numpy()).max()))
        assert int((env.overflow & 511).max()) == 0
    assert max(errors) < 0.002, errors
    # Extreme residual requests must remain bounded, target clipping is logged controller behavior.
    env.reset(torch.ones(16, device="cuda", dtype=torch.bool))
    env.obs = env.observe()
    env.residual_amplitude = 1.0
    with torch.no_grad():
        _, _, _, info = env.step(
            torch.full((16, 12), 100.0, device="cuda"), auto_reset=False
        )
    limit = max(LIMITS[task]["residual_rad"], LIMITS[task]["stationary_residual_rad"])
    maxres = float(env.current_joint_residual.abs().max())
    assert maxres <= limit + 1e-6
    assert float((env.target[:, 12:18] - env.home[19:25]).abs().max()) == 0
    assert not bool(info["partial_goal"])
    env.reset(torch.ones(16, device="cuda", dtype=torch.bool))
    env.obs = env.observe()
    env.residual_amplitude = 0.0
    for _ in range(6):
        with torch.no_grad():
            env.step(torch.zeros((16, 12), device="cuda"), auto_reset=True)
        assert torch.equal(env.command_observation, env.obs[:, :320]), (
            "Teacher history double-append mismatch"
        )
    # Explicit test RESET into forbidden collision, never accepted task progress.
    env.reset(torch.ones(16, device="cuda", dtype=torch.bool))
    env.obs = env.observe()
    if task == "climb":
        env.q[:, 0] = env.q[:, 27] - 0.15
        env.q[:, 2] = 0.12
    else:
        env.q[:, 0] = env.center_x
        env.q[:, 2] = env.normal_base
    env.v.zero_()
    wp.to_torch(env.wd.qacc_warmstart).zero_()
    with torch.no_grad():
        _, _, _, bad = env.step(torch.zeros((16, 12), device="cuda"), auto_reset=False)
    assert (
        float(env.audit[:, 4].max()) > 2
        and bool(bad["failure"].any())
        and not bool(bad["success"].any())
    )
    for k, v in env.teacher.state_dict().items():
        assert torch.equal(v.detach().cpu(), before[k]), k
    assert hashlib.sha256(Path(reference).read_bytes()).hexdigest() == sha
    return dict(
        schema=SCHEMA,
        task=task,
        obs_dim=OBS_DIM,
        zero_initial_residual=True,
        climb_first_ascent_target_matches_v1_max_error=baseline_error,
        cpu_gpu_fixed_action_qpos_max_errors_rad_or_m=errors,
        static_reference_fk_error_max_m=env.reference_fk_error,
        requested_residual_bound_rad=limit,
        observed_max_residual_rad=maxres,
        frozen_teacher_unchanged=True,
        continuing_world_teacher_history_matches_policy_observation=True,
        arm_target_max_change_rad=0.0,
        collision_negative_test_passed=True,
        full_eval_starts_upright_on_floor=True,
        reference_sha256=sha,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        scope="Engineering preflight only, no learned full skill success; negative poses are test resets; no hardware",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--task", choices=("climb", "down"), required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = validate(a.runtime, a.task, a.reference, a.reference_sha256)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
