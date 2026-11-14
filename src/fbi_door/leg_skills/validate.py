"""Preflight checks, NOT a task-success evaluation or hardware qualification."""

import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
import torch
import warp as wp
from .contracts import OBS_DIM, LEG_SHA256
from .gpu_env import LegSkillEnv
from .policy import ActorCritic


def validate(runtime, task):
    runtime = Path(runtime)
    env = LegSkillEnv(runtime, task, 16, 123, split="heldout", difficulty=1.0)
    policy = ActorCritic()
    policy.initialize(
        runtime / "reference/models/policies/arm_moving_clock/model_7999.pt"
    )
    policy.cuda()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    onnx = ort.InferenceSession(
        str(
            runtime / "reference/models/policies/arm_moving_clock/exported/policy.onnx"
        ),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    torch.manual_seed(123)
    x = torch.randn((64, OBS_DIM), device="cuda")
    with torch.no_grad():
        expected = np.concatenate(
            [onnx.run(None, {"obs": row[None]})[0] for row in x[:, :300].cpu().numpy()],
            axis=0,
        )
    actor_error = float(
        np.max(np.abs(policy.actor(x).detach().cpu().numpy() - expected))
    )
    if actor_error > 2e-5:
        raise AssertionError(("ONNX actor parity", actor_error))
    model = mujoco.MjModel.from_xml_path(str(runtime / "assets/scene.xml"))
    data = mujoco.MjData(model)
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
        getattr(model, name)[:] = (
            wp.to_torch(getattr(env.wm, name))[0]
            .cpu()
            .numpy()
            .reshape(getattr(model, name).shape)
        )
    mujoco.mj_setConst(model, data)
    data.qpos[:] = env.q[0].cpu().numpy()
    data.qvel[:] = env.v[0].cpu().numpy()
    mujoco.mj_forward(model, data)
    target = env.target[0].cpu().numpy().copy()
    errors = []
    flags = []
    for _ in range(5):
        env.audit.zero_()
        env.overflow.zero_()
        wp.capture_launch(env.graph)
        wp.synchronize()
        for _ in range(10):
            data.ctrl[:12] = np.clip(
                float(env.motor[0])
                * (
                    20 * float(env.kp[0]) * (target[:12] - data.qpos[7:19])
                    - 1.5 * float(env.kd[0]) * data.qvel[6:18]
                ),
                model.actuator_ctrlrange[:12, 0],
                model.actuator_ctrlrange[:12, 1],
            )
            data.ctrl[12:18] = np.clip(
                50 * (target[12:18] - data.qpos[19:25])
                - 4 * data.qvel[18:24]
                + data.qfrc_bias[18:24],
                model.actuator_ctrlrange[12:18, 0],
                model.actuator_ctrlrange[12:18, 1],
            )
            data.ctrl[18] = np.clip(
                800 * (0.006 - data.qpos[25]) - 5 * data.qvel[24], -20, 20
            )
            mujoco.mj_step(model, data)
        errors.append(float(np.max(np.abs(data.qpos - env.q[0].cpu().numpy()))))
        flags.append(int(env.overflow[0]))
    if max(errors) > 0.002 or any(f & 511 for f in flags):
        raise AssertionError(("CPU/GPU integration/capacity", errors, flags))
    # Explicit negative-test RESET injection. No training trajectory/return/success is recorded.
    new = env.q.clone()
    if task == "down":
        new[:, 0] = env.center_x
        new[:, 2] = env.normal_base
    else:
        new[:, 0] = env.q[:, 27] - 0.15
        new[:, 2] = 0.12
    env.q.copy_(new)
    env.v.zero_()
    wp.to_torch(env.wd.qacc_warmstart).zero_()
    _, _, _, info = env.step(torch.zeros((16, 12), device="cuda"), auto_reset=False)
    collision = float(env.audit[:, 4].max())
    if (
        collision <= 2.0
        or not bool(info["failure"].any())
        or bool(info["success"].any())
    ):
        raise AssertionError(("Forbidden-contact negative test", collision))
    return {
        "task": task,
        "worlds": 16,
        "original_checkpoint_sha256": LEG_SHA256,
        "onnx_initial_actor_max_error": actor_error,
        "cpu_gpu_same_action_qpos_max_errors_rad_or_m": errors,
        "physics_flags": flags,
        "negative_test_forbidden_contact_peak_n": collision,
        "negative_test_failure_detected": True,
        "arm_target_max_change_rad": float(
            (env.target[:, 12:18] - env.home[19:25]).abs().max()
        ),
        "purpose": "preflight ONLY, no successful climb/down claim; collision pose is an explicit negative-test reset",
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "hardware_validated": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--task", choices=("climb", "down"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    report = validate(a.runtime, a.task)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
