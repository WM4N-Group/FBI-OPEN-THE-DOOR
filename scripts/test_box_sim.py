#!/usr/bin/env python3
"""Legacy v1 task-only regression, NOT self-collision acceptance.

Use test_box_clearance_sim.py for the current controller. Missing assets fail.
"""

import json
import hashlib
import numpy as np
import mujoco
from gymnasium.utils.env_checker import check_env
from fbi_door.sim.box_env import BoxPushEnv
from fbi_door.paths import RUNTIME_ROOT as ROOT
from fbi_door.box_task import LEG_CHECKPOINT_SHA256
from fbi_door.storage import atomic_json
from fbi_door.sim.core import _original, ort


def main():
    print("WARNING: legacy v1 task-only check; self-collision is NOT accepted here")
    env = BoxPushEnv(randomization=0)
    try:
        check_env(env, skip_render_check=True)
        p = env.p
        assert p.model.nu == 19, "12 leg + 6 arm + gripper; box has no actuator"
        assert (
            p.model.jnt_type[p.model.joint("box_free").id] == mujoco.mjtJoint.mjJNT_FREE
        )
        # Only the retained gripper coupling; no task weld or other box constraints.
        assert p.model.neq == 1
        assert p.model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
        for i in range(p.model.nu):
            assert p.model.actuator_trnid[i, 0] != p.model.joint("box_free").id
        policy = ROOT / "reference/models/policies/arm_moving_clock"
        assert (
            hashlib.sha256((policy / "model_7999.pt").read_bytes()).hexdigest()
            == LEG_CHECKPOINT_SHA256
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = _original(
            str(policy / "exported/policy.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        assert sess.get_inputs()[0].shape[-1] == 300
        assert sess.get_outputs()[0].shape[-1] == 12
        a, _ = env.reset(seed=72000)
        b, _ = env.reset(seed=72000)
        np.testing.assert_allclose(a, b, atol=1e-7)
        obs, _ = env.reset(seed=72000)
        done = False
        push_steps = 0
        while not done:
            action = np.zeros(6)
            if env.locked_arm is not None:
                action[3:] = 1  # deliberately try to move the locked arm via residuals
                push_steps += 1
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["phase"] == "push":
                np.testing.assert_array_equal(env.arm_command, env.locked_arm)
        assert info["is_success"], info
        assert push_steps > 10
        assert info["box_displacement_m"] >= 0.5
        assert info["arm_box_contact_s"] >= 0.4
        assert info["non_arm_box_impulse_Ns"] < 0.5
        assert info["push_lock_max_target_delta_rad"] == 0
        assert info["phase_history"] == [
            "approach",
            "reach",
            "push",
            "stop",
            "retract",
            "success",
        ]
        real_contact_s = p.arm_contact_s
        p.arm_contact_s = 0
        assert not env._valid_push(), "No-contact motion must fail"
        p.arm_contact_s = real_contact_s
        p.non_arm_impulse = 1.0
        assert not env._valid_push(), "Base/leg ramming must fail"
        p.non_arm_impulse = 0.0
        atomic_json(
            ROOT / "reports/box_sim_regression.json",
            {
                "passed": True,
                "checks": [
                    "gym API",
                    "deterministic reset",
                    "free box",
                    "no box actuator or weld",
                    "frozen actor300->12",
                    "masked arm residual during hold",
                    "physical push",
                    "full retract",
                    "no-contact and body-ram negatives",
                ],
                "rollout": info,
            },
        )
        print(json.dumps(info, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
