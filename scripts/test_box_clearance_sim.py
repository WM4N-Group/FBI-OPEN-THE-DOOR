#!/usr/bin/env python3
"""Asset-dependent v2 collision/withdrawal regression. Does not connect to hardware."""

import json
import numpy as np
import mujoco
from gymnasium.utils.env_checker import check_env
from fbi_door.sim.box_clearance_env import ClearanceBoxPushEnv
from fbi_door.box_clearance import PHASES
from fbi_door.paths import RUNTIME_ROOT as ROOT
from fbi_door.storage import atomic_json


def main():
    env = ClearanceBoxPushEnv(randomization=1)
    try:
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=72000)
        again, _ = env.reset(seed=72000)
        np.testing.assert_allclose(obs, again, atol=1e-7)
        p, guard = env.p, env.audit
        assert p.model.nu == 19 and p.model.neq == 1
        assert (
            p.model.jnt_type[p.model.joint("box_free").id] == mujoco.mjtJoint.mjJNT_FREE
        )
        # Actual v1 retract configuration. Writes only the guard's scratch FK data.
        bad = np.array(
            [
                -0.024103700560915653,
                1.215926691075454,
                -0.119743252244225,
                0.0007670033567497285,
                -0.0552110808480714,
                0.4951421896824487,
            ]
        )
        assert not guard.configuration_ok(bad)
        distances, _ = guard.distances(guard.work)
        assert distances["self"] < 0, distances
        try:
            guard.plan(env.home_arm, bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Must reject a collision-invalid endpoint")
        env.reset(seed=72000)
        done = False
        while not done:
            action = np.zeros(6)
            if env.locked_arm is not None:
                action[3:] = 1  # deliberately try moving the arm after it is locked
            _, _, term, trunc, info = env.step(action)
            if info["phase"] in ("touch", "push", "stop", "retreat", "withdraw_settle"):
                np.testing.assert_array_equal(env.arm_command, env.locked_arm)
            done = term or trunc
        assert info["is_success"], info
        assert info["phase_history"] == list(PHASES)
        assert info["minimum_clearance_m"]["self"] >= 0.02
        assert info["minimum_clearance_m"]["body_box"] >= 0.16
        assert info["non_arm_box_impulse_Ns"] == 0
        assert info["self_contact_peak_N"] == 0
        assert info["non_pad_box_contact_peak_N"] == 0
        assert info["push_lock_max_target_delta_rad"] == 0
        assert info["max_arm_target_speed_rad_s"] <= 0.3 + 1e-9
        assert info["max_base_vx_command_accel_m_s2"] <= 0.06 + 1e-9
        withdrawal = info["withdrawal_before_retract"]
        assert withdrawal["base_retreat_m"] >= 0.2
        assert withdrawal["pad_gap_m"] >= 0.08
        assert withdrawal["force_N"] < 0.15 and withdrawal["quiet_s"] >= 0.3
        assert info["minimum_retract_pad_box_gap_m"] >= 0.08
        assert info["box_displacement_m"] >= 0.5
        assert info["retract_joint_error_rad"] < 0.12
        atomic_json(
            ROOT / "reports/clearance_sim_regression.json",
            {
                "passed": True,
                "checks": [
                    "Gym API/determinism",
                    "free unactuated box",
                    "v1 collision rejected",
                    "no margin relaxation",
                    "latched arm",
                    "self/body/camera geometry clearance",
                    "unload and withdraw before retract",
                    "zero forbidden contacts",
                    "bounded target speeds",
                    "full physical push and retract",
                ],
                "rollout": info,
            },
        )
        print(json.dumps(info, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
