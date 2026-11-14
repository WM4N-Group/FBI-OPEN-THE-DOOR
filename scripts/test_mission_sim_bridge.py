#!/usr/bin/env python3
"""Standalone physical box rollout + READ-ONLY interface regression, not navigation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from fbi_door.sim import box_clearance_env
from fbi_door.sim.box_clearance_env import ClearanceBoxPushEnv
from fbi_door.mission import (
    NavigationGoal,
    Pose2D,
    PoseSample,
    Profile,
    Requirement as R,
    SkillCoordinator,
    Source,
    WorldState,
    assess_navigation_goal,
)
from fbi_door.mission.boundaries import box_v2_boundary_facts
from fbi_door.mission.catalogue import research_contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-controller-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    controller_hash = hashlib.sha256(
        Path(box_clearance_env.__file__).read_bytes()
    ).hexdigest()
    if (
        args.expected_controller_sha256
        and controller_hash != args.expected_controller_sha256
    ):
        raise RuntimeError("Controller source mismatch")
    env = ClearanceBoxPushEnv(randomization=1.0)
    try:
        env.reset(seed=72000)
        for _ in range(1000):
            _, _, term, trunc, info = env.step(np.zeros(6))
            if term or trunc:
                break
        assert info["is_success"], info
        before_q = env.p.data.qpos.copy()
        before_v = env.p.data.qvel.copy()
        before_ctrl = env.p.data.ctrl.copy()
        clock = "standalone:mission-bridge-72000"
        facts = box_v2_boundary_facts(
            info, request_id="box-report", at_s=info["elapsed_s"], clock=clock
        )
        np.testing.assert_array_equal(before_q, env.p.data.qpos)
        np.testing.assert_array_equal(before_v, env.p.data.qvel)
        np.testing.assert_array_equal(before_ctrl, env.p.data.ctrl)
        w, x, y, z = map(float, env.p.data.qpos[3:7])
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pose = Pose2D("map", float(before_q[0]), float(before_q[1]), yaw)
        state = WorldState(
            "standalone-world",
            clock,
            env.t,
            1,
            0,
            "map",
            facts,
            PoseSample(pose, env.t, clock, Source.SIM_TRUTH, 1.0),
        )
        goal = NavigationGoal(
            "interface-probe",
            state.world_id,
            clock,
            Pose2D("map", pose.x_m + 10.0, pose.y_m, pose.yaw_rad),
            0.2,
            None,
            (R("robot/arm_stowed"),),
        )
        complete, reasons = assess_navigation_goal(
            goal, state, Profile.SIM_TEACHER, env.t
        )
        assert not complete and "destination_not_reached" in reasons
        coordinator = SkillCoordinator.for_goal(goal, Profile.SIM_TEACHER)
        for contract in research_contracts():
            coordinator.register(contract)
        assert (
            not coordinator.backends
        )  # no navigation or shared-world executors invented
        assert "robot/base_stopped" not in {f.key for f in facts}
        assert "safety/motion_permitted" not in {f.key for f in facts}
        result = {
            "passed": True,
            "controller_source_sha256": controller_hash,
            "standalone_box_task_completed": True,
            "navigation_task_completed": False,
            "shared_world_execution_backend_implemented": False,
            "learned_manager_implemented": False,
            "physical_state_unchanged_by_adapter": True,
            "bridge_fact_keys": [f.key for f in facts],
            "bridge_source": "sim_truth",
            "navigation_goal_rejection_reasons": list(reasons),
            "box_rollout": info,
            "scope": "One standalone physical regression only, NOT long-horizon navigation/skill-chain execution or hardware acceptance",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
