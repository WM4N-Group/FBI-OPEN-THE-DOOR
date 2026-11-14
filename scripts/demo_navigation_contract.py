#!/usr/bin/env python3
"""Synthetic protocol-only test: no navigation, physics, learning or hardware execution."""

import argparse
import json
from pathlib import Path
from fbi_door.mission import (
    Fact,
    NavigationGoal,
    Pose2D,
    PoseSample,
    Profile,
    Requirement as R,
    SkillContract,
    SkillCoordinator,
    SkillFeedback,
    SkillRequest,
    Source,
    WorldState,
    Outcome,
)
from fbi_door.mission.catalogue import research_contracts
from fbi_door.mission.evaluation import mission_summary, smdp_transition


def snapshot(at, revision, x, request_done=None):
    keys = [
        "safety/motion_permitted",
        "robot/base_stopped",
        "robot/arm_stowed",
        "robot/contact_released",
    ]
    if request_done:
        keys.append("request/" + request_done + "/done")
    return WorldState(
        "synthetic-world",
        "synthetic-clock",
        at,
        revision,
        0,
        "map",
        tuple(
            Fact(k, True, at, "synthetic-clock", Source.ESTIMATED, 1.0) for k in keys
        ),
        PoseSample(
            Pose2D("map", x, 0, 0), at, "synthetic-clock", Source.ESTIMATED, 1.0
        ),
    )


def demo():
    goal = NavigationGoal(
        "demo",
        "synthetic-world",
        "synthetic-clock",
        Pose2D("map", 10, 0, 0),
        0.2,
        0.2,
        (R("robot/base_stopped"), R("robot/arm_stowed"), R("robot/contact_released")),
    )
    coordinator = SkillCoordinator.for_goal(goal, Profile.OBSERVED_ONLY)
    catalogue = research_contracts()
    for contract in catalogue:
        coordinator.register(contract)  # explicitly no executors
    for skill in ("test.interaction_report", "test.navigation_report"):
        coordinator.register(
            SkillContract(
                skill, 1, ("base", "arm"), (), (R("request/{request}/done"),), 10.0
            ),
            "test_double",
        )
    start = snapshot(0, 0, 0)
    first = SkillRequest(
        "interaction1", "demo", goal.world_id, "test.interaction_report", 1, 0, 0
    )
    intent = coordinator.begin(first, start, 0)
    after_interaction = snapshot(1, 1, 0, "interaction1")
    update = coordinator.poll(
        "interaction1",
        after_interaction,
        SkillFeedback("interaction1", goal.world_id, Outcome.SUCCEEDED, True),
        1,
    )
    intermediate = mission_summary(
        goal,
        after_interaction,
        Profile.OBSERVED_ONLY,
        1,
        coordinator.results,
        mission_started_s=0,
        active_request_ids=tuple(coordinator.active),
    )
    assert not intermediate["mission_complete"]
    second = SkillRequest(
        "nav1", "demo", goal.world_id, "test.navigation_report", 1, 1, 0
    )
    coordinator.begin(second, after_interaction, 1)
    # Synthetic fixture only; this is not a navigation executor updating physical pose.
    destination_fixture = snapshot(3, 2, 10, "nav1")
    coordinator.poll(
        "nav1",
        destination_fixture,
        SkillFeedback("nav1", goal.world_id, Outcome.SUCCEEDED, True),
        3,
    )
    final = mission_summary(
        goal,
        destination_fixture,
        Profile.OBSERVED_ONLY,
        3,
        coordinator.results,
        mission_started_s=0,
        active_request_ids=tuple(coordinator.active),
    )
    assert final["mission_complete"] and final["safe_mission_success"] is None
    return {
        "synthetic_protocol_test": True,
        "physical_navigation_executed": False,
        "learned_manager": False,
        "available_real_backends": [],
        "catalogue_contracts_only": [c.skill_id for c in catalogue],
        "first_start_intent": intent,
        "after_interaction": intermediate,
        "after_synthetic_navigation_fixture": final,
        "example_smdp_transition": smdp_transition(
            update.result,
            ((1.0, -1.0),),
            0.99,
            mission_terminal=False,
            mission_truncated=False,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = demo()
    text = json.dumps(value, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
