"""Contract catalogue, NOT an implementation of these skills or a fixed skill chain."""

from .contracts import ParameterBound, Requirement as R, SkillContract


def research_contracts():
    """Example simulation bounds; must be replaced by validated backend capabilities."""
    stopped = (
        R("robot/base_stopped"),
        R("robot/arm_stowed"),
        R("robot/contact_released"),
    )
    return (
        SkillContract(
            "navigation.move_to",
            1,
            ("base",),
            stopped + (R("localization/valid"), R("request/{request}/route_clear")),
            (R("request/{request}/target_reached"), R("robot/base_stopped")),
            120.0,
            requires_target_pose=True,
            invariants=(R("localization/valid"),),
        ),
        SkillContract(
            "navigation.replan",
            1,
            ("planner",),
            (R("localization/valid"),),
            (R("request/{request}/route_ready"),),
            10.0,
            requires_target_pose=True,
            invalidate_on_map_change=False,
        ),
        SkillContract(
            "perception.reobserve",
            1,
            ("perception",),
            (),
            (R("request/{request}/observation_ready"),),
            10.0,
            invalidate_on_map_change=False,
        ),
        SkillContract(
            "manipulation.open_door",
            1,
            ("base", "arm", "gripper"),
            stopped
            + (
                R("object/{entity}/localized"),
                R("object/{entity}/handle_localized"),
                R("capability/door_self_collision_audited"),
            ),
            (R("request/{request}/door_goal_reached"),) + stopped,
            60.0,
            parameters=(ParameterBound("opening_angle_rad", 0.5, 1.4, "rad"),),
            requires_entity=True,
        ),
        SkillContract(
            "manipulation.push_box",
            1,
            ("base", "arm", "gripper"),
            stopped
            + (
                R("object/{entity}/localized"),
                R("object/{entity}/push_side_accessible"),
                R("capability/shared_world_push_backend"),
            ),
            (R("request/{request}/object_goal_reached"),) + stopped,
            100.0,
            parameters=(
                ParameterBound("distance_m", 0.1, 0.6, "m"),
                ParameterBound(
                    "direction_rad", -3.141592653589793, 3.141592653589793, "rad"
                ),
            ),
            requires_entity=True,
        ),
    )
