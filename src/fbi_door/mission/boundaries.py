"""Read-only bridge for recorded simulation info; NEVER an execution/reset adapter."""

from .contracts import ContractError, Fact, Source, finite, identifier


def box_v2_boundary_facts(info, *, request_id, at_s, clock):
    """Export only evidence actually present. No base-stop, ownership or route claims.

    Legacy BoxEnv owns/resets its world; its episode success is not by itself a
    callable shared-world skill or an acknowledgement of released controllers.
    """
    identifier(request_id, "request")
    if not clock.startswith("standalone:"):
        raise ContractError(
            "Standalone episode evidence needs an isolated standalone: clock"
        )
    if (
        "elapsed_s" not in info
        or abs(finite(info["elapsed_s"], "episode time", 0) - at_s) > 1e-8
    ):
        raise ContractError(
            "Do not restamp recorded standalone evidence as fresh mission evidence"
        )
    if info.get("controller") != "clearance_v2":
        raise ContractError("Only audited v2 info is accepted")
    if type(info.get("is_success")) is not bool:
        raise ContractError("Missing outcome")
    required = (
        "box_displacement_m",
        "retract_joint_error_rad",
        "self_contact_peak_N",
        "non_pad_box_contact_peak_N",
    )
    for name in required:
        if name not in info:
            raise ContractError("Missing box evidence: " + name)
        finite(info[name], name, 0)
    clear = info.get("minimum_clearance_m", {})
    for name in ("self", "body_box", "protected_box", "arm_floor"):
        if name not in clear:
            raise ContractError("Missing proxy clearance")
        finite(clear[name], name)
    inspected_clear = (
        clear["self"] >= 0.02
        and clear["body_box"] >= 0.16
        and clear["protected_box"] >= 0.015
        and clear["arm_floor"] >= 0.07
        and info["self_contact_peak_N"] == 0
        and info["non_pad_box_contact_peak_N"] == 0
    )
    achieved = (
        info["is_success"]
        and info["box_displacement_m"] >= 0.5
        and not info.get("failure")
    )
    return (
        Fact(
            "request/" + request_id + "/standalone_box_task_completed",
            bool(achieved),
            at_s,
            clock,
            Source.SIM_TRUTH,
            1.0,
        ),
        Fact(
            "request/" + request_id + "/proxy_clearance_passed",
            bool(inspected_clear),
            at_s,
            clock,
            Source.SIM_TRUTH,
            1.0,
        ),
        Fact(
            "robot/arm_stowed",
            bool(info["is_success"] and info["retract_joint_error_rad"] < 0.12),
            at_s,
            clock,
            Source.SIM_TRUTH,
            1.0,
        ),
    )
