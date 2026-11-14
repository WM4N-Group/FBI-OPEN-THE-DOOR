"""Navigation-goal metrics and semi-Markov trace records, not a learned manager."""

from .contracts import (
    SCHEMA,
    ContractError,
    assess_navigation_goal,
    finite,
    identifier,
    wire,
)
from .runtime import Outcome, SkillResult


def smdp_transition(
    result, reward_events, gamma_per_second, *, mission_terminal, mission_truncated
):
    """Integrate rewards at actual offsets; do not apply one fixed gamma per skill.

    Mission terminal/truncated flags are supplied by the mission evaluator, NOT
    derived from skill outcome. Snapshot pairs are stored by coordinator ID.
    """
    if (
        not isinstance(result, SkillResult)
        or not result.resources_released
        or not result.duration_valid
    ):
        raise ContractError("Need an acknowledged transition with a valid clock")
    if (
        type(mission_terminal) is not bool
        or type(mission_truncated) is not bool
        or (mission_terminal and mission_truncated)
    ):
        raise ContractError(
            "Explicit, mutually exclusive mission terminal/truncated flags required"
        )
    if not 0 < finite(gamma_per_second, "discount") <= 1:
        raise ContractError("Discount outside (0,1]")
    total = discounted = 0.0
    for offset, reward in reward_events:
        finite(offset, "reward event time", 0)
        finite(reward, "reward event")
        if offset > result.elapsed_s:
            raise ContractError("Reward event outside option duration")
        total += reward
        discounted += (gamma_per_second**offset) * reward
    return {
        "schema": result.schema,
        "request_id": result.request.request_id,
        "action": result.request.to_dict(),
        "elapsed_s": result.elapsed_s,
        "undiscounted_reward_sum": total,
        "option_discounted_reward": discounted,
        "continuation_discount": gamma_per_second**result.elapsed_s,
        "skill_outcome": result.outcome.value,
        "before_observation_revision": result.before_observation_revision,
        "after_observation_revision": result.after_observation_revision,
        "mission_terminal": mission_terminal,
        "mission_truncated": mission_truncated,
    }


def mission_summary(
    goal,
    final_state,
    profile,
    now_s,
    results,
    *,
    mission_started_s,
    active_request_ids,
    measured_path_length_m=None,
    confirmed_safety_violations=None,
):
    """Separate destination, completed handoff, and independently audited safety.

    Correct protective stops can be recovered from; they are not automatically
    collisions. No path length from sparse option endpoints; no invented SPL.
    """
    finite(mission_started_s, "mission start", 0)
    finite(now_s, "summary time", 0)
    if now_s < mission_started_s:
        raise ContractError("Mission clock went backwards")
    if measured_path_length_m is not None:
        finite(measured_path_length_m, "measured path length", 0)
    if confirmed_safety_violations is not None:
        if (
            isinstance(confirmed_safety_violations, bool)
            or not isinstance(confirmed_safety_violations, int)
            or confirmed_safety_violations < 0
        ):
            raise ContractError("Invalid independently audited violation count")
    if not isinstance(active_request_ids, tuple) or len(set(active_request_ids)) != len(
        active_request_ids
    ):
        raise ContractError("Explicit unique active request IDs required")
    for request_id in active_request_ids:
        identifier(request_id, "active request")
    rows = tuple(results)
    if any(
        not isinstance(r, SkillResult)
        or r.request.mission_id != goal.mission_id
        or r.request.world_id != goal.world_id
        for r in rows
    ):
        raise ContractError("Mixed mission/world results")
    if len({r.request.request_id for r in rows}) != len(rows):
        raise ContractError("Duplicate attempts")
    if set(active_request_ids) & {r.request.request_id for r in rows}:
        raise ContractError("Active request reported terminal")
    arrived, reasons = assess_navigation_goal(goal, final_state, profile, now_s)
    incomplete = bool(active_request_ids) or any(
        not r.resources_released or not r.duration_valid for r in rows
    )
    complete = arrived and not incomplete
    reasons = list(reasons)
    if incomplete:
        reasons.append("incomplete_or_invalid_handoff")
    return {
        "schema": SCHEMA,
        "mission_id": goal.mission_id,
        "world_id": goal.world_id,
        "clock": goal.clock,
        "profile": profile.value,
        "budget_exhausted": goal.deadline_s is not None and now_s >= goal.deadline_s,
        "navigation_goal_reached": arrived,
        "mission_complete": complete,
        "safe_mission_success": None
        if confirmed_safety_violations is None
        else complete and confirmed_safety_violations == 0,
        "goal_evidence_reasons": reasons,
        "elapsed_s_all_attempts": now_s - mission_started_s,
        "measured_path_length_m": measured_path_length_m,
        "skill_attempts": len(rows),
        "successful_skill_attempts": sum(r.outcome == Outcome.SUCCEEDED for r in rows),
        "unsuccessful_attempts": sum(r.outcome != Outcome.SUCCEEDED for r in rows),
        "protective_stop_attempts": sum(r.outcome == Outcome.UNSAFE for r in rows),
        "active_request_ids": list(active_request_ids),
        "confirmed_safety_violations": confirmed_safety_violations,
        "spl": None,
        "warning": "No navigator, physical mission execution or safety certification is implied",
        "attempts": [wire(r) for r in rows],
    }
