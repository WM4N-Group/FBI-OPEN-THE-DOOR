"""Pure protocol coordinator: intents and leases, never motor commands or world resets."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from .contracts import (
    SCHEMA,
    ContractError,
    Profile,
    SkillContract,
    SkillRequest,
    WorldState,
    finite,
    identifier,
    natural,
    freeze_tuple,
    requirement_problems,
    evidence_problem,
    wire,
)


class Outcome(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    WORLD_CHANGED = "world_changed"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class Assessment:
    allowed: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class SkillFeedback:
    request_id: str
    world_id: str
    outcome: Outcome
    quiescent: bool
    detail: str = ""

    def __post_init__(self):
        identifier(self.request_id, "feedback request")
        identifier(self.world_id, "feedback world")
        if self.outcome not in (
            Outcome.RUNNING,
            Outcome.SUCCEEDED,
            Outcome.FAILED,
        ) or not isinstance(self.outcome, Outcome):
            raise ContractError("Backend may report only running/succeeded/failed")
        if self.outcome == Outcome.RUNNING and self.quiescent:
            raise ContractError(
                "A running backend cannot acknowledge final control release"
            )
        if type(self.quiescent) is not bool or not isinstance(self.detail, str):
            raise ContractError(
                "Explicit quiescent acknowledgement and detail required"
            )


@dataclass(frozen=True)
class SkillResult:
    request: SkillRequest
    outcome: Outcome
    started_at_s: float
    ended_at_s: float
    elapsed_s: float
    before_observation_revision: int
    after_observation_revision: int
    start_map_revision: int
    end_map_revision: int
    reasons: Tuple[str, ...]
    resources_released: bool
    duration_valid: bool
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA or not isinstance(self.request, SkillRequest):
            raise ContractError("Invalid result schema/request")
        if not isinstance(self.outcome, Outcome) or self.outcome == Outcome.RUNNING:
            raise ContractError("Terminal result needs terminal Outcome")
        for name in ("started_at_s", "ended_at_s", "elapsed_s"):
            finite(getattr(self, name), name, 0)
        if abs(self.ended_at_s - self.started_at_s - self.elapsed_s) > 1e-8:
            raise ContractError("Inconsistent result duration")
        for name in (
            "before_observation_revision",
            "after_observation_revision",
            "start_map_revision",
            "end_map_revision",
        ):
            natural(getattr(self, name), name)
        freeze_tuple(self.reasons, "result reasons", str)
        if (
            type(self.resources_released) is not bool
            or type(self.duration_valid) is not bool
        ):
            raise ContractError("Explicit handoff/duration flags required")


@dataclass(frozen=True)
class Update:
    request_id: str
    state: str  # running, stop_requested, terminal
    stop_required: bool
    reasons: Tuple[str, ...]
    result: Optional[SkillResult] = None


@dataclass
class _Active:
    request: SkillRequest
    contract: SkillContract
    before: WorldState
    latest: WorldState
    started_now_s: float
    latest_now_s: float
    duration_valid: bool = True
    pending: Optional[Outcome] = None
    reasons: Tuple[str, ...] = ()


class SkillCoordinator:
    """Backend registration declares plumbing, not safety certification.

    A real executor must independently consume intents and acknowledge that
    control has stopped/transferred. This package supplies no such executor.
    """

    def __init__(
        self,
        mission_id,
        world_id,
        profile,
        clock,
        max_state_age_s=0.5,
        mission_deadline_s=None,
    ):
        identifier(mission_id, "mission")
        identifier(world_id, "world")
        if not isinstance(profile, Profile):
            raise ContractError("No hardware profile/backend is implemented")
        identifier(clock, "coordinator clock")
        finite(max_state_age_s, "state age", 0)
        if mission_deadline_s is not None:
            finite(mission_deadline_s, "mission deadline", 0)
        self.mission_deadline_s = mission_deadline_s
        self.clock = clock
        self.max_state_age_s = max_state_age_s
        self.mission_id = mission_id
        self.world_id = world_id
        self.profile = profile
        self.contracts = {}
        self.backends = {}
        self.active = {}
        self.seen_requests = set()
        self.results = []
        self.handoff_states = {}
        self.finished = {}
        self.rejected = []

    @classmethod
    def for_goal(cls, goal, profile, max_state_age_s=0.5):
        from .contracts import NavigationGoal

        if not isinstance(goal, NavigationGoal):
            raise ContractError("Typed navigation goal required")
        return cls(
            goal.mission_id,
            goal.world_id,
            profile,
            goal.clock,
            max_state_age_s,
            goal.deadline_s,
        )

    def register(self, contract, backend_kind=None):
        if (
            not isinstance(contract, SkillContract)
            or contract.skill_id in self.contracts
        ):
            raise ContractError("Unique typed skill contract required")
        if backend_kind not in (None, "test_double", "simulation"):
            raise ContractError("Only test/simulation declarations are supported")
        self.contracts[contract.skill_id] = contract
        if backend_kind is not None:
            self.backends[contract.skill_id] = backend_kind

    def assess(self, request, state, now_s):
        if not isinstance(request, SkillRequest) or not isinstance(state, WorldState):
            raise ContractError("Typed request/state required")
        finite(now_s, "decision time", 0)
        reasons = []
        if self.mission_deadline_s is not None and now_s >= self.mission_deadline_s:
            reasons.append("mission_deadline")
        if state.clock != self.clock:
            reasons.append("wrong_clock")
        if not 0 <= now_s - state.at_s <= self.max_state_age_s:
            reasons.append("state_stale_or_future")
        contract = self.contracts.get(request.skill_id)
        if contract is None:
            return Assessment(False, ("unknown_skill",))
        if request.mission_id != self.mission_id:
            reasons.append("wrong_mission")
        if request.world_id != self.world_id or state.world_id != self.world_id:
            reasons.append("wrong_world")
        if request.skill_version != contract.version:
            reasons.append("skill_version_mismatch")
        if request.request_id in self.seen_requests:
            reasons.append("request_already_used")
        if request.parent_request_id is not None:
            parent = self.finished.get(request.parent_request_id)
            if parent is None:
                reasons.append("parent_not_finished")
            elif (
                request.intent in ("retry", "recovery")
                and parent.outcome == Outcome.SUCCEEDED
            ):
                reasons.append("retry_parent_was_successful")
            elif request.intent == "retry" and (
                parent.request.skill_id != request.skill_id
                or parent.request.entity_id != request.entity_id
            ):
                reasons.append("retry_target_or_skill_changed")
        if request.observation_revision != state.observation_revision:
            reasons.append("stale_decision")
        if request.map_revision != state.map_revision:
            reasons.append("map_changed")
        if request.skill_id not in self.backends:
            reasons.append("backend_unavailable")
        if contract.requires_entity and request.entity_id is None:
            reasons.append("entity_required")
        if contract.requires_target_pose and request.target_pose is None:
            reasons.append("target_pose_required")
        if request.target_pose is not None and request.target_pose.frame != state.frame:
            reasons.append("target_frame_mismatch")
        bounds = {b.name: b for b in contract.parameters}
        supplied = dict(request.parameters)
        if set(supplied) != set(bounds):
            reasons.append("parameter_set_mismatch")
        for name, value in supplied.items():
            if (
                name in bounds
                and not bounds[name].minimum <= value <= bounds[name].maximum
            ):
                reasons.append("parameter_out_of_bounds:" + name)
        if not (contract.requires_entity and request.entity_id is None):
            reasons.extend(
                requirement_problems(
                    contract.preconditions + contract.invariants,
                    state,
                    self.profile,
                    request.entity_id,
                    now_s,
                    request.request_id,
                    request.mission_id,
                )
            )
        for run in self.active.values():
            if set(contract.resources) & set(run.contract.resources):
                reasons.append("control_resource_busy")
        # Any motion resource requires fresh permission; pure sensing/planning may recover while motion is blocked.
        from .contracts import Requirement

        if set(contract.resources) & {"base", "arm", "gripper"}:
            reasons.extend(
                requirement_problems(
                    (Requirement("safety/motion_permitted"),),
                    state,
                    self.profile,
                    now_s=now_s,
                )
            )
        return Assessment(not reasons, tuple(reasons))

    def action_mask(self, candidates, state, now_s):
        """Candidate order is explicit; does not choose a skill or reveal oracle costs."""
        if any(not isinstance(r, SkillRequest) for r in candidates) or len(
            {r.request_id for r in candidates}
        ) != len(candidates):
            raise ContractError("Candidate request IDs must be unique")
        return tuple(self.assess(r, state, now_s).allowed for r in candidates)

    def begin(self, request, state, now_s):
        assessment = self.assess(request, state, now_s)
        if not assessment.allowed:
            self.rejected.append((request.request_id, assessment.reasons))
            raise ContractError("Request rejected: " + str(assessment.reasons))
        self.seen_requests.add(request.request_id)
        self.active[request.request_id] = _Active(
            request, self.contracts[request.skill_id], state, state, now_s, now_s
        )
        # Start intent only. No calls into controllers, reset(), simulator or SDK.
        return {
            "schema": request.schema,
            "kind": "start_skill_intent",
            "backend_kind": self.backends[request.skill_id],
            "request": request.to_dict(),
            "resources": list(self.contracts[request.skill_id].resources),
        }

    def cancel(self, request_id, reason="manager_cancelled"):
        if request_id in self.finished:
            return Update(
                request_id,
                "terminal",
                False,
                self.finished[request_id].reasons,
                self.finished[request_id],
            )
        if request_id not in self.active:
            raise ContractError("Unknown active request")
        run = self.active[request_id]
        if run.pending is None:
            run.pending = Outcome.CANCELLED
            run.reasons = (reason,)
        return Update(request_id, "stop_requested", True, run.reasons)

    def poll(self, request_id, state, feedback, now_s):
        if not isinstance(state, WorldState) or not isinstance(feedback, SkillFeedback):
            raise ContractError("Typed state and feedback required")
        if request_id in self.finished:
            result = self.finished[request_id]
            if (
                feedback.request_id != request_id
                or feedback.world_id != result.request.world_id
            ):
                raise ContractError("Feedback does not acknowledge this lease/world")
            return Update(request_id, "terminal", False, result.reasons, result)
        if request_id not in self.active:
            raise ContractError("Unknown active request")
        run = self.active[request_id]
        if (
            feedback.request_id != request_id
            or feedback.world_id != run.before.world_id
        ):
            raise ContractError("Feedback does not acknowledge this lease/world")
        finite(now_s, "poll time", 0)
        latest = run.latest
        invalid_clock = (
            state.clock != run.before.clock
            or state.at_s < latest.at_s
            or state.observation_revision < latest.observation_revision
            or state.map_revision < latest.map_revision
            or now_s < run.latest_now_s
            or (
                state.at_s > latest.at_s
                and state.observation_revision <= latest.observation_revision
            )
        )
        reset = state.world_id != run.before.world_id
        changed = (
            state.map_revision != run.before.map_revision
            and run.contract.invalidate_on_map_change
        )
        from .contracts import Requirement

        global_motion = (
            (Requirement("safety/motion_permitted"),)
            if set(run.contract.resources) & {"base", "arm", "gripper"}
            else ()
        )
        unsafe = requirement_problems(
            global_motion + run.contract.invariants,
            state,
            self.profile,
            run.request.entity_id,
            now_s,
            run.request.request_id,
            run.request.mission_id,
        )
        if not 0 <= now_s - state.at_s <= self.max_state_age_s:
            unsafe += ("state_stale_or_future",)
        if invalid_clock or reset:
            run.duration_valid = False
        budget_exhausted = (
            self.mission_deadline_s is not None and now_s >= self.mission_deadline_s
        )
        if run.pending == Outcome.SUCCEEDED:
            if reset or invalid_clock or changed:
                run.pending = Outcome.WORLD_CHANGED
                run.reasons = ("handoff_world_or_clock_changed",)
            elif unsafe:
                run.pending = Outcome.UNSAFE
                run.reasons = unsafe
            elif (
                budget_exhausted or now_s - run.started_now_s >= run.contract.timeout_s
            ):
                run.pending = Outcome.TIMED_OUT
                run.reasons = (
                    "handoff_mission_deadline"
                    if budget_exhausted
                    else "handoff_skill_deadline",
                )
        if run.pending is None:
            if reset or invalid_clock or changed:
                run.pending = Outcome.WORLD_CHANGED
                run.reasons = ("world_reset_clock_or_map_changed",)
            elif unsafe:
                run.pending = Outcome.UNSAFE
                run.reasons = unsafe
            elif budget_exhausted:
                run.pending = Outcome.TIMED_OUT
                run.reasons = ("mission_deadline",)
            elif now_s - run.started_now_s >= run.contract.timeout_s:
                run.pending = Outcome.TIMED_OUT
                run.reasons = ("skill_deadline",)
            elif feedback.outcome == Outcome.FAILED:
                run.pending = Outcome.FAILED
                run.reasons = (feedback.detail or "backend_failure",)
            elif feedback.outcome == Outcome.SUCCEEDED:
                problems = requirement_problems(
                    run.contract.postconditions,
                    state,
                    self.profile,
                    run.request.entity_id,
                    now_s,
                    run.request.request_id,
                    run.request.mission_id,
                )
                run.pending = Outcome.FAILED if problems else Outcome.SUCCEEDED
                run.reasons = problems
        # Never replace the saved monotonic clock/world with a reset clock.
        if not invalid_clock and not reset:
            run.latest = state
            run.latest_now_s = now_s
        if run.pending is None:
            return Update(request_id, "running", False, ())
        if not feedback.quiescent:
            return Update(request_id, "stop_requested", True, run.reasons)
        # A delayed stop acknowledgement cannot turn stale earlier success into success.
        if run.pending == Outcome.SUCCEEDED:
            problems = requirement_problems(
                run.contract.postconditions,
                state,
                self.profile,
                run.request.entity_id,
                now_s,
                run.request.request_id,
                run.request.mission_id,
            )
            if problems or unsafe or reset or invalid_clock or changed:
                run.pending = Outcome.FAILED
                run.reasons = problems or ("handoff_state_invalid",)
        final = run.latest
        elapsed = run.latest_now_s - run.started_now_s
        result = SkillResult(
            run.request,
            run.pending,
            run.started_now_s,
            run.latest_now_s,
            elapsed,
            run.before.observation_revision,
            final.observation_revision,
            run.before.map_revision,
            final.map_revision,
            run.reasons,
            True,
            run.duration_valid,
        )
        del self.active[request_id]
        self.results.append(result)
        self.finished[request_id] = result
        self.handoff_states[request_id] = (run.before, final)
        return Update(request_id, "terminal", False, result.reasons, result)


def actor_view(state, profile, now_s, max_age_s=0.5, min_confidence=0.9):
    """No privileged critic fields in an observed-only manager view, including its mask."""
    finite(now_s, "decision time", 0)
    finite(max_age_s, "maximum evidence age", 0)
    finite(min_confidence, "confidence", 0)
    if min_confidence > 1:
        raise ContractError("Confidence outside [0,1]")
    # evidence_problem validates profile even if the state has no facts.
    if not isinstance(profile, Profile):
        raise ContractError("Unsupported profile")
    usable = 0 <= now_s - state.at_s <= max_age_s
    facts = [
        wire(f)
        for f in state.facts
        if usable
        and evidence_problem(f, state, profile, max_age_s, min_confidence, now_s)
        is None
    ]
    base = state.base
    visible_base = (
        usable
        and base is not None
        and base.pose.frame == state.frame
        and evidence_problem(base, state, profile, max_age_s, min_confidence, now_s)
        is None
    )
    return {
        "schema": SCHEMA,
        "world_id": state.world_id,
        "clock": state.clock,
        "at_s": state.at_s,
        "map_revision": state.map_revision,
        "observation_revision": state.observation_revision,
        "state_usable": usable,
        "facts": facts,
        "base": wire(base) if visible_base else None,
    }
