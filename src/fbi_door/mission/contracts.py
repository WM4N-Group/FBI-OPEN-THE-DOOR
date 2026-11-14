"""Versioned navigation-first skill contracts; standard library, no robot imports."""

import math
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Optional, Tuple

SCHEMA = "navigation-skills/v1"


class ContractError(ValueError):
    pass


def finite(value, name, minimum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ContractError(name + " must be finite numeric")
    if minimum is not None and value < minimum:
        raise ContractError(name + " is below its minimum")
    return value


def identifier(value, name="id"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", value):
        raise ContractError("Invalid " + name)
    return value


def natural(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(name + " must be a nonnegative integer")


def freeze_tuple(value, name, item_type):
    if not isinstance(value, tuple) or any(not isinstance(v, item_type) for v in value):
        raise ContractError(name + " must be an immutable typed tuple")


class Source(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    SIM_TRUTH = "sim_truth"


class Profile(str, Enum):
    SIM_TEACHER = "sim_teacher"
    OBSERVED_ONLY = "observed_only"
    # No hardware profile, motion transport, or hardware enable flag.


@dataclass(frozen=True)
class Pose2D:
    frame: str
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self):
        identifier(self.frame, "frame")
        for name in ("x_m", "y_m", "yaw_rad"):
            finite(getattr(self, name), name)


@dataclass(frozen=True)
class Fact:
    key: str
    value: bool
    at_s: float
    clock: str
    source: Source
    confidence: float

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key or len(self.key) > 240:
            raise ContractError("Invalid fact key")
        if type(self.value) is not bool or not isinstance(self.source, Source):
            raise ContractError("Fact needs bool value and explicit Source")
        finite(self.at_s, "fact timestamp", 0)
        identifier(self.clock, "clock")
        if not 0 <= finite(self.confidence, "confidence") <= 1:
            raise ContractError("Confidence outside [0,1]")


@dataclass(frozen=True)
class PoseSample:
    pose: Pose2D
    at_s: float
    clock: str
    source: Source
    confidence: float

    def __post_init__(self):
        if not isinstance(self.pose, Pose2D):
            raise ContractError("PoseSample needs Pose2D")
        Fact("pose", True, self.at_s, self.clock, self.source, self.confidence)


@dataclass(frozen=True)
class WorldState:
    world_id: str
    clock: str
    at_s: float
    observation_revision: int
    map_revision: int
    frame: str
    facts: Tuple[Fact, ...]
    base: Optional[PoseSample] = None

    def __post_init__(self):
        for name in ("world_id", "clock", "frame"):
            identifier(getattr(self, name), name)
        finite(self.at_s, "state timestamp", 0)
        natural(self.observation_revision, "observation revision")
        natural(self.map_revision, "map revision")
        freeze_tuple(self.facts, "facts", Fact)
        if len({f.key for f in self.facts}) != len(self.facts):
            raise ContractError("Duplicate fact keys")
        if self.base is not None and not isinstance(self.base, PoseSample):
            raise ContractError("Invalid base sample")

    def fact(self, key):
        return next((f for f in self.facts if f.key == key), None)


@dataclass(frozen=True)
class Requirement:
    key: str
    expected: bool = True
    max_age_s: float = 0.5
    min_confidence: float = 0.9

    def __post_init__(self):
        if (
            not isinstance(self.key, str)
            or not self.key
            or type(self.expected) is not bool
        ):
            raise ContractError("Invalid requirement")
        self.resolve("ENTITY", "REQUEST", "MISSION")
        finite(self.max_age_s, "max age", 0)
        if not 0 <= finite(self.min_confidence, "minimum confidence") <= 1:
            raise ContractError("Invalid confidence threshold")

    def resolve(self, entity_id=None, request_id=None, mission_id=None):
        key = self.key
        for token, value in (
            ("entity", entity_id),
            ("request", request_id),
            ("mission", mission_id),
        ):
            if "{" + token + "}" in key:
                if value is None:
                    raise ContractError("Missing predicate context: " + token)
                key = key.replace("{" + token + "}", value)
        if "{" in key or "}" in key:
            raise ContractError("Unknown predicate placeholder")
        return key


@dataclass(frozen=True)
class ParameterBound:
    name: str
    minimum: float
    maximum: float
    unit: str

    def __post_init__(self):
        identifier(self.name, "parameter name")
        if finite(self.minimum, "parameter minimum") > finite(
            self.maximum, "parameter maximum"
        ):
            raise ContractError("Reversed parameter range")
        if not isinstance(self.unit, str) or not self.unit:
            raise ContractError("Explicit parameter unit required")


@dataclass(frozen=True)
class SkillContract:
    skill_id: str
    version: int
    resources: Tuple[str, ...]
    preconditions: Tuple[Requirement, ...]
    postconditions: Tuple[Requirement, ...]
    timeout_s: float
    parameters: Tuple[ParameterBound, ...] = ()
    requires_entity: bool = False
    requires_target_pose: bool = False
    invalidate_on_map_change: bool = True
    invariants: Tuple[Requirement, ...] = ()

    def __post_init__(self):
        identifier(self.skill_id, "skill")
        natural(self.version, "skill version")
        if self.version == 0:
            raise ContractError("Skill version starts at1")
        freeze_tuple(self.resources, "resources", str)
        if not self.resources or len(set(self.resources)) != len(self.resources):
            raise ContractError("Need unique control resources")
        if any(
            r not in ("base", "arm", "gripper", "perception", "planner")
            for r in self.resources
        ):
            raise ContractError("Unknown resource")
        freeze_tuple(self.preconditions, "preconditions", Requirement)
        freeze_tuple(self.postconditions, "postconditions", Requirement)
        freeze_tuple(self.invariants, "invariants", Requirement)
        freeze_tuple(self.parameters, "parameters", ParameterBound)
        if not self.postconditions:
            raise ContractError("Explicit terminal postconditions required")
        if (
            any(
                "{entity}" in r.key
                for r in self.preconditions + self.postconditions + self.invariants
            )
            and not self.requires_entity
        ):
            raise ContractError("Entity predicates require an entity-scoped contract")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ContractError("Duplicate parameter bounds")
        if finite(self.timeout_s, "timeout", 0) == 0:
            raise ContractError("Timeout must be positive")
        for name in (
            "requires_entity",
            "requires_target_pose",
            "invalidate_on_map_change",
        ):
            if type(getattr(self, name)) is not bool:
                raise ContractError("Contract flags must be bool")


@dataclass(frozen=True)
class SkillRequest:
    request_id: str
    mission_id: str
    world_id: str
    skill_id: str
    skill_version: int
    observation_revision: int
    map_revision: int
    entity_id: Optional[str] = None
    target_pose: Optional[Pose2D] = None
    parameters: Tuple[Tuple[str, float], ...] = ()
    parent_request_id: Optional[str] = None
    intent: str = "normal"
    route_candidate_id: Optional[str] = None
    schema: str = SCHEMA

    def __post_init__(self):
        if self.schema != SCHEMA:
            raise ContractError("Unsupported request schema")
        for name in ("request_id", "mission_id", "world_id", "skill_id"):
            identifier(getattr(self, name), name)
        for name in ("skill_version", "observation_revision", "map_revision"):
            natural(getattr(self, name), name)
        if self.intent not in ("normal", "retry", "recovery"):
            raise ContractError("Unknown request intent")
        for name in ("parent_request_id", "route_candidate_id"):
            if getattr(self, name) is not None:
                identifier(getattr(self, name), name)
        if self.parent_request_id == self.request_id or (
            self.intent != "normal" and self.parent_request_id is None
        ):
            raise ContractError("Retry/recovery needs a distinct parent request")
        if self.entity_id is not None:
            identifier(self.entity_id, "entity")
        if self.target_pose is not None and not isinstance(self.target_pose, Pose2D):
            raise ContractError("Target pose must be explicit Pose2D")
        freeze_tuple(self.parameters, "request parameters", tuple)
        names = []
        for pair in self.parameters:
            if len(pair) != 2:
                raise ContractError("Parameter must be name/value pair")
            identifier(pair[0], "parameter")
            finite(pair[1], "parameter value")
            names.append(pair[0])
        if len(set(names)) != len(names):
            raise ContractError("Duplicate request parameter")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ContractError("Request must be an object")
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ContractError("Unknown request fields: " + str(sorted(unknown)))
        if value.get("schema") != SCHEMA:
            raise ContractError("Explicit supported schema required")
        data = dict(value)
        if data.get("target_pose") is not None:
            try:
                data["target_pose"] = Pose2D(**data["target_pose"])
            except (TypeError, ValueError) as exc:
                raise ContractError("Invalid target pose") from exc
        raw = data.get("parameters", {})
        if not isinstance(raw, dict):
            raise ContractError("Wire parameters must be a named object")
        data["parameters"] = tuple(sorted(raw.items()))
        try:
            return cls(**data)
        except TypeError as exc:
            raise ContractError("Missing/invalid request fields") from exc

    def to_dict(self):
        value = wire(self)
        value["parameters"] = dict(self.parameters)
        return value


@dataclass(frozen=True)
class NavigationGoal:
    mission_id: str
    world_id: str
    clock: str
    destination: Pose2D
    position_tolerance_m: float
    heading_tolerance_rad: Optional[float]
    completion_requirements: Tuple[Requirement, ...]
    max_pose_age_s: float = 0.5
    min_pose_confidence: float = 0.9
    deadline_s: Optional[float] = None

    def __post_init__(self):
        identifier(self.mission_id, "mission")
        identifier(self.world_id, "world")
        identifier(self.clock, "clock")
        if not isinstance(self.destination, Pose2D):
            raise ContractError("Destination must be Pose2D")
        if finite(self.position_tolerance_m, "goal tolerance", 0) == 0:
            raise ContractError("Goal tolerance must be positive")
        if self.heading_tolerance_rad is not None:
            finite(self.heading_tolerance_rad, "heading tolerance", 0)
        freeze_tuple(
            self.completion_requirements, "completion requirements", Requirement
        )
        if not self.completion_requirements:
            raise ContractError("Explicit mission completion requirements required")
        for requirement in self.completion_requirements:
            requirement.resolve(mission_id=self.mission_id)
        Requirement("pose", True, self.max_pose_age_s, self.min_pose_confidence)
        if self.deadline_s is not None:
            finite(self.deadline_s, "mission deadline", 0)


def wire(value):
    """JSON primitives only; no pickle, simulation state writes or control commands."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: wire(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [wire(v) for v in value]
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    return value


def evidence_problem(sample, state, profile, max_age, minimum_confidence, now_s=None):
    if not isinstance(profile, Profile):
        raise ContractError(
            "Explicit supported profile required; hardware is unavailable"
        )
    # Check source BEFORE value: action masks must not leak teacher-only truth values.
    if profile == Profile.OBSERVED_ONLY and sample.source == Source.SIM_TRUTH:
        return "teacher_source_not_allowed"
    if sample.clock != state.clock:
        return "clock_mismatch"
    if sample.at_s > state.at_s:
        return "evidence_newer_than_snapshot"
    age = (state.at_s if now_s is None else now_s) - sample.at_s
    if age < 0:
        return "future_evidence"
    if age > max_age:
        return "stale_evidence"
    if sample.confidence < minimum_confidence:
        return "low_confidence"
    return None


def requirement_problems(
    requirements,
    state,
    profile,
    entity_id=None,
    now_s=None,
    request_id=None,
    mission_id=None,
):
    result = []
    for r in requirements:
        key = r.resolve(entity_id, request_id, mission_id)
        fact = state.fact(key)
        reason = (
            "missing_evidence"
            if fact is None
            else evidence_problem(
                fact, state, profile, r.max_age_s, r.min_confidence, now_s
            )
        )
        if reason is None and fact.value != r.expected:
            reason = "predicate_not_satisfied"
        if reason:
            result.append(key + ":" + reason)
    return tuple(result)


def assess_navigation_goal(goal, state, profile, now_s):
    if not isinstance(profile, Profile):
        raise ContractError("Unsupported profile")
    finite(now_s, "decision time", 0)
    reasons = list(
        requirement_problems(
            goal.completion_requirements,
            state,
            profile,
            now_s=now_s,
            mission_id=goal.mission_id,
        )
    )
    if goal.deadline_s is not None and now_s >= goal.deadline_s:
        reasons.append("mission_deadline_exceeded")
    if state.world_id != goal.world_id or state.clock != goal.clock:
        reasons.append("mission_world_or_clock_mismatch")
    if not 0 <= now_s - state.at_s <= goal.max_pose_age_s:
        reasons.append("state_stale_or_future")
    pose = state.base
    if pose is None:
        reasons.append("missing_base_pose")
    else:
        problem = evidence_problem(
            pose, state, profile, goal.max_pose_age_s, goal.min_pose_confidence, now_s
        )
        if problem:
            reasons.append("base_pose:" + problem)
        elif (
            pose.pose.frame != state.frame or pose.pose.frame != goal.destination.frame
        ):
            reasons.append("goal_frame_mismatch")
        else:
            distance = math.hypot(
                pose.pose.x_m - goal.destination.x_m,
                pose.pose.y_m - goal.destination.y_m,
            )
            angle = pose.pose.yaw_rad - goal.destination.yaw_rad
            if distance > goal.position_tolerance_m:
                reasons.append("destination_not_reached")
            if (
                goal.heading_tolerance_rad is not None
                and abs(math.atan2(math.sin(angle), math.cos(angle)))
                > goal.heading_tolerance_rad
            ):
                reasons.append("goal_heading_not_reached")
    return not reasons, tuple(reasons)
