"""Extension points. No planner, manager network or motion executor is supplied."""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple
from .contracts import (
    ContractError,
    NavigationGoal,
    Pose2D,
    SkillRequest,
    Source,
    WorldState,
    finite,
    freeze_tuple,
    identifier,
    natural,
    evidence_problem,
)
from .runtime import SkillFeedback


@dataclass(frozen=True)
class RouteCandidate:
    candidate_id: str
    world_id: str
    clock: str
    at_s: float
    map_revision: int
    waypoints: Tuple[Pose2D, ...]
    intervention_entities: Tuple[str, ...]
    estimated_travel_s: Optional[float]
    estimate_source: Source
    confidence: float

    def __post_init__(self):
        identifier(self.candidate_id, "route candidate")
        identifier(self.world_id, "route world")
        identifier(self.clock, "route clock")
        finite(self.at_s, "route timestamp", 0)
        natural(self.map_revision, "map revision")
        freeze_tuple(self.waypoints, "waypoints", Pose2D)
        freeze_tuple(self.intervention_entities, "intervention entities", str)
        if not self.waypoints:
            raise ContractError("Need route waypoints")
        if len({p.frame for p in self.waypoints}) != 1:
            raise ContractError("Route frames differ")
        if len(set(self.intervention_entities)) != len(self.intervention_entities):
            raise ContractError("Duplicate route intervention")
        for entity in self.intervention_entities:
            identifier(entity, "route entity")
        if self.estimated_travel_s is not None:
            finite(self.estimated_travel_s, "estimated travel time", 0)
        if not isinstance(self.estimate_source, Source):
            raise ContractError("Explicit estimate source required")
        if not 0 <= finite(self.confidence, "route confidence") <= 1:
            raise ContractError("Invalid confidence")

    @property
    def source(self):
        return self.estimate_source


class WorldReader(Protocol):
    def snapshot(self) -> WorldState:
        """Read the persistent mission world; no per-skill reset() interface."""
        ...


class NavigationPlanner(Protocol):
    def propose(
        self, goal: NavigationGoal, observed_state: dict
    ) -> Tuple[RouteCandidate, ...]:
        """May propose clear detours and routes requiring intervention, not a fixed chain."""
        ...


class ManagerPolicy(Protocol):
    def choose(
        self,
        goal: NavigationGoal,
        observed_state: dict,
        routes: Tuple[RouteCandidate, ...],
        candidates: Tuple[SkillRequest, ...],
        action_mask: Tuple[bool, ...],
    ) -> Optional[SkillRequest]:
        """Rule/search/BC/RL interface; revalidate choices. None means no dispatch, NOT a brake."""
        ...


class SimulationSkillExecutor(Protocol):
    def start(self, request: SkillRequest, world: WorldReader) -> None:
        """Bind to a shared world; do not reset physics or the frozen leg history."""
        ...

    def poll(self, request_id: str) -> SkillFeedback: ...

    def request_stop(self, request_id: str, reason: str) -> None:
        """Stop confirmation is asynchronous and must acknowledge the same request/world."""
        ...


def observed_routes(
    candidates, profile, state, now_s, max_age_s=0.5, min_confidence=0.9
):
    """Filter actor route estimates by provenance, world/revision, age and confidence."""
    from .contracts import Profile

    if not isinstance(profile, Profile):
        raise ContractError("Unsupported profile")
    finite(now_s, "decision time", 0)
    finite(max_age_s, "route maximum age", 0)
    if not 0 <= finite(min_confidence, "minimum route confidence") <= 1:
        raise ContractError("Invalid confidence")
    if not 0 <= now_s - state.at_s <= max_age_s:
        return ()
    return tuple(
        c
        for c in candidates
        if c.world_id == state.world_id
        and c.map_revision == state.map_revision
        and c.waypoints[0].frame == state.frame
        and evidence_problem(c, state, profile, max_age_s, min_confidence, now_s)
        is None
    )
