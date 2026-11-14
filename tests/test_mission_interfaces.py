import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from fbi_door.mission import (
    ContractError,
    Fact,
    NavigationGoal,
    ParameterBound,
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
    actor_view,
    assess_navigation_goal,
    wire,
)
from fbi_door.mission.catalogue import research_contracts
from fbi_door.mission.evaluation import mission_summary, smdp_transition
from fbi_door.mission.interfaces import RouteCandidate, observed_routes
from fbi_door.mission.boundaries import box_v2_boundary_facts


class MissionInterfaceTests(unittest.TestCase):
    def fact(
        self,
        key,
        value=True,
        t=0,
        source=Source.ESTIMATED,
        confidence=1.0,
        clock="simclock",
    ):
        return Fact(key, value, t, clock, source, confidence)

    def state(
        self, t=0, rev=0, maprev=0, x=0, extra=(), world="world", clock="simclock"
    ):
        facts = (
            tuple(
                self.fact(k, t=t, clock=clock)
                for k in (
                    "safety/motion_permitted",
                    "robot/base_stopped",
                    "robot/arm_stowed",
                    "robot/contact_released",
                )
            )
            + extra
        )
        return WorldState(
            world,
            clock,
            t,
            rev,
            maprev,
            "map",
            facts,
            PoseSample(Pose2D("map", x, 0, 0), t, clock, Source.ESTIMATED, 1.0),
        )

    def contract(self, skill="mock.skill", resources=("base", "arm"), timeout=10):
        return SkillContract(
            skill,
            1,
            resources,
            (R("robot/arm_stowed"),),
            (R("request/{request}/done"), R("robot/base_stopped")),
            timeout,
        )

    def request(self, rid="r1", skill="mock.skill", rev=0, maprev=0):
        return SkillRequest(rid, "mission", "world", skill, 1, rev, maprev)

    def coordinator(
        self, contract=None, backend="test_double", profile=Profile.OBSERVED_ONLY
    ):
        c = SkillCoordinator("mission", "world", profile, "simclock")
        c.register(contract or self.contract(), backend)
        return c

    def feedback(self, outcome=Outcome.SUCCEEDED, quiet=True, rid="r1", world="world"):
        return SkillFeedback(rid, world, outcome, quiet)

    def done_state(self, t=1, rid="r1", rev=1):
        return self.state(t, rev, extra=(self.fact("request/" + rid + "/done", t=t),))

    def result(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        return c.poll("r1", self.done_state(), self.feedback(), 1).result

    def goal(self):
        return NavigationGoal(
            "mission",
            "world",
            "simclock",
            Pose2D("map", 10, 0, 0),
            0.2,
            0.2,
            (
                R("robot/base_stopped"),
                R("robot/arm_stowed"),
                R("robot/contact_released"),
            ),
        )

    def test_request_wire_roundtrip_and_version(self):
        r = replace(
            self.request(),
            target_pose=Pose2D("map", 2, 3, 0.1),
            parameters=(("distance_m", 0.5),),
        )
        self.assertEqual(SkillRequest.from_dict(json.loads(json.dumps(r.to_dict()))), r)
        for mutate in (
            lambda d: d.update(schema="v999"),
            lambda d: d.pop("schema"),
            lambda d: d.update(unknown=True),
        ):
            d = r.to_dict()
            mutate(d)
            with self.assertRaises(ContractError):
                SkillRequest.from_dict(d)

    def test_numeric_validation_and_immutable_inputs(self):
        for bad in (True, math.nan, math.inf, -math.inf):
            with self.assertRaises(ContractError):
                Pose2D("map", bad, 0, 0)
            with self.assertRaises(ContractError):
                replace(self.request(), parameters=(("x", bad),))
        with self.assertRaises(ContractError):
            replace(self.request(), parameters=(("x", 1), ("x", 2)))
        with self.assertRaises(ContractError):
            replace(self.state(), facts=list(self.state().facts))
        with self.assertRaises(ContractError):
            replace(
                self.state(),
                facts=self.state().facts + (self.fact("safety/motion_permitted"),),
            )
        with self.assertRaises(ContractError):
            replace(self.state(), map_revision=True)

    def test_backend_absence_is_not_implementation(self):
        c = self.coordinator(backend=None)
        self.assertIn(
            "backend_unavailable", c.assess(self.request(), self.state(), 0).reasons
        )
        with self.assertRaises(ContractError):
            c.begin(self.request(), self.state(), 0)
        self.assertFalse(c.active)
        c = SkillCoordinator("mission", "world", Profile.SIM_TEACHER, "simclock")
        for contract in research_contracts():
            c.register(contract)
        self.assertTrue(
            all(
                contract.skill_id not in c.backends for contract in research_contracts()
            )
        )

    def test_hardware_is_explicitly_unavailable(self):
        with self.assertRaises(ContractError):
            SkillCoordinator("mission", "world", "hardware", "simclock")
        with self.assertRaises(ContractError):
            self.coordinator(backend="hardware")

    def test_source_mask_cannot_leak_truth_value(self):
        c = self.coordinator(replace(self.contract(), preconditions=(R("hidden"),)))
        states = [
            self.state(extra=(self.fact("hidden", v, source=Source.SIM_TRUTH),))
            for v in (True, False)
        ]
        a, b = [c.assess(self.request(), s, 0) for s in states]
        self.assertEqual(a, b)
        self.assertFalse(a.allowed)
        self.assertEqual(c.action_mask((self.request(),), states[0], 0), (False,))
        self.assertFalse(
            any(
                f["key"] == "hidden"
                for f in actor_view(states[0], Profile.OBSERVED_ONLY, 0)["facts"]
            )
        )
        teacher = self.coordinator(
            replace(self.contract(), preconditions=(R("hidden"),)),
            profile=Profile.SIM_TEACHER,
        )
        self.assertTrue(teacher.assess(self.request(), states[0], 0).allowed)

    def test_clock_age_confidence_and_independent_now(self):
        c = self.coordinator(replace(self.contract(), preconditions=(R("fresh"),)))
        for fact in (
            self.fact("fresh", t=-0.0, clock="other"),
            self.fact("fresh", t=2),
            self.fact("fresh", confidence=0.2),
            self.fact("fresh", t=0),
        ):
            s = self.state(t=1, extra=(fact,))
            self.assertFalse(c.assess(self.request(), s, 1).allowed)
        s = self.state(extra=(self.fact("fresh"),))
        self.assertFalse(c.assess(self.request(), s, 2).allowed)
        self.assertFalse(actor_view(s, Profile.OBSERVED_ONLY, 2)["state_usable"])

    def test_stale_decision_map_entity_frame_and_parameter_guards(self):
        contract = replace(
            self.contract(),
            requires_entity=True,
            requires_target_pose=True,
            parameters=(ParameterBound("distance_m", 0.1, 0.6, "m"),),
        )
        c = self.coordinator(contract)
        r = replace(
            self.request(),
            entity_id="box1",
            target_pose=Pose2D("map", 0, 0, 0),
            parameters=(("distance_m", 0.5),),
        )
        self.assertTrue(c.assess(r, self.state(), 0).allowed)
        for bad in (
            replace(r, observation_revision=1),
            replace(r, map_revision=1),
            replace(r, entity_id=None),
            replace(r, target_pose=None),
            replace(r, target_pose=Pose2D("camera", 0, 0, 0)),
            replace(r, parameters=()),
            replace(r, parameters=(("distance_m", 1),)),
        ):
            self.assertFalse(c.assess(bad, self.state(), 0).allowed)

    def test_global_permission_always_required(self):
        c = self.coordinator(replace(self.contract(), preconditions=()))
        s = replace(self.state(), facts=())
        self.assertFalse(c.assess(self.request(), s, 0).allowed)

    def test_resource_lease_and_duplicate_request(self):
        c = self.coordinator()
        c.register(self.contract("mock.nav", ("base",)), "test_double")
        intent = c.begin(self.request(), self.state(), 0)
        self.assertEqual(intent["kind"], "start_skill_intent")
        self.assertIn(
            "control_resource_busy",
            c.assess(self.request("r2", "mock.nav"), self.state(), 0).reasons,
        )
        c.poll("r1", self.done_state(), self.feedback(), 1)
        self.assertFalse(c.assess(self.request(), self.state(), 0).allowed)
        self.assertTrue(
            c.assess(self.request("r2", "mock.nav", rev=1), self.state(1, 1), 1).allowed
        )

    def test_backend_success_does_not_create_postconditions(self):
        c = self.coordinator()
        s = self.state()
        c.begin(self.request(), s, 0)
        result = c.poll("r1", self.state(1, 1), self.feedback(), 1).result
        self.assertEqual(result.outcome, Outcome.FAILED)
        self.assertIsNone(s.fact("request/r1/done"))

    def test_postconditions_are_request_scoped(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        s = self.state(1, 1, extra=(self.fact("request/other/done", t=1),))
        self.assertEqual(
            c.poll("r1", s, self.feedback(), 1).result.outcome, Outcome.FAILED
        )

    def test_delayed_stop_ack_keeps_resources(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        u = c.poll("r1", self.done_state(), self.feedback(quiet=False), 1)
        self.assertTrue(u.stop_required)
        self.assertIn("r1", c.active)
        self.assertFalse(
            c.assess(self.request("r2", rev=1), self.state(1, 1), 1).allowed
        )
        u = c.poll("r1", self.done_state(1.1, rev=2), self.feedback(), 1.1)
        self.assertEqual(u.result.outcome, Outcome.SUCCEEDED)
        self.assertNotIn("r1", c.active)
        self.assertEqual(c.handoff_states["r1"][0].world_id, "world")

    def test_success_is_rechecked_at_handoff(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        c.poll("r1", self.done_state(), self.feedback(quiet=False), 1)
        r = c.poll("r1", self.state(2, 2), self.feedback(), 2).result
        self.assertEqual(r.outcome, Outcome.FAILED)

    def test_timeout_and_cancel_are_not_mission_termination(self):
        c = self.coordinator(self.contract(timeout=1))
        c.begin(self.request(), self.state(), 0)
        u = c.poll("r1", self.state(1, 1), self.feedback(Outcome.RUNNING, False), 1)
        self.assertTrue(u.stop_required)
        self.assertIn("r1", c.active)
        r = c.poll("r1", self.state(1.1, 2), self.feedback(Outcome.FAILED), 1.1).result
        self.assertEqual(r.outcome, Outcome.TIMED_OUT)
        transition = smdp_transition(
            r, ((1.0, -1.0),), 0.99, mission_terminal=False, mission_truncated=False
        )
        self.assertFalse(transition["mission_terminal"])
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        c.cancel("r1")
        r = c.poll("r1", self.state(1, 1), self.feedback(Outcome.FAILED), 1).result
        self.assertEqual(r.outcome, Outcome.CANCELLED)

    def test_wrong_ack_cannot_release_a_lease(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        for feedback in (self.feedback(rid="r2"), self.feedback(world="other")):
            with self.assertRaises(ContractError):
                c.poll("r1", self.done_state(), feedback, 1)
        self.assertIn("r1", c.active)
        with self.assertRaises(ContractError):
            self.feedback(Outcome.RUNNING, True)

    def test_world_reset_invalidates_duration_and_stops(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        c.poll("r1", self.state(1, 1), self.feedback(Outcome.RUNNING, False), 1)
        reset = self.state(world="newworld")
        u = c.poll("r1", reset, self.feedback(Outcome.RUNNING, False), 1.1)
        self.assertTrue(u.stop_required)
        self.assertIn("r1", c.active)
        r = c.poll("r1", reset, self.feedback(Outcome.FAILED), 1.2).result
        self.assertEqual(r.outcome, Outcome.WORLD_CHANGED)
        self.assertFalse(r.duration_valid)
        with self.assertRaises(ContractError):
            smdp_transition(
                r, (), 0.99, mission_terminal=False, mission_truncated=False
            )

    def test_map_change_and_continuation_invariant(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        r = c.poll("r1", self.state(1, 1, maprev=1), self.feedback(), 1).result
        self.assertEqual(r.outcome, Outcome.WORLD_CHANGED)
        c = self.coordinator(
            replace(self.contract(), invariants=(R("localization/valid"),))
        )
        c.begin(self.request(), self.state(extra=(self.fact("localization/valid"),)), 0)
        r = c.poll("r1", self.done_state(), self.feedback(), 1).result
        self.assertEqual(r.outcome, Outcome.UNSAFE)

    def test_smdp_discount_uses_real_skill_duration(self):
        r = self.result()
        t = smdp_transition(
            r,
            ((0.0, 1.0), (1.0, 2.0)),
            0.9,
            mission_terminal=False,
            mission_truncated=False,
        )
        self.assertAlmostEqual(t["option_discounted_reward"], 2.8)
        self.assertAlmostEqual(t["continuation_discount"], 0.9)
        r2 = replace(r, ended_at_s=2, elapsed_s=2)
        t2 = smdp_transition(
            r2, (), 0.9, mission_terminal=False, mission_truncated=False
        )
        self.assertAlmostEqual(t2["continuation_discount"], 0.81)
        with self.assertRaises(ContractError):
            smdp_transition(
                r, ((2.0, 1.0),), 0.9, mission_terminal=False, mission_truncated=False
            )
        with self.assertRaises(ContractError):
            smdp_transition(r, (), 0.9, mission_terminal=True, mission_truncated=True)

    def test_manipulation_success_is_not_navigation_success(self):
        summary = mission_summary(
            self.goal(),
            self.state(1, 1),
            Profile.OBSERVED_ONLY,
            1,
            (self.result(),),
            mission_started_s=0,
            active_request_ids=(),
            confirmed_safety_violations=0,
        )
        self.assertEqual(summary["successful_skill_attempts"], 1)
        self.assertFalse(summary["mission_complete"])
        self.assertIsNone(summary["measured_path_length_m"])
        self.assertIsNone(summary["spl"])

    def test_navigation_needs_pose_frame_confidence_and_clean_handoff(self):
        g = self.goal()
        s = self.state(1, 1, x=10)
        self.assertTrue(assess_navigation_goal(g, s, Profile.OBSERVED_ONLY, 1)[0])
        for bad in (
            replace(s, base=None),
            replace(s, world_id="other"),
            replace(s, base=replace(s.base, pose=Pose2D("odom", 10, 0, 0))),
            replace(s, base=replace(s.base, confidence=0.2)),
        ):
            self.assertFalse(
                assess_navigation_goal(g, bad, Profile.OBSERVED_ONLY, 1)[0]
            )
        a = mission_summary(
            g,
            s,
            Profile.OBSERVED_ONLY,
            1,
            (),
            mission_started_s=0,
            active_request_ids=("active",),
        )
        self.assertFalse(a["mission_complete"])
        a = mission_summary(
            g,
            s,
            Profile.OBSERVED_ONLY,
            1,
            (),
            mission_started_s=0,
            active_request_ids=(),
        )
        self.assertTrue(a["mission_complete"])
        self.assertIsNone(a["safe_mission_success"])
        a = mission_summary(
            g,
            s,
            Profile.OBSERVED_ONLY,
            1,
            (),
            mission_started_s=0,
            active_request_ids=(),
            confirmed_safety_violations=1,
        )
        self.assertFalse(a["safe_mission_success"])

    def test_navigation_heading_wraparound(self):
        g = replace(
            self.goal(),
            destination=Pose2D("map", 10, 0, math.pi),
            heading_tolerance_rad=0.1,
        )
        s = self.state(1, 1, x=10)
        s = replace(s, base=replace(s.base, pose=Pose2D("map", 10, 0, -math.pi)))
        self.assertTrue(assess_navigation_goal(g, s, Profile.OBSERVED_ONLY, 1)[0])

    def test_route_alternatives_are_not_fixed_skill_order(self):
        direct = RouteCandidate(
            "door_route",
            "world",
            "simclock",
            0,
            0,
            (Pose2D("map", 10, 0, 0),),
            ("door1",),
            20,
            Source.ESTIMATED,
            0.9,
        )
        detour = replace(
            direct,
            candidate_id="detour",
            intervention_entities=(),
            estimated_travel_s=40,
        )
        oracle = replace(
            direct, candidate_id="oracle", estimate_source=Source.SIM_TRUTH
        )
        self.assertEqual(
            observed_routes(
                (direct, detour, oracle), Profile.OBSERVED_ONLY, self.state(), 0
            ),
            (direct, detour),
        )
        self.assertEqual(
            observed_routes((direct,), Profile.OBSERVED_ONLY, self.state(maprev=1), 0),
            (),
        )
        self.assertEqual(
            observed_routes((direct,), Profile.OBSERVED_ONLY, self.state(t=2), 2), ()
        )
        self.assertEqual(
            observed_routes(
                (replace(direct, world_id="other"),),
                Profile.OBSERVED_ONLY,
                self.state(),
                0,
            ),
            (),
        )

    def test_real_box_boundary_bridge_does_not_invent_navigation_or_release(self):
        root = Path(__file__).resolve().parents[1]
        info = json.loads(
            (root / "provenance/box_clearance_20261114/v2_regression8.json").read_text()
        )["results"][0]
        facts = box_v2_boundary_facts(
            info, request_id="push1", at_s=info["elapsed_s"], clock="standalone:71000"
        )
        self.assertTrue(all(f.source == Source.SIM_TRUTH for f in facts))
        self.assertTrue(
            next(
                f.value
                for f in facts
                if f.key.endswith("standalone_box_task_completed")
            )
        )
        keys = {f.key for f in facts}
        self.assertNotIn("request/push1/object_goal_reached", keys)
        with self.assertRaises(ContractError):
            box_v2_boundary_facts(
                info, request_id="push1", at_s=info["elapsed_s"], clock="simclock"
            )
        with self.assertRaises(ContractError):
            box_v2_boundary_facts(
                info, request_id="push1", at_s=1, clock="standalone:71000"
            )
        self.assertNotIn("robot/base_stopped", keys)
        self.assertNotIn("safety/motion_permitted", keys)
        self.assertFalse(any("route" in key for key in keys))
        with self.assertRaises(ContractError):
            box_v2_boundary_facts(
                dict(info, controller="legacy"),
                request_id="r",
                at_s=1,
                clock="simclock",
            )

    def test_serialization_contains_no_hardware_side_effects(self):
        self.assertEqual(
            json.loads(json.dumps(wire(self.result())))["outcome"], "succeeded"
        )
        import subprocess, sys, os

        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=str(root / "src"))
        check = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; import fbi_door.mission; import fbi_door.mission.catalogue; import fbi_door.mission.interfaces; "
                "import fbi_door.mission.evaluation; import fbi_door.mission.boundaries; "
                "assert not any(n.split('.')[0] in ('torch','mujoco','gymnasium','rclpy','can') for n in sys.modules)",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_duplicate_feedback_is_idempotent(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        first = c.poll("r1", self.done_state(), self.feedback(), 1)
        again = c.poll("r1", self.done_state(), self.feedback(), 1)
        self.assertEqual(first, again)
        self.assertEqual(len(c.results), 1)
        self.assertEqual(c.cancel("r1"), again)
        with self.assertRaises(ContractError):
            c.poll("unknown", self.state(), self.feedback(), 1)
        with self.assertRaises(ContractError):
            c.action_mask((self.request(), self.request()), self.state(), 0)

    def test_stop_latency_counts_towards_option_deadline(self):
        c = self.coordinator(self.contract(timeout=1.5))
        c.begin(self.request(), self.state(), 0)
        c.poll("r1", self.done_state(), self.feedback(quiet=False), 1)
        result = c.poll("r1", self.done_state(2, rev=2), self.feedback(), 2).result
        self.assertEqual(result.outcome, Outcome.TIMED_OUT)
        self.assertEqual(result.elapsed_s, 2)

    def test_failed_attempt_can_be_recovered_before_navigation_goal(self):
        c = self.coordinator()
        c.begin(self.request(), self.state(), 0)
        failed = c.poll("r1", self.state(1, 1), self.feedback(Outcome.FAILED), 1).result
        retry = replace(
            self.request("r2", rev=1), parent_request_id="r1", intent="retry"
        )
        self.assertTrue(c.assess(retry, self.state(1, 1), 1).allowed)
        c.begin(retry, self.state(1, 1), 1)
        success = c.poll(
            "r2", self.done_state(2, "r2", 2), self.feedback(rid="r2"), 2
        ).result
        self.assertEqual(success.outcome, Outcome.SUCCEEDED)
        bad = replace(self.request("r3", rev=2), parent_request_id="r2", intent="retry")
        self.assertFalse(c.assess(bad, self.state(2, 2), 2).allowed)
        at_goal = self.state(3, 3, x=10)
        summary = mission_summary(
            self.goal(),
            at_goal,
            Profile.OBSERVED_ONLY,
            3,
            (failed, success),
            mission_started_s=0,
            active_request_ids=(),
            confirmed_safety_violations=0,
        )
        self.assertTrue(summary["safe_mission_success"])
        self.assertEqual(summary["unsuccessful_attempts"], 1)
        self.assertEqual(summary["elapsed_s_all_attempts"], 3)

    def test_planner_may_update_map_without_claiming_world_motion(self):
        contract = replace(self.contract(), invalidate_on_map_change=False)
        c = self.coordinator(contract)
        c.begin(self.request(), self.state(), 0)
        state = replace(self.done_state(), map_revision=1)
        result = c.poll("r1", state, self.feedback(), 1).result
        self.assertEqual(result.outcome, Outcome.SUCCEEDED)
        self.assertEqual(result.end_map_revision, 1)

    def test_entity_and_request_placeholders_are_validated(self):
        with self.assertRaises(ContractError):
            R("request/{unrecognized}/done")
        with self.assertRaises(ContractError):
            replace(self.contract(), postconditions=(R("object/{entity}/done"),))
        with self.assertRaises(ContractError):
            replace(self.request(), intent="retry")
        with self.assertRaises(ContractError):
            replace(self.request(), parent_request_id="r1")

    def test_pure_reobservation_is_allowed_while_motion_is_blocked(self):
        c = self.coordinator()
        c.register(self.contract("mock.observe", ("perception",)), "test_double")
        c.begin(self.request(), self.state(), 0)
        blocked = self.state(1, 1)
        blocked = replace(
            blocked,
            facts=tuple(
                replace(f, value=False) if f.key == "safety/motion_permitted" else f
                for f in blocked.facts
            ),
        )
        self.assertTrue(
            c.poll(
                "r1", blocked, self.feedback(Outcome.RUNNING, False), 1
            ).stop_required
        )
        self.assertIn("r1", c.active)
        request = self.request("observe", "mock.observe", rev=1)
        self.assertTrue(c.assess(request, blocked, 1).allowed)
        c.begin(request, blocked, 1)
        self.assertIn("observe", c.active)
        self.assertFalse(
            c.assess(self.request("another-motion", rev=1), blocked, 1).allowed
        )

    def test_mission_budget_stops_option_and_blocks_new_dispatch(self):
        goal = replace(self.goal(), deadline_s=1)
        c = SkillCoordinator.for_goal(goal, Profile.OBSERVED_ONLY)
        c.register(self.contract(), "test_double")
        c.begin(self.request(), self.state(), 0)
        update = c.poll(
            "r1", self.state(1, 1), self.feedback(Outcome.RUNNING, False), 1
        )
        self.assertTrue(update.stop_required)
        self.assertIn("mission_deadline", update.reasons)
        result = c.poll(
            "r1", self.state(1.1, 2), self.feedback(Outcome.FAILED), 1.1
        ).result
        self.assertEqual(result.outcome, Outcome.TIMED_OUT)
        self.assertFalse(
            c.assess(self.request("r2", rev=2), self.state(1.1, 2), 1.1).allowed
        )
        arrived = self.state(1.1, 2, x=10)
        self.assertFalse(
            assess_navigation_goal(goal, arrived, Profile.OBSERVED_ONLY, 1.1)[0]
        )

    def test_evidence_cannot_be_newer_than_its_snapshot(self):
        coordinator = self.coordinator(
            replace(self.contract(), preconditions=(R("fresh"),))
        )
        state = self.state(1, 1, extra=(self.fact("fresh", t=1.1),))
        request = self.request(rev=1)
        self.assertFalse(coordinator.assess(request, state, 1.2).allowed)

    def test_world_change_during_success_handoff_is_not_generic_success(self):
        coordinator = self.coordinator()
        coordinator.begin(self.request(), self.state(), 0)
        coordinator.poll("r1", self.done_state(), self.feedback(quiet=False), 1)
        changed = replace(self.done_state(1.1, rev=2), map_revision=1)
        result = coordinator.poll("r1", changed, self.feedback(), 1.1).result
        self.assertEqual(result.outcome, Outcome.WORLD_CHANGED)

    def test_route_permission_is_specific_to_each_navigation_candidate(self):
        contract = research_contracts()[0]
        coordinator = self.coordinator(contract, backend="simulation")
        state = self.state(
            extra=(
                self.fact("localization/valid"),
                self.fact("request/clear/route_clear"),
                self.fact("request/blocked/route_clear", False),
            )
        )
        clear = SkillRequest(
            "clear",
            "mission",
            "world",
            contract.skill_id,
            1,
            0,
            0,
            target_pose=Pose2D("map", 10, 0, 0),
            route_candidate_id="detour",
        )
        blocked = replace(
            clear, request_id="blocked", route_candidate_id="short-door-route"
        )
        self.assertEqual(
            coordinator.action_mask((clear, blocked), state, 0), (True, False)
        )
