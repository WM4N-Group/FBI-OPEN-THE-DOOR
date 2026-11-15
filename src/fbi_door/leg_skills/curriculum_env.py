"""Protected leg references + bounded residual PPO with explicit reset curricula.

Climb retains the exact v1 actor/history/command baseline during ascent; v3 learns
joint corrections and a separate stationary controller, not a complete learned
manager. Down uses an offline static-load reference before a hand-designed low crawl + compressed pretrained gait.
"""

import json, math
from pathlib import Path
import torch
import warp as wp
from .gpu_env import LegSkillEnv, rotate_inverse
from .style_env import SmallStepEnv
from .curriculum_contracts import SCHEMA, OBS_DIM, LIMITS, validate_stage
from .curriculum_policy import frozen_skill
from .posture_reference import table
from .contracts import MJ_TO_POLICY, POLICY_TO_MJ


class CurriculumEnv(SmallStepEnv):
    def __init__(
        self,
        runtime,
        task,
        count,
        seed,
        reference,
        reference_sha,
        stage="full",
        split="train",
        difficulty=1.0,
    ):
        validate_stage(task, stage)
        self._curr_ready = False
        self.training_stage = stage
        self._current_ref = None
        self.residual_amplitude = 0.0
        # Preparatory skill gates use an easier, explicitly labeled geometry distribution.
        # Final full eval retains full randomized clearance/heights, never partial reset success.
        difficulty = (
            0.0 if stage == "crouch" else 0.25 if stage == "crawl" else difficulty
        )
        self.curriculum_difficulty = difficulty
        super().__init__(runtime, task, count, seed, split=split, difficulty=difficulty)
        self.teacher = frozen_skill(reference, reference_sha, task).to(self.device)
        self.reference_sha = reference_sha
        self.reference_path = Path(reference)
        self._style_ready = False  # v3 deliberately does not patch v1 history/commands.
        self.speed = self.tensor([d["command_speed"] for d in self.domains])
        t = table(self.m)
        self.ref_heights = self.tensor(t["heights"])
        self.ref_poses = self.tensor(t["poses"])
        self.ref_torques = self.tensor(t["torques"])
        self.reference_fk_error = t["fk_error_max_m"]
        self.nominal_target = torch.zeros((self.n, 12), device=self.device)
        self.pending_target = torch.zeros_like(self.nominal_target)
        self.emitted_action = torch.zeros_like(self.nominal_target)
        self.current_joint_residual = torch.zeros_like(self.nominal_target)
        self.previous_joint_residual = torch.zeros_like(self.nominal_target)
        self.command_observation = torch.zeros((self.n, 320), device=self.device)
        self.initial_height = torch.full_like(self.age, self.normal_base)
        self.reference_height = torch.full_like(self.age, self.normal_base)
        self.crawl_reference_height = self.low_target.clamp(0.155, 0.235)
        self.phase_id = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.phase_elapsed = torch.zeros_like(self.age)
        self.stage_hold = torch.zeros_like(self.age)
        self.style_hold = torch.zeros_like(self.age)
        self.task_hold = torch.zeros_like(self.age)
        self.contact_now = torch.zeros_like(self.was_contact)
        self.crossing_witness = torch.zeros_like(self.under_seen)
        self.crouch_verified = torch.zeros_like(self.under_seen)
        self.task_awarded = torch.zeros_like(self.under_seen)
        self.best_support_count = torch.zeros_like(self.age)
        self.stage_return = torch.zeros_like(self.age)
        self.episode_swing = torch.zeros_like(self.age)
        self.episode_foreaft = torch.zeros_like(self.age)
        self.episode_lateral = torch.zeros_like(self.age)
        self.episode_speed = torch.zeros_like(self.age)
        self.episode_impact = torch.zeros_like(self.age)
        self.episode_arms = torch.zeros_like(self.age)
        self.physical_self_peak = torch.zeros_like(self.age)
        self.max_crawl_swing = torch.zeros_like(self.age)
        self.crouch_shift = torch.zeros_like(self.age)
        self.torso_rel_height = torch.zeros_like(self.age)
        self._curr_ready = True
        self.reset(torch.ones(self.n, device=self.device, dtype=torch.bool))
        self.obs = self.observe()
        assert self.obs.shape == (self.n, OBS_DIM), self.obs.shape

    def pose_reference(self, height):
        pos = (
            (height - self.ref_heights[0]) / (self.ref_heights[1] - self.ref_heights[0])
        ).clamp(0, len(self.ref_heights) - 1.001)
        i = pos.long()
        a = (pos - i.float())[:, None]
        q = self.ref_poses[i] * (1 - a) + self.ref_poses[i + 1] * a
        tau = self.ref_torques[i] * (1 - a) + self.ref_torques[i + 1] * a
        return q + tau / (20 * self.kp * self.motor)[:, None]

    def reset(self, mask):
        # v1 reset is used unchanged for full-task starts. Partial goals are reset-only.
        LegSkillEnv.reset(self, mask)
        if not self._curr_ready:
            return
        for v in (
            self.phase_id,
            self.phase_elapsed,
            self.stage_hold,
            self.style_hold,
            self.task_hold,
            self.best_support_count,
            self.stage_return,
            self.episode_swing,
            self.episode_foreaft,
            self.episode_lateral,
            self.episode_speed,
            self.episode_impact,
            self.episode_arms,
            self.physical_self_peak,
            self.max_crawl_swing,
            self.crouch_shift,
        ):
            v.masked_fill_(mask, 0)
        self.crouch_verified.masked_fill_(mask, False)
        self.crossing_witness.masked_fill_(mask, False)
        self.task_awarded.masked_fill_(mask, False)
        self.current_joint_residual.masked_fill_(mask[:, None], 0.0)
        self.previous_joint_residual.masked_fill_(mask[:, None], 0.0)
        self.last_stride.masked_fill_(mask[:, None], 0.0)
        self.airtime.masked_fill_(mask[:, None], 0.0)
        self.was_contact.masked_fill_(mask[:, None], True)
        if self.task == "climb" and self.training_stage == "settle":
            # This is an on-box TRAINING start, never a full climb success.
            self.q[mask, 0] = self.q[mask, 27] + (self.rand(self.n) - 0.5)[mask] * 0.08
            self.q[mask, 1] = self.q[mask, 28] + (self.rand(self.n) - 0.5)[mask] * 0.05
            self.q[mask, 2] = self.height[mask] + self.normal_base
            self.q[mask, 3:7] = self.q[mask, 30:34]
            self.phase_id[mask] = 1
            self.phase_elapsed[mask] = 1.0
        if self.task == "down" and self.training_stage == "crawl":
            # Explicit low-stance reset goal; full eval ALWAYS starts upright.
            h = self.crawl_reference_height
            i = (
                (
                    (h - self.ref_heights[0])
                    / (self.ref_heights[1] - self.ref_heights[0])
                )
                .long()
                .clamp(0, len(self.ref_heights) - 1)
            )
            self.q[mask, 2] = h[mask]
            self.q[mask, 7:19] = self.ref_poses[i[mask]]
            self.phase_id[mask] = 1
            self.crouch_verified[mask] = True
            self.phase_elapsed[mask] = 0.0
        self.v.masked_fill_(mask[:, None], 0.0)
        wp.to_torch(self.wd.qacc_warmstart).masked_fill_(mask[:, None], 0.0)
        from mujoco_warp import fwd_kinematics

        fwd_kinematics(self.wm, self.wd)
        feet = self.xpos[:, self.foot_geoms].clone()
        self.previous_feet.copy_(
            torch.where(mask[:, None, None], feet, self.previous_feet)
        )
        self.liftoff.copy_(
            torch.where(mask[:, None, None], feet[:, :, :2], self.liftoff)
        )
        self.start_xy.copy_(torch.where(mask[:, None], self.q[:, :2], self.start_xy))
        self.initial_height[mask] = self.q[mask, 2]
        self.reference_height[mask] = (
            self.q[mask, 2] - self.height[mask]
            if self.task == "climb"
            else self.q[mask, 2]
        )
        self.nominal_target.copy_(
            torch.where(mask[:, None], self.home[7:19], self.nominal_target)
        )
        self.pending_target.copy_(
            torch.where(mask[:, None], self.home[7:19], self.pending_target)
        )
        for v in (self.foot_current, self.foot_peak, self.foot_impulse):
            v.masked_fill_(mask[:, None], 0.0)

    def leg_targets(self, action):
        if not self._curr_ready:
            return LegSkillEnv.leg_targets(self, action)
        return self.pending_target

    def executed_history_action(self, action):
        if not self._curr_ready:
            return LegSkillEnv.executed_history_action(self, action)
        return self.emitted_action.detach().clone()

    def observe(self):
        base = LegSkillEnv.observe(self)
        if not self._curr_ready:
            return self._augment(base) if getattr(self, "_style_ready", False) else base
        return self.augment(base)

    def augment(self, base):
        # First 320 features, including v1 clock/speed/dwell, are left untouched for teacher.
        self.command_observation = base[:, :320].clone()
        local = self.foot_local()
        phase = torch.nn.functional.one_hot(self.phase_id, 3).float()
        stage_i = {"settle": 0, "crouch": 0, "crawl": 1, "full": 2}[self.training_stage]
        stage = torch.zeros((self.n, 3), device=self.device)
        stage[:, stage_i] = 1
        extra = torch.cat(
            (
                phase,
                stage,
                self.nominal_target - self.home[7:19],
                self.current_joint_residual,
                self.q[:, 7:19] - self.home[7:19],
                local[:, :, 0] - self.nominal_feet[:, 0],
                local[:, :, 1] - self.nominal_feet[:, 1],
                self.contact_now.float(),
                self.last_stride,
                torch.stack(
                    (
                        self.stage_hold,
                        self.style_hold,
                        self.phase_elapsed / 45.0,
                        self.crawl_reference_height,
                        self.reference_height,
                        self.crouch_verified.float(),
                    ),
                    -1,
                ),
            ),
            -1,
        )
        result = torch.cat((base[:, :320], extra), -1)
        if result.shape[-1] != OBS_DIM:
            raise AssertionError(result.shape)
        return torch.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0).clamp(
            -100, 100
        )

    def balanced_pose(self, height, desired_v, gait=None):
        # Model-control reference, not the PPO action. Keep all terms explicit in provenance.
        body_vel = rotate_inverse(self.q[:, 3:7], self.v[:, :3])
        w, x, y, z = self.q[:, 3], self.q[:, 4], self.q[:, 5], self.q[:, 6]
        pitch = torch.asin((2 * (w * y - z * x)).clamp(-0.8, 0.8))
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        actual_height = self.q[:, 2] - (self.height if self.task == "climb" else 0.0)
        virtual_h = (height + 0.5 * (height - actual_height)).clamp(0.13, 0.31)
        result = self.pose_reference(virtual_h)
        hleg = (
            virtual_h[:, None]
            + (0.6 * pitch + 0.08 * self.v[:, 4])[:, None] * self.nominal_feet[:, 0]
            - (0.6 * roll + 0.08 * self.v[:, 3])[:, None] * self.nominal_feet[:, 1]
        ).clamp(0.13, 0.31)
        shift = (
            (0.12 * (body_vel[:, 0] - desired_v))
            .clamp(-0.025, 0.025)[:, None]
            .expand_as(hleg)
        )
        lift = torch.zeros_like(hleg)
        if gait is not None:
            # Conservative nominal four-beat crawl:7cm fore-aft stroke,2cm lift,
            # 80% stance duty,1.6s period. This is explicitly HAND-DESIGNED, not learned.
            phase = (
                self.phase_elapsed[:, None] / 1.6 + self.tensor([0.0, 0.5, 0.75, 0.25])
            ) % 1.0
            swing = (phase - 0.8).clamp(min=0) / 0.2
            dx = torch.where(
                phase < 0.8, 0.035 - 0.07 * phase / 0.8, -0.035 + 0.07 * swing
            )
            lift = torch.where(
                phase < 0.8, torch.zeros_like(phase), 0.020 * torch.sin(math.pi * swing)
            )
            enable = (
                gait.float()
                * (self.phase_elapsed / 1.0).clamp(0, 1)
                * (desired_v / 0.055).clamp(0, 1)
            )
            shift = shift + dx * enable[:, None]
            lift = lift * enable[:, None]

        def sagittal(h, delta):
            zleg = (h - 0.022).clamp(0.10, 0.30)
            distance = torch.sqrt(zleg.square() + delta.square())
            angle = torch.acos((distance / 0.426).clamp(0.15, 0.95))
            return torch.atan2(-delta, zleg) + angle, -2 * angle

        h0, k0 = sagittal(virtual_h[:, None], torch.full_like(hleg, -0.002))
        h1, k1 = sagittal(hleg - lift, -0.002 + shift)
        result[:, [1, 4, 7, 10]] += h1 - h0
        result[:, [2, 5, 8, 11]] += k1 - k0
        result[:, [0, 3, 6, 9]] += (0.08 * body_vel[:, 1]).clamp(-0.015, 0.015)[:, None]
        return result

    def reference_control(self):
        with torch.no_grad():
            a = self.teacher(self.command_observation).clamp(-6, 6)
        old_target = (
            self.home[7:19] + 0.5 * (0.8 * a + 0.2 * self.last)[:, POLICY_TO_MJ]
        )
        if self.task == "climb":
            # Identical pretrained baseline until four feet actually reach the top.
            standing = self.balanced_pose(
                torch.full_like(self.age, self.normal_base), torch.zeros_like(self.age)
            )
            stationary = self.phase_id == 1
            blend = (self.phase_elapsed / 1.0).clamp(0, 1)[:, None]
            gathered = old_target * (1 - blend) + standing * blend
            return torch.where(stationary[:, None], gathered, old_target), a
        low = self.crawl_reference_height
        target_height = torch.where(
            (self.phase_id == 2) & (self.training_stage != "crawl"),
            torch.full_like(low, 0.30),
            low,
        )
        # Gradual physical crouch/rise: no base pose or velocity writes in rollout.
        step = 0.035 * self.dt
        self.reference_height += (target_height - self.reference_height).clamp(
            -step, step
        )
        desired_v = torch.where(
            self.phase_id == 1, torch.full_like(low, 0.055), torch.zeros_like(low)
        )
        if self.training_stage == "crawl":
            desired_v *= ((self.start_xy[:, 0] + 0.25 - self.q[:, 0]) / 0.10).clamp(
                0, 1
            )
        static = self.balanced_pose(
            self.reference_height, desired_v, gait=self.phase_id == 1
        )
        high = self.pose_reference(torch.full_like(low, 0.30))
        # A retargeted PRETRAINED gait, not a claim the residual learned it from scratch.
        gait_weight = 0.10 * (self.phase_elapsed / 1.0).clamp(0, 1)
        if self.training_stage == "crawl":
            remaining = (self.start_xy[:, 0] + 0.25 - self.q[:, 0]).clamp(min=0)
            gait_weight *= (remaining / 0.10).clamp(0, 1)
        crawl = static + gait_weight[:, None] * (old_target - high)
        # Nonzero phases/reference are supplied explicitly to residual actor.
        return torch.where((self.phase_id == 1)[:, None], crawl, static), a

    def step(self, action, auto_reset=True):
        cfg = LIMITS[self.task]
        xy = self.q[:, :2].clone()
        z = self.q[:, 2].clone()
        old_phase = self.phase_id.clone()
        previous_dwell = self.task_hold.clone()
        old_refheight = self.reference_height.clone()
        self.nominal_target, teacher_action = self.reference_control()
        amplitude = torch.where(
            self.phase_id == 1 if self.task == "climb" else self.phase_id != 1,
            torch.full_like(self.age, cfg["stationary_residual_rad"]),
            torch.full_like(self.age, cfg["residual_rad"]),
        )
        self.previous_joint_residual = self.current_joint_residual.clone()
        requested = action.tanh() * amplitude[:, None] * self.residual_amplitude
        self.current_joint_residual = (
            0.3 * requested + 0.7 * self.previous_joint_residual
        )
        proposed = self.nominal_target + self.current_joint_residual
        self.pending_target = torch.maximum(
            torch.minimum(proposed, self.upper), self.lower
        )
        # v1 actions/history are preserved in climb ascent; stance/crawl use executed joint equivalents.
        joint_action = ((self.pending_target - self.home[7:19]) / 0.5)[:, MJ_TO_POLICY]
        self.emitted_action = torch.where(
            ((self.phase_id == 0) & (self.task == "climb"))[:, None],
            teacher_action,
            joint_action,
        )
        self.foot_peak.zero_()
        self.foot_impulse.zero_()
        _, _, _, base = LegSkillEnv.step(self, teacher_action, auto_reset=False)
        feet = self.xpos[:, self.foot_geoms].clone()
        local = self.foot_local()
        contact = self.foot_current > 0.5
        takeoff = self.was_contact & ~contact
        self.liftoff = torch.where(
            takeoff[:, :, None], self.previous_feet[:, :, :2], self.liftoff
        )
        landed = ~self.was_contact & contact
        stride = (feet[:, :, :2] - self.liftoff).norm(dim=-1)
        self.last_stride = torch.where(landed, stride, self.last_stride)
        swing = (stride * (~contact | landed)).max(-1).values
        foreaft = (local[:, :, 0] - self.nominal_feet[:, 0]).abs().max(-1).values
        lateral = (local[:, :, 1] - self.nominal_feet[:, 1]).abs().max(-1).values
        speed = self.v[:, :2].norm(dim=-1)
        vertical = self.v[:, 2].abs()
        angular = self.v[:, 3:6].norm(dim=-1)
        up = 1 - 2 * (self.q[:, 4] ** 2 + self.q[:, 5] ** 2)
        arm = (self.q[:, 19:25] - self.home[19:25]).abs().max(-1).values
        self.episode_swing = torch.maximum(self.episode_swing, swing)
        self.episode_foreaft = torch.maximum(self.episode_foreaft, foreaft)
        self.episode_lateral = torch.maximum(self.episode_lateral, lateral)
        self.episode_speed = torch.maximum(self.episode_speed, speed)
        self.episode_impact = torch.maximum(
            self.episode_impact, self.foot_peak.max(-1).values
        )
        self.episode_arms = torch.maximum(self.episode_arms, arm)
        self.physical_self_peak = torch.maximum(
            self.physical_self_peak, self.audit[:, 5]
        )
        style = (
            (self.episode_swing <= cfg["swing_m"])
            & (self.episode_foreaft <= cfg["foreaft_m"])
            & (self.episode_lateral <= cfg["lateral_m"])
            & (self.episode_speed <= cfg["peak_speed_m_s"])
        )
        # Do NOT terminate for style during learning: physical failures still terminate,
        # and full skill qualification always separately demands style_clean.
        compact = (foreaft < (0.095 if self.task == "climb" else 0.12)) & (
            lateral < 0.085
        )
        phase_failure = torch.zeros_like(self.crouch_verified)
        if self.task == "climb":
            boxq = self.q[:, 30:34]
            pos = rotate_inverse(
                boxq[:, None].expand(-1, 4, -1).reshape(-1, 4),
                (feet - self.q[:, None, 27:30]).reshape(-1, 3),
            ).reshape(self.n, 4, 3)
            top = (
                (pos[:, :, 0].abs() < self.length[:, None] / 2 - 0.025)
                & (pos[:, :, 1].abs() < self.width[:, None] / 2 - 0.025)
                & (pos[:, :, 2] > self.height[:, None] / 2 + 0.005)
                & (pos[:, :, 2] < self.height[:, None] / 2 + 0.05)
            )
            support = (self.audit[:, :4] > 0.5) & top & contact
            support_count = support.sum(-1).float()
            reached = support.all(-1)
            self.phase_id = torch.where(
                (self.phase_id == 0) & reached,
                torch.ones_like(self.phase_id),
                self.phase_id,
            )
            boxup = 1 - 2 * (boxq[:, 1] ** 2 + boxq[:, 2] ** 2)
            boxslow = self.v[:, 26:29].norm(dim=-1) < 0.03
            stable = (
                reached
                & (speed < 0.045)
                & (vertical < 0.035)
                & (angular < 0.20)
                & (up > 0.98)
                & (boxup > math.cos(0.1))
                & boxslow
                & (arm < 0.12)
            )
            goal = self.q[:, 27:29]
            progress = (xy - goal).norm(dim=-1) - (self.q[:, :2] - goal).norm(dim=-1)
            support_progress = (support_count - self.best_support_count).clamp(min=0)
            self.best_support_count = torch.maximum(
                self.best_support_count, support_count
            )
            reward = 10 * progress + support_progress * 0.5 + 2 * (self.q[:, 2] - z)
            reward -= torch.where(
                self.phase_id == 1,
                2 * speed.square()
                + 0.3 * angular.square()
                + 0.4 * (self.q[:, 7:19] - self.ref_poses[28]).square().mean(-1),
                torch.zeros_like(speed),
            )
            desired_height = self.height + self.normal_base
            if self.training_stage == "settle":
                phase_goal = stable & compact
            else:
                phase_goal = stable
            hold = 2.0
        else:
            minimum, maximum = self.robot_bounds(rail_frame=True)
            low = self.crawl_reference_height
            crouched = (
                (self.q[:, 2] < low + 0.015)
                & (maximum[:, 2] < self.clearance - 0.012)
                & (speed < 0.045)
                & (angular < 0.30)
                & (contact.sum(-1) >= 3)
                & (arm < 0.12)
            )
            self.stage_hold = torch.where(
                (self.phase_id == 0) & crouched,
                self.stage_hold + self.dt,
                torch.where(
                    self.phase_id == 0, torch.zeros_like(self.age), self.stage_hold
                ),
            )
            start = (self.phase_id == 0) & (self.stage_hold >= 0.30)
            self.crouch_verified |= start
            if self.training_stage != "crouch":
                self.phase_id = torch.where(
                    start, torch.ones_like(self.phase_id), self.phase_id
                )
            # Required physical ordering remains hard; no early advance cheat.
            self.crouch_shift = torch.maximum(
                self.crouch_shift,
                torch.where(
                    old_phase == 0,
                    (self.q[:, :2] - self.start_xy).norm(dim=-1),
                    torch.zeros_like(self.age),
                ),
            )
            phase_failure = (old_phase == 0) & (self.crouch_shift > 0.18)
            rel = self.q[:, :2] - torch.stack((self.center_x, self.center_y), -1)
            localx = rel[:, 0] * torch.cos(self.yaw) + rel[:, 1] * torch.sin(self.yaw)
            inside = (
                (minimum[:, 1] > -self.width / 2 + 0.015)
                & (maximum[:, 1] < self.width / 2 - 0.015)
                & (maximum[:, 2] < self.clearance - 0.005)
            )
            self.crossing_witness |= (
                (localx.abs() < self.length / 2) & inside & self.crouch_verified
            )
            clear = (
                (minimum[:, 0] > self.length / 2 + 0.12)
                & self.crossing_witness
                & self.crouch_verified
            )
            if self.training_stage == "crawl":
                clear = self.q[:, 0] > self.start_xy[:, 0] + 0.25
            self.phase_id = torch.where(
                (self.phase_id == 1) & clear,
                torch.full_like(self.phase_id, 2),
                self.phase_id,
            )
            recovered = (self.q[:, 2] - self.normal_base).abs() < 0.025
            stable = (
                (speed < 0.05)
                & (angular < 0.20)
                & (up > 0.98)
                & (arm < 0.12)
                & (contact.sum(-1) >= 3)
            )
            if self.training_stage == "crouch":
                phase_goal = crouched & compact
            elif self.training_stage == "crawl":
                phase_goal = clear & (self.q[:, 2] < low + 0.02) & stable & compact
            else:
                phase_goal = clear & recovered & stable & compact
            hold = 1.0
            goal = (
                self.start_xy[:, 0] + 0.25
                if self.training_stage == "crawl"
                else self.center_x + 0.8
            )
            progress = (goal - xy[:, 0]).abs() - (goal - self.q[:, 0]).abs()
            reward = 12 * torch.where(
                old_phase == 1, progress, torch.zeros_like(progress)
            )
            reward += 15 * (
                (z - old_refheight).abs() - (self.q[:, 2] - old_refheight).abs()
            )
            desired_height = torch.where(
                (self.phase_id == 2) & (self.training_stage != "crawl"),
                torch.full_like(low, 0.30),
                low,
            )
            reward -= 1.5 * (self.q[:, 2] - desired_height).square() + torch.where(
                old_phase == 0, 3 * speed.square(), torch.zeros_like(speed)
            )
            support_count = contact.sum(-1).float()
            self.max_crawl_swing = torch.maximum(
                self.max_crawl_swing,
                torch.where(old_phase == 1, swing, torch.zeros_like(swing)),
            )
        self.phase_elapsed = torch.where(
            self.phase_id != old_phase,
            torch.zeros_like(self.age),
            self.phase_elapsed + self.dt,
        )
        self.task_hold = torch.where(
            phase_goal, self.task_hold + self.dt, torch.zeros_like(self.task_hold)
        )
        self.style_hold = torch.where(
            phase_goal & style & compact,
            self.style_hold + self.dt,
            torch.zeros_like(self.style_hold),
        )
        # Signed dwell progress cannot be farmed by repeated broken holds.
        reward += 5 * (self.task_hold - previous_dwell) / hold
        reward += 3 * (self.phase_id != old_phase).float()
        strength = 0.15 + 0.85 * self.residual_amplitude
        stride_penalty = torch.relu(swing - cfg["swing_m"] * 0.7).square()
        spread_penalty = (
            torch.relu(foreaft - 0.11).square()
            + 2 * torch.relu(lateral - 0.055).square()
        )
        speed_penalty = torch.relu(
            speed - (cfg["speed_m_s"] if self.task == "down" else 0.22)
        ).square()
        force_penalty = 0.000002 * torch.relu(self.foot_peak - 130).square().mean(-1)
        reward -= (
            strength * (2 * stride_penalty + 2 * spread_penalty + speed_penalty)
            + force_penalty
        )
        reward -= (
            0.005
            + 0.000001 * self.torque[:, :12].square().sum(-1)
            + 0.01 * self.current_joint_residual.square().mean(-1)
        )
        reward -= 0.05 * (
            self.current_joint_residual - self.previous_joint_residual
        ).square().mean(-1)
        failure = base["failure"] | phase_failure
        # Task goal for learning is recorded separately from FINAL full style qualification.
        task_success = (self.task_hold >= hold) & ~failure
        full = self.training_stage == "full"
        success = (self.style_hold >= hold) & ~failure if full else task_success
        timeout = (
            (
                self.age
                >= (
                    8.0
                    if self.training_stage == "crouch"
                    else 10.0
                    if self.training_stage == "settle"
                    else cfg["timeout_s"]
                )
            )
            & ~success
            & ~failure
        )
        task_milestone = task_success & ~self.task_awarded
        reward += (
            25 * task_milestone.float() - 15 * failure.float() - 5 * timeout.float()
        )
        self.task_awarded |= task_success
        reward = torch.nan_to_num(reward, nan=-15.0, posinf=-15.0, neginf=-15.0)
        self.stage_return += reward
        self.return_.copy_(self.stage_return)
        self.was_contact = contact
        self.contact_now = contact
        self.previous_feet = feet
        done = success | failure | timeout
        terminal = self.augment(base["terminal_obs"][:, :320])
        info = dict(base)
        info.update(
            success=success,
            failure=failure,
            timeout=timeout,
            done=done,
            terminal_obs=terminal,
            **{"return": self.stage_return.clone()},
            dwell=self.task_hold.clone(),
            style_dwell=self.style_hold.clone(),
            task_success=task_success,
            task_milestone=task_milestone,
            phase=self.phase_id.clone(),
            phase_violation=phase_failure,
            style_clean=style,
            crouch_achieved=self.crouch_verified.clone(),
            under_seen=self.crossing_witness.clone(),
            max_stride_m=self.episode_swing.clone(),
            max_foreaft_m=self.episode_foreaft.clone(),
            max_lateral_m=self.episode_lateral.clone(),
            max_speed_m_s=self.episode_speed.clone(),
            max_foot_force_n=self.episode_impact.clone(),
            arm_error=self.episode_arms.clone(),
            self_peak=self.physical_self_peak.clone(),
            support_count=support_count,
            compact=compact,
            stage=self.training_stage,
            partial_goal=not full,
            reference_height=self.reference_height.clone(),
        )
        if auto_reset:
            self.reset(done)
            history = self.history.clone()
            new_obs = self.observe()
            self.history.copy_(torch.where(done[:, None, None], self.history, history))
            self.obs = torch.where(done[:, None], new_obs, terminal)
            # The reset observation is computed for all worlds; continuing worlds must
            # NOT feed its duplicate frame into the frozen teacher's history.
            self.command_observation = self.obs[:, :320].clone()
        else:
            self.obs = terminal
        return self.obs, reward, done, info

    def save_domains(self, path):
        LegSkillEnv.save_domains(self, path)
        data = json.loads(Path(path).read_text())
        data.update(
            schema=SCHEMA,
            training_stage=self.training_stage,
            reference_sha256=self.reference_sha,
            initialization="on-box reset for settle, low reset for crawl; full ALWAYS starts upright on floor",
            reference_controller="v1 frozen actor plus offline static-load reference, hand-designed low crawl + compressed pretrained gait, learned bounded 12D joint residual",
            reference_fk_error_max_m=self.reference_fk_error,
        )
        Path(path).write_text(json.dumps(data, indent=2) + "\n")
