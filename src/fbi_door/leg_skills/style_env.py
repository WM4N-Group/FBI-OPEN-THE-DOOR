"""Small-step v2: learn compact legs; down must crouch before advancing.

No inverse-kinematics gait, base pose writes, weld, or arm-policy actions. The
original physics/contact/fall checks remain active, with stricter style checks.
"""

import math
import json
from pathlib import Path
import mujoco
import torch
import warp as wp
import mujoco_warp as mw
from .gpu_env import LegSkillEnv, rotate_inverse
from .kernels import pd, audit, overflow_audit
from .style_kernels import foot_contacts
from .style_contracts import STYLE_OBS_DIM, style_limits, target_increment


class SmallStepEnv(LegSkillEnv):
    def __init__(self, *args, **kwargs):
        self._style_ready = False
        super().__init__(*args, **kwargs)
        self.limits = style_limits(self.task)
        self.speed = self.tensor(
            [
                self.limits["command_speed_min"]
                + (d["command_speed"] - 0.12)
                / 0.13
                * (self.limits["command_speed_max"] - self.limits["command_speed_min"])
                for d in self.domains
            ]
        )
        mujoco.mj_resetDataKeyframe(self.m, self.cpu, self.m.key("home").id)
        mujoco.mj_forward(self.m, self.cpu)
        self.nominal_feet = self.tensor(
            self.cpu.geom_xpos[self.foot_geoms] - self.cpu.qpos[:3]
        )
        shape = (self.n, 4)
        self.foot_current = torch.zeros(shape, device=self.device)
        self.foot_peak = torch.zeros(shape, device=self.device)
        self.foot_impulse = torch.zeros(shape, device=self.device)
        self.foot_current_wp = wp.from_torch(self.foot_current)
        self.foot_peak_wp = wp.from_torch(self.foot_peak)
        self.foot_impulse_wp = wp.from_torch(self.foot_impulse)
        self.foot_args = self.audit_args[:10] + [
            self.classes_wp,
            self.feet_wp,
            self.foot_current_wp,
            self.foot_peak_wp,
            self.foot_impulse_wp,
        ]
        self.phase = torch.zeros(self.n, device=self.device, dtype=torch.long)
        self.phase_age = torch.zeros_like(self.age)
        self.crouch_dwell = torch.zeros_like(self.age)
        self.crouch_achieved = torch.zeros_like(self.under_seen)
        self.style_dwell = torch.zeros_like(self.age)
        self.was_contact = torch.ones(shape, device=self.device, dtype=torch.bool)
        self.airtime = torch.zeros(shape, device=self.device)
        self.liftoff = torch.zeros((self.n, 4, 2), device=self.device)
        self.last_stride = torch.zeros(shape, device=self.device)
        self.previous_feet = torch.zeros((self.n, 4, 3), device=self.device)
        self.start_xy = torch.zeros((self.n, 2), device=self.device)
        self.best_support = torch.zeros_like(self.age)
        self.max_stride = torch.zeros_like(self.age)
        self.max_foreaft = torch.zeros_like(self.age)
        self.max_lateral = torch.zeros_like(self.age)
        self.max_speed = torch.zeros_like(self.age)
        self.max_foot_force = torch.zeros_like(self.age)
        self.max_vertical_speed = torch.zeros_like(self.age)
        self.max_target_speed = torch.zeros_like(self.age)
        self.style_return = torch.zeros_like(self.age)
        self.peak_style_violation = torch.zeros_like(self.age)
        self.low_target = (
            self.normal_base - (self.normal_top - self.clearance) - 0.025
        ).clamp(0.115, self.normal_base - 0.045)
        self._style_ready = True
        self.reset(torch.ones(self.n, device=self.device, dtype=torch.bool))
        # Keep v1 compilation probe separate; replace only this NEW env graph.
        self._style_physics()
        self.reset(torch.ones(self.n, device=self.device, dtype=torch.bool))
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            self._style_physics()
        self.graph = cap.graph
        self.reset(torch.ones(self.n, device=self.device, dtype=torch.bool))
        self.obs = self.observe()
        assert self.obs.shape == (self.n, STYLE_OBS_DIM), self.obs.shape

    def save_domains(self, path):
        super().save_domains(path)
        data = json.loads(Path(path).read_text())
        data.update(
            style_schema="leg-skills/small-step-v2",
            style_limits=self.limits,
            actual_command_speed_m_s=self.speed.cpu().tolist(),
            crouch_target_height_m=self.low_target.cpu().tolist()
            if self.task == "down"
            else None,
        )
        Path(path).write_text(json.dumps(data, indent=2) + "\n")

    def _style_physics(self):
        for _ in range(10):
            wp.launch(pd, dim=(self.n, 19), inputs=self.pd_args)
            mw.step(self.wm, self.wd)
            wp.launch(audit, dim=self.wd.naconmax, inputs=self.audit_args)
            self.foot_current_wp.zero_()
            wp.launch(foot_contacts, dim=self.wd.naconmax, inputs=self.foot_args)
            wp.launch(
                overflow_audit,
                dim=self.n,
                inputs=[self.wd.overflow, wp.from_torch(self.overflow)],
            )
        mw.fwd_kinematics(self.wm, self.wd)

    def reset(self, mask):
        super().reset(mask)
        if not self._style_ready:
            return
        for value in (
            self.phase,
            self.phase_age,
            self.crouch_dwell,
            self.style_dwell,
            self.best_support,
            self.max_stride,
            self.max_foreaft,
            self.max_lateral,
            self.max_speed,
            self.max_foot_force,
            self.max_vertical_speed,
            self.max_target_speed,
            self.style_return,
            self.peak_style_violation,
        ):
            value.masked_fill_(mask, 0)
        self.crouch_achieved.masked_fill_(mask, False)
        self.was_contact.masked_fill_(mask[:, None], True)
        self.airtime.masked_fill_(mask[:, None], 0.0)
        self.last_stride.masked_fill_(mask[:, None], 0.0)
        for value in (self.foot_current, self.foot_peak, self.foot_impulse):
            value.masked_fill_(mask[:, None], 0.0)
        feet = self.xpos[:, self.foot_geoms].clone()
        self.liftoff.copy_(
            torch.where(mask[:, None, None], feet[:, :, :2], self.liftoff)
        )
        self.previous_feet.copy_(
            torch.where(mask[:, None, None], feet, self.previous_feet)
        )
        self.start_xy.copy_(torch.where(mask[:, None], self.q[:, :2], self.start_xy))

    def leg_targets(self, action):
        proposed = super().leg_targets(action)
        if not self._style_ready:
            return proposed
        delta = target_increment(self.task, self.dt)
        return self.target[:, :12] + (proposed - self.target[:, :12]).clamp(
            -delta, delta
        )

    def geometry(self):
        center, local, desired = super().geometry()
        if self._style_ready and self.task == "down":
            desired = torch.where(
                self.phase < 2,
                self.low_target,
                torch.full_like(self.age, self.normal_base),
            )
        return center, local, desired

    def foot_local(self):
        return rotate_inverse(
            self.q[:, None, 3:7].expand(-1, 4, -1).reshape(-1, 4),
            (self.xpos[:, self.foot_geoms] - self.q[:, None, :3]).reshape(-1, 3),
        ).reshape(self.n, 4, 3)

    def _patch_commands(self):
        center, _, _ = self.geometry()
        cmd = self.history[:, -1, 9:12]
        if self.task == "down":
            forward = (self.center_x + 0.8 - self.q[:, 0]).clamp(min=0) * 0.7
            cmd[:, 0] = torch.where(
                self.phase == 1,
                torch.minimum(self.speed, forward),
                torch.zeros_like(forward),
            )
            move = self.phase == 1
        else:
            cmd[:, 0] = torch.where(
                self.phase == 0,
                torch.minimum(
                    self.speed, (center[:, 0] - self.q[:, 0]).clamp(min=0) * 0.7
                ),
                torch.zeros_like(self.age),
            )
            move = self.phase == 0
        cmd[:, 1] = torch.where(
            move,
            ((center[:, 1] - self.q[:, 1]) * 0.5).clamp(-0.025, 0.025),
            torch.zeros_like(self.age),
        )
        yaw = torch.atan2(
            2 * (self.q[:, 3] * self.q[:, 6] + self.q[:, 4] * self.q[:, 5]),
            1 - 2 * (self.q[:, 5] ** 2 + self.q[:, 6] ** 2),
        )
        cmd[:, 2] = torch.where(
            move, (-yaw * 0.5).clamp(-0.04, 0.04), torch.zeros_like(self.age)
        )
        frequency = 0.85 if self.task == "down" else 1.1
        clock = (
            (self.age[:, None] + self.dt) * frequency
            + self.tensor([0.0, 0.5, 0.5, 0.0])
        ) % 1.0
        self.history[:, -1, 50:54] = torch.where(
            cmd.norm(dim=-1, keepdim=True) < 0.01, -torch.ones_like(clock), clock
        )

    def _augment(self, base):
        self._patch_commands()
        # Include actual slew-limited targets and visible rule phase; do not hide dynamics memory.
        local = self.foot_local()
        _, _, desired = self.geometry()
        extra = torch.cat(
            (
                torch.nn.functional.one_hot(self.phase, 3).float(),
                (self.crouch_dwell / 0.30)[:, None],
                self.target[:, :12] - self.home[7:19],
                local[:, :, 0] - self.nominal_feet[:, 0],
                local[:, :, 1] - self.nominal_feet[:, 1],
                self.was_contact.float(),
                self.last_stride,
                self.phase_age[:, None] / 45.0,
                desired[:, None],
                self.speed[:, None],
                self.max_stride[:, None],
                self.max_foreaft[:, None],
                self.max_lateral[:, None],
                (self.style_dwell / self.limits["hold_s"])[:, None],
                self.crouch_achieved[:, None].float(),
            ),
            -1,
        )
        base = base.clone()
        base[:, :300] = self.history.flatten(1)
        base[:, 307] = desired
        base[:, 317] = self.style_dwell / 2.0
        result = torch.cat((base[:, :320], extra), -1)
        return torch.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0).clamp(
            -100.0, 100.0
        )

    def observe(self):
        base = super().observe()
        return self._augment(base) if self._style_ready else base

    def step(self, action, auto_reset=True):
        cfg = self.limits
        old_xy = self.q[:, :2].clone()
        old_z = self.q[:, 2].clone()
        old_phase = self.phase.clone()
        old_target = self.target[:, :12].clone()
        old_dwell = self.style_dwell.clone()
        old_center = (
            self.q[:, 27:29].clone()
            if self.task == "climb"
            else torch.stack((self.center_x, self.center_y), -1)
        )
        _, _, old_desired = self.geometry()
        self.foot_peak.zero_()
        self.foot_impulse.zero_()
        _, _, _, base_info = super().step(action, auto_reset=False)
        # The parent reward/20s timeout are deliberately not used; its physics/failure gates are.
        feet = self.xpos[:, self.foot_geoms].clone()
        local = self.foot_local()
        contact = self.foot_current > 0.5
        liftoff = self.was_contact & ~contact
        self.liftoff = torch.where(
            liftoff[:, :, None], self.previous_feet[:, :, :2], self.liftoff
        )
        self.airtime = torch.where(
            contact, torch.zeros_like(self.airtime), self.airtime + self.dt
        )
        landed = (~self.was_contact) & contact
        touchdown = (feet[:, :, :2] - self.liftoff).norm(dim=-1)
        self.last_stride = torch.where(landed, touchdown, self.last_stride)
        live_stride = (feet[:, :, :2] - self.liftoff).norm(dim=-1) * ~contact
        swing_now = (
            torch.maximum(
                live_stride, torch.where(landed, touchdown, torch.zeros_like(touchdown))
            )
            .max(-1)
            .values
        )
        self.max_stride = torch.maximum(self.max_stride, swing_now)
        foreaft = (local[:, :, 0] - self.nominal_feet[:, 0]).abs()
        lateral = (local[:, :, 1] - self.nominal_feet[:, 1]).abs()
        self.max_foreaft = torch.maximum(self.max_foreaft, foreaft.max(-1).values)
        self.max_lateral = torch.maximum(self.max_lateral, lateral.max(-1).values)
        speed = self.v[:, :2].norm(dim=-1)
        angular = self.v[:, 3:6].norm(dim=-1)
        self.max_speed = torch.maximum(self.max_speed, speed)
        self.max_foot_force = torch.maximum(
            self.max_foot_force, self.foot_peak.max(-1).values
        )
        self.max_vertical_speed = torch.maximum(
            self.max_vertical_speed, self.v[:, 2].abs()
        )
        self.max_target_speed = torch.maximum(
            self.max_target_speed,
            ((self.target[:, :12] - old_target).abs() / self.dt).max(-1).values,
        )
        self.phase_age += self.dt
        up = 1 - 2 * (self.q[:, 4] ** 2 + self.q[:, 5] ** 2)
        arm_error = (self.q[:, 19:25] - self.home[19:25]).abs().max(-1).values
        compact = (
            foreaft.max(-1).values < (0.095 if self.task == "climb" else 0.12)
        ) & (lateral.max(-1).values < cfg["lateral_accept_m"])
        no_flight = contact.sum(-1) >= 2
        phase_violation = torch.zeros(self.n, device=self.device, dtype=torch.bool)
        style_clean = (
            (self.max_stride <= cfg["swing_accept_m"])
            & (self.max_foreaft <= cfg["foreaft_accept_m"])
            & (self.max_lateral <= cfg["lateral_accept_m"])
            & (self.max_speed <= cfg["max_horizontal_speed_m_s"])
        )
        if self.task == "down":
            bounds_min, bounds_max = self.robot_bounds(rail_frame=True)
            crouched = (
                (self.q[:, 2] <= self.low_target + 0.012)
                & (bounds_max[:, 2] <= self.clearance - 0.012)
                & (speed < 0.045)
                & (angular < 0.30)
                & (contact.sum(-1) >= 3)
                & (arm_error < 0.12)
            )
            self.crouch_dwell = torch.where(
                (self.phase == 0) & crouched,
                self.crouch_dwell + self.dt,
                torch.where(
                    self.phase == 0,
                    torch.zeros_like(self.crouch_dwell),
                    self.crouch_dwell,
                ),
            )
            start_crawl = (self.phase == 0) & (self.crouch_dwell >= 0.30)
            self.crouch_achieved |= start_crawl
            self.phase = torch.where(
                start_crawl, torch.ones_like(self.phase), self.phase
            )
            drift = (self.q[:, :2] - self.start_xy).norm(dim=-1)
            phase_violation = (old_phase == 0) & (drift > 0.18)
            clear = (
                (bounds_min[:, 0] > self.length / 2 + 0.12)
                & self.under_seen
                & self.crouch_achieved
            )
            self.phase = torch.where(
                (self.phase == 1) & clear, torch.full_like(self.phase, 2), self.phase
            )
            ready = (
                (self.phase == 2)
                & clear
                & (self.q[:, 2] - self.normal_base).abs().lt(0.025)
                & (speed < 0.05)
                & (angular < 0.20)
                & (up > 0.98)
                & (arm_error < 0.12)
                & (contact.sum(-1) >= 3)
                & compact
                & style_clean
            )
            goal = self.center_x + 0.8
            progress = (goal - old_xy[:, 0]).abs() - (goal - self.q[:, 0]).abs()
            # Forward reward is unavailable until the physical crouch precondition was met.
            progress = torch.where(old_phase == 1, progress, torch.zeros_like(progress))
            reward = 14 * progress
            reward -= torch.where(
                old_phase == 0, 5 * speed.square(), torch.zeros_like(speed)
            )
            # Dense NEGATIVE tracking costs help learn the preparatory crouch without
            # granting a fixed positive reward for remaining low forever.
            reward -= 2 * (self.q[:, 2] - old_desired).square()
            reward -= torch.where(
                old_phase < 2,
                2 * torch.relu(bounds_max[:, 2] - self.clearance + 0.012).square(),
                torch.zeros_like(speed),
            )
            inside_but_tall = (
                (old_phase == 1)
                & (bounds_min[:, 0] < self.length / 2 + 0.12)
                & (bounds_max[:, 2] > self.clearance - 0.005)
            )
            reward -= 0.12 * inside_but_tall.float()
            surface = torch.zeros((self.n, 4), device=self.device)
            allowed_lift = cfg["swing_height_m"]
            support_progress = torch.zeros_like(self.age)
        else:
            center = self.q[:, 27:29]
            box_q = self.q[:, 30:34]
            foot_box = rotate_inverse(
                box_q[:, None].expand(-1, 4, -1).reshape(-1, 4),
                (feet - self.q[:, None, 27:30]).reshape(-1, 3),
            ).reshape(self.n, 4, 3)
            contained = (foot_box[:, :, 0].abs() < self.length[:, None] / 2 - 0.025) & (
                foot_box[:, :, 1].abs() < self.width[:, None] / 2 - 0.025
            )
            on_top = (foot_box[:, :, 2] > self.height[:, None] / 2 + 0.005) & (
                foot_box[:, :, 2] < self.height[:, None] / 2 + 0.05
            )
            support = (self.audit[:, :4] > 0.5) & contact & contained & on_top
            count = support.sum(-1).float()
            self.phase = torch.where(
                (self.phase == 0) & support.all(-1),
                torch.ones_like(self.phase),
                self.phase,
            )
            box_up = 1 - 2 * (box_q[:, 1] ** 2 + box_q[:, 2] ** 2)
            box_speed = self.v[:, 26:29].norm(dim=-1)
            ready = (
                support.all(-1)
                & (speed < 0.045)
                & (self.v[:, 2].abs() < 0.035)
                & (angular < 0.20)
                & (up > 0.98)
                & (box_up > math.cos(0.10))
                & (box_speed < 0.03)
                & (arm_error < 0.12)
                & compact
                & style_clean
            )
            goal = center
            progress = (old_xy - old_center).norm(dim=-1) - (
                self.q[:, :2] - center
            ).norm(dim=-1)
            elevation = self.q[:, 2] - old_z
            support_progress = (count - self.best_support).clamp(min=0)
            self.best_support = torch.maximum(self.best_support, count)
            reward = 14 * progress + 3 * elevation + support_progress * 0.8
            surface = torch.where(
                contained, self.height[:, None], torch.zeros_like(feet[:, :, 2])
            )
            allowed_lift = torch.where(
                self.phase == 0, self.height + 0.055, torch.full_like(self.age, 0.055)
            )[:, None]
            # Slow down and gather the feet once supported, rather than keep trotting.
            reward -= torch.where(
                self.phase == 1,
                1.5 * speed.square() + 0.2 * angular.square(),
                torch.zeros_like(speed),
            )
        changed = self.phase != old_phase
        self.phase_age = torch.where(
            changed, torch.zeros_like(self.phase_age), self.phase_age
        )
        self.style_dwell = torch.where(
            ready, self.style_dwell + self.dt, torch.zeros_like(self.style_dwell)
        )
        reward += (
            4 * changed.float() + 8 * (self.style_dwell - old_dwell) / cfg["hold_s"]
        )
        # Signed height error improvement, not a positive reward for waiting low/on top.
        _, _, desired = self.geometry()
        height_progress = (old_z - old_desired).abs() - (
            self.q[:, 2] - old_desired
        ).abs()
        reward += (
            torch.where(
                changed, torch.zeros_like(height_progress), 20 * height_progress
            )
            if self.task == "down"
            else 0.0
        )
        foot_speed = (feet - self.previous_feet).norm(dim=-1) / self.dt
        leg_rate = self.v[:, 6:18].abs()
        limb_cost = 8 * torch.relu(foreaft - cfg["foreaft_soft_m"]).square().mean(
            -1
        ) + 12 * torch.relu(lateral - cfg["lateral_soft_m"]).square().mean(-1)
        stride_cost = 12 * (
            torch.relu(touchdown - cfg["swing_soft_m"]).square() * landed
        ).sum(-1)
        lift = torch.relu(feet[:, :, 2] - surface - 0.022)
        lift_cost = 8 * torch.relu(lift - allowed_lift).square().mean(-1)
        slip_cost = 0.002 * (foot_speed.square() * contact).mean(-1)
        force_cost = 0.000002 * torch.relu(self.foot_peak - 100.0).square().mean(-1)
        vertical_cost = (
            0.10
            * torch.relu(
                self.v[:, 2].abs() - (0.32 if self.task == "climb" else 0.12)
            ).square()
        )
        speed_cost = 2 * torch.relu(speed - self.speed).square()
        stationary = (self.phase != 0) if self.task == "climb" else (self.phase != 1)
        speed_cost += torch.where(
            stationary, 2 * speed.square(), torch.zeros_like(speed)
        )
        # Quasi-static posture prior is a reward ONLY, never a leg command/IK gait.
        # Symmetric 2x0.213m leg geometry keeps the feet near the hips while crouching.
        reference = self.home[7:19].expand(self.n, -1).clone()
        if self.task == "down":
            hip = torch.acos(((desired - 0.022) / 0.426).clamp(0.15, 0.95))
            reference[:, [1, 4, 7, 10]] = hip[:, None]
            reference[:, [2, 5, 8, 11]] = -2 * hip[:, None]
            posture_cost = 0.15 * (self.q[:, 7:19] - reference).square().mean(-1)
        else:
            posture_cost = (
                0.04
                * (self.q[:, 7:19] - reference).square().mean(-1)
                * (self.phase == 1)
            )
        reward -= (
            posture_cost
            + 0.08 * (1 - up).clamp(min=0)
            + 0.015
            + limb_cost
            + stride_cost
            + lift_cost
            + slip_cost
            + force_cost
            + vertical_cost
            + speed_cost
            + 0.0005 * torch.relu(leg_rate - 2.5).square().mean(-1)
            + 0.000001 * self.torque[:, :12].square().sum(-1)
        )
        reward -= 0.06 * ~no_flight
        style_violation = (
            (self.max_stride > cfg["swing_abort_m"])
            | (foreaft.max(-1).values > cfg["foreaft_accept_m"] + 0.10)
            | (lateral.max(-1).values > cfg["lateral_accept_m"] + 0.07)
        )
        failure = base_info["failure"] | phase_violation | style_violation
        success = (self.style_dwell >= cfg["hold_s"]) & ~failure
        timeout = (self.age >= cfg["episode_timeout_s"]) & ~success & ~failure
        reward += (
            50 * success.float()
            - (40 if self.task == "down" else 20) * failure.float()
            - 15 * timeout.float()
        )
        reward = torch.nan_to_num(reward, nan=-8.0, posinf=-8.0, neginf=-8.0)
        self.style_return += reward
        self.return_.copy_(self.style_return)
        self.was_contact = contact
        self.previous_feet = feet
        self.peak_style_violation = torch.maximum(
            self.peak_style_violation, (~style_clean).float()
        )
        done = success | failure | timeout
        # Parent observation has already appended exactly one history frame.
        terminal_obs = self._augment(base_info["terminal_obs"][:, :320])
        info = dict(base_info)
        info.update(
            success=success,
            failure=failure,
            timeout=timeout,
            done=done,
            terminal_obs=terminal_obs,
            **{"return": self.style_return.clone()},
            dwell=self.style_dwell.clone(),
            phase=self.phase.clone(),
            phase_violation=phase_violation,
            style_failure=style_violation,
            max_stride_m=self.max_stride.clone(),
            max_foreaft_m=self.max_foreaft.clone(),
            max_lateral_m=self.max_lateral.clone(),
            max_speed_m_s=self.max_speed.clone(),
            max_foot_force_n=self.max_foot_force.clone(),
            max_vertical_speed_m_s=self.max_vertical_speed.clone(),
            max_target_rate_rad_s=self.max_target_speed.clone(),
            style_clean=style_clean,
            crouch_achieved=self.crouch_achieved.clone(),
            crouch_dwell=self.crouch_dwell.clone(),
            foot_support_count=contact.sum(-1),
            limb_cost=limb_cost,
            lift_cost=lift_cost,
            stride_cost=stride_cost,
        )
        if auto_reset:
            self.reset(done)
            old_history = self.history.clone()
            new_obs = self.observe()
            self.history.copy_(
                torch.where(done[:, None, None], self.history, old_history)
            )
            self.obs = torch.where(done[:, None], new_obs, terminal_obs)
        else:
            self.obs = terminal_obs
        return self.obs, reward, done, info
