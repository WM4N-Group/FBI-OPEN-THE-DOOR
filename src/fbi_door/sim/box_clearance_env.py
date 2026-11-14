"""Collision-aware simulation variant; preserves v1 and the frozen 12D leg actor."""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .box_env import BoxPushEnv
from .box_physics import BoxPhysics
from .box_collision import BoxCollisionGuard
from ..box_clearance import PHASES, ARM_SPEED, ARM_ACCEL, mask, withdrawal_ready


TOOL_FORWARD = Rotation.from_euler("y", 120, degrees=True).as_matrix()


class AuditedBoxPhysics(BoxPhysics):
    """Original bounded PD, with additional contact auditing at each 2 ms substep."""

    def solve_pose(self, position, orientation):
        m, work = self.model, self.work
        seeds = [
            self.data.qpos[self.arm_q].copy(),
            np.array([0, 1.6, -1.1, 0, -0.1, 0]),
            np.array([0, 2.3, -1.3, 0, -0.4, 0]),
        ]
        best = None
        diagnostics = []
        jp = np.zeros((3, m.nv))
        jr = jp.copy()
        for seed in seeds:
            work.qpos[:] = self.data.qpos
            work.qpos[self.arm_q] = seed
            for _ in range(120):
                mujoco.mj_forward(m, work)
                ep = np.asarray(position) - work.site_xpos[self.site]
                er = Rotation.from_matrix(
                    orientation @ work.site_xmat[self.site].reshape(3, 3).T
                ).as_rotvec()
                if np.linalg.norm(ep) < 0.002 and np.linalg.norm(er) < 0.025:
                    break
                mujoco.mj_jacSite(m, work, jp, jr, self.site)
                J = np.vstack([jp[:, self.arm_v], 0.25 * jr[:, self.arm_v]])
                dq = J.T @ np.linalg.solve(
                    J @ J.T + 0.0025 * np.eye(6), np.r_[ep, 0.25 * er]
                )
                work.qpos[self.arm_q] = np.clip(
                    work.qpos[self.arm_q] + np.clip(dq, -0.10, 0.10),
                    m.jnt_range[self.arm_jids, 0] + 0.015,
                    m.jnt_range[self.arm_jids, 1] - 0.015,
                )
            q = work.qpos[self.arm_q].copy()
            diagnostics.append(
                (
                    float(np.linalg.norm(ep)),
                    float(np.linalg.norm(er)),
                    q.round(2).tolist(),
                    self.audit.configuration_ok(q),
                    self.audit.distances(self.audit.work)[0],
                )
            )
            score = np.linalg.norm(ep) + 0.25 * np.linalg.norm(er)
            if (
                np.linalg.norm(ep) < 0.015
                and np.linalg.norm(er) < 0.08
                and self.audit.configuration_ok(q)
            ):
                if best is None or score < best[0]:
                    best = (score, q, float(np.linalg.norm(ep)))
        if best is None:
            raise RuntimeError(
                "No collision-valid full-pose IK solution: " + str(diagnostics)
            )
        return best[1], best[2]

    def step_pd(self, legs, arm, kp=20, kd=1.5):
        d, m = self.data, self.model
        self.arm_force = 0.0
        for _ in range(10):
            d.ctrl[:12] = np.clip(
                kp * (legs - d.qpos[7:19]) - kd * d.qvel[6:18],
                m.actuator_ctrlrange[:12, 0],
                m.actuator_ctrlrange[:12, 1],
            )
            d.ctrl[12:18] = np.clip(
                50 * (arm - d.qpos[self.arm_q])
                - 4 * d.qvel[self.arm_v]
                + d.qfrc_bias[self.arm_v],
                m.actuator_ctrlrange[12:18, 0],
                m.actuator_ctrlrange[12:18, 1],
            )
            d.ctrl[18] = np.clip(
                800 * (self.gripper_target - d.qpos[25]) - 5 * d.qvel[24], -20, 20
            )
            self.arm_torque_peak = np.maximum(
                self.arm_torque_peak, np.abs(d.ctrl[12:18])
            )
            mujoco.mj_step(m, d)
            force = 0.0
            for ci, c in enumerate(d.contact):
                if self.box_geom not in (c.geom1, c.geom2):
                    continue
                other = int(c.geom2 if c.geom1 == self.box_geom else c.geom1)
                if other not in self.robot_geoms:
                    continue
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                if other in self.arm_geoms:
                    force += max(0.0, float(f[0]))
                else:
                    self.non_arm_impulse += max(0.0, float(f[0])) * m.opt.timestep
            if force > 0.15:
                self.arm_contact_s += m.opt.timestep
            self.arm_force = max(self.arm_force, force)
            self.max_arm_force = max(self.max_arm_force, force)
            if hasattr(self, "audit"):
                self.audit.contacts()


class ClearanceBoxPushEnv(BoxPushEnv):
    def __init__(self, randomization=0.0, render_mode=None):
        gym.Env.__init__(self)
        if not 0 <= randomization <= 1:
            raise ValueError("randomization must be in [0,1]")
        self.p = AuditedBoxPhysics()
        self.audit = BoxCollisionGuard(self.p)
        self.p.audit = self.audit
        self.randomization = float(randomization)
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = spaces.Box(-1, 1, (6,), np.float32)
        self.reset(seed=42)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, self._obs().shape, np.float32
        )

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        p = self.p
        p.reset(seed or 0)
        r = self.randomization
        p.data.qpos[:2] += self.np_random.uniform(-1, 1, 2) * [0.025 * r, 0.025 * r]
        yaw = self.np_random.uniform(-0.04, 0.04) * r
        p.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        p.data.qpos[p.box_q : p.box_q + 2] += self.np_random.uniform(-1, 1, 2) * [
            0.04 * r,
            0.025 * r,
        ]
        mujoco.mj_forward(p.model, p.data)
        p.home = p.data.qpos.copy()
        p.prepare()
        self.audit.reset()
        self.initial_box = self.box_pos
        self.phase = 0
        self.phase_start = self.t = 0.0
        self.arm_command = p.home[p.arm_q].copy()
        self.home_arm = self.arm_command.copy()
        self.arm_velocity = np.zeros(6)
        self.base_command = np.zeros(3)
        self.locked_arm = None
        self.last_action = np.zeros(6)
        self.ik_error = 0.0
        self.failure = ""
        self.success = False
        self.phase_history = [PHASES[0]]
        self._reward_phase_count = 1
        self.push_lock_max_delta = 0.0
        self.retract_error = 0.0
        self.best_displacement = 0.0
        self.arm_contact_at_lock = 0.0
        self.contact_quiet_s = 0.0
        self.reach_path = [self.arm_command.copy()]
        self.retract_index = 0
        self.reach_offset = None
        self.planned_path = None
        self.plan_index = 0
        self.retreat_start = None
        self.withdrawal_at_retract = None
        self.clearance = self.audit.observe()
        self.min_retract_pad_gap = 1.0
        self.max_arm_target_speed = 0.0
        self.max_base_command_accel = 0.0
        return self._obs(), self._info()

    def _obs(self):
        p, d = self.p, self.p.data
        rpy = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")
        box_rpy = Rotation.from_quat(d.qpos[p.box_q + np.array([4, 5, 6, 3])]).as_euler(
            "xyz"
        )
        return np.r_[
            np.eye(len(PHASES))[self.phase],
            self.box_pos - d.qpos[:3],
            self.displacement,
            d.qpos[2],
            rpy,
            d.qvel[:6],
            d.qpos[p.arm_q],
            d.qvel[p.arm_v],
            d.site_xpos[p.site] - self.box_pos,
            box_rpy,
            p.arm_force / 20,
            p.arm_contact_s,
            self.last_action,
            (self.t - self.phase_start) / 10,
            self.base_command,
            self.clearance["self"],
            self.clearance["pad_box"],
            np.zeros(3) if self.reach_offset is None else self.reach_offset,
            self.arm_command - d.qpos[p.arm_q],
        ].astype(np.float32)

    def _advance(self):
        self.phase += 1
        self.phase_start = self.t
        phase = PHASES[self.phase]
        self.phase_history.append(phase)
        if phase == "touch":
            self.locked_arm = self.arm_command.copy()
            self.arm_contact_at_lock = self.p.arm_contact_s
        if phase == "retreat":
            self.retreat_start = float(self.p.data.qpos[0])
            self.contact_quiet_s = 0.0
        if phase == "retract":
            self.withdrawal_at_retract = {
                "base_retreat_m": self.retreat_start - float(self.p.data.qpos[0]),
                "pad_gap_m": self.clearance["pad_box"],
                "force_N": self.p.arm_force,
                "quiet_s": self.contact_quiet_s,
            }
            self.retract_index = len(self.reach_path) - 1
            self.arm_velocity[:] = 0

    def _arm_step(self, target):
        delta = target - self.arm_command
        desired = np.sign(delta) * np.minimum(
            ARM_SPEED, np.sqrt(2 * ARM_ACCEL * np.abs(delta))
        )
        velocity = self.arm_velocity + np.clip(
            desired - self.arm_velocity, -ARM_ACCEL * 0.02, ARM_ACCEL * 0.02
        )
        candidate = self.arm_command + velocity * 0.02
        # Do not overshoot an endpoint while braking.
        near = np.abs(target - self.arm_command) < np.abs(velocity * 0.02)
        candidate[near] = target[near]
        if not self.audit.swept_arm(candidate):
            self.arm_velocity[:] = 0
            return False
        self.max_arm_target_speed = max(
            self.max_arm_target_speed,
            float(np.max(np.abs(candidate - self.arm_command)) / 0.02),
        )
        self.arm_command = candidate
        self.arm_velocity = velocity
        return True

    def _reach(self):
        p, d = self.p, self.p.data
        if self.reach_offset is None:
            return  # Wait for the first policy observation in the reach phase.
        if self.planned_path is None:
            offset = self.reach_offset
            face = self.initial_box[0] - 0.20
            low = np.array([face - 0.09, self.initial_box[1], 0.34]) + offset
            high = low.copy()
            high[2] = 0.63
            try:
                q_high, _ = p.solve_pose(high, TOOL_FORWARD)
                q_low, self.ik_error = p.solve_pose(low, TOOL_FORWARD)
                first = self.audit.plan(self.arm_command, q_high, seed=10)
                second = self.audit.plan(q_high, q_low, seed=11)
                self.planned_path = first + second[1:]
                self.plan_index = 1
            except RuntimeError as exc:
                self.failure = str(exc)
                return
        q = self.planned_path[self.plan_index]
        if self._arm_step(q):
            self.reach_path.append(self.arm_command.copy())
        if np.max(np.abs(self.arm_command - q)) < 0.003:
            if self.plan_index < len(self.planned_path) - 1:
                self.plan_index += 1
            elif (
                np.max(np.abs(d.qpos[p.arm_q] - q)) < 0.055
                and np.max(np.abs(self.arm_velocity)) < 0.03
            ):
                self.arm_velocity[:] = 0
                self._advance()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("Six finite residuals required")
        a = np.clip(action, -1, 1) * mask(PHASES[self.phase])
        self.last_action = a.copy()
        if PHASES[self.phase] == "reach" and self.reach_offset is None:
            self.reach_offset = a[3:] * [0.005, 0.01, 0.01]
        reward = -0.02 - 0.002 * float(a @ a)
        for _ in range(5):
            p, d = self.p, self.p.data
            phase = PHASES[self.phase]
            age = self.t - self.phase_start
            target = np.zeros(3)
            yaw = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")[2]
            if phase == "approach":
                error = np.array(
                    [
                        self.box_pos[0] - 0.20 - 0.63 - d.qpos[0],
                        self.box_pos[1] - d.qpos[1],
                    ]
                )
                target = np.array(
                    [
                        np.clip(0.7 * error[0], 0, 0.12),
                        np.clip(0.7 * error[1], -0.04, 0.04),
                        np.clip(-yaw, -0.08, 0.08),
                    ]
                ) + a[:3] * [0.02, 0.015, 0.025]
                braking_distance = max(
                    0.0, float(d.qvel[0]), float(self.base_command[0])
                ) ** 2 / (2 * 0.06)
                if error[0] < braking_distance + 0.015:
                    target[0] = 0
                if (
                    abs(error[0]) < 0.045
                    and abs(error[1]) < 0.03
                    and age > 1
                    and np.linalg.norm(d.qvel[:2]) < 0.035
                ):
                    self._advance()
            elif phase == "settle":
                if age > 0.8 and np.linalg.norm(d.qvel[:2]) < 0.035:
                    self._advance()
            elif phase == "reach":
                self._reach()
            elif phase in ("touch", "push", "stop", "retreat"):
                self.arm_command = self.locked_arm.copy()
                self.push_lock_max_delta = max(
                    self.push_lock_max_delta,
                    float(np.max(np.abs(self.arm_command - self.locked_arm))),
                )
                if phase == "touch":
                    target[0] = 0.025
                    if p.arm_contact_s - self.arm_contact_at_lock >= 0.20:
                        self._advance()
                elif phase == "push":
                    remaining = max(0.0, 0.53 - self.displacement)
                    target = np.array(
                        [
                            min(0.075, np.sqrt(2 * 0.06 * remaining)),
                            np.clip(
                                -0.35 * (self.box_pos[1] - self.initial_box[1]),
                                -0.02,
                                0.02,
                            ),
                            np.clip(-yaw, -0.06, 0.06),
                        ]
                    ) + a[:3] * [0.02, 0.01, 0.025]
                    target[0] = np.clip(target[0], 0.015, 0.095)
                    if self.displacement >= 0.52:
                        self._advance()
                elif phase == "stop":
                    if age > 1 and np.linalg.norm(d.qvel[:2]) < 0.035:
                        self._advance()
                elif phase == "retreat":
                    target[0] = -0.06
                    self.contact_quiet_s = (
                        self.contact_quiet_s + 0.02 if p.arm_force < 0.15 else 0.0
                    )
                    retreat = self.retreat_start - float(d.qpos[0])
                    if withdrawal_ready(
                        retreat,
                        self.clearance["pad_box"],
                        p.arm_force,
                        self.contact_quiet_s,
                    ):
                        self._advance()
                        target[:] = 0
            elif phase == "withdraw_settle":
                self.arm_command = self.locked_arm.copy()
                if age > 0.8 and np.linalg.norm(d.qvel[:2]) < 0.035:
                    self._advance()
            elif phase == "retract":
                self.min_retract_pad_gap = min(
                    self.min_retract_pad_gap, self.clearance["pad_box"]
                )
                # Retrace checked reach commands, rather than a direct joint-space fold near the box.
                # Look ahead on the dense checked path instead of braking at every 20 ms sample.
                while (
                    self.retract_index > 0
                    and np.max(
                        np.abs(self.arm_command - self.reach_path[self.retract_index])
                    )
                    < 0.08
                ):
                    self.retract_index -= 1
                q = self.reach_path[self.retract_index]
                self._arm_step(q)
                self.retract_error = float(
                    np.max(np.abs(d.qpos[p.arm_q] - self.home_arm))
                )
                if (
                    self.retract_index == 0
                    and np.max(np.abs(self.arm_command - self.home_arm)) < 0.02
                    and self.retract_error < 0.12
                ):
                    self._advance()
                    self.success = self._valid_push()
                    if not self.success:
                        self.failure = "push_evidence_failed"
            previous = self.base_command.copy()
            self.base_command += np.clip(
                target - self.base_command,
                -np.array([0.06, 0.06, 0.15]) * 0.02,
                np.array([0.06, 0.06, 0.15]) * 0.02,
            )
            self.max_base_command_accel = max(
                self.max_base_command_accel,
                abs(float(self.base_command[0] - previous[0])) / 0.02,
            )
            if phase == "reach" and np.linalg.norm(d.qvel[:2]) > 0.08:
                self.failure = "base_moving_during_ik"
            if self.clearance["body_box"] < 0.16 and self.base_command[0] > 0:
                self.failure = "base_box_clearance"
            p.policy_step(self.base_command, self.arm_command)
            self.t += 0.02
            self.clearance = self.audit.observe()
            rpy = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")
            if d.qpos[2] < 0.14 or np.max(np.abs(rpy[:2])) > 0.55:
                self.failure = "unstable_base"
            if self.clearance["arm_floor"] < 0.05:
                self.failure = "arm_floor_clearance_stop"
            if self.clearance["self"] < 0.012:
                self.failure = "self_clearance_stop"
            if (
                self.clearance["protected_box"] < 0.015
                or self.audit.forbidden_box_force_peak > 1
            ):
                self.failure = "non_pad_box_stop"
            if p.non_arm_impulse >= 0.1:
                self.failure = "body_or_leg_ramming"
            if (
                self._box_tilt() > 0.25
                or abs(self.box_pos[1] - self.initial_box[1]) > 0.15
            ):
                self.failure = "box_tilt_or_lateral_escape"
            if p.arm_force > 25:
                self.failure = "simulation_contact_force_stop"
            if self.audit.self_force_peak > 1:
                self.failure = "self_contact_stop"
            if (
                age
                > {
                    "approach": 14,
                    "settle": 5,
                    "reach": 30,
                    "touch": 6,
                    "push": 18,
                    "stop": 5,
                    "retreat": 9,
                    "withdraw_settle": 5,
                    "retract": 32,
                    "success": 1,
                }[phase]
            ):
                self.failure = "phase_timeout_" + phase
            if self.failure:
                self.success = False
            if self.success or self.failure:
                break
        progress = max(self.best_displacement, self.displacement)
        reward += 30 * (progress - self.best_displacement)
        self.best_displacement = progress
        reward += 2 * max(0, len(self.phase_history) - self._reward_phase_count)
        self._reward_phase_count = len(self.phase_history)
        reward += 50 if self.success else (-20 if self.failure else 0)
        truncated = self.t >= 100 and not (self.success or self.failure)
        if truncated:
            self.failure = "episode_timeout"
        return (
            self._obs(),
            float(reward),
            bool(self.success or self.failure) and not truncated,
            bool(truncated),
            self._info(),
        )

    def _valid_push(self):
        return (
            super()._valid_push()
            and self.audit.minimum["self"] >= 0.02
            and self.audit.minimum["protected_box"] >= 0.015
            and self.audit.minimum["body_box"] >= 0.16
            and self.audit.minimum["arm_floor"] >= 0.07
            and self.audit.self_force_peak < 1
            and self.audit.forbidden_box_force_peak < 1
            and self.withdrawal_at_retract is not None
            and self.min_retract_pad_gap >= 0.08
        )

    def _info(self):
        return {
            "is_success": self.success,
            "failure": self.failure,
            "phase": PHASES[self.phase],
            "phase_history": self.phase_history.copy(),
            "elapsed_s": self.t,
            "box_displacement_m": self.displacement,
            "arm_box_contact_s": self.p.arm_contact_s,
            "non_arm_box_impulse_Ns": self.p.non_arm_impulse,
            "peak_arm_box_normal_N": self.p.max_arm_force,
            "peak_commanded_arm_torque_Nm": self.p.arm_torque_peak.tolist(),
            "push_lock_max_target_delta_rad": self.push_lock_max_delta,
            "retract_joint_error_rad": self.retract_error,
            "ik_error_m": self.ik_error,
            "minimum_clearance_m": self.audit.minimum.copy(),
            "nearest_self_pair": self.audit.nearest_self_pair,
            "self_contact_peak_N": self.audit.self_force_peak,
            "non_pad_box_contact_peak_N": self.audit.forbidden_box_force_peak,
            "arm_path_guard_rejections": self.audit.rejections,
            "withdrawal_before_retract": self.withdrawal_at_retract,
            "minimum_retract_pad_box_gap_m": self.min_retract_pad_gap,
            "max_arm_target_speed_rad_s": self.max_arm_target_speed,
            "max_base_vx_command_accel_m_s2": self.max_base_command_accel,
            "controller": "clearance_v2",
            "scope": "privileged geometry; proxy/sampled collision guards; simulation only, NOT safety certification",
        }
