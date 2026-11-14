"""Rule-managed IK/hold/push/retract task with bounded high-level residual PPO.

All box pose observations are privileged. Only the original frozen leg actor
outputs 12 joint targets. The 6D residual is NOT a new 18D whole-body policy.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .box_physics import BoxPhysics
from ..box_task import PHASES, push_valid, residual_mask

TOOL_FORWARD = Rotation.from_euler("y", 30, degrees=True).as_matrix() @ np.array(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float
)


class BoxPushEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, randomization=0.0, render_mode=None):
        super().__init__()
        self.p = BoxPhysics()
        self.randomization = float(randomization)
        if not 0 <= self.randomization <= 1:
            raise ValueError("randomization must be in [0,1]")
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = spaces.Box(-1, 1, (6,), np.float32)
        self.reset(seed=42)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, self._obs().shape, np.float32
        )

    @property
    def box_pos(self):
        return self.p.data.qpos[self.p.box_q : self.p.box_q + 3].copy()

    @property
    def displacement(self):
        return float(self.box_pos[0] - self.initial_box[0])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p = self.p
        p.reset(seed or 0)
        r = self.randomization
        p.data.qpos[0:2] += self.np_random.uniform(-1, 1, 2) * [0.025 * r, 0.025 * r]
        yaw = self.np_random.uniform(-0.04, 0.04) * r
        p.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        p.data.qpos[p.box_q : p.box_q + 2] += self.np_random.uniform(-1, 1, 2) * [
            0.04 * r,
            0.025 * r,
        ]
        mujoco.mj_forward(p.model, p.data)
        p.home = p.data.qpos.copy()
        p.prepare()
        self.initial_box = self.box_pos
        self.phase = 0
        self.phase_start = 0.0
        self.t = 0.0
        self.arm_command = p.home[p.arm_q].copy()
        self.home_arm = self.arm_command.copy()
        self.locked_arm = None
        self.last_action = np.zeros(6)
        self.ik_error = 0.0
        self.failure = ""
        self.success = False
        self.phase_history = ["approach"]
        self._reward_phase_count = 1
        self.push_lock_max_delta = 0.0
        self.retract_error = 0.0
        self.best_displacement = 0.0
        self.reach_origin = None
        self.arm_contact_at_lock = 0.0
        return self._obs(), self._info()

    def _obs(self):
        p, d = self.p, self.p.data
        base_rpy = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")
        box_rpy = Rotation.from_quat(d.qpos[p.box_q + np.array([4, 5, 6, 3])]).as_euler(
            "xyz"
        )
        return np.r_[
            np.eye(len(PHASES))[self.phase],
            self.box_pos - d.qpos[:3],
            self.displacement,
            d.qpos[2],
            base_rpy,
            d.qvel[:6],
            d.qpos[p.arm_q],
            d.qvel[p.arm_v],
            d.site_xpos[p.site] - self.box_pos,
            box_rpy,
            p.arm_force / 20,
            p.arm_contact_s,
            self.last_action,
            (self.t - self.phase_start) / 10,
        ].astype(np.float32)

    def _advance(self):
        self.phase += 1
        self.phase_start = self.t
        self.phase_history.append(PHASES[self.phase])
        if PHASES[self.phase] == "reach":
            self.reach_origin = self.p.data.site_xpos[self.p.site].copy()
        if PHASES[self.phase] == "push":
            self.locked_arm = self.arm_command.copy()
            self.arm_contact_at_lock = self.p.arm_contact_s

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (6,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected six finite bounded residual actions")
        a = np.clip(action, -1, 1) * residual_mask(PHASES[self.phase])
        self.last_action = a.copy()
        reward = -0.025 - 0.002 * float(a @ a)
        for _ in range(5):
            p, d = self.p, self.p.data
            phase = PHASES[self.phase]
            age = self.t - self.phase_start
            pos = self.box_pos
            command = np.zeros(3)
            if phase == "approach":
                # Stop at an arm-reachable distance, before any robot-box collision.
                error = np.array([pos[0] - 0.20 - 0.55 - d.qpos[0], pos[1] - d.qpos[1]])
                yaw = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")[2]
                command = np.array(
                    [
                        np.clip(1.0 * error[0], 0, 0.22),
                        np.clip(0.8 * error[1], -0.08, 0.08),
                        np.clip(-yaw, -0.12, 0.12),
                    ]
                )
                command += a[:3] * [0.04, 0.025, 0.04]
                if abs(error[0]) < 0.035 and abs(error[1]) < 0.04 and age > 1:
                    self._advance()
            elif phase == "reach":
                goal = np.array([pos[0] - 0.20 - 0.075, pos[1], 0.27]) + a[3:] * [
                    0.01,
                    0.015,
                    0.015,
                ]
                alpha = min(1.0, age / 4.0)
                pre = goal - np.array([0.12, 0, 0])
                if age < 2.5:
                    blend = min(1.0, age / 2.5)
                    waypoint = (1 - blend) * self.reach_origin + blend * pre
                else:
                    blend = min(1.0, (age - 2.5) / 1.5)
                    waypoint = (1 - blend) * pre + blend * goal
                q, self.ik_error = p.ik(waypoint, TOOL_FORWARD, iterations=35)
                self.arm_command += np.clip(q - self.arm_command, -0.016, 0.016)
                actual_err = float(np.linalg.norm(d.site_xpos[p.site] - goal))
                if alpha >= 1 and actual_err < 0.04 and self.ik_error < 0.025:
                    self._advance()
            elif phase == "push":
                # Latched position commands: body movement translates the held arm.
                self.arm_command = self.locked_arm.copy()
                self.push_lock_max_delta = max(
                    self.push_lock_max_delta,
                    float(np.max(np.abs(self.arm_command - self.locked_arm))),
                )
                lateral = pos[1] - self.initial_box[1]
                yaw = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")[2]
                command = np.array(
                    [
                        0.13,
                        np.clip(-0.5 * lateral, -0.035, 0.035),
                        np.clip(-yaw, -0.1, 0.1),
                    ]
                )
                command += a[:3] * [0.055, 0.025, 0.05]
                command[0] = np.clip(command[0], 0.04, 0.20)
                if self.displacement >= 0.52:
                    self._advance()
            elif phase == "stop":
                self.arm_command = self.locked_arm.copy()
                if age >= 0.7 and np.linalg.norm(d.qvel[:2]) < 0.07:
                    self._advance()
            elif phase == "retract":
                self.arm_command += np.clip(
                    self.home_arm - self.arm_command, -0.012, 0.012
                )
                self.retract_error = float(
                    np.max(np.abs(d.qpos[p.arm_q] - self.home_arm))
                )
                if (
                    age > 2
                    and np.max(np.abs(self.arm_command - self.home_arm)) < 0.02
                    and self.retract_error < 0.15
                ):
                    self._advance()
                    self.success = self._valid_push()
                    if not self.success:
                        self.failure = "push_evidence_failed"
            p.policy_step(command, self.arm_command)
            self.t += 0.02
            rpy = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")
            if (
                not np.all(np.isfinite(d.qpos))
                or d.qpos[2] < 0.14
                or np.max(np.abs(rpy[:2])) > 0.65
            ):
                self.failure = "unstable_base"
            if p.non_arm_impulse >= 0.5:
                self.failure = "body_or_leg_ramming"
            if (
                self._box_tilt() > 0.35
                or abs(self.box_pos[1] - self.initial_box[1]) > 0.18
            ):
                self.failure = "box_tilt_or_lateral_escape"
            if (
                phase == "push"
                and age > 2.5
                and p.arm_contact_s - self.arm_contact_at_lock < 0.05
            ):
                self.failure = "no_arm_contact"
            if (
                age
                > {
                    "approach": 12,
                    "reach": 9,
                    "push": 14,
                    "stop": 4,
                    "retract": 9,
                    "success": 1,
                }[phase]
            ):
                self.failure = "phase_timeout_" + phase
            if self.failure or self.success:
                break
        progress = max(self.best_displacement, self.displacement)
        reward += 30 * (progress - self.best_displacement)
        self.best_displacement = progress
        # Phase advancement rewards are handled once per external step.
        count = len(self.phase_history)
        previous_count = getattr(self, "_reward_phase_count", 1)
        reward += 2 * max(0, count - previous_count)
        self._reward_phase_count = count
        if self.success:
            reward += 50
        if self.failure:
            reward -= 20
        truncated = self.t >= 42 and not (self.success or self.failure)
        if truncated:
            self.failure = "episode_timeout"
        return (
            self._obs(),
            float(reward),
            bool(self.success or self.failure) and not truncated,
            bool(truncated),
            self._info(),
        )

    def _box_tilt(self):
        p = self.p
        q = p.data.qpos[p.box_q + np.array([4, 5, 6, 3])]
        return float(np.max(np.abs(Rotation.from_quat(q).as_euler("xyz")[:2])))

    def _valid_push(self):
        return push_valid(
            self.displacement,
            self.p.arm_contact_s,
            self.p.non_arm_impulse,
            self.box_pos[1] - self.initial_box[1],
            self._box_tilt(),
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
            "scope": "privileged geometry; rule manager + IK + frozen leg12; simulation only",
        }

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.p.model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [self.p.data.qpos[0] + 0.6, 0, 0.3]
        cam.distance = 2.8
        cam.azimuth = 90
        cam.elevation = -23
        opt = mujoco.MjvOption()
        opt.sitegroup[:] = 0
        self.renderer.update_scene(self.p.data, camera=cam, scene_option=opt)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
