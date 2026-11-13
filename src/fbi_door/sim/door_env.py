"""Physical two-finger grasp candidate; scripted task manager + learned residual interface.
Teacher uses privileged geometry. Learned perception still has privileged scoring/phase gates.
No weld, magnetic grasp, kinematic handle motion, or direct door/handle actuation.
"""

import numpy as np, mujoco
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from .core import Physics, utils

PHASES = [
    "approach",
    "pregrasp",
    "insert",
    "grasp",
    "press",
    "push",
    "release",
    "retract",
    "traverse",
    "success",
]
GRIP_R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)


class DoorEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, perception="teacher", curriculum="grasp", render_mode=None):
        super().__init__()
        self.p = Physics()
        self.perception = perception
        self.curriculum = curriculum
        self.render_mode = render_mode
        self.camera = None
        self.detector = None
        self.viewer = None
        if perception != "teacher":
            raise ValueError(
                "Visual closed-loop control is not validated. Use camera_probe for observer-only validation."
            )
        self.action_space = spaces.Box(-1, 1, (10,), dtype=np.float32)
        self.reset(seed=42)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, self._obs().shape, dtype=np.float32
        )

    def _sense(self):
        p = self.p
        d = p.data
        theta = float(d.qpos[p.handle.qposadr[0]])
        angle = float(d.qpos[p.hinge.qposadr[0]])
        pos = d.site_xpos[p.model.site("handle_grasp").id].copy()
        self.detected = (pos, angle, theta)
        self.last_seen = self.t

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p = self.p
        p.reset(seed or 0)
        p.data.qpos[0] = self.np_random.uniform(0.03, 0.09)
        p.data.qpos[1] += self.np_random.uniform(-0.025, 0.025)
        yaw = self.np_random.uniform(-0.04, 0.04)
        p.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        p.data.qpos[25:27] = [0.04, -0.04]
        mujoco.mj_forward(p.model, p.data)
        p.home = p.data.qpos.copy()
        p.prepare()
        self.t = 0.0
        self.phase = 0
        self.phase_start = 0.0
        self.detected = None
        self.last_seen = -100.0
        self.last_action = np.zeros(10)
        self.commanded_arm = p.data.qpos[p.arm_q].copy()
        self.hold = 0
        self.previous_dist = None
        self.previous_angle = 0.0
        self.phase_times = {"approach": 0.0}
        self.stage_success = False
        self.max_door = 0.0
        self.max_handle_down = 0.0
        self.bilateral_duration = 0.0
        self.ever_grasp = False
        self.grasp_retries = 0
        self.press_start_theta = 0.0
        self.press_pivot = None
        self.pressed_theta = -0.43
        self._sense()
        return self._obs(), {
            "perception_mode": self.perception,
            "task": "door-grasp-v3",
        }

    def _next(self):
        self.phase += 1
        self.phase_start = self.t
        self.hold = 0
        self.previous_dist = None
        self.phase_times[PHASES[self.phase]] = round(self.t, 2)
        if self.phase == 4 and self.detected is not None:
            pos, angle, theta = self.detected
            Rz = Rotation.from_euler("z", angle).as_matrix()
            self.press_pivot = pos - Rz @ Rotation.from_euler("x", theta).apply(
                [0, 0.105, 0]
            )
            self.press_start_theta = theta

    def _obs(self):
        p = self.p
        d = p.data
        R = utils.heading_orientation_SO3(d)
        if self.detected is None:
            rel = np.zeros(3)
            angle = theta = valid = 0.0
        else:
            pos, angle, theta = self.detected
            rel = R.T @ (pos - d.qpos[:3])
            valid = float(self.t - self.last_seen < 0.8)
        return np.r_[
            rel,
            np.sin(angle),
            np.cos(angle),
            np.sin(theta),
            np.cos(theta),
            valid,
            R.T @ (d.site_xpos[p.site] - d.qpos[:3]),
            d.qpos[p.arm_q],
            d.qvel[p.arm_v],
            utils.base_lin_vel(d, "base"),
            utils.base_ang_vel(d, "base"),
            p.ctrl._get_projected_gravity(d.qpos[3:7]),
            d.qpos[25],
            np.clip(p.finger_forces / 20, 0, 2),
            min(p.grasp_hold_s, 1),
            p.grasp_speed,
            np.eye(len(PHASES))[self.phase],
            min((self.t - self.phase_start) / 5, 1),
            self.last_action,
        ].astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError("Expected ten finite residual actions")
        a = np.clip(action, -1, 1)
        p = self.p
        d = p.data
        self._sense()  # teacher geometry; visual control is explicitly rejected by this environment
        valid = self.detected is not None and self.t - self.last_seen < 0.8
        command = np.zeros(3)
        goal = None
        rot = None
        opening = 0.04
        target_arm = p.home[p.arm_q].copy()
        tau = self.t - self.phase_start
        if valid:
            pos, angle, theta = self.detected
            Rz = Rotation.from_euler("z", angle).as_matrix()
            normal = Rz[:, 0]
            currentrot = Rz @ Rotation.from_euler("x", theta).as_matrix() @ GRIP_R
            if self.phase == 0:
                err = pos - d.qpos[:3] - [0.48, 0, 0]
                command = np.array(
                    [
                        np.clip(0.8 * err[0], -0.05, 0.12),
                        np.clip(0.8 * err[1], -0.06, 0.06),
                        np.clip(-utils.base_ori_euler_xyz(d)[2], -0.15, 0.15),
                    ]
                )
                if abs(err[0]) < 0.035 and abs(err[1]) < 0.045:
                    self._next()
            if self.phase == 1:
                goal = pos - normal * 0.105
                rot = currentrot
            elif self.phase == 2:
                u = np.clip(tau / 1.6, 0, 1)
                u = u * u * (3 - 2 * u)
                goal = pos - normal * 0.105 * (1 - u)
                rot = currentrot
            elif self.phase == 3:
                goal = pos
                rot = currentrot
                u = np.clip(tau / 0.7, 0, 1)
                u = u * u * (3 - 2 * u)
                opening = (1 - u) * 0.04 + u * np.clip(0.001 + 0.002 * a[9], 0, 0.004)
            elif self.phase == 4:
                # A rigidly grasped lever requires a circular position path AND wrist rotation.
                u = np.clip(tau / 2.5, 0, 1)
                u = u * u * (3 - 2 * u)
                desired = self.press_start_theta + (-0.43 - self.press_start_theta) * u
                goal = self.press_pivot + Rz @ Rotation.from_euler("x", desired).apply(
                    [0, 0.105, 0]
                )
                rot = Rz @ Rotation.from_euler("x", desired).as_matrix() @ GRIP_R
                opening = np.clip(0.001 + 0.002 * a[9], 0, 0.004)
            elif self.phase == 5:
                # Follow a small opening lead about the door hinge, maintaining depressed grasp.
                # This release deliberately supports teacher geometry only.
                pivot = pos - Rz @ Rotation.from_euler("x", theta).apply([0, 0.105, 0])
                if self.perception == "teacher":
                    hinge = d.xpos[p.model.body("door").id].copy()
                else:
                    hinge = pivot - Rz @ np.array([-0.065, -0.73, 0.83])
                desiredRz = Rotation.from_euler("z", angle + 0.055).as_matrix()
                offset = Rz.T @ (pivot - hinge)
                goal = hinge + desiredRz @ (
                    offset + Rotation.from_euler("x", -0.43).apply([0, 0.105, 0])
                )
                rot = desiredRz @ Rotation.from_euler("x", -0.43).as_matrix() @ GRIP_R
                opening = np.clip(0.001 + 0.002 * a[9], 0, 0.004)
                err = pos - d.qpos[:3] - [0.46, 0, 0]
                command = np.array(
                    [
                        np.clip(0.8 * err[0] + 0.04, -0.04, 0.16),
                        np.clip(0.7 * err[1], -0.08, 0.08),
                        np.clip(-utils.base_ori_euler_xyz(d)[2], -0.15, 0.15),
                    ]
                )
            elif self.phase == 6:
                goal = pos - normal * 0.06
                rot = currentrot
                opening = 0.04
            elif self.phase == 7:
                opening = 0.04
            elif self.phase == 8:
                command = np.array(
                    [
                        0.18,
                        np.clip(-d.qpos[1] * 0.25, -0.06, 0.06),
                        np.clip(-utils.base_ori_euler_xyz(d)[2], -0.15, 0.15),
                    ]
                )
            if self.phase in [0, 1, 2, 3, 4, 5, 8]:
                command += a[:3] * [0.025, 0.025, 0.08]
                command = np.clip(command, [-0.08, -0.09, -0.25], [0.20, 0.09, 0.25])
            if goal is not None:
                goal = goal + a[3:6] * 0.012
                rot = Rotation.from_rotvec(a[6:9] * 0.07).as_matrix() @ rot
                desired_arm, ikerror = p.ik(goal, rot, iterations=65)
                target_arm = self.commanded_arm + np.clip(
                    desired_arm - self.commanded_arm, -0.075, 0.075
                )
            else:
                target_arm = self.commanded_arm + np.clip(
                    target_arm - self.commanded_arm, -0.075, 0.075
                )
        else:
            command[:] = 0
            target_arm = d.qpos[p.arm_q].copy()
            opening = p.gripper_target
        self.commanded_arm = target_arm.copy()
        p.gripper_target = opening
        for _ in range(5):
            p.policy_step(command, target_arm)
        self.t += 0.1
        angle = float(d.qpos[p.hinge.qposadr[0]])
        theta = float(d.qpos[p.handle.qposadr[0]])
        self.max_door = max(self.max_door, angle)
        self.max_handle_down = max(self.max_handle_down, -theta)
        self.ever_grasp |= p.grasp_valid
        self.bilateral_duration += 0.1 * float(p.bilateral)
        reward = (
            -0.015
            - 0.005 * float(np.square(a).mean())
            - 0.003 * float(np.square(a - self.last_action).mean())
        )
        term = trunc = False
        self.stage_success = False
        dist = None
        if goal is not None:
            dist = float(np.linalg.norm(d.site_xpos[p.site] - goal))
            reward += 0.06 * np.exp(-dist / 0.05)
            if self.previous_dist is not None:
                reward += 3 * np.clip(self.previous_dist - dist, -0.1, 0.1)
            self.previous_dist = dist
        orierr = (
            0.0
            if rot is None
            else np.linalg.norm(
                Rotation.from_matrix(
                    rot @ d.site_xmat[p.site].reshape(3, 3).T
                ).as_rotvec()
            )
        )
        if self.phase in [1, 2]:
            if (
                dist is not None
                and dist < 0.008
                and orierr < 0.10
                and (self.phase != 2 or tau >= 1.6)
            ):
                self.hold += 1
            else:
                self.hold = 0
            if self.hold >= 3:
                reward += 3
                if self.curriculum == "reach" and self.phase == 2:
                    self.stage_success = True
                    term = True
                else:
                    self._next()
        elif self.phase == 3:
            reward += 0.08 * float(p.bilateral) + 0.2 * min(p.grasp_hold_s, 0.5)
            if p.grasp_valid:
                reward += 10
                if self.curriculum == "grasp":
                    self.stage_success = True
                    term = True
                else:
                    self._next()
        elif self.phase == 4:
            reward += 0.1 * max(0, -theta)
            if p.unlocked:
                reward += 20
                if self.curriculum == "press":
                    self.stage_success = True
                    term = True
                else:
                    self._next()
        elif self.phase == 5:
            reward += 8 * np.clip(angle - self.previous_angle, -0.02, 0.06)
            if angle > 1.1:
                reward += 25
                if self.curriculum == "push":
                    self.stage_success = True
                    term = True
                else:
                    self._next()
        elif self.phase == 6:
            if tau > 0.6 and not p.bilateral:
                self._next()
        elif self.phase == 7:
            if np.max(np.abs(d.qpos[p.arm_q] - p.home[p.arm_q])) < 0.15:
                self._next()
                reward += 5
        elif self.phase == 8:
            # Tail and every moving robot body must be beyond the frame, not just base center.
            # Conservative bounding spheres include the whole trunk, arm and every leg collider.
            rear = min(
                float(d.geom_xpos[g, 0] - p.model.geom_rbound[g])
                for g in p.robot_collision_geoms
            )
            if (
                rear > p.model.body_pos[p.model.body("door").id, 0] + 0.08
                and abs(d.qpos[1]) < 0.28
            ):
                self._next()
                reward += 100
                term = True
        failure = None
        rpy = utils.base_ori_euler_xyz(d)
        if not np.isfinite(d.qpos).all():
            failure = "nonfinite"
        elif d.qpos[2] < 0.12 or np.max(np.abs(rpy[:2])) > 0.65:
            failure = "fall"
        elif self.t > 6 and not valid:
            failure = "perception_lost"
        elif self.phase in [4, 5] and p.contact_loss_s > 0.6:
            failure = "grasp_lost"
        if failure:
            term = True
            reward -= 15
        if self.t >= 55:
            trunc = True
        self.previous_angle = angle
        self.last_action = a.copy()
        info = {
            "perception_mode": self.perception,
            "teacher_privileged": self.perception == "teacher",
            "phase": PHASES[self.phase],
            "phase_times": self.phase_times.copy(),
            "failure": failure,
            "full_success": bool(self.phase == 9 and not failure),
            "stage_success": self.stage_success and not failure,
            "latch_unlocked": p.unlocked,
            "door_angle_rad": angle,
            "handle_angle_rad": theta,
            "max_door_angle_rad": self.max_door,
            "max_handle_down_rad": self.max_handle_down,
            "grasp_ever_valid": self.ever_grasp,
            "grasp_valid": p.grasp_valid,
            "bilateral_contact": p.bilateral,
            "finger_normal_forces_N": p.finger_forces.tolist(),
            "bilateral_duration_s": self.bilateral_duration,
            "grasp_relative_speed_m_s": p.grasp_speed,
            "tcp_error_m": dist,
            "orientation_error_rad": float(orierr),
            "gripper_q_m": float(d.qpos[25]),
            "gripper_target_m": float(opening),
            "sim_only": True,
            "manager": "scripted stages + residual RL interface",
        }
        return self._obs(), float(reward), term, trunc, info

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.Renderer(self.p.model, height=720, width=1280)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.85, -0.05, 0.7]
        cam.distance = 2.65
        cam.azimuth = 45
        cam.elevation = -20
        self.viewer.update_scene(self.p.data, camera=cam)
        return self.viewer.render().copy()

    def close(self):
        if self.camera:
            self.camera.close()
        if self.viewer:
            self.viewer.close()
