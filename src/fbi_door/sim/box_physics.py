"""Frozen 12D locomotion + separately controlled arm; no door model dependencies."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from .core import Physics, wrapper
from ..paths import RUNTIME_ROOT as ROOT


class BoxPhysics(Physics):
    # Reuse exactly the tested IK, policy wrapper and one-second preparation.
    # Door-specific reset/contact/constraint code is deliberately not invoked.
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/box_scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.work = mujoco.MjData(self.model)
        self.site = self.model.site("attachment_site").id
        self.arm_jids = [self.model.joint("joint" + str(i)).id for i in range(1, 7)]
        self.arm_q = np.array([self.model.jnt_qposadr[i] for i in self.arm_jids])
        self.arm_v = np.array([self.model.jnt_dofadr[i] for i in self.arm_jids])
        self.box_q = int(self.model.joint("box_free").qposadr[0])
        self.box_geom = self.model.geom("push_box_geom").id
        self.arm_geoms = set()
        self.robot_geoms = set()
        for i in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[i])
            while bid > 0:
                if self.model.body(bid).name == "link1":
                    self.arm_geoms.add(i)
                if self.model.body(bid).name == "base":
                    self.robot_geoms.add(i)
                bid = int(self.model.body_parentid[bid])
        self.reset()

    def reset(self, seed=0):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[25:27] = [0.006, -0.006]
        mujoco.mj_forward(self.model, self.data)
        self.home = self.data.qpos.copy()
        self.ctrl = wrapper.LocomotionPolicyWrapper(self.model)
        self.gripper_target = 0.006
        self.arm_contact_s = 0.0
        self.non_arm_impulse = 0.0
        self.arm_force = 0.0
        self.max_arm_force = 0.0
        self.arm_torque_peak = np.zeros(6)

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
            arm_force = 0.0
            for ci, c in enumerate(d.contact):
                if self.box_geom not in (c.geom1, c.geom2):
                    continue
                other = int(c.geom2 if c.geom1 == self.box_geom else c.geom1)
                if other not in self.robot_geoms:
                    continue
                f = np.zeros(6)
                mujoco.mj_contactForce(m, d, ci, f)
                if other in self.arm_geoms:
                    arm_force += max(0.0, float(f[0]))
                else:
                    self.non_arm_impulse += max(0.0, float(f[0])) * m.opt.timestep
            if arm_force > 0.15:
                self.arm_contact_s += m.opt.timestep
            self.arm_force = max(self.arm_force, arm_force)
            self.max_arm_force = max(self.max_arm_force, arm_force)
