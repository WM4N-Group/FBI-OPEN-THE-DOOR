"""GPU physics and task rewards. Only resets write qpos; policies actuate 12 legs."""

import json
import math
from pathlib import Path
import mujoco
import mujoco_warp as mw
import numpy as np
import torch
import warp as wp
from .contracts import domain, MJ_TO_POLICY, POLICY_TO_MJ, OBS_DIM
from .kernels import pd, audit, overflow_audit


def rotate_inverse(q, v):
    w, xyz = q[:, :1], q[:, 1:]
    return (
        v
        - 2 * w * torch.cross(xyz, v, dim=-1)
        + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1), dim=-1)
    )


def geometry_extents(model, data, geom_ids):
    result = []
    for g in geom_ids:
        rot = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rot) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = np.repeat(size[0], 3)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0] + np.abs(rot[:, 2]) * size[1]
        else:
            extent = np.repeat(model.geom_rbound[g], 3)
        result.append(extent)
    return np.asarray(result)


class LegSkillEnv:
    dt = 0.02

    def __init__(
        self,
        runtime,
        task,
        count=1024,
        seed=300,
        split="train",
        difficulty=1.0,
        capture=True,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU physics required; do not silently use CPU")
        self.device = torch.device("cuda:0")
        wp.init()
        self.task, self.n, self.seed, self.split = task, count, seed, split
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.m = mujoco.MjModel.from_xml_path(str(Path(runtime) / "assets/scene.xml"))
        self.cpu = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.cpu, self.m.key("home").id)
        self.home = self.tensor(self.cpu.qpos.copy())
        mujoco.mj_forward(self.m, self.cpu)
        m = self.m
        base = m.body("base").id

        def descendants(b):
            result = set()
            for i in range(1, m.nbody):
                k = i
                while k > 0:
                    if k == b:
                        result.add(i)
                        break
                    k = int(m.body_parentid[k])
            return result

        robot_bodies = descendants(base)
        arm_bodies = descendants(m.body("link1").id)
        classes = np.zeros(m.ngeom, np.int32)
        feet = np.full(m.ngeom, -1, np.int32)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in robot_bodies:
                classes[g] = (
                    4
                    if m.geom_bodyid[g] in arm_bodies
                    else (3 if m.geom_bodyid[g] == base else 2)
                )
        self.foot_geoms = [m.geom(s).id for s in ("FL", "FR", "RL", "RR")]
        for i, g in enumerate(self.foot_geoms):
            classes[g] = 1
            feet[g] = i
        self.robot_geoms = [
            g for g in range(m.ngeom) if classes[g] > 0 and m.geom_contype[g]
        ]
        self.arm_geoms = [g for g in self.robot_geoms if classes[g] == 4]
        self.extents = self.tensor(geometry_extents(m, self.cpu, self.robot_geoms))
        self.normal_top = float(
            max(
                self.cpu.geom_xpos[g, 2] + e[2]
                for g, e in zip(self.robot_geoms, self.extents.cpu().numpy())
            )
        )
        self.normal_base = float(self.cpu.qpos[2])
        self.domains = [
            domain(
                task,
                seed * 10000 + i,
                difficulty
                if i % 5 == 0 or split != "train"
                else difficulty * ((i % 5) - 1) / 4,
                split,
            )
            for i in range(count)
        ]
        batches = {
            k: count
            for k in (
                "geom_size",
                "geom_pos",
                "geom_quat",
                "geom_aabb",
                "geom_rbound",
                "geom_friction",
            )
        }
        if task == "climb":
            batches.update(
                {
                    k: count
                    for k in (
                        "body_mass",
                        "body_inertia",
                        "body_subtreemass",
                        "body_invweight0",
                        "dof_invweight0",
                        "actuator_acc0",
                    )
                }
            )
        self.wm = mw.put_model(m, batch_sizes=batches)
        self.wm.opt.warn_overflow = (
            False  # Flags are audited, never hidden from metrics.
        )
        self.wd = mw.put_data(m, self.cpu, nworld=count, nconmax=64, njmax=256)
        self.q = wp.to_torch(self.wd.qpos)
        self.v = wp.to_torch(self.wd.qvel)
        self.xpos = wp.to_torch(self.wd.geom_xpos)
        self.xmat = wp.to_torch(self.wd.geom_xmat)
        self.target = torch.zeros((count, 19), device=self.device)
        self.target[:, :18] = self.home[7:25]
        self.target[:, 18] = 0.006
        self.audit = torch.zeros((count, 8), device=self.device)
        self.torque = torch.zeros((count, 19), device=self.device)
        self.overflow = torch.zeros(count, device=self.device, dtype=torch.int32)
        self.motor = self.tensor([d["motor_scale"] for d in self.domains])
        self.kp = self.tensor([d["kp_scale"] for d in self.domains])
        self.kd = self.tensor([d["kd_scale"] for d in self.domains])
        self.speed = self.tensor([d["command_speed"] for d in self.domains])
        self.start_x = self.tensor([d["start_x"] for d in self.domains])
        self.start_y = self.tensor([d["start_y"] for d in self.domains])
        self.start_yaw = self.tensor([d["start_yaw"] for d in self.domains])
        self.center_x = self.tensor([d["center_x"] for d in self.domains])
        self.center_y = self.tensor([d["obstacle_y"] for d in self.domains])
        self.yaw = self.tensor([d["obstacle_yaw"] for d in self.domains])
        self.length = self.tensor(
            [d.get("length", d.get("thickness")) for d in self.domains]
        )
        self.width = self.tensor([d["width"] for d in self.domains])
        self.height = self.tensor([d["height"] for d in self.domains])
        self.clearance = self.tensor(
            [self.normal_top - d.get("clearance_drop", 0.0) for d in self.domains]
        )
        if task == "climb":
            self.obstacle = m.geom("step_box_geom").id
            classes[self.obstacle] = 5
        else:
            self.obstacle = m.geom("rail_bar").id
            for name in ("rail_bar", "rail_post_-1", "rail_post_1"):
                classes[m.geom(name).id] = 6
        self._set_domains()
        forbidden = np.zeros((m.ngeom, m.ngeom), np.int32)

        def chain(b):
            result = []
            while b:
                result.append(b)
                b = int(m.body_parentid[b])
            return result + [0]

        chains = {b: chain(b) for b in robot_bodies}
        for g in self.robot_geoms:
            for h in self.robot_geoms:
                a, b = int(m.geom_bodyid[g]), int(m.geom_bodyid[h])
                distance = min(
                    chains[a].index(c) + chains[b].index(c)
                    for c in set(chains[a]) & set(chains[b])
                )
                if distance > 2:
                    forbidden[g, h] = 1
        self.classes_wp = wp.array(classes, dtype=wp.int32, device="cuda:0")
        self.feet_wp = wp.array(feet, dtype=wp.int32, device="cuda:0")
        self.forbidden_wp = wp.array(forbidden, dtype=wp.int32, device="cuda:0")
        self.pd_args = [
            self.wd.qpos,
            self.wd.qvel,
            self.wd.qfrc_bias,
            self.wd.ctrl,
            self.wm.actuator_ctrlrange,
            wp.from_torch(self.target),
            wp.from_torch(self.motor),
            wp.from_torch(self.kp),
            wp.from_torch(self.kd),
            wp.from_torch(self.torque),
        ]
        contact = self.wd.contact
        self.audit_args = [
            self.wm.opt.cone,
            self.wd.nacon,
            contact.worldid,
            contact.geom,
            contact.frame,
            contact.friction,
            contact.dim,
            contact.efc_address,
            self.wd.efc.force,
            self.wd.njmax,
            contact.dist,
            self.classes_wp,
            self.feet_wp,
            self.forbidden_wp,
            wp.from_torch(self.audit),
        ]
        self.history = torch.zeros((count, 5, 60), device=self.device)
        self.last = torch.zeros((count, 12), device=self.device)
        self.age = torch.zeros(count, device=self.device)
        self.dwell = torch.zeros_like(self.age)
        self.lowest = torch.full_like(self.age, self.normal_base)
        self.rail_peak = torch.zeros_like(self.age)
        self.arm_peak = torch.zeros_like(self.age)
        self.under_seen = torch.zeros(count, device=self.device, dtype=torch.bool)
        self.initial_box = torch.zeros((count, 2), device=self.device)
        self.return_ = torch.zeros_like(self.age)
        self.episode_steps = torch.zeros_like(self.age)
        self.lower = self.tensor(m.jnt_range[1:13, 0] + 0.01)
        self.upper = self.tensor(m.jnt_range[1:13, 1] - 0.01)
        self.graph = None
        self.kin_graph = None
        self.reset(torch.ones(count, device=self.device, dtype=torch.bool))
        # Compile kernels before capture, then reset: compile probes are not episodes.
        self._physics()
        self.reset(torch.ones(count, device=self.device, dtype=torch.bool))
        wp.synchronize()
        if capture:
            with wp.ScopedCapture() as cap:
                self._physics()
            self.graph = cap.graph
        self.reset(torch.ones(count, device=self.device, dtype=torch.bool))
        self.obs = self.observe()
        if self.obs.shape != (count, OBS_DIM):
            raise AssertionError(self.obs.shape)

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def _set_domains(self):
        geom_size = wp.to_torch(self.wm.geom_size)
        geom_pos = wp.to_torch(self.wm.geom_pos)
        geom_quat = wp.to_torch(self.wm.geom_quat)
        aabb = wp.to_torch(self.wm.geom_aabb)
        rbound = wp.to_torch(self.wm.geom_rbound)
        friction = wp.to_torch(self.wm.geom_friction)
        friction[:, :, 0] = self.tensor([d["friction"] for d in self.domains])[:, None]
        g = self.obstacle
        if self.task == "climb":
            geom_size[:, g] = torch.stack(
                (self.length / 2, self.width / 2, self.height / 2), -1
            )
            b = self.m.body("step_box").id
            mass = self.tensor([d["mass"] for d in self.domains])
            wp.to_torch(self.wm.body_mass)[:, b] = mass
            wp.to_torch(self.wm.body_inertia)[:, b] = (
                mass[:, None]
                / 12
                * torch.stack(
                    (
                        self.width**2 + self.height**2,
                        self.length**2 + self.height**2,
                        self.length**2 + self.width**2,
                    ),
                    -1,
                )
            )
            mw.set_const(self.wm, self.wd)
        else:
            # Body remains fixed; all randomized obstacle shapes/poses are genuine colliders.
            geom_pos[:, g, 0] = self.center_x - 1
            geom_pos[:, g, 1] = self.center_y
            geom_pos[:, g, 2] = self.clearance + 0.05
            geom_size[:, g] = torch.stack(
                (self.length / 2, self.width / 2, torch.full_like(self.width, 0.05)), -1
            )
            geom_quat[:, g, 0] = torch.cos(self.yaw / 2)
            geom_quat[:, g, 3] = torch.sin(self.yaw / 2)
            for side in (-1, 1):
                p = self.m.geom("rail_post_" + str(side)).id
                geom_pos[:, p, 0] = (
                    self.center_x
                    - 1
                    - side * (self.width / 2 + 0.025) * torch.sin(self.yaw)
                )
                geom_pos[:, p, 1] = self.center_y + side * (
                    self.width / 2 + 0.025
                ) * torch.cos(self.yaw)
                geom_pos[:, p, 2] = (self.clearance + 0.1) / 2
                geom_size[:, p] = torch.stack(
                    (
                        self.length / 2,
                        torch.full_like(self.width, 0.025),
                        (self.clearance + 0.1) / 2,
                    ),
                    -1,
                )
                geom_quat[:, p] = geom_quat[:, g]
                aabb[:, p, 0] = 0
                aabb[:, p, 1] = geom_size[:, p]
                rbound[:, p] = geom_size[:, p].norm(dim=-1)
        aabb[:, g, 0] = 0
        aabb[:, g, 1] = geom_size[:, g]
        rbound[:, g] = geom_size[:, g].norm(dim=-1)

    def _physics(self):
        for _ in range(10):
            wp.launch(pd, dim=(self.n, 19), inputs=self.pd_args)
            mw.step(self.wm, self.wd)
            wp.launch(audit, dim=self.wd.naconmax, inputs=self.audit_args)
            wp.launch(
                overflow_audit,
                dim=self.n,
                inputs=[self.wd.overflow, wp.from_torch(self.overflow)],
            )
        mw.fwd_kinematics(self.wm, self.wd)

    def rand(self, *shape):
        return torch.rand(shape, device=self.device, generator=self.rng)

    def reset(self, mask):
        new = self.home.repeat(self.n, 1)
        new[:, 0] = self.start_x + (self.rand(self.n) - 0.5) * 0.03
        new[:, 1] = self.start_y + (self.rand(self.n) - 0.5) * 0.03
        yaw = self.start_yaw + (self.rand(self.n) - 0.5) * 0.04
        new[:, 3] = torch.cos(yaw / 2)
        new[:, 6] = torch.sin(yaw / 2)
        new[:, 7:19] += (self.rand(self.n, 12) - 0.5) * 0.04
        new[:, 25], new[:, 26] = 0.006, -0.006
        if self.task == "climb":
            new[:, 27] = self.center_x
            new[:, 28] = self.center_y
            new[:, 29] = self.height / 2 + 0.001
            new[:, 30] = torch.cos(self.yaw / 2)
            new[:, 33] = torch.sin(self.yaw / 2)
        self.q.copy_(torch.where(mask[:, None], new, self.q))
        self.v.masked_fill_(mask[:, None], 0.0)
        wp.to_torch(self.wd.qacc_warmstart).masked_fill_(mask[:, None], 0.0)
        wp.to_torch(self.wd.time).masked_fill_(mask, 0.0)
        self.history.masked_fill_(mask[:, None, None], 0.0)
        self.last.masked_fill_(mask[:, None], 0.0)
        for arr in (
            self.age,
            self.dwell,
            self.rail_peak,
            self.arm_peak,
            self.return_,
            self.episode_steps,
        ):
            arr.masked_fill_(mask, 0.0)
        self.lowest[mask] = self.normal_base
        self.under_seen.masked_fill_(mask, False)
        self.target[:, :18] = torch.where(
            mask[:, None], self.home[7:25], self.target[:, :18]
        )
        if self.task == "climb":
            self.initial_box = torch.where(
                mask[:, None], new[:, 27:29], self.initial_box
            )
        mw.fwd_kinematics(self.wm, self.wd)

    def body_local(self, delta):
        yaw = torch.atan2(
            2 * (self.q[:, 3] * self.q[:, 6] + self.q[:, 4] * self.q[:, 5]),
            1 - 2 * (self.q[:, 5] ** 2 + self.q[:, 6] ** 2),
        )
        c, s = torch.cos(yaw), torch.sin(yaw)
        return torch.stack(
            (c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]), -1
        )

    def geometry(self):
        if self.task == "climb":
            center = self.q[:, 27:30]
        else:
            center = torch.stack((self.center_x, self.center_y, self.clearance), -1)
        delta = center[:, :2] - self.q[:, :2]
        local = self.body_local(delta)
        desired_height = torch.full_like(self.age, self.normal_base)
        if self.task == "down":
            near = (self.q[:, 0] > self.center_x - 0.60) & (
                self.q[:, 0] < self.center_x + self.length / 2 + 0.50
            )
            low = (self.normal_base - (self.normal_top - self.clearance) - 0.025).clamp(
                0.115, self.normal_base
            )
            desired_height = torch.where(near, low, desired_height)
        return center, local, desired_height

    def observe(self):
        center, local, desired_height = self.geometry()
        q = self.q[:, 3:7]
        gravity = rotate_inverse(q, self.tensor([0.0, 0.0, -1.0]).expand(self.n, -1))
        # Same 60-feature ordering/units and oldest->newest history as original wrapper.
        cmd = torch.zeros((self.n, 3), device=self.device)
        goal_x = center[:, 0] if self.task == "climb" else self.center_x + 0.8
        cmd[:, 0] = torch.minimum(
            self.speed, (goal_x - self.q[:, 0]).clamp(min=0) * 0.7
        )
        cmd[:, 1] = ((center[:, 1] - self.q[:, 1]) * 0.7).clamp(-0.06, 0.06)
        phase = (
            ((self.age[:, None] + self.dt) * 1.4) + self.tensor([0.0, 0.5, 0.5, 0.0])
        ) % 1.0
        phase = torch.where(
            cmd.norm(dim=-1, keepdim=True) < 0.01, -torch.ones_like(phase), phase
        )
        single = torch.cat(
            (
                rotate_inverse(q, self.v[:, :3]),
                self.v[:, 3:6],
                gravity,
                cmd,
                torch.zeros((self.n, 2), device=self.device),
                (self.q[:, 7:19] - self.home[7:19])[:, MJ_TO_POLICY],
                self.v[:, 6:18][:, MJ_TO_POLICY],
                self.last,
                phase,
                self.q[:, 19:25] - self.home[19:25],
            ),
            -1,
        )
        self.history = torch.cat((self.history[:, 1:], single[:, None]), 1)
        feet_pos = self.xpos[:, self.foot_geoms]
        extras = torch.cat(
            (
                local,
                (center[:, 2] - self.q[:, 2])[:, None],
                torch.stack(
                    (
                        self.length,
                        self.width,
                        self.height,
                        self.clearance,
                        desired_height,
                    ),
                    -1,
                ),
                self.audit[:, :4] / 50.0,
                (feet_pos[:, :, 2] - self.height[:, None]),
                self.q[:, 2:3],
                self.dwell[:, None] / 2.0,
                torch.stack(
                    (
                        torch.full_like(self.age, float(self.task == "climb")),
                        torch.full_like(self.age, float(self.task == "down")),
                    ),
                    -1,
                ),
            ),
            -1,
        )
        return torch.nan_to_num(
            torch.cat((self.history.flatten(1), extras), -1),
            nan=0.0,
            posinf=100.0,
            neginf=-100.0,
        ).clamp(-100.0, 100.0)

    def leg_targets(self, action):
        filtered = 0.8 * action.clamp(-6, 6) + 0.2 * self.last
        targets = self.home[7:19] + 0.5 * filtered[:, POLICY_TO_MJ]
        return torch.maximum(torch.minimum(targets, self.upper), self.lower)

    def step(self, action, auto_reset=True):
        old_position = self.q[:, :2].clone()
        self.target[:, :12] = self.leg_targets(action)
        # All six arm targets stay at the original home for the entire episode.
        self.audit.zero_()
        self.torque.zero_()
        self.overflow.zero_()
        if self.graph is None:
            self._physics()
        else:
            wp.capture_launch(self.graph)
        self.age += self.dt
        self.episode_steps += 1
        self.lowest = torch.minimum(self.lowest, self.q[:, 2])
        self.rail_peak = torch.maximum(self.rail_peak, self.audit[:, 4])
        arm_error = (self.q[:, 19:25] - self.home[19:25]).abs().max(-1).values
        self.arm_peak = torch.maximum(self.arm_peak, arm_error)
        up = 1 - 2 * (self.q[:, 4] ** 2 + self.q[:, 5] ** 2)
        center, local, desired_height = self.geometry()
        body_speed = self.v[:, :3].norm(dim=-1)
        angular_speed = self.v[:, 3:6].norm(dim=-1)
        feet_pos = self.xpos[:, self.foot_geoms]
        if self.task == "climb":
            # Use actual free box orientation, not initial box coordinates.
            box_q = self.q[:, 30:34]
            foot_delta = feet_pos - self.q[:, None, 27:30]
            foot_local = rotate_inverse(
                box_q[:, None].expand(-1, 4, -1).reshape(-1, 4),
                foot_delta.reshape(-1, 3),
            ).reshape(self.n, 4, 3)
            contained = (
                foot_local[:, :, 0].abs() < self.length[:, None] / 2 - 0.025
            ) & (foot_local[:, :, 1].abs() < self.width[:, None] / 2 - 0.025)
            on_top = (foot_local[:, :, 2] > self.height[:, None] / 2 + 0.005) & (
                foot_local[:, :, 2] < self.height[:, None] / 2 + 0.05
            )
            support = (self.audit[:, :4] > 0.5) & contained & on_top
            box_up = 1 - 2 * (box_q[:, 1] ** 2 + box_q[:, 2] ** 2)
            box_speed = self.v[:, 26:29].norm(dim=-1)
            ready = (
                support.all(-1)
                & (body_speed < 0.06)
                & (angular_speed < 0.25)
                & (up > 0.97)
                & (box_up > math.cos(0.10))
                & (box_speed < 0.03)
                & (arm_error < 0.12)
            )
            hold = 2.0
            near_goal = ((self.q[:, :2] - center[:, :2]) ** 2).sum(-1)
            elevation = ((self.q[:, 2] - self.normal_base) / self.height).clamp(0, 1)
            progress = (old_position - center[:, :2]).norm(dim=-1) - (
                self.q[:, :2] - center[:, :2]
            ).norm(dim=-1)
            reward = (
                12 * progress
                + 0.04 * support.float().sum(-1)
                + 0.025 * elevation
                + 0.025 * torch.exp(-near_goal / 0.10)
                + 0.10 * ready.float()
            )
            height_error = (
                self.q[:, 2] - (self.normal_base + elevation.detach() * self.height)
            ).square()
            obstacle_failed = (
                (box_up < math.cos(0.25))
                | ((self.q[:, 27:29] - self.initial_box).norm(dim=-1) > 0.25)
                | (self.q[:, 0] > center[:, 0] + self.length / 2 + 0.3)
            )
        else:
            # Full collision envelope: use current rotation-aware primitive extents below.
            bounds_min, bounds_max = self.robot_bounds(rail_frame=True)
            relative = self.q[:, :2] - torch.stack((self.center_x, self.center_y), -1)
            local_x = relative[:, 0] * torch.cos(self.yaw) + relative[:, 1] * torch.sin(
                self.yaw
            )
            inside = (
                (bounds_min[:, 1] > -self.width / 2 + 0.015)
                & (bounds_max[:, 1] < self.width / 2 - 0.015)
                & (bounds_max[:, 2] < self.clearance - 0.005)
            )
            self.under_seen |= (local_x.abs() < self.length / 2) & inside
            ready = (
                (bounds_min[:, 0] > self.length / 2 + 0.12)
                & self.under_seen
                & (self.lowest < self.normal_base - 0.04)
                & ((self.q[:, 2] - self.normal_base).abs() < 0.04)
                & (body_speed < 0.07)
                & (angular_speed < 0.25)
                & (up > 0.97)
                & (arm_error < 0.12)
                & (self.rail_peak < 2.0)
            )
            hold = 1.0
            height_error = (self.q[:, 2] - desired_height).square()
            goal = self.center_x + 0.8
            distance = (goal - self.q[:, 0]).abs()
            progress = (goal - old_position[:, 0]).abs() - distance
            reward = (
                12 * progress
                + 0.04 * torch.exp(-height_error / 0.003)
                + 0.02 * torch.exp(-distance.square() / 0.1)
                + 0.10 * ready.float()
            )
            obstacle_failed = torch.zeros_like(ready)
        self.dwell = torch.where(
            ready, self.dwell + self.dt, torch.zeros_like(self.dwell)
        )
        success = self.dwell >= hold
        finite = torch.isfinite(self.q).all(-1) & torch.isfinite(self.v).all(-1)
        forbidden = self.audit[:, 4] > (5.0 if self.task == "climb" else 2.0)
        failure_reasons = torch.stack(
            (
                ~finite,
                (up < 0.65) | (self.q[:, 2] < 0.09),
                self.arm_peak > 0.25,
                forbidden,
                self.audit[:, 5] > 10.0,
                obstacle_failed,
                (self.q[:, 1].abs() > 1.0) | (self.q[:, 0] < -0.6),
            ),
            -1,
        )
        failure = failure_reasons.any(-1)
        success &= ~failure
        timeout = (self.age >= 20.0) & ~success & ~failure
        reward -= (
            0.005
            + 0.01 * (1 - up)
            + 0.2 * height_error
            + 0.000003 * (self.torque[:, :12] ** 2).sum(-1)
            + 0.001 * ((action - self.last) ** 2).sum(-1)
        )
        reward += 10 * success.float() - 5 * failure.float()
        reward = torch.nan_to_num(reward, nan=-5.0, posinf=-5.0, neginf=-5.0)
        self.return_ += reward
        self.last = action.clamp(-6, 6).detach().clone()
        done = success | failure | timeout
        terminal_obs = self.observe()
        info = {
            "success": success,
            "failure": failure,
            "failure_reasons": failure_reasons,
            "base_x": self.q[:, 0].clone(),
            "base_z": self.q[:, 2].clone(),
            "physics_overflow": self.overflow.clone(),
            "timeout": timeout,
            "done": done,
            "return": self.return_.clone(),
            "duration": self.age.clone(),
            "arm_error": self.arm_peak.clone(),
            "forbidden_peak": self.rail_peak.clone(),
            "self_peak": self.audit[:, 5].clone(),
            "difficulty_height": self.height,
            "terminal_obs": terminal_obs,
            "dwell": self.dwell.clone(),
            "under_seen": self.under_seen.clone(),
            "support_count": (self.audit[:, :4] > 0.5).sum(-1),
        }
        if auto_reset:
            self.reset(done)
            # Do not append an extra history frame for continuing episodes.
            old_history = self.history.clone()
            new_obs = self.observe()
            self.history.copy_(
                torch.where(done[:, None, None], self.history, old_history)
            )
            self.obs = torch.where(done[:, None], new_obs, terminal_obs)
        else:
            self.obs = terminal_obs
        return self.obs, reward, done, info

    def robot_bounds(self, rail_frame=False):
        # Primitive-specific support functions, valid for arbitrary live orientations.
        ids = self.robot_geoms
        mats = self.xmat[:, ids].reshape(self.n, len(ids), 3, 3)
        positions = self.xpos[:, ids]
        if rail_frame:
            c, s = torch.cos(self.yaw), torch.sin(self.yaw)
            r = torch.zeros((self.n, 3, 3), device=self.device)
            r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1], r[:, 2, 2] = (
                c,
                s,
                -s,
                c,
                1.0,
            )
            positions = (
                positions
                - torch.stack((self.center_x, self.center_y, torch.zeros_like(c)), -1)[
                    :, None
                ]
            )
            positions = torch.einsum("nij,nkj->nki", r, positions)
            mats = torch.einsum("nij,nkjl->nkil", r, mats)
        sizes = wp.to_torch(self.wm.geom_size)[:, ids]
        kinds = self.m.geom_type[ids]
        ext = torch.zeros((self.n, len(ids), 3), device=self.device)
        for kind in np.unique(kinds):
            mask = np.flatnonzero(kinds == kind)
            r, s = mats[:, mask], sizes[:, mask]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                ext[:, mask] = (r.abs() * s[:, :, None, :]).sum(-1)
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                ext[:, mask] = s[:, :, :1]
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                ext[:, mask] = s[:, :, :1] + r[:, :, :, 2].abs() * s[:, :, 1:2]
            else:
                ext[:, mask] = self.tensor(self.m.geom_rbound[ids])[None, mask, None]
        return (positions - ext).min(1).values, (positions + ext).max(1).values

    def save_domains(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "normal_collision_top": self.normal_top,
                    "normal_base": self.normal_base,
                    "domains": self.domains,
                },
                indent=2,
            )
            + "\n"
        )
