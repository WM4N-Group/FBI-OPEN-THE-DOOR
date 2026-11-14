"""Geometry-distance guard independent of MuJoCo collision bitmasks.

Uses retained collision proxies, not visual meshes. Mechanically neighbouring
arm bodies (tree distance<=2) are excluded explicitly. This is discretized
simulation checking, NOT continuous/hardware collision-safety certification.
"""

import mujoco
import numpy as np
from ..box_clearance import SELF_MARGIN


class BoxCollisionGuard:
    def __init__(self, physics):
        self.p = p = physics
        self.m = m = p.model
        self.work = mujoco.MjData(m)
        self.arm = sorted(g for g in p.arm_geoms if m.geom_contype[g])
        self.body = sorted(g for g in p.robot_geoms - p.arm_geoms if m.geom_contype[g])
        self.pads = {m.geom(n).id for n in ("finger_left_pad", "finger_right_pad")}
        self.protected = [g for g in self.arm if g not in self.pads]
        self.self_pairs = []
        for i in self.arm:
            for j in self.arm + self.body:
                if i == j or (j in self.arm and j < i):
                    continue
                if self.tree_distance(int(m.geom_bodyid[i]), int(m.geom_bodyid[j])) > 2:
                    self.self_pairs.append((i, j))
        self.pair_set = {tuple(sorted(pair)) for pair in self.self_pairs}
        self.categories = {
            "self": self.self_pairs,
            "body_box": [(g, p.box_geom) for g in self.body],
            "protected_box": [(g, p.box_geom) for g in self.protected],
            "pad_box": [(g, p.box_geom) for g in sorted(self.pads)],
            "arm_floor": [(g, m.geom("ground").id) for g in self.arm],
        }
        self.arrays = {k: np.array(v, dtype=int) for k, v in self.categories.items()}
        self.minimum = {k: 1.0 for k in self.categories}
        self.nearest_self_pair = None
        self.self_force_peak = 0.0
        self.forbidden_box_force_peak = 0.0
        self.rejections = 0
        self.reset()

    def tree_distance(self, i, j):
        def ancestors(b):
            a = [b]
            while b:
                b = int(self.m.body_parentid[b])
                a.append(b)
            return a

        a, b = ancestors(i), ancestors(j)
        return min(a.index(x) + b.index(x) for x in set(a) & set(b))

    def reset(self):
        self.minimum = {k: 1.0 for k in self.categories}
        self.nearest_self_pair = None
        self.self_force_peak = 0.0
        self.forbidden_box_force_peak = 0.0
        self.rejections = 0

    def distances(self, data, limit=0.3):
        values = {}
        nearest = None
        for name, pairs in self.arrays.items():
            best = limit
            # Bounding spheres reject only provably distant pairs; plane is unbounded.
            a, b = pairs[:, 0], pairs[:, 1]
            lower = (
                np.linalg.norm(data.geom_xpos[a] - data.geom_xpos[b], axis=1)
                - self.m.geom_rbound[a]
                - self.m.geom_rbound[b]
            )
            indices = (
                range(len(pairs))
                if name == "arm_floor"
                else np.flatnonzero(lower < limit)
            )
            for k in indices:
                i, j = pairs[k]
                dist = float(
                    mujoco.mj_geomDistance(self.m, data, int(i), int(j), best, None)
                )
                if dist < best:
                    best = dist
                    if name == "self":
                        nearest = (int(i), int(j))
            values[name] = best
        return values, nearest

    def observe(self):
        self.work.qpos[:] = self.p.data.qpos
        mujoco.mj_kinematics(self.m, self.work)
        values, pair = self.distances(self.work)
        if values["self"] < self.minimum["self"]:
            self.nearest_self_pair = pair
        for name, v in values.items():
            self.minimum[name] = min(self.minimum[name], v)
        return values

    def admissible(self, values, touching=False):
        return (
            values["self"] >= SELF_MARGIN
            and values["body_box"] >= 0.16
            and values["protected_box"] >= 0.035
            and values["arm_floor"] >= 0.07
            and values["pad_box"] >= (-0.003 if touching else 0.025)
        )

    def swept_arm(self, candidate, touching=False):
        """Check actual->command segment at <=0.01 rad joint spacing, including endpoints."""
        p = self.p
        start = p.data.qpos[p.arm_q].copy()
        n = max(2, int(np.ceil(np.max(np.abs(candidate - start)) / 0.01)) + 1)
        self.work.qpos[:] = p.data.qpos
        for alpha in np.linspace(0, 1, n):
            self.work.qpos[p.arm_q] = (1 - alpha) * start + alpha * candidate
            mujoco.mj_kinematics(self.m, self.work)
            values, _ = self.distances(self.work, limit=0.17)
            if not self.admissible(values, touching):
                self.rejections += 1
                return False
        return True

    def contacts(self):
        """Substep contact forces for excluded physical interactions; caller chooses audit cadence."""
        d = self.p.data
        for ci, c in enumerate(d.contact):
            pair = tuple(sorted((int(c.geom1), int(c.geom2))))
            forbidden_box = self.p.box_geom in pair and any(
                g in self.protected for g in pair
            )
            if pair not in self.pair_set and not forbidden_box:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(self.m, d, ci, f)
            if pair in self.pair_set:
                self.self_force_peak = max(self.self_force_peak, float(f[0]))
            if forbidden_box:
                self.forbidden_box_force_peak = max(
                    self.forbidden_box_force_peak, float(f[0])
                )

    def configuration_ok(self, q):
        self.work.qpos[:] = self.p.data.qpos
        self.work.qpos[self.p.arm_q] = q
        mujoco.mj_kinematics(self.m, self.work)
        values, _ = self.distances(self.work, limit=0.17)
        return self.admissible(values)

    def segment_ok(self, start, goal):
        n = max(2, int(np.ceil(np.max(np.abs(goal - start)) / 0.025)) + 1)
        return all(
            self.configuration_ok((1 - a) * start + a * goal)
            for a in np.linspace(0, 1, n)
        )

    def plan(self, start, goal, seed=0):
        """Bounded joint-space RRT fallback; never relax collision margins to get a path."""
        if not self.configuration_ok(start) or not self.configuration_ok(goal):
            raise RuntimeError("Collision-invalid arm path endpoint")
        if self.segment_ok(start, goal):
            return [start.copy(), goal.copy()]
        rng = np.random.default_rng(seed)
        nodes = [start.copy()]
        parents = [-1]
        limits = self.m.jnt_range[self.p.arm_jids]
        for _ in range(700):
            target = (
                goal
                if rng.random() < 0.35
                else rng.uniform(limits[:, 0] + 0.015, limits[:, 1] - 0.015)
            )
            nearest = int(np.argmin(np.linalg.norm(np.asarray(nodes) - target, axis=1)))
            delta = target - nodes[nearest]
            q = nodes[nearest] + delta * min(
                1.0, 0.22 / max(np.linalg.norm(delta), 1e-9)
            )
            if not self.segment_ok(nodes[nearest], q):
                continue
            nodes.append(q)
            parents.append(nearest)
            if np.linalg.norm(q - goal) < 0.4 and self.segment_ok(q, goal):
                path = [goal.copy()]
                i = len(nodes) - 1
                while i >= 0:
                    path.append(nodes[i])
                    i = parents[i]
                path.reverse()
                return path
        raise RuntimeError("Bounded collision-free arm planner exhausted")
