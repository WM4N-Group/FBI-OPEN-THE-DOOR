import os

os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import mujoco, numpy as np, torch
from scipy.spatial.transform import Rotation
from ..paths import RUNTIME_ROOT as ROOT
from ..paths import ensure_runtime

ensure_runtime()
REF = ROOT / "reference/third_party/locomotion_policy/deploy"
if not (REF / "config.py").is_file():
    raise RuntimeError(
        "Missing private reference assets. Run python scripts/import_reference.py --archive PATH first."
    )
sys.path[:0] = [str(REF), str(REF / "mujoco")]
import mujoco_utils as utils, locomotion_policy_wrapper as wrapper
import onnxruntime as ort

_original = ort.InferenceSession


def _session(path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return _original(path, sess_options=opts, providers=["CPUExecutionProvider"])


wrapper.ort.InferenceSession = _session
torch.set_num_threads(1)


class Physics:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/door_scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.work = mujoco.MjData(self.model)
        self.site = self.model.site("attachment_site").id
        self.hinge = self.model.joint("door_hinge")
        self.handle = self.model.joint("handle_hinge")
        self.latch = self.model.equality("physical_latch").id
        baseid = self.model.body("base").id

        def on_robot(bid):
            while bid > 0:
                if bid == baseid:
                    return True
                bid = self.model.body_parentid[bid]
            return False

        self.robot_collision_geoms = [
            i
            for i in range(self.model.ngeom)
            if self.model.geom_contype[i] and on_robot(int(self.model.geom_bodyid[i]))
        ]
        self.arm_jids = [self.model.joint("joint" + str(i)).id for i in range(1, 7)]
        self.arm_q = np.array([self.model.jnt_qposadr[i] for i in self.arm_jids])
        self.arm_v = np.array([self.model.jnt_dofadr[i] for i in self.arm_jids])
        self.reset()

    def reset(self, seed=0):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.data.qpos[1] = -0.175
        self.data.eq_active[self.latch] = 1
        mujoco.mj_forward(self.model, self.data)
        self.home = self.data.qpos.copy()
        self.ctrl = wrapper.LocomotionPolicyWrapper(self.model)
        self.handle_contact = False
        self.unlocked = False
        self.gripper_target = 0.04
        self.finger_forces = np.zeros(2)
        self.bilateral = False
        self.grasp_hold_s = 0.0
        self.grasp_valid = False
        self.grasp_ever_valid = False
        self.grasp_speed = 0.0
        self.contact_loss_s = 0.0
        self.leg_target = self.home[7:19].copy()
        self.arm_target = self.home[self.arm_q].copy()

    def ik(self, position, orientation=None, iterations=60):
        """Full pose DLS when orientation supplied; position-only for feasibility checks."""
        d = self.work
        m = self.model
        d.qpos[:] = self.data.qpos
        qmin = m.jnt_range[self.arm_jids, 0] + 0.015
        qmax = m.jnt_range[self.arm_jids, 1] - 0.015
        jp = np.zeros((3, m.nv))
        jr = jp.copy()
        for _ in range(iterations):
            mujoco.mj_forward(m, d)
            errp = np.asarray(position) - d.site_xpos[self.site]
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            if orientation is None:
                err = errp
                J = jp[:, self.arm_v]
            else:
                errr = Rotation.from_matrix(
                    np.asarray(orientation) @ d.site_xmat[self.site].reshape(3, 3).T
                ).as_rotvec()
                err = np.r_[errp, 0.25 * errr]
                J = np.vstack([jp[:, self.arm_v], 0.25 * jr[:, self.arm_v]])
            if np.linalg.norm(err) < 0.003:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0025 * np.eye(J.shape[0]), err)
            d.qpos[self.arm_q] = np.clip(
                d.qpos[self.arm_q] + np.clip(dq, -0.12, 0.12), qmin, qmax
            )
        mujoco.mj_forward(m, d)
        self.ik_rotation_error = (
            0.0
            if orientation is None
            else float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        np.asarray(orientation) @ d.site_xmat[self.site].reshape(3, 3).T
                    ).as_rotvec()
                )
            )
        )
        return d.qpos[self.arm_q].copy(), float(
            np.linalg.norm(np.asarray(position) - d.site_xpos[self.site])
        )

    def policy_step(self, command, arm):
        d = self.data
        lin, ang = utils.target_base_vel(
            d, np.r_[command[:2], 0], command[2], frame="world"
        )
        target = self.ctrl.compute_control(
            base_pos=d.qpos[:3].copy(),
            base_ori_euler_xyz=utils.base_ori_euler_xyz(d),
            base_quat_wxyz=d.qpos[3:7].copy(),
            base_lin_vel=utils.base_lin_vel(d, "base"),
            base_ang_vel=utils.base_ang_vel(d, "base"),
            heading_orientation_SO3=utils.heading_orientation_SO3(d),
            joints_pos_leg=d.qpos[7:19].copy(),
            joints_vel_leg=d.qvel[6:18].copy(),
            ref_base_lin_vel=lin,
            ref_base_ang_vel=ang,
            ref_pose_command=np.zeros(2),
            joints_pos_arm=d.qpos[self.arm_q].copy(),
        )
        self.step_pd(target, arm)

    def step_pd(self, legs, arm, kp=20, kd=1.5):
        d = self.data
        m = self.model
        self.handle_contact = False
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
            mujoco.mj_step(m, d)
            self.finger_forces[:] = 0
            self.handle_contact = False
            for ci, c in enumerate(d.contact):
                names = [m.geom(g).name for g in [c.geom1, c.geom2]]
                if any(n in ["handle_lever", "handle_hub"] for n in names):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, force)
                    for k, pad in enumerate(["finger_left_pad", "finger_right_pad"]):
                        if pad in names:
                            self.finger_forces[k] += max(0, float(force[0]))
                    if any(n in ["finger_left_pad", "finger_right_pad"] for n in names):
                        self.handle_contact = True
            self.bilateral = bool(np.all(self.finger_forces > 0.15))
            # A contact normal alone is not a stable grasp: also constrain relative slip.
            hv = np.zeros(6)
            tv = np.zeros(6)
            mujoco.mj_objectVelocity(
                m, d, mujoco.mjtObj.mjOBJ_SITE, m.site("handle_grasp").id, hv, 0
            )
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, self.site, tv, 0)
            self.grasp_speed = float(np.linalg.norm(hv[3:] - tv[3:]))
            if (
                self.bilateral
                and self.grasp_speed < 0.07
                and self.gripper_target < 0.018
            ):
                self.grasp_hold_s += m.opt.timestep
            else:
                self.grasp_hold_s = 0.0
            self.grasp_valid = bool(self.grasp_hold_s >= 0.25)
            self.grasp_ever_valid |= self.grasp_valid
            self.contact_loss_s = (
                0.0 if self.bilateral else self.contact_loss_s + m.opt.timestep
            )
            if self.grasp_valid and d.qpos[self.handle.qposadr[0]] < -0.32:
                self.unlocked = True
                d.eq_active[self.latch] = 0

    def prepare(self):
        for _ in range(50):
            self.step_pd(self.home[7:19], self.home[self.arm_q], 60, 5)


class RGBD:
    def __init__(self, physics, width=160, height=120, camera="head_rgbd"):
        self.p = physics
        self.w = width
        self.h = height
        self.r = mujoco.Renderer(physics.model, height=height, width=width)
        self.sensor_options = mujoco.MjvOption()
        self.sensor_options.sitegroup[:] = (
            0  # no debug TCP/handle markers in sensor RGB
        )
        self.camera_name = camera
        self.camera = physics.model.camera(camera).id
        self.ids = [physics.model.geom(n).id for n in ["handle_lever", "handle_hub"]]

    def capture(self, labels=False):
        self.r.update_scene(
            self.p.data, camera=self.camera_name, scene_option=self.sensor_options
        )
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(
            self.p.data, camera=self.camera_name, scene_option=self.sensor_options
        )
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        mask = None
        if labels:
            self.r.enable_segmentation_rendering()
            self.r.update_scene(
                self.p.data, camera=self.camera_name, scene_option=self.sensor_options
            )
            seg = self.r.render().copy()
            self.r.disable_segmentation_rendering()
            mask = np.isin(seg[:, :, 0], self.ids) & (
                seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            )
        return rgb, depth, mask

    def pose(self, mask, depth):
        """Estimated visible handle pose from predicted RGB mask + metric depth only."""
        v, u = np.where(mask & np.isfinite(depth) & (depth > 0.05) & (depth < 2.5))
        if len(v) < 6:
            return None
        z = depth[v, u]
        f = (
            0.5
            * self.h
            / np.tan(np.deg2rad(float(self.p.model.cam_fovy[self.camera])) / 2)
        )
        optical = np.stack(
            [(u - (self.w - 1) / 2) * z / f, -(v - (self.h - 1) / 2) * z / f, -z],
            axis=1,
        )
        rot = self.p.data.cam_xmat[self.camera].reshape(3, 3)
        pts = optical @ rot.T + self.p.data.cam_xpos[self.camera]
        center = np.median(pts, axis=0)
        cov = (pts - center).T @ (pts - center)
        _, vec = np.linalg.eigh(cov)
        axis = vec[:, -1]
        if axis[1] < 0:
            axis = -axis
        # Depth geometry lever axis plus calibrated gravity; normal sign faces camera.
        normal = np.cross(axis, np.array([0, 0, 1.0]))
        normal /= max(np.linalg.norm(normal), 1e-8)
        if np.dot(normal, self.p.data.cam_xpos[self.camera] - center) < 0:
            normal = -normal
        up = np.cross(normal, axis)
        up /= max(np.linalg.norm(up), 1e-8)
        grasp = center + axis * (np.quantile(pts @ axis, 0.80) - np.dot(center, axis))
        return {
            "position": center,
            "grasp_position": grasp,
            "rotation": np.column_stack([normal, axis, up]),
            "pixels": len(v),
        }

    def close(self):
        self.r.close()
