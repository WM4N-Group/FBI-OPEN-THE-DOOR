"""No-learning height ablation: unchanged controller vs geometric IK retargeting."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .box_clearance_env import ClearanceBoxPushEnv, TOOL_FORWARD
from ..box_height import contact_height
from ..box_task import LEG_CHECKPOINT_SHA256
from ..paths import RUNTIME_ROOT as ROOT
from ..storage import atomic_json


class HeightRetargetedEnv(ClearanceBoxPushEnv):
    """Only the low IK target is height-aware; all v2 guards/limits/phase logic inherited."""

    top_margin_m = 0.04
    tool_tilt_deg = 30.0

    def _reach(self):
        p, d = self.p, self.p.data
        if self.reach_offset is None:
            return
        if self.planned_path is None:
            half = p.model.geom_size[p.box_geom]
            face = self.initial_box[0] - float(half[0])
            z = contact_height(2 * float(half[2]), self.top_margin_m)
            low = np.array([face - 0.09, self.initial_box[1], z]) + self.reach_offset
            high = low.copy()
            high[2] = 0.63
            try:
                orientation = (
                    TOOL_FORWARD
                    if self.tool_tilt_deg == 30.0
                    else Rotation.from_euler(
                        "y", 90 + self.tool_tilt_deg, degrees=True
                    ).as_matrix()
                )
                q_high, _ = p.solve_pose(high, orientation)
                q_low, self.ik_error = p.solve_pose(low, orientation)
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


class PoseRetargetedEnv(HeightRetargetedEnv):
    """Geometry-only early stand-off and tool-pitch proposal; original safety gates retained."""

    standoff_m = 0.70
    tool_tilt_deg = 45.0
    top_margin_m = 0.02

    def step(self, action):
        if self.phase == 0:
            d = self.p.data
            gap = float(
                self.box_pos[0] - self.p.model.geom_size[self.p.box_geom, 0] - d.qpos[0]
            )
            braking = max(0.0, float(d.qvel[0]), float(self.base_command[0])) ** 2 / (
                2 * 0.06
            )
            yaw = Rotation.from_quat(d.qpos[[4, 5, 6, 3]]).as_euler("xyz")[2]
            if (
                gap <= self.standoff_m + braking + 0.015
                and abs(self.box_pos[1] - d.qpos[1]) < 0.04
                and abs(yaw) < 0.08
                and self.t > 1
            ):
                self._advance()  # request the existing, acceleration-limited settle phase earlier
        return super().step(action)


class ContactFaces:
    """Read-only pad contact audit at the existing 2ms callback; no force/state writes."""

    def __init__(self, env):
        self.p = env.p
        self.pads = env.audit.pads
        self.original = env.audit.contacts
        env.audit.contacts = self.observe
        self.reset()

    def reset(self):
        self.impulse = {name: 0.0 for name in ("front", "top", "other")}
        self.seconds = {name: 0.0 for name in self.impulse}

    def observe(self):
        self.original()
        p = self.p
        m, d = p.model, p.data
        groups = {name: 0.0 for name in self.impulse}
        R = d.geom_xmat[p.box_geom].reshape(3, 3)
        for ci, c in enumerate(d.contact):
            if p.box_geom not in (c.geom1, c.geom2):
                continue
            other = int(c.geom2 if c.geom1 == p.box_geom else c.geom1)
            if other not in self.pads:
                continue
            local = R.T @ (c.pos - d.geom_xpos[p.box_geom])
            normal = R.T @ c.frame[:3]
            axis = int(np.argmax(np.abs(normal)))
            face = (
                "front"
                if axis == 0 and local[0] < 0
                else ("top" if axis == 2 and local[2] > 0 else "other")
            )
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, f)
            groups[face] += max(0.0, float(f[0]))
        for face, force in groups.items():
            self.impulse[face] += force * m.opt.timestep
            if force > 0.15:
                self.seconds[face] += m.opt.timestep

    def report(self):
        total = sum(self.impulse.values())
        return {
            "pad_impulse_Ns_by_face": self.impulse.copy(),
            "pad_contact_s_by_face": self.seconds.copy(),
            "front_impulse_fraction": self.impulse["front"] / total
            if total > 0
            else 0.0,
            "classification": "dominant box-local contact-normal axis + contact-point side, per2ms; edges approximate",
        }


def policy_fingerprint(model):
    h = hashlib.sha256()
    for name, value in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--checkpoint", type=Path, required=True)
    pa.add_argument("--expected-checkpoint-sha256", required=True)
    pa.add_argument(
        "--modes",
        nargs="+",
        choices=["strict_ppo", "height_ppo", "height_ik", "pose_ppo", "pose_ik"],
        default=["strict_ppo", "height_ppo", "height_ik"],
    )
    pa.add_argument("--count", type=int, default=8)
    pa.add_argument("--seed", type=int, default=920100)
    pa.add_argument("--top-margin", type=float, default=0.04)
    pa.add_argument("--pose-top-margin", type=float, default=0.02)
    pa.add_argument("--standoff", type=float, default=0.70)
    pa.add_argument("--tilt-deg", type=float, default=45.0)
    pa.add_argument("--output", type=Path, required=True)
    pa.add_argument("--video", type=Path)
    a = pa.parse_args()
    if a.count < 1 or len(set(a.modes)) != len(a.modes):
        pa.error("Positive count and unique modes required")
    if (
        not 0.63 <= a.standoff <= 0.78
        or not 20 <= a.tilt_deg <= 60
        or not 0.015 <= a.pose_top_margin <= 0.06
    ):
        pa.error("Pose proposal outside this ablation range")
    if a.video and len(a.modes) != 1:
        pa.error("Video requires exactly one mode")
    for p in (a.output, a.video):
        if p:
            if p.exists():
                raise FileExistsError(p)
            p.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha != a.expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA mismatch")
    leg = ROOT / "reference/models/policies/arm_moving_clock/model_7999.pt"
    if hashlib.sha256(leg.read_bytes()).hexdigest() != LEG_CHECKPOINT_SHA256:
        raise ValueError("Frozen leg checkpoint mismatch")
    from stable_baselines3 import PPO

    model = PPO.load(a.checkpoint, device="cpu")
    before = policy_fingerprint(model)
    steps_before = model.num_timesteps
    height_manifest = json.loads((ROOT / "height_manifest.json").read_text())
    height = float(height_manifest["box_size_m"][2])
    contact_height(height, a.top_margin)
    results = []
    for mode in a.modes:
        cls = (
            ClearanceBoxPushEnv
            if mode == "strict_ppo"
            else (
                PoseRetargetedEnv if mode.startswith("pose_") else HeightRetargetedEnv
            )
        )
        env = cls(randomization=1.0)
        if isinstance(env, PoseRetargetedEnv):
            env.top_margin_m = a.pose_top_margin
            env.standoff_m = a.standoff
            env.tool_tilt_deg = a.tilt_deg
        elif isinstance(env, HeightRetargetedEnv):
            env.top_margin_m = a.top_margin
        if (
            env.observation_space.shape != model.observation_space.shape
            or env.action_space.shape != model.action_space.shape
        ):
            raise ValueError("High-level policy observation/action contract changed")
        faces = ContactFaces(env)
        writer = None
        if a.video:
            import cv2

            writer = cv2.VideoWriter(
                str(a.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720)
            )
            if not writer.isOpened():
                raise RuntimeError("Video encoding unavailable")
        try:
            for n in range(a.count):
                obs, _ = env.reset(seed=a.seed + n)
                faces.reset()
                last_phase = None
                for _ in range(1100):
                    action = (
                        np.zeros(6)
                        if mode.endswith("_ik")
                        else model.predict(obs, deterministic=True)[0]
                    )
                    obs, _, term, trunc, info = env.step(action)
                    if info["phase"] != last_phase:
                        print(
                            "PHASE",
                            mode,
                            a.seed + n,
                            round(env.t, 2),
                            info["phase"],
                            flush=True,
                        )
                        last_phase = info["phase"]
                    if writer and n == 0:
                        import cv2

                        frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            frame,
                            f"SIM | H={height:.2f}m | {mode} | NO RETRAINING | privileged geometry",
                            (16, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (20, 20, 20),
                            2,
                        )
                        cv2.putText(
                            frame,
                            f"{info['phase']} | t={env.t:.1f}s | box dx={info['box_displacement_m']:.2f}m",
                            (16, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (20, 20, 20),
                            2,
                        )
                        writer.write(frame)
                    if term or trunc:
                        break
                else:
                    raise RuntimeError("Episode exceeded its configured horizon")
                contact = faces.report()
                row = {
                    "mode": mode,
                    "seed": a.seed + n,
                    **info,
                    **contact,
                    "front_push_accepted": bool(
                        info["is_success"]
                        and contact["front_impulse_fraction"] >= 0.9
                        and contact["pad_contact_s_by_face"]["front"] >= 0.4
                    ),
                    "nominal_ik_contact_height_m": 0.34
                    if mode == "strict_ppo"
                    else contact_height(height, env.top_margin_m),
                    "geometric_tool_tilt_deg": env.tool_tilt_deg
                    if isinstance(env, HeightRetargetedEnv)
                    else 30.0,
                    "geometric_approach_standoff_m": env.standoff_m
                    if isinstance(env, PoseRetargetedEnv)
                    else 0.63,
                    "final_reach_offset_m": None
                    if env.reach_offset is None
                    else env.reach_offset.tolist(),
                }
                results.append(row)
                print("RESULT", json.dumps(row), flush=True)
        finally:
            if writer:
                writer.release()
            env.close()
    after = policy_fingerprint(model)
    assert before == after and steps_before == model.num_timesteps
    assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    result = {
        "height_manifest": height_manifest,
        "height_m": height,
        "top_margin_m": a.top_margin,
        "pose_top_margin_m": a.pose_top_margin,
        "pose_standoff_m": a.standoff,
        "pose_tool_tilt_deg": a.tilt_deg,
        "frozen_leg_checkpoint_sha256": LEG_CHECKPOINT_SHA256,
        "checkpoint_sha256": checkpoint_sha,
        "policy_parameters_before": before,
        "policy_parameters_after": after,
        "training_steps_before": steps_before,
        "training_steps_after": model.num_timesteps,
        "learning_performed": False,
        "summary": {
            mode: {
                "task_passed": sum(
                    r["is_success"] for r in results if r["mode"] == mode
                ),
                "front_push_passed": sum(
                    r["front_push_accepted"] for r in results if r["mode"] == mode
                ),
                "total": a.count,
            }
            for mode in a.modes
        },
        "results": results,
        "scope": "Height-only, fixed footprint2kg/friction .35, same small initial-pose perturbations; NOT vision or stepping/load-bearing validation",
        "adaptation_note": "strict_ppo leaves controller/weights unchanged; height modes retarget only low IK height; pose modes additionally request earlier stand-off and tool-pitch targets; no v2 safety gate is weakened",
    }
    atomic_json(a.output, result)
    print("SUMMARY", json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
