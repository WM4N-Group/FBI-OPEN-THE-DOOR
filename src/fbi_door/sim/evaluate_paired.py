"""Same-condition paired rules/IK vs trained PPO-residual evaluation.
Privileged handle geometry, one door family and small reset-pose variation only.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import hashlib
import json
import re
import time
import subprocess
from pathlib import Path
import numpy as np
import torch
import mujoco
import cv2
from stable_baselines3 import PPO
from .door_env import DoorEnv
from ..paths import RUNTIME_ROOT, artifact_hashes
from ..storage import atomic_json, valid_checkpoint_bytes
from ..evaluation import summarize_paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trusted PPO ZIP; never load an untrusted pickle",
    )
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--policy-seed",
        type=int,
        help="Provenance label only, not used to select a checkpoint",
    )
    parser.add_argument(
        "--selection-note",
        default="User-provided candidate; selection provenance not asserted by this command",
    )
    parser.add_argument("--seed-start", type=int, default=990000)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument(
        "--output-name", default="paired_eval_" + time.strftime("%Y%m%d_%H%M%S")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the first candidate episode at 1x speed; requires ffmpeg",
    )
    a = parser.parse_args()
    if a.seed_start < 0 or not 1 <= a.episodes <= 10000:
        parser.error("Need a nonnegative starting seed and 1..10000 paired episodes")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", a.output_name):
        parser.error(
            "Use a fresh simple output name, not an absolute path or traversal"
        )
    checkpoint = Path(a.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error("Checkpoint does not exist")
    data = checkpoint.read_bytes()
    valid_checkpoint_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if a.expected_sha256 and digest != a.expected_sha256:
        parser.error("Checkpoint SHA256 mismatch")
    torch.set_num_threads(1)
    OUT = RUNTIME_ROOT / "reports" / a.output_name
    OUT.mkdir(parents=True, exist_ok=False)
    seeds = list(range(a.seed_start, a.seed_start + a.episodes))
    meta = {
        "observed_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "checkpoint_sha256": digest,
        "policy_seed": a.policy_seed,
        "selection_rule": a.selection_note,
        "initial_pose_seeds": seeds,
        "test_usage_note": "Caller must choose reserved seeds; this command cannot attest to their previous use.",
        "perception": "privileged simulated handle pose/axes; NOT visual closed-loop",
        "comparison": "zero residual rules/IK vs deterministic PPO residual; identical frozen leg policy and limits",
        "scope": "same single door/handle, limited reset-pose distribution; not randomized door physics, force safety, or hardware acceptance",
        "episode_horizon_s": 55,
        "diagnostic_sampling_hz": 10,
        "source_sha256": artifact_hashes(
            [
                "core.py",
                "door_env.py",
                "evaluate_paired.py",
                "assets/robot.xml",
                "assets/door_scene.xml",
            ]
        ),
    }
    atomic_json(OUT / "metadata.json", meta)
    policy = PPO.load(str(checkpoint), device=a.device)
    env = DoorEnv(perception="teacher", curriculum="full")
    results = []
    for kind in ["rule_ik", "rl_residual"]:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            episode_reward = 0.0
            contacts = {}
            peak_contact = 0.0
            peak_finger = np.zeros(2)
            peak_torque = 0.0
            trace = []
            record = a.record and kind == "rl_residual" and seed == seeds[0]
            if record:
                encoder = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        "1280x720",
                        "-r",
                        "10",
                        "-i",
                        "-",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "21",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(OUT / "trained_rl.mp4"),
                    ],
                    stdin=subprocess.PIPE,
                )
            for step in range(550):
                if kind == "rule_ik":
                    action = np.zeros(10)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                episode_reward += reward
                p = env.p
                d = p.data
                m = p.model
                assert np.isfinite(obs).all() and np.isfinite(d.qpos).all()
                peak_finger = np.maximum(peak_finger, p.finger_forces)
                peak_torque = max(peak_torque, float(np.abs(d.ctrl[12:18]).max()))
                # Diagnostics sampled at 10Hz are NOT peak force safety certification.
                for ci, c in enumerate(d.contact):
                    bodies = [m.body(m.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
                    force = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, force)
                    norm = float(np.linalg.norm(force[:3]))
                    peak_contact = max(peak_contact, norm)
                    external = (
                        "world" in bodies or "door" in bodies or "handle" in bodies
                    )
                    intended = "handle" in bodies and any(
                        b in ["link7", "link8"] for b in bodies
                    )
                    feet_ground = "world" in bodies and any(
                        b.endswith("_foot") for b in bodies
                    )
                    if external and not intended and not feet_ground and norm > 5:
                        key = "|".join(sorted(bodies))
                        contacts[key] = max(contacts.get(key, 0), norm)
                if record:
                    image = env.render()
                    image[:92] = (20, 25, 35)
                    cv2.putText(
                        image,
                        "TRAINED PPO RESIDUAL + SCRIPTED IK | PRIVILEGED GEOMETRY",
                        (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (245, 245, 245),
                        2,
                    )
                    cv2.putText(
                        image,
                        f"candidate {a.policy_seed} | test {seed} | {info['phase']} | {env.t:.1f}s | door {np.degrees(info['door_angle_rad']):.1f} deg",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (100, 230, 180),
                        1,
                    )
                    encoder.stdin.write(image.tobytes())
                    if step in [60, 90, 120, 150]:
                        cv2.imwrite(
                            str(OUT / f"rl_frame_{step}.jpg"),
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        )
                    trace.append(
                        {
                            "time": env.t,
                            "phase": info["phase"],
                            "residual_action": np.asarray(action).tolist(),
                            "finger_forces_N": p.finger_forces.tolist(),
                            "grasp_valid": p.grasp_valid,
                            "door_angle_rad": info["door_angle_rad"],
                        }
                    )
                if term or trunc:
                    break
            if record:
                encoder.stdin.close()
                assert encoder.wait() == 0
                atomic_json(OUT / "video_trace.json", trace)
            row = {
                "controller": kind,
                "seed": seed,
                "seconds": round(env.t, 3),
                "full_success": info["full_success"],
                "failure": info["failure"],
                "last_phase": info["phase"],
                "phase_times": info["phase_times"],
                "grasp_ever_valid": info["grasp_ever_valid"],
                "max_door_angle_rad": info["max_door_angle_rad"],
                "diagnostic_external_nonfinger_contacts_over5N_sampled10Hz": contacts,
                "diagnostic_max_contact_force_N_sampled10Hz": peak_contact,
                "diagnostic_peak_finger_forces_N_sampled10Hz": peak_finger.tolist(),
                "diagnostic_peak_arm_command_torque_Nm_sampled10Hz": peak_torque,
                "episode_reward_not_acceptance": episode_reward,
            }
            results.append(row)
            atomic_json(OUT / "results.json", results)
            print(json.dumps(row), flush=True)

    summary = summarize_paired(results)
    summary.update(
        observed_at=time.strftime("%Y-%m-%d %H:%M:%S %z"),
        scope=meta["scope"],
        perception=meta["perception"],
    )
    atomic_json(OUT / "summary.json", summary)
    print("SUMMARY", json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    main()
