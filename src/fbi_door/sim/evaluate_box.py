"""Evaluate zero-residual IK baseline or a trusted, compatible box PPO checkpoint."""

import argparse
import json
from pathlib import Path
import numpy as np
from .box_env import BoxPushEnv
from ..storage import atomic_json


def evaluate(
    policy=None, count=8, seed=70000, randomization=1.0, video=None, trace=None
):
    env = BoxPushEnv(randomization=randomization)
    results = []
    writer = None
    stream = None
    if trace:
        stream = Path(trace).open("w")
    if video:
        import cv2

        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720)
        )
        if not writer.isOpened():
            raise RuntimeError("Video writer failed")
    try:
        for n in range(count):
            obs, _ = env.reset(seed=seed + n)
            done = False
            last_phase = None
            while not done:
                action = (
                    np.zeros(6)
                    if policy is None
                    else policy.predict(obs, deterministic=True)[0]
                )
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if info["phase"] != last_phase:
                    print(
                        "PHASE",
                        seed + n,
                        info["elapsed_s"],
                        info["phase"],
                        "base",
                        env.p.data.qpos[:3].round(3).tolist(),
                        "tcp",
                        env.p.data.site_xpos[env.p.site].round(3).tolist(),
                        flush=True,
                    )
                    last_phase = info["phase"]
                if stream:
                    stream.write(
                        json.dumps(
                            {
                                **info,
                                "base": env.p.data.qpos[:7].tolist(),
                                "box": env.box_pos.tolist(),
                                "arm_actual": env.p.data.qpos[env.p.arm_q].tolist(),
                                "arm_target": env.arm_command.tolist(),
                                "tcp": env.p.data.site_xpos[env.p.site].tolist(),
                            }
                        )
                        + "\n"
                    )
                if writer and n == 0:
                    import cv2

                    frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
                    cv2.putText(
                        frame,
                        "SIM | privileged pose | "
                        + ("IK baseline" if policy is None else "residual PPO + IK"),
                        (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (20, 20, 20),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{info['phase']}  box dx={info['box_displacement_m']:.2f}m  t={info['elapsed_s']:.1f}s",
                        (20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (20, 20, 20),
                        2,
                    )
                    writer.write(frame)
            results.append({"seed": seed + n, **info})
            print("BOX_EVAL", json.dumps(results[-1]), flush=True)
    finally:
        if writer:
            writer.release()
        if stream:
            stream.close()
        env.close()
    return {
        "passed": sum(r["is_success"] for r in results),
        "total": count,
        "results": results,
        "randomization": randomization,
        "policy": "zero residual IK baseline"
        if policy is None
        else "PPO residual + IK",
        "scope": "initial-pose perturbations only; fixed 2kg box, friction .35; privileged geometry, not vision/hardware",
    }


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--checkpoint", type=Path)
    pa.add_argument("--count", type=int, default=8)
    pa.add_argument("--seed", type=int, default=70000)
    pa.add_argument("--randomization", type=float, default=1.0)
    pa.add_argument("--output", type=Path, required=True)
    pa.add_argument("--video", type=Path)
    pa.add_argument("--trace", type=Path)
    a = pa.parse_args()
    if a.count < 1:
        pa.error("count must be positive")
    for path in (a.output, a.video, a.trace):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
    policy = None
    if a.checkpoint:
        from stable_baselines3 import PPO

        policy = PPO.load(a.checkpoint, device="cpu")
    result = evaluate(policy, a.count, a.seed, a.randomization, a.video, a.trace)
    atomic_json(a.output, result)
    print("BOX_SUMMARY", result["passed"], "/", result["total"], flush=True)


if __name__ == "__main__":
    main()
