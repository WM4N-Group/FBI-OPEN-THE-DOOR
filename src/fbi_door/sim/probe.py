from ..paths import artifact_hashes
import argparse, json, subprocess
import numpy as np, cv2, mujoco
from .core import ROOT
from .door_env import DoorEnv

pa = argparse.ArgumentParser()
pa.add_argument("--mode", default="teacher")
pa.add_argument("--curriculum", default="full")
pa.add_argument("--seed", type=int, default=111)
pa.add_argument("--record", action="store_true")
pa.add_argument("--steps", type=int, default=550)
a = pa.parse_args()
out = ROOT / "reports"
out.mkdir(exist_ok=True)
e = DoorEnv(perception=a.mode, curriculum=a.curriculum)
o, _ = e.reset(seed=a.seed)
if a.record:
    enc = subprocess.Popen(
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
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out / f"grasp_{a.mode}_{a.curriculum}_seed{a.seed}.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
rows = []
last = None
if a.record:
    closeview = mujoco.Renderer(e.p.model, height=240, width=400)
for i in range(a.steps):
    o, r, t, tr, info = e.step(np.zeros(10))
    assert np.isfinite(o).all() and np.isfinite(r)
    row = {"time": e.t, "reward": r, **info}
    rows.append(row)
    if info["phase"] != last or i % 20 == 0:
        print(json.dumps(row), flush=True)
        last = info["phase"]
    if a.record:
        im = e.render()
        im[:88] = (20, 25, 35)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = e.p.data.site_xpos[e.p.model.site("handle_grasp").id]
        cam.distance = 0.46
        cam.azimuth = 45
        cam.elevation = -12
        closeview.update_scene(e.p.data, camera=cam)
        close = closeview.render().copy()
        im[115:355, 850:1250] = close
        cv2.rectangle(im, (848, 113), (1252, 357), (55, 195, 145), 2)
        cv2.putText(
            im,
            "FINGER / HANDLE CLOSE-UP",
            (850, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (55, 195, 145),
            1,
        )
        cv2.putText(
            im,
            f"PHYSICAL TWO-FINGER GRASP | {a.mode.upper()} | {info['phase']} | {e.t:.1f}s",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        text = f"Finger forces {info['finger_normal_forces_N'][0]:.1f}/{info['finger_normal_forces_N'][1]:.1f} N | grip stable {info['grasp_valid']} | door {np.degrees(info['door_angle_rad']):.1f} deg"
        cv2.putText(
            im, text, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 240, 180), 1
        )
        enc.stdin.write(im.tobytes())
        if i in [60, 80, 100, 120, 180]:
            cv2.imwrite(
                str(out / f"probe_{a.mode}_{i}.jpg"),
                cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
            )
    if t or tr:
        break
if a.record:
    enc.stdin.close()
    assert enc.wait() == 0
    closeview.close()
result = {
    "status": "physical candidate; NOT trained policy or hardware acceptance",
    "mode": a.mode,
    "seed": a.seed,
    "curriculum": a.curriculum,
    "source_sha256": artifact_hashes(
        ["core.py", "door_env.py", "assets/door_scene.xml", "assets/robot.xml"]
    ),
    "seconds": e.t,
    "final": info,
    "trajectory": rows,
}
(out / f"probe_{a.mode}_{a.curriculum}_seed{a.seed}.json").write_text(
    json.dumps(result, indent=2)
)
print(
    "RESULT",
    json.dumps({k: v for k, v in result.items() if k != "trajectory"}),
    flush=True,
)
e.close()
