"""Preserve the robot; build an independent free support box or collidable rail."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as E
from .contracts import LEG_SHA256, SCHEMA


def build(source_runtime, destination, task):
    src, dst = Path(source_runtime).resolve(), Path(destination).resolve()
    if task not in ("climb", "down"):
        raise ValueError(task)
    checkpoint = src / "reference/models/policies/arm_moving_clock/model_7999.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != LEG_SHA256:
        raise ValueError("Reference leg checkpoint mismatch")
    robot_path = src / "assets/box_robot.xml"
    robot = E.parse(robot_path)
    # The source box robot contains no box body, only its appended free qpos.
    for key in robot.findall(".//keyframe/key"):
        q = key.get("qpos").split()
        if len(q) != 34:
            raise ValueError("Expected original 27 robot + 7 free box coordinates")
        key.set(
            "qpos",
            " ".join(
                q[:27]
                + (["1.0", "0", ".081", "1", "0", "0", "0"] if task == "climb" else [])
            ),
        )
    dst.mkdir(parents=True, exist_ok=False)
    (dst / "reference").symlink_to(src / "reference", target_is_directory=True)
    assets = dst / "assets"
    assets.mkdir()
    robot.write(assets / "robot.xml")
    scene = E.Element("mujoco", model="go2_piper_l_" + task)
    E.SubElement(scene, "include", file="robot.xml")
    option = E.SubElement(
        scene,
        "option",
        timestep=".002",
        gravity="0 0 -9.81",
        iterations="80",
        ls_iterations="50",
        tolerance=".00001",
    )
    E.SubElement(option, "flag", multiccd="disable", nativeccd="disable")
    visual = E.SubElement(scene, "visual")
    E.SubElement(visual, "global", offwidth="1280", offheight="720")
    world = E.SubElement(scene, "worldbody")
    E.SubElement(world, "light", pos="0 0 4", directional="true", dir="0 0 -1")
    E.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="0 0 .05",
        rgba=".7 .73 .76 1",
        friction=".8 .01 .001",
    )
    if task == "climb":
        box = E.SubElement(world, "body", name="step_box", pos="1 0 .081")
        E.SubElement(box, "freejoint", name="step_box_free")
        E.SubElement(
            box,
            "geom",
            name="step_box_geom",
            type="box",
            size=".5 .4 .08",
            mass="40",
            rgba=".8 .45 .15 1",
            friction=".8 .01 .001",
            condim="4",
        )
    else:
        rail = E.SubElement(world, "body", name="rail", pos="1 0 0")
        E.SubElement(
            rail,
            "geom",
            name="rail_bar",
            type="box",
            pos="0 0 .7",
            size=".06 .6 .05",
            rgba=".2 .4 .8 1",
            friction=".6 .01 .001",
        )
        for side in (-1, 1):
            E.SubElement(
                rail,
                "geom",
                name="rail_post_" + str(side),
                type="box",
                pos="0 %s .4" % (0.62 * side),
                size=".06 .025 .4",
                rgba=".2 .4 .8 1",
            )
    E.ElementTree(scene).write(assets / "scene.xml")
    manifest = {
        "schema": SCHEMA,
        "task": task,
        "source_robot_sha256": hashlib.sha256(robot_path.read_bytes()).hexdigest(),
        "leg_checkpoint_sha256": LEG_SHA256,
        "robot_collision_and_inertia": "unchanged",
        "arm": "fixed home six-joint position target, bounded PD, not a weld",
        "perception": "privileged geometry teacher",
        "hardware_enabled": False,
        "solver": {
            "iterations": 80,
            "ls_iterations": 50,
            "tolerance": 0.00001,
            "timestep": 0.002,
            "multiccd": False,
            "nativeccd": False,
            "reason": "Warp 3.12 rejects MULTICCD/NATIVECCD with original nonzero geom margins; margins preserved",
        },
        "obstacle": "free 25-60kg load-bearing box, NOT original 2kg push box"
        if task == "climb"
        else "fixed physical horizontal rail and two posts",
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-runtime", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--task", choices=("climb", "down"), required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source_runtime, a.destination, a.task), indent=2))


if __name__ == "__main__":
    main()
