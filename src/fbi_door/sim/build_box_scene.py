"""Independent free-body box scene, reusing the documented derived Piper L robot."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as E
from pathlib import Path
from ..box_task import LEG_CHECKPOINT_SHA256
from ..paths import RUNTIME_ROOT as ROOT, ensure_runtime


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--robot-source", required=True, type=Path)
    a = pa.parse_args()
    ensure_runtime()
    out = ROOT / "assets"
    if (out / "box_scene.xml").exists():
        raise FileExistsError("Use a new runtime; do not overwrite experiment assets")
    policy = ROOT / "reference/models/policies/arm_moving_clock/model_7999.pt"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if digest != LEG_CHECKPOINT_SHA256:
        raise ValueError("Wrong frozen leg checkpoint")
    robot = E.parse(a.robot_source)
    # Preserve source-derived robot geometry, cameras, actuators and TCP unchanged.
    # Extend both robot keyframes by the new free-joint box initial state.
    for key in robot.findall(".//keyframe/key"):
        key.set("qpos", key.get("qpos") + " 1.15 0 .201 1 0 0 0")
    robot.write(out / "box_robot.xml")
    scene = E.Element("mujoco", model="go2_piper_l_physical_box_push")
    E.SubElement(scene, "include", file="box_robot.xml")
    E.SubElement(scene, "option", timestep=".002", gravity="0 0 -9.81", iterations="80")
    vis = E.SubElement(scene, "visual")
    E.SubElement(vis, "global", offwidth="1280", offheight="720")
    E.SubElement(vis, "headlight", ambient=".5 .5 .5", diffuse=".65 .65 .65")
    world = E.SubElement(scene, "worldbody")
    E.SubElement(world, "light", pos="0 0 4", directional="true", dir="0 0 -1")
    E.SubElement(
        world,
        "geom",
        name="ground",
        type="plane",
        size="0 0 .05",
        rgba=".75 .77 .79 1",
        friction=".7 .01 .001",
    )
    box = E.SubElement(world, "body", name="push_box", pos="1.15 0 .201")
    E.SubElement(box, "freejoint", name="box_free")
    E.SubElement(
        box,
        "geom",
        name="push_box_geom",
        type="box",
        size=".20 .25 .20",
        mass="2",
        rgba=".75 .42 .13 1",
        friction=".35 .005 .0001",
        condim="4",
        priority="1",
    )
    E.ElementTree(scene).write(out / "box_scene.xml")
    manifest = {
        "task": "approach -> IK reach -> latched arm joint targets -> base push -> stop -> retract",
        "leg_checkpoint_sha256": digest,
        "robot_source_sha256": hashlib.sha256(a.robot_source.read_bytes()).hexdigest(),
        "box_size_m": [0.4, 0.5, 0.4],
        "box_mass_kg": 2,
        "box_sliding_friction": 0.35,
        "box_actuators": 0,
        "box_constraints": "freejoint only; no weld/attraction; pose writes at reset only",
        "robot": "legacy Piper L; source-derived robot unchanged apart from keyframe extension",
        "perception": "privileged simulator box pose, NOT vision",
        "lock": "latched joint position target with bounded PD; not a mechanical brake or weld",
        "hardware_validated": False,
    }
    (ROOT / "box_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out / "box_scene.xml")


if __name__ == "__main__":
    main()
