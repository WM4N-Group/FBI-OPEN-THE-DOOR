"""Height-only box ablation helpers. No dynamics or learning dependencies."""

import hashlib
import json
import math
import xml.etree.ElementTree as E
from pathlib import Path


def finite_positive(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(name + " must be finite and positive")
    return float(value)


def contact_height(box_height_m, top_margin_m=0.04):
    """Flat z=0 floor, unchanged .34m target when possible; otherwise below the top.

    This is a geometric proposal, not a guarantee of IK, clearance, or load stability.
    The original collision/force/withdrawal gates must still accept the rollout.
    """
    height = finite_positive(box_height_m, "height")
    margin = finite_positive(top_margin_m, "top margin")
    if margin >= height:
        raise ValueError("Top margin must be smaller than box height")
    return min(0.34, height - margin)


def build_height_variant(source_runtime, destination, height_m):
    """Copy only scene XML; change box half-height and reset height, never a rollout.

    Original robot geometry, actuators, mass and friction are not changed. Private
    meshes/policies remain read-only references and never enter the source repo.
    """
    height = finite_positive(height_m, "height")
    if not 0.10 <= height <= 0.40:
        raise ValueError("This ablation is limited to .10.. .40m boxes")
    source = Path(source_runtime).resolve()
    out = Path(destination).resolve()
    if out.exists():
        raise FileExistsError("Use a new evaluation runtime")
    scene_path = source / "assets/box_scene.xml"
    robot_path = source / "assets/box_robot.xml"
    scene = E.parse(scene_path)
    robot = E.parse(robot_path)
    body = scene.find(".//body[@name='push_box']")
    geom = scene.find(".//geom[@name='push_box_geom']")
    if (
        body is None
        or geom is None
        or body.find("freejoint") is None
        or geom.get("type") != "box"
    ):
        raise ValueError("Expected the independent free-box scene")
    half = list(map(float, geom.get("size").split()))
    if len(half) != 3:
        raise ValueError("Malformed box size")
    if scene.find("include").get("file") != "box_robot.xml":
        raise ValueError("Unexpected robot include")
    center = list(map(float, body.get("pos").split()))
    center[2] = height / 2 + 0.001
    half[2] = height / 2
    geom.set("size", " ".join(map(str, half)))
    body.set("pos", " ".join(map(str, center)))
    for key in robot.findall(".//keyframe/key"):
        q = key.get("qpos").split()
        if len(q) != 34:
            raise ValueError("Unexpected frozen robot/free-box coordinate layout")
        q[-5] = str(center[2])
        key.set("qpos", " ".join(q))
    reference = source / "reference"
    if not reference.is_dir():
        raise ValueError("Missing authorized reference assets")
    (out / "assets").mkdir(parents=True)
    (out / "reports").mkdir()
    (out / "reference").symlink_to(reference.resolve(), target_is_directory=True)
    scene.write(out / "assets/box_scene.xml")
    robot.write(out / "assets/box_robot.xml")
    manifest = {
        "ablation": "box height only; no retraining",
        "box_size_m": [2 * x for x in half],
        "box_mass_kg": float(geom.get("mass")),
        "box_friction": list(map(float, geom.get("friction").split())),
        "box_free_joint": True,
        "box_actuator_added": False,
        "robot_collision_geometries_modified": False,
        "reset_box_center_z_m": center[2],
        "changed_fields": ["box.geom.size.z", "box.body.pos.z", "box keyframe qpos z"],
        "source_xml_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (scene_path, robot_path)
        },
        "scope": "fixed footprint/mass/friction, flat floor; NOT load-bearing or stepping/climbing validation",
    }
    (out / "height_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
