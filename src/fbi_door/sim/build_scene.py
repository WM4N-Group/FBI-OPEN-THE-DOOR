"""Build an independent physical door probe, preserving frozen policy artifacts."""

import hashlib, json, xml.etree.ElementTree as E
import numpy as np
from ..paths import RUNTIME_ROOT as ROOT
from ..paths import ensure_runtime

ensure_runtime()
SRC = ROOT / "reference/third_party/locomotion_policy/deploy/mujoco/models"
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)
tree = E.parse(SRC / "go2_piper_l.xml")
robot = tree.getroot()
# Resolve mesh paths without modifying the reference assets.
for el in robot.findall(".//mesh"):
    el.set("file", str((SRC / el.get("file")).resolve()))
base = robot.find(".//body[@name='base']")
if base is None:
    base = robot.find("worldbody/body")
E.SubElement(
    base,
    "camera",
    name="head_rgbd",
    pos="0.24 0 0.22",
    xyaxes="0 -1 0 -0.57357644 0 0.81915204",
    fovy="58",
)
# Candidate gripper control/observability; original assets remain untouched.
robot.find(".//site[@name='attachment_site']").set("pos", "0 0 0.11003")
robot.find(".//motor[@name='gripper']").set("ctrlrange", "-20 20")
for finger, padname in [("link7", "finger_left_pad"), ("link8", "finger_right_pad")]:
    for geom in robot.find(".//body[@name='%s']" % finger).findall("geom"):
        if geom.get("class") == "collision":
            geom.set("name", padname)
            geom.set("friction", "1.0 .01 .001")
            geom.set("condim", "4")
# Nominal eye-in-hand extrinsics based on the retained wrist camera housing.
# This is a simulation approximation, NOT a measured hardware hand-eye calibration.
wrist = robot.find(".//body[@name='link6']")
camera_pos = np.array([-0.0760961, -0.03, 0.058])
look_at = np.array([0, 0, 0.145])
forward = look_at - camera_pos
forward /= np.linalg.norm(forward)
z = -forward
up = np.array([0, 1.0, 0])
cx = np.cross(up, z)
cx /= np.linalg.norm(cx)
cy = np.cross(z, cx)
E.SubElement(
    wrist,
    "camera",
    name="wrist_rgbd",
    pos=" ".join(map(str, camera_pos)),
    xyaxes=" ".join(map(str, np.r_[cx, cy])),
    fovy="75",
)
tree.write(OUT / "robot.xml")
scene = E.Element("mujoco", model="frozen_go2_piper_l_grasp_door_wrist_v4")
E.SubElement(scene, "include", file=str(OUT / "robot.xml"))
E.SubElement(scene, "option", timestep="0.002", gravity="0 0 -9.81", iterations="80")
vis = E.SubElement(scene, "visual")
E.SubElement(vis, "global", offwidth="1280", offheight="720")
E.SubElement(vis, "map", znear=".01", zfar="20")
E.SubElement(vis, "headlight", ambient=".45 .45 .45", diffuse=".6 .6 .6")
asset = E.SubElement(scene, "asset")
E.SubElement(asset, "material", name="door_paint", rgba=".38 .25 .13 1")
E.SubElement(
    asset,
    "material",
    name="handle_metal",
    rgba=".60 .62 .64 1",
    specular=".6",
    shininess=".7",
)
w = E.SubElement(scene, "worldbody")
E.SubElement(w, "light", pos="0 0 3", dir="0 0 -1", directional="true")
E.SubElement(
    w,
    "geom",
    name="ground",
    type="plane",
    size="0 0 .05",
    rgba=".75 .77 .79 1",
    friction=".9 .01 .001",
)
# +z hinge: a leaf extending toward -y opens toward +x when pushed.
x = 0.85
width = 0.90
height = 1.65
hy = 0.45
handle_h = 0.83
for y in [-0.50, 0.52]:
    E.SubElement(
        w,
        "geom",
        name="frame_" + str(y),
        type="box",
        pos=f"{x} {y} {height / 2}",
        size=f".045 .03 {height / 2}",
        rgba=".18 .19 .20 1",
    )
E.SubElement(
    w,
    "geom",
    name="frame_top",
    type="box",
    pos=f"{x} 0 {height + 0.03}",
    size=".045 .56 .03",
    rgba=".18 .19 .20 1",
)
door = E.SubElement(w, "body", name="door", pos=f"{x} {hy} 0")
E.SubElement(
    door,
    "joint",
    name="door_hinge",
    type="hinge",
    axis="0 0 1",
    range="0 1.65",
    damping="0",
    frictionloss="0",
    stiffness="0",
    springref="0",
    armature=".03",
)
E.SubElement(
    door,
    "geom",
    name="door_panel",
    type="box",
    pos=f"0 {-width / 2} {height / 2}",
    size=f".022 {width / 2 - 0.008} {height / 2 - 0.02}",
    mass="8",
    material="door_paint",
    friction=".7 .01 .001",
)
lever = E.SubElement(door, "body", name="handle", pos=f"-.065 -.73 {handle_h}")
E.SubElement(
    lever,
    "joint",
    name="handle_hinge",
    type="hinge",
    axis="1 0 0",
    range="-.65 .05",
    stiffness="1.0",
    springref="0",
    damping=".12",
    armature=".002",
)
E.SubElement(
    lever,
    "geom",
    name="handle_hub",
    type="cylinder",
    pos="0 0 0",
    size=".027 .022",
    quat=".70710678 0 .70710678 0",
    mass=".08",
    material="handle_metal",
)
E.SubElement(
    lever,
    "geom",
    name="handle_lever",
    type="capsule",
    fromto="0 0 0 0 .145 0",
    size=".012",
    mass=".1",
    material="handle_metal",
    friction="1.2 .03 .003",
)
E.SubElement(
    lever, "site", name="handle_grasp", pos="0 .105 0", size=".008", rgba="0 1 0 1"
)
eq = E.SubElement(scene, "equality")
E.SubElement(
    eq,
    "joint",
    name="physical_latch",
    joint1="door_hinge",
    polycoef="0 0 0 0 0",
    active="true",
    solref=".004 1",
    solimp=".99 .999 .001",
)
E.ElementTree(scene).write(OUT / "door_scene.xml")
p = ROOT / "reference/models/policies/arm_moving_clock"
manifest = {
    "task_version": "door-grasp-wrist-v4-clean",
    "door": "inward push door, horizontal lever; not pull/knob/sliding doors",
    "door_width_m": width,
    "handle_height_m": handle_h,
    "door_mass_kg": 8,
    "door_hinge": {
        "damping_Nm_s_per_rad": 0,
        "frictionloss_Nm": 0,
        "stiffness_Nm_per_rad": 0,
        "auto_close": False,
        "note": "free vertical hinge; inertia and physical contacts remain, no restoring closer",
    },
    "handle_hinge": {
        "stiffness_Nm_per_rad": 1.0,
        "damping_Nm_s_per_rad": 0.12,
        "note": "handle return spring retained; separate from door hinge",
    },
    "cameras": {
        "head": "nominal base-mounted RGB-D, fovy 58 deg, upward tilt 35 deg",
        "wrist": "link6-mounted RGB-D, fovy 75 deg",
        "wrist_position_in_link6_m": camera_pos.tolist(),
        "wrist_look_at_in_link6_m": look_at.tolist(),
        "hand_eye_calibration": "nominal simulation extrinsics, NOT measured on hardware",
        "debug_markers_in_sensor_RGB": False,
        "inference": "predicted RGB masks + rendered depth + simulated joint-based camera transform; segmentation IDs never supplied at inference",
    },
    "gripper": {
        "candidate_motor_force_limit_N": 20,
        "kp_N_per_m": 800,
        "kd_Ns_per_m": 5,
        "tcp_wrist_z_m": 0.11003,
        "contact_rule": "both finger pads must contact handle with positive normal forces; no weld or attraction",
        "hardware_limits_verified": False,
    },
    "clearances": {
        "panel_floor_gap_m": 0.02,
        "panel_top_gap_m": 0.02,
        "jamb_centers_y_m": [-0.50, 0.52],
        "reason": "remove unintended hinge-side penetration and floor/top rubbing, preserving contacts",
    },
    "unlock_threshold_rad": -0.32,
    "robot": "legacy Piper L; NOT standard PiPER",
    "leg_policy_output": 12,
    "arm_controller": "separate 6-DOF damped-least-squares IK and PD",
    "latch_model": "joint equality active until actual handle angle passes threshold, with contact prerequisite; never set door qpos during a task rollout",
    "source_hashes": {
        str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            p / "model_7999.pt",
            p / "exported/policy.pt",
            p / "exported/policy.onnx",
            SRC / "go2_piper_l.xml",
        ]
    },
}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(OUT / "door_scene.xml")
