import json, numpy as np, mujoco
from .core import Physics, RGBD, ROOT

p = Physics()
m = p.model
d = p.data
c = m.camera("wrist_rgbd").id
b = m.body("link6").id
assert m.cam_bodyid[c] == b
cam = RGBD(p, camera="wrist_rgbd")
assert not np.any(cam.sensor_options.sitegroup)
poses = []
for angle in [-0.7, 0, 0.7]:
    d.qpos[p.arm_q[-1]] = angle
    mujoco.mj_forward(m, d)
    actual = d.xmat[b].reshape(3, 3).T @ (d.cam_xpos[c] - d.xpos[b])
    assert np.allclose(actual, m.cam_pos[c], atol=1e-9)
    rgb, depth, mask = cam.capture(False)
    assert (
        mask is None
        and rgb.shape == (120, 160, 3)
        and depth.shape == (120, 160)
        and np.isfinite(depth).all()
    )
    poses.append(d.cam_xpos[c].tolist())
assert np.linalg.norm(np.array(poses[0]) - poses[2]) > 0.02
cam.close()
out = {
    "passed": True,
    "camera_attached_to": "link6 / J6 wrist",
    "camera_moves_with_wrist": True,
    "metric_depth_output": True,
    "semantic_labels_disabled_at_inference": True,
    "all_debug_site_markers_disabled_in_sensor_RGB": True,
    "hardware_hand_eye_calibration_verified": False,
    "note": "Nominal simulated extrinsics only.",
}
(ROOT / "reports/camera_contract_tests.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
