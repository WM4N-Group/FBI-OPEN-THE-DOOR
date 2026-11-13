"""Render online dual-camera inference over a TEACHER-CONTROLLED rollout.
This does NOT claim the detector controls the arm or that visual grasping succeeds.
"""

import json, hashlib
import cv2, numpy as np
from .core import ROOT
from .door_env import DoorEnv
from .dual_perception import DualCameraObserver

out = ROOT / "reports"
e = DoorEnv(perception="teacher", curriculum="full")
e.reset(seed=111)
observer = DualCameraObserver(e.p)
saved = set()
records = []
for step in range(550):
    _, _, term, trunc, info = e.step(np.zeros(10))
    if (
        info["phase"] in ["pregrasp", "grasp", "press", "push", "retract"]
        and info["phase"] not in saved
        and e.t - e.phase_start > 0.5
    ):
        saved.add(info["phase"])
        observations = observer.observe()
        features = observer.features(observations)
        assert np.isfinite(features).all() and len(features) == 12
        panels = []
        measurements = []
        for obs in observations:
            im = obs["rgb"].copy()
            mask = obs["predicted_mask"]
            im[mask] = (im[mask] * 0.55 + np.array([40, 220, 120]) * 0.45).astype(
                np.uint8
            )
            panels.append(cv2.resize(im, (640, 480), interpolation=cv2.INTER_NEAREST))
            surf = obs["surface"]
            measurements.append(
                {
                    "camera": obs["camera"],
                    "accepted": surf is not None,
                    "accepted_confidence": obs["accepted_confidence"],
                    "mask_pixels": int(mask.sum()),
                    "visible_center_world": None
                    if surf is None
                    else surf["visible_center_world"].tolist(),
                }
            )
        header = np.full((85, 1280, 3), (20, 25, 35), dtype=np.uint8)
        image = np.vstack([header, np.hstack(panels)])
        cv2.putText(
            image,
            "HEAD RGB-D",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 245, 245),
            2,
        )
        cv2.putText(
            image,
            "WRIST RGB-D",
            (660, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 245, 245),
            2,
        )
        cv2.putText(
            image,
            f"{info['phase']} | ONLINE RGB DETECTION | TEACHER MOTION, NOT VISION-CONTROLLED",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (100, 230, 180),
            1,
        )
        cv2.imwrite(
            str(out / f"online_pair_{info['phase']}.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
        record = {
            "phase": info["phase"],
            "time": e.t,
            "cameras": measurements,
            "TCP_relative_visible_surface_features_12D": features.tolist(),
            "controls_motion": False,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    if term or trunc:
        break
observer.close()
e.close()
(out / "online_camera_probe.json").write_text(
    json.dumps(
        {
            "motion_controller": "privileged teacher scripted + IK; detector overlays/features ONLY",
            "camera_probe": records,
            "weights_sha256": hashlib.sha256(
                (ROOT / "runs/dual_perception/handle_best.pt").read_bytes()
            ).hexdigest(),
            "debug_markers_disabled": True,
            "not_visual_task_acceptance": True,
        },
        indent=2,
    )
)
