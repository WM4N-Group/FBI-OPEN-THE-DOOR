"""Eye-in-hand + head-camera handle segmentation candidate.
Inference uses RGB, calibrated depth and camera transforms. Rendered semantic IDs are labels ONLY.
Output is a visible-surface observation, NOT a guaranteed grasp pose or slip certificate.
"""

from ..paths import artifact_hashes


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from .core import ROOT

CAMERAS = ["head_rgbd", "wrist_rgbd"]


class DualHandleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return F.interpolate(
            self.dec(self.enc(x)),
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    @torch.inference_mode()
    def detect(self, rgb, camera_index, device="cpu"):
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float().to(device) / 255
        cid = torch.full((1, 1, *x.shape[-2:]), float(camera_index), device=device)
        prob = self(torch.cat([x, cid], 1)).sigmoid()[0, 0].cpu().numpy()
        return prob > 0.5, prob


def backproject(camera, mask, depth):
    """Per-camera visible surface, using ONLY predicted mask at inference."""
    v, u = np.where(mask & np.isfinite(depth) & (depth > 0.025) & (depth < 3.0))
    if len(v) < 6:
        return None
    z = depth[v, u]
    f = (
        0.5
        * camera.h
        / np.tan(np.deg2rad(float(camera.p.model.cam_fovy[camera.camera])) / 2)
    )
    pts = np.c_[(u - (camera.w - 1) / 2) * z / f, -(v - (camera.h - 1) / 2) * z / f, -z]
    R = camera.p.data.cam_xmat[camera.camera].reshape(3, 3)
    cloud = pts @ R.T + camera.p.data.cam_xpos[camera.camera]
    return {
        "cloud_world": cloud,
        "visible_center_world": np.median(cloud, axis=0),
        "pixels": len(v),
    }


class DualCameraObserver:
    def __init__(self, physics, checkpoint=None, device="cpu"):
        from .core import RGBD

        self.physics = physics
        self.device = device
        self.net = DualHandleNet().to(device)
        checkpoint = checkpoint or ROOT / "runs/dual_perception/handle_best.pt"
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.net.load_state_dict(state["model"])
        self.net.eval()
        self.cameras = [RGBD(physics, camera=c) for c in CAMERAS]

    def observe(self):
        import cv2

        observations = []
        for i, cam in enumerate(self.cameras):
            rgb, depth, _ = cam.capture(labels=False)
            mask, prob = self.net.detect(rgb, i, self.device)
            # Conservative connected-component gate. Reject ambiguous/too-small predictions.
            labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), 8
            )
            candidates = []
            for lab in range(1, labels_n):
                region = labels == lab
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < (12 if i == 0 else 32):
                    continue
                confidence = float(np.mean(prob[region]))
                if confidence < 0.75:
                    continue
                validdepth = (
                    region
                    & np.isfinite(depth)
                    & (depth > 0.025)
                    & (depth < (2.5 if i == 0 else 1.2))
                )
                if validdepth.sum() < 0.8 * area:
                    continue
                candidates.append((confidence * np.sqrt(area), region, confidence))
            accepted = max(candidates, key=lambda x: x[0]) if candidates else None
            mask = np.zeros_like(mask) if accepted is None else accepted[1]
            points = backproject(cam, mask, depth)
            observations.append(
                {
                    "camera": CAMERAS[i],
                    "rgb": rgb,
                    "depth": depth,
                    "predicted_mask": mask,
                    "probability": prob,
                    "accepted_confidence": 0.0 if accepted is None else accepted[2],
                    "surface": points,
                }
            )
        return observations

    def features(self, observations):
        # Known TCP from arm FK, not a simulator object coordinate. These are visible-surface
        # offsets, not a canonical grasp center; do not blindly average between cameras.
        tcp = self.physics.data.site_xpos[self.physics.site]
        R = self.physics.data.site_xmat[self.physics.site].reshape(3, 3)
        features = []
        for obs in observations:
            surf = obs["surface"]
            valid = surf is not None
            offset = (
                np.zeros(3)
                if not valid
                else np.clip(R.T @ (surf["visible_center_world"] - tcp), -2, 2)
            )
            area = 0 if not valid else surf["pixels"] / (160 * 120)
            features.extend(
                [*offset, obs["accepted_confidence"], float(area), float(valid)]
            )
        return np.array(features, dtype=np.float32)

    def close(self):
        for c in self.cameras:
            c.close()


def collect_bank(out):
    from .door_env import DoorEnv
    from .core import RGBD
    import cv2

    e = DoorEnv(perception="teacher", curriculum="full")
    e.reset(seed=111)
    cams = [RGBD(e.p, camera=c) for c in CAMERAS]
    states = []
    names = []
    saved = set()
    visibility = []
    for step in range(550):
        _, _, term, trunc, info = e.step(np.zeros(10))
        if step % 3 == 0:
            states.append(e.p.data.qpos.copy())
            names.append(info["phase"])
        if (
            info["phase"] in ["pregrasp", "insert", "grasp", "press", "push", "retract"]
            and info["phase"] not in saved
        ):
            if info["phase"] in ["grasp", "press"] and e.t - e.phase_start < 0.5:
                continue
            saved.add(info["phase"])
            pairs = []
            counts = []
            for cam in cams:
                rgb, depth, mask = cam.capture(True)
                counts.append(int(mask.sum()))
                pairs.append(cv2.resize(rgb, (640, 480)))
            sheet = np.vstack(
                [np.full((50, 1280, 3), (20, 25, 35), dtype=np.uint8), np.hstack(pairs)]
            )
            cv2.putText(
                sheet,
                "HEAD RGB-D",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                sheet,
                "WRIST RGB-D",
                (660, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imwrite(
                str(out / f"camera_pair_{info['phase']}.jpg"),
                cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
            )
            visibility.append(
                {
                    "phase": info["phase"],
                    "time": e.t,
                    "head_label_pixels": counts[0],
                    "wrist_label_pixels": counts[1],
                }
            )
        if term or trunc:
            break
    for c in cams:
        c.close()
    np.savez_compressed(
        out / "rollout_bank.npz", qpos=np.stack(states), phase=np.array(names)
    )
    (out / "camera_visibility.json").write_text(
        json.dumps(
            {
                "teacher_rollout": True,
                "full_success": info["full_success"],
                "phase_visibility": visibility,
                "note": "Semantic pixels used for visibility diagnostics only; not inference.",
            },
            indent=2,
        )
    )
    print("CAMERA_VISIBILITY", json.dumps(visibility), flush=True)
    e.close()


def generate(out, n):
    from .core import Physics, RGBD
    import mujoco

    bank = np.load(out / "rollout_bank.npz")
    bankq = bank["qpos"]
    p = Physics()
    m = p.model
    d = p.data
    cams = [RGBD(p, camera=c) for c in CAMERAS]
    bpos = m.body_pos.copy()
    fov = m.cam_fovy.copy()
    cpos = m.cam_pos.copy()
    door = m.body("door").id
    handle = m.body("handle").id
    xs = []
    ys = []
    ds = []
    intrinsics = []
    for i in range(n):
        rng = np.random.default_rng(910000 + i)
        d.qpos[:] = bankq[rng.integers(len(bankq))]
        d.qvel[:] = 0
        d.qpos[0:2] += rng.uniform(-0.035, 0.035, 2)
        d.qpos[p.arm_q] += rng.normal(0, 0.025 if rng.random() < 0.75 else 0.12, 6)
        d.qpos[p.arm_q] = np.clip(
            d.qpos[p.arm_q],
            m.jnt_range[p.arm_jids, 0] + 0.01,
            m.jnt_range[p.arm_jids, 1] - 0.01,
        )
        if rng.random() < 0.12:
            yaw = rng.uniform(-2.4, 2.4)
            d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        m.body_pos[door] = bpos[door] + [
            rng.uniform(-0.06, 0.06),
            rng.uniform(-0.03, 0.03),
            0,
        ]
        m.body_pos[handle] = bpos[handle] + [0, 0, rng.uniform(-0.06, 0.06)]
        m.mat_rgba[m.material("door_paint").id, :3] = rng.uniform(0.08, 0.85, 3)
        gray = rng.uniform(0.15, 0.95)
        m.mat_rgba[m.material("handle_metal").id, :3] = np.clip(
            gray + rng.uniform(-0.04, 0.04, 3), 0.05, 1
        )
        for cam in cams:
            m.cam_fovy[cam.camera] = fov[cam.camera] + rng.uniform(-2, 2)
            m.cam_pos[cam.camera] = cpos[cam.camera] + rng.uniform(-0.003, 0.003, 3)
        mujoco.mj_forward(m, d)
        xx = []
        yy = []
        dd = []
        ff = []
        for cam in cams:
            rgb, depth, mask = cam.capture(True)
            rgb = np.clip(
                rgb.astype(float) * rng.uniform(0.75, 1.2)
                + rng.normal(0, 2, rgb.shape),
                0,
                255,
            ).astype(np.uint8)
            xx.append(rgb)
            yy.append(mask)
            dd.append(np.minimum(depth, 5).astype(np.float16))
            ff.append(float(m.cam_fovy[cam.camera]))
        xs.append(xx)
        ys.append(yy)
        ds.append(dd)
        intrinsics.append(ff)
        if (i + 1) % 128 == 0:
            print("DATA", i + 1, "/", n, flush=True)
    for c in cams:
        c.close()
    np.savez_compressed(
        out / "dual_rgbd.npz",
        rgb=np.array(xs),
        mask=np.array(ys),
        depth=np.array(ds),
        fovy=np.array(intrinsics),
    )
    cfg = {
        "scenes": n,
        "views_per_scene": 2,
        "cameras": CAMERAS,
        "resolution": [160, 120],
        "split_rule": "scene-level split; both camera views from each scene stay in the same fold",
        "train_scenes": [0, n - 1024],
        "validation_scenes": [n - 1024, n - 512],
        "test_scenes": [n - 512, n],
        "seed_rule": "910000+i",
        "semantic_truth": "labels/scoring only, never detector input",
        "debug_site_markers_in_RGB": False,
        "source_sha256": artifact_hashes(
            ["core.py", "assets/door_scene.xml", "assets/robot.xml"]
        ),
        "warning": "synthetic single-door-family dataset, nominal hand-eye calibration, no real camera validation",
    }
    (out / "dataset.json").write_text(json.dumps(cfg, indent=2))


def batch_input(x, device):
    # x is [scene,camera,H,W,RGB]; camera identity is a known sensor ID, not a semantic label.
    n, c, h, w, _ = x.shape
    image = (
        torch.from_numpy(x.reshape(n * c, h, w, 3))
        .permute(0, 3, 1, 2)
        .float()
        .to(device)
        / 255
    )
    ids = (
        torch.arange(c, device=device)
        .repeat(n)[:, None, None, None]
        .expand(-1, 1, h, w)
        .float()
    )
    return torch.cat([image, ids], 1)


@torch.inference_mode()
def score(net, x, y, depth, fovy, device):
    net.eval()
    counts = [
        {"inter": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0, "errors": []}
        for _ in CAMERAS
    ]
    for i in range(0, len(x), 16):
        pred = (
            (net(batch_input(x[i : i + 16], device)).sigmoid()[:, 0] > 0.5)
            .cpu()
            .numpy()
            .reshape(-1, 2, *y.shape[-2:])
        )
        for j in range(len(pred)):
            for k in range(2):
                a = pred[j, k]
                b = y[i + j, k]
                s = counts[k]
                s["inter"] += np.logical_and(a, b).sum()
                s["union"] += np.logical_or(a, b).sum()
                av = a.sum() >= 6
                bv = b.sum() >= 6
                s["tp"] += int(av and bv)
                s["fp"] += int(av and not bv)
                s["fn"] += int(bv and not av)
                if av and bv:
                    z = depth[i + j, k].astype(float)
                    h, w = a.shape
                    f = 0.5 * h / np.tan(np.deg2rad(fovy[i + j, k]) / 2)

                    def center(mask):
                        v, u = np.where(mask)
                        zz = z[v, u]
                        return np.median(
                            np.c_[
                                (u - (w - 1) / 2) * zz / f,
                                -(v - (h - 1) / 2) * zz / f,
                                -zz,
                            ],
                            axis=0,
                        )

                    s["errors"].append(float(np.linalg.norm(center(a) - center(b))))
    results = {}
    for name, s in zip(CAMERAS, counts):
        err = s["errors"]
        results[name] = {
            "iou": float(s["inter"] / max(s["union"], 1)),
            "presence_precision": s["tp"] / max(s["tp"] + s["fp"], 1),
            "presence_recall": s["tp"] / max(s["tp"] + s["fn"], 1),
            "visible_center_error_median_m": float(np.median(err)) if err else None,
            "visible_center_error_p95_m": float(np.percentile(err, 95))
            if err
            else None,
            "visible_matches": len(err),
        }
    return results


def train(out, epochs, device, seed, batch_scenes=16, datafile=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.load(datafile if datafile is not None else out / "dual_rgbd.npz")
    x = data["rgb"]
    y = data["mask"]
    dep = data["depth"]
    fov = data["fovy"]
    split = len(x) - 1024
    v_end = len(x) - 512
    net = DualHandleNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=7e-4)
    best = -1
    history = []
    for epoch in range(epochs):
        net.train()
        losses = []
        for ids in np.array_split(
            np.random.permutation(split), (split + batch_scenes - 1) // batch_scenes
        ):
            xx = batch_input(x[ids], device)
            yy = (
                torch.from_numpy(y[ids].reshape(-1, *y.shape[-2:]))
                .float()
                .to(device)[:, None]
            )
            logits = net(xx)
            prob = logits.sigmoid()
            bce = F.binary_cross_entropy_with_logits(
                logits, yy, pos_weight=torch.tensor(15.0, device=device)
            )
            dice = (
                1
                - (2 * (prob * yy).sum((1, 2, 3)) + 1)
                / ((prob + yy).sum((1, 2, 3)) + 1)
            ).mean()
            loss = bce + dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = score(
            net,
            x[split:v_end],
            y[split:v_end],
            dep[split:v_end],
            fov[split:v_end],
            device,
        )
        miou = np.mean([v["iou"] for v in val.values()])
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation": val}
        history.append(row)
        (out / "training_history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
        if miou > best:
            best = miou
            torch.save(
                {
                    "model": net.cpu().state_dict(),
                    "epoch": epoch + 1,
                    "validation": val,
                    "architecture": "RGB+camera-ID shared segmentation v1",
                    "calibration": "nominal, not hardware-calibrated",
                },
                out / "handle_best.pt",
            )
            net.to(device)
        torch.save(
            {
                "model": net.state_dict(),
                "epoch": epoch + 1,
                "optimizer": opt.state_dict(),
            },
            out / "handle_last.pt",
        )
    net.load_state_dict(
        torch.load(out / "handle_best.pt", map_location=device, weights_only=False)[
            "model"
        ]
    )
    test = score(net, x[v_end:], y[v_end:], dep[v_end:], fov[v_end:], device)
    result = {
        "heldout_scene_count": 512,
        "results": test,
        "scope": "synthetic visible handle surface only, NOT grasp/rotation/slip/full-task success or sim2real acceptance",
    }
    (out / "heldout_test.json").write_text(json.dumps(result, indent=2))
    print("HELDOUT", json.dumps(result), flush=True)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--scenes", type=int, default=3072)
    pa.add_argument("--epochs", type=int, default=60)
    pa.add_argument("--device", default="cuda:0")
    pa.add_argument("--seed", type=int, default=45)
    pa.add_argument("--skip-data", action="store_true")
    pa.add_argument("--output", default="dual_perception")
    pa.add_argument(
        "--datafile",
        help="Completed trusted scene-paired dataset; requires --skip-data",
    )
    pa.add_argument("--batch-scenes", type=int, default=16)
    a = pa.parse_args()
    if a.batch_scenes < 1 or a.epochs < 1 or (not a.skip_data and a.scenes < 1152):
        pa.error("positive batch/epochs and at least 1152 paired scenes required")
    if a.datafile and not a.skip_data:
        pa.error("--datafile requires --skip-data")
    out = ROOT / "runs" / a.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(
        json.dumps(
            {
                "seed": a.seed,
                "device": a.device,
                "epochs": a.epochs,
                "batch_scenes": a.batch_scenes,
                "datafile": a.datafile,
                "requested_scenes_if_generated": a.scenes,
                "debug_markers_in_sensor_rgb": False,
                "visual_closed_loop_control": False,
            },
            indent=2,
        )
    )
    if not a.skip_data:
        collect_bank(out)
        generate(out, a.scenes)
    train(out, a.epochs, a.device, a.seed, a.batch_scenes, a.datafile)
