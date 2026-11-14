"""Dependency-free skill contracts and reproducible domain sampling."""

import math
import random

SCHEMA = "leg-skills/v1"
LEG_SHA256 = "408f96e58a3ea544cdf9a2a27ba0e913c7d02c6a4346cc6202e5c1f3351b0556"
HISTORY_ORDER = "oldest_to_newest;5x60;original_wrapper_v1"
POLICY_TO_MJ = (0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11)
MJ_TO_POLICY = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
TASK_FEATURES = 20
OBS_DIM = 300 + TASK_FEATURES


def domain(task, seed, difficulty=1.0, split="train"):
    if task not in ("climb", "down"):
        raise ValueError("Unknown skill")
    if split not in ("train", "heldout", "ood"):
        raise ValueError("Unknown split")
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Difficulty must be finite in [0,1]")
    # Disjoint streams, not a claim of unseen geometry families.
    r = random.Random(seed + {"train": 0, "heldout": 10000000, "ood": 20000000}[split])
    d = difficulty
    out = {
        "task": task,
        "seed": seed,
        "split": split,
        "start_y": r.uniform(-0.08, 0.08),
        "start_yaw": r.uniform(-0.12, 0.12),
        "friction": r.uniform(0.5, 1.1),
        "motor_scale": r.uniform(0.9, 1.1),
        "kp_scale": r.uniform(0.9, 1.1),
        "kd_scale": r.uniform(0.9, 1.1),
        "command_speed": r.uniform(0.12, 0.25),
        "obstacle_y": r.uniform(-0.04, 0.04),
        "obstacle_yaw": r.uniform(-0.08, 0.08) * d,
        "start_x": r.uniform(-0.12, 0.05),
    }
    if task == "climb":
        out.update(
            height=r.uniform(0.035 + 0.085 * d, 0.065 + 0.175 * d),
            length=r.uniform(0.85, 1.25),
            width=r.uniform(0.70, 1.05),
            mass=r.uniform(25.0, 60.0),
            center_x=r.uniform(0.95, 1.10),
        )
        if split == "ood":
            out.update(height=r.uniform(0.24, 0.28), friction=r.uniform(0.35, 0.5))
    else:
        # Relative to normal locked-arm collision-envelope top. Not base height.
        out.update(
            clearance_drop=r.uniform(0.015 + 0.06 * d, 0.035 + 0.12 * d),
            thickness=r.uniform(0.07, 0.16),
            width=r.uniform(0.95, 1.30),
            center_x=r.uniform(0.95, 1.10),
            height=0.10,
            mass=0.0,
        )
        if split == "ood":
            out.update(
                clearance_drop=r.uniform(0.155, 0.18), friction=r.uniform(0.35, 0.5)
            )
    return out


def climb_ready(
    foot_support,
    contained,
    linear_speed,
    angular_speed,
    upright,
    arm_error,
    box_tilt,
    box_speed,
):
    return (
        len(foot_support) == 4
        and all(f > 0.5 for f in foot_support)
        and contained
        and linear_speed < 0.06
        and angular_speed < 0.25
        and upright > 0.97
        and arm_error < 0.12
        and box_tilt < 0.10
        and box_speed < 0.03
    )


def down_ready(
    trailing_x,
    rail_far_x,
    lowest_base,
    initial_base,
    base_height,
    linear_speed,
    angular_speed,
    upright,
    arm_error,
    rail_peak,
    under_seen,
):
    return (
        trailing_x > rail_far_x + 0.12
        and lowest_base < initial_base - 0.04
        and abs(base_height - initial_base) < 0.04
        and linear_speed < 0.07
        and angular_speed < 0.25
        and upright > 0.97
        and arm_error < 0.12
        and rail_peak < 2.0
        and under_seen
    )
