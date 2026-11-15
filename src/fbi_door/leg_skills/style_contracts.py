"""Small-step v2 units, phase contract, and fail-closed style predicates."""

import math

STYLE_SCHEMA = "leg-skills/small-step-v2"
STYLE_FEATURES = 40
STYLE_OBS_DIM = 320 + STYLE_FEATURES
STYLE_PHASES = ("crouch", "crawl", "recover")
STYLE_LIMITS = {
    "climb": {
        "command_speed_min": 0.09,
        "command_speed_max": 0.15,
        "target_rate_rad_s": 6.0,
        "swing_soft_m": 0.15,
        "swing_accept_m": 0.23,
        "swing_abort_m": 0.32,
        "foreaft_soft_m": 0.13,
        "foreaft_accept_m": 0.21,
        "lateral_soft_m": 0.055,
        "lateral_accept_m": 0.10,
        "max_horizontal_speed_m_s": 0.25,
        "swing_height_m": 0.10,
        "episode_timeout_s": 35.0,
        "hold_s": 2.0,
    },
    "down": {
        "command_speed_min": 0.065,
        "command_speed_max": 0.10,
        "target_rate_rad_s": 1.2,
        "swing_soft_m": 0.08,
        "swing_accept_m": 0.14,
        "swing_abort_m": 0.22,
        "foreaft_soft_m": 0.09,
        "foreaft_accept_m": 0.16,
        "lateral_soft_m": 0.045,
        "lateral_accept_m": 0.085,
        "max_horizontal_speed_m_s": 0.16,
        "swing_height_m": 0.045,
        "episode_timeout_s": 45.0,
        "hold_s": 1.0,
    },
}


def style_limits(task):
    if task not in STYLE_LIMITS:
        raise ValueError("Unknown skill")
    return dict(STYLE_LIMITS[task])


def target_increment(task, dt=0.02):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite control interval required")
    return style_limits(task)["target_rate_rad_s"] * dt


def crouch_ready(
    base_z, target_z, top_z, opening_z, speed, angular_speed, supporting_feet, arm_error
):
    values = (base_z, target_z, top_z, opening_z, speed, angular_speed, arm_error)
    return (
        all(math.isfinite(v) for v in values)
        and base_z <= target_z + 0.012
        and top_z <= opening_z - 0.012
        and speed < 0.045
        and angular_speed < 0.30
        and supporting_feet >= 3
        and arm_error < 0.12
    )


def style_acceptable(task, swing_m, foreaft_m, lateral_m, peak_speed):
    cfg = style_limits(task)
    values = (swing_m, foreaft_m, lateral_m, peak_speed)
    return (
        all(math.isfinite(v) and v >= 0 for v in values)
        and swing_m <= cfg["swing_accept_m"]
        and foreaft_m <= cfg["foreaft_accept_m"]
        and lateral_m <= cfg["lateral_accept_m"]
        and peak_speed <= cfg["max_horizontal_speed_m_s"]
    )
