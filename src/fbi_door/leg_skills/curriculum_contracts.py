"""V3 stage budgets are gated, not a claim that the whole task passed."""

import math

SCHEMA = "leg-skills/protected-curriculum-v3"
OBS_DIM = 384
STAGES = {"climb": ("settle", "full"), "down": ("crouch", "crawl", "full")}
LIMITS = {
    "climb": {
        "residual_rad": 0.10,
        "stationary_residual_rad": 0.08,
        "timeout_s": 35.0,
        "speed_m_s": 0.12,
        "swing_m": 0.23,
        "foreaft_m": 0.21,
        "lateral_m": 0.10,
        "peak_speed_m_s": 0.25,
    },
    "down": {
        "residual_rad": 0.30,
        "stationary_residual_rad": 0.08,
        "timeout_s": 60.0,
        "speed_m_s": 0.08,
        "swing_m": 0.14,
        "foreaft_m": 0.16,
        "lateral_m": 0.085,
        "peak_speed_m_s": 0.16,
    },
}


def validate_stage(task, stage):
    if task not in STAGES or stage not in STAGES[task]:
        raise ValueError("Unknown task/stage")


def next_stage(task, stage, successes, count, minimum=0.75):
    validate_stage(task, stage)
    if (
        count < 1
        or successes < 0
        or successes > count
        or not math.isfinite(minimum)
        or not 0 < minimum <= 1
    ):
        raise ValueError("Invalid stage evidence")
    if successes / count < minimum:
        return stage
    stages = STAGES[task]
    index = stages.index(stage)
    return stages[min(index + 1, len(stages) - 1)]


def usable_style(task, stride, foreaft, lateral, speed):
    if task not in LIMITS:
        raise ValueError(task)
    values = (stride, foreaft, lateral, speed)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        return False
    c = LIMITS[task]
    return (
        stride <= c["swing_m"]
        and foreaft <= c["foreaft_m"]
        and lateral <= c["lateral_m"]
        and speed <= c["peak_speed_m_s"]
    )
