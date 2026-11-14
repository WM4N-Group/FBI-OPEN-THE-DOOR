"""Pure-Python contract for the collision-aware box controller (v2)."""

import math

PHASES = (
    "approach",
    "settle",
    "reach",
    "touch",
    "push",
    "stop",
    "retreat",
    "withdraw_settle",
    "retract",
    "success",
)
ARM_SPEED = 0.30  # rad/s, simulation candidate, not a verified hardware limit
ARM_ACCEL = 0.60  # rad/s^2
SELF_MARGIN = 0.025  # m, independent of simulator collision filtering


def mask(phase):
    if phase in ("approach", "push"):
        return (1, 1, 1, 0, 0, 0)
    if phase == "reach":
        return (0, 0, 0, 1, 1, 1)
    if phase in PHASES:
        return (0,) * 6
    raise ValueError("Unknown clearance-controller phase")


def withdrawal_ready(distance, gap, force, quiet_seconds):
    return (
        all(math.isfinite(x) for x in (distance, gap, force, quiet_seconds))
        and distance >= 0.20
        and gap >= 0.08
        and 0 <= force < 0.15
        and quiet_seconds >= 0.3
    )


def ramp(previous, target, acceleration, dt):
    if (
        not all(math.isfinite(x) for x in (previous, target, acceleration, dt))
        or acceleration <= 0
        or dt <= 0
    ):
        raise ValueError("Finite values and positive acceleration/dt required")
    return previous + max(-acceleration * dt, min(acceleration * dt, target - previous))
