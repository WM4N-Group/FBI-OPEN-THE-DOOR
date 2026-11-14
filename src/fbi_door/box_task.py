"""Box task contract. Pure Python so safety/phase rules test without simulator assets."""

import math

PHASES = ("approach", "reach", "push", "stop", "retract", "success")
LEG_CHECKPOINT_SHA256 = (
    "408f96e58a3ea544cdf9a2a27ba0e913c7d02c6a4346cc6202e5c1f3351b0556"
)


def residual_mask(phase):
    """3 base residuals + 3 pre-contact Cartesian offsets. No arm residual in push."""
    if phase == "approach":
        return (1, 1, 1, 0, 0, 0)
    if phase == "reach":
        return (0, 0, 0, 1, 1, 1)
    if phase == "push":
        return (1, 1, 1, 0, 0, 0)
    if phase in PHASES:
        return (0,) * 6
    raise ValueError("Unknown box phase")


def push_valid(displacement, contact_seconds, non_arm_impulse, lateral, tilt):
    """Reject base/leg ramming and non-contact motion as a successful arm push."""
    if not all(
        math.isfinite(v)
        for v in (displacement, contact_seconds, non_arm_impulse, lateral, tilt)
    ):
        return False
    return (
        displacement >= 0.5
        and contact_seconds >= 0.4
        and 0 <= non_arm_impulse < 0.5
        and abs(lateral) < 0.18
        and abs(tilt) < 0.35
    )
