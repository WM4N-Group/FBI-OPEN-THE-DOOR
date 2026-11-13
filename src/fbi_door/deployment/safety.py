"""Pure validation only: importing or calling this module cannot connect to a robot."""

import math
from dataclasses import dataclass
from typing import Tuple


class SafetyError(ValueError):
    pass


def vector6(values, name="joints"):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise SafetyError("%s must contain exactly six values" % name)
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        raise SafetyError("%s must contain finite numeric values" % name)
    return tuple(float(v) for v in values)


@dataclass(frozen=True)
class JointFeedback:
    positions_rad: Tuple[float, ...]
    timestamp_monotonic_s: float


class CommandGuard:
    """Position-command limits are not a replacement for force/contact safety."""

    def __init__(
        self,
        limits_rad,
        max_speed_rad_s=0.05,
        max_step_rad=0.01,
        feedback_timeout_s=0.10,
        max_tracking_error_rad=0.03,
    ):
        if len(limits_rad) != 6:
            raise SafetyError("Need six joint-limit pairs")
        self.limits = []
        for pair in limits_rad:
            if (
                len(pair) != 2
                or not all(
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                    for v in pair
                )
                or pair[0] >= pair[1]
            ):
                raise SafetyError("Invalid joint limits")
            self.limits.append(tuple(pair))
        for v in (
            max_speed_rad_s,
            max_step_rad,
            feedback_timeout_s,
            max_tracking_error_rad,
        ):
            if isinstance(v, bool) or not math.isfinite(v) or v <= 0:
                raise SafetyError("Guard parameters must be finite and positive")
        self.max_speed = max_speed_rad_s
        self.max_step = max_step_rad
        self.timeout = feedback_timeout_s
        self.max_tracking_error = max_tracking_error_rad

    def validate(
        self, target, feedback, now_monotonic_s, previous_target=None, dt_s=0.02
    ):
        q = vector6(target, "target")
        actual = vector6(feedback.positions_rad, "feedback")
        if (
            not all(
                math.isfinite(v)
                for v in (now_monotonic_s, feedback.timestamp_monotonic_s, dt_s)
            )
            or dt_s <= 0
        ):
            raise SafetyError("Invalid monotonic time or command interval")
        age = now_monotonic_s - feedback.timestamp_monotonic_s
        if age < 0 or age > self.timeout:
            raise SafetyError("Feedback is stale or from the future")
        previous = (
            actual
            if previous_target is None
            else vector6(previous_target, "previous target")
        )
        for i, (lo, hi) in enumerate(self.limits):
            if not lo <= actual[i] <= hi:
                raise SafetyError(
                    "Actual joint %d is outside limits; no automatic recovery" % (i + 1)
                )
            if not lo <= q[i] <= hi:
                raise SafetyError("Joint %d target exceeds limits" % (i + 1))
            if abs(q[i] - actual[i]) > self.max_tracking_error:
                raise SafetyError("Joint %d tracking error too large" % (i + 1))
            if (
                abs(q[i] - previous[i])
                > min(self.max_step, self.max_speed * dt_s) + 1e-12
            ):
                raise SafetyError("Joint %d step/speed limit exceeded" % (i + 1))
        return q


def preflight(config):
    """Fail-closed checklist, NOT hardware attestation. Flags require measured evidence."""
    failures = [
        "Full hardware door transport and audited gripper/base adapter are not implemented"
    ]
    robot, policy = config.get("robot", {}), config.get("policy", {})
    if not robot.get("variant") or robot.get("variant") != policy.get("robot_variant"):
        failures.append(
            "Robot/model mismatch: legacy Piper L policy is not validated for standard PiPER"
        )
    if config.get("mode") != "dry_run":
        failures.append(
            "Autonomous hardware door actuation is not implemented/approved"
        )
    for name in (
        "joint_limits_verified",
        "tcp_calibrated",
        "wrist_camera_calibrated",
        "gripper_command_and_feedback_verified",
        "force_limits_verified",
        "emergency_stop_tested",
        "watchdog_tested",
        "base_stop_interface_verified",
        "camera_time_sync_verified",
        "operator_present",
    ):
        if config.get("checks", {}).get(name) is not True:
            failures.append("Missing measured verification: " + name)
    force = robot.get("gripper_force_limit_N")
    if (
        isinstance(force, bool)
        or not isinstance(force, (int, float))
        or not math.isfinite(force)
        or force <= 0
    ):
        failures.append(
            "Real gripper force limit is unknown; do not copy simulation 20 N"
        )
    calibration = config.get("wrist_camera", {}).get("calibration_file")
    if not calibration:
        failures.append("A measured hand-eye calibration file is required")
    else:
        try:
            from .transforms import load_calibration

            load_calibration(calibration, require_measured=True)
        except (ValueError, OSError) as exc:
            failures.append("Calibration rejected: " + str(exc))
    return {
        "ready_for_hardware_door_trial": not failures,
        "autonomous_motion_implemented": False,
        "mode": config.get("mode"),
        "failures": failures,
        "note": "Checklist evidence is not a safety certification. No hardware commands are sent.",
    }


class DryRunBackend:
    """Deliberately no CAN/DDS/ROS import and no hardware transport."""

    def __init__(self):
        self.commands = []

    def send_arm_positions(self, joints_rad):
        q = vector6(joints_rad)
        item = {
            "transport": "dry_run",
            "arm_positions_rad": list(q),
            "sent_to_hardware": False,
        }
        self.commands.append(item)
        return item

    def set_gripper(self, *args, **kwargs):
        raise SafetyError(
            "Real gripper unit/force/readback adapter is not verified; no command sent"
        )

    def send_base_velocity(self, *args, **kwargs):
        raise SafetyError(
            "Base takeover/stop contract is not verified; no command sent"
        )
