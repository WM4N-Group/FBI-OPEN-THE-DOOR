"""Safe entry point: no network/hardware transport in default commands."""

import argparse
import json
import runpy
import sys
import time
from pathlib import Path
from .paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser(
        "doctor", help="Offline migration checklist; does not read hardware"
    )
    doctor.add_argument(
        "--config",
        default=str(
            (REPO_ROOT / "configs/real_robot.example.json")
            if (REPO_ROOT / "configs/real_robot.example.json").is_file()
            else Path(__file__).parent / "data/real_robot.example.json"
        ),
    )
    sub.add_parser("dry-run", help="Validate a tiny synthetic command; no CAN/DDS/ROS")
    modules = {
        "sim-build": "build_scene",
        "sim-probe": "probe",
        "sim-train": "train",
        "camera-train": "dual_perception",
        "camera-probe": "camera_probe",
        "check-door": "check_freehinge",
        "check-grasp": "test_grasp_contract",
        "check-camera": "check_wrist_camera",
    }
    for name in modules:
        sub.add_parser(name, add_help=False)
    args, rest = parser.parse_known_args()
    if args.command == "doctor":
        if rest:
            parser.error("Unknown arguments: " + " ".join(rest))
        from .deployment.safety import preflight

        with open(args.config, encoding="utf-8") as stream:
            result = preflight(json.load(stream))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ready_for_hardware_door_trial"] else 2
    if args.command == "dry-run":
        if rest:
            parser.error("Unknown arguments: " + " ".join(rest))
        from .deployment.safety import CommandGuard, JointFeedback, DryRunBackend

        q = [0.0, 0.9, -1.8, 0.0, 0.0, 0.0]
        limits = [
            (-2.0, 2.0),
            (0, 3.0),
            (-2.8, 0),
            (-1.7, 1.7),
            (-1.2, 1.2),
            (-2.0, 2.0),
        ]
        now = time.monotonic()
        target = q.copy()
        target[0] += 0.0005
        command = CommandGuard(limits).validate(
            target, JointFeedback(tuple(q), now), now, dt_s=0.02
        )
        print(json.dumps(DryRunBackend().send_arm_positions(command), indent=2))
        return 0
    sys.argv = [args.command] + rest
    runpy.run_module("fbi_door.sim." + modules[args.command], run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
