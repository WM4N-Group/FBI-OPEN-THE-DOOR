import copy
import json
import tempfile
import unittest
from pathlib import Path
from fbi_door.deployment.safety import (
    CommandGuard,
    JointFeedback,
    SafetyError,
    DryRunBackend,
    preflight,
    vector6,
)
from fbi_door.deployment.transforms import (
    rigid_transform,
    compose,
    transform_point,
    load_calibration,
)

IDENTITY = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.guard = CommandGuard([(-2, 2)] * 6)
        self.feedback = JointFeedback((0.0,) * 6, 10.0)

    def test_small_command(self):
        self.assertEqual(
            self.guard.validate([0.0005] * 6, self.feedback, 10.0), (0.0005,) * 6
        )

    def test_wrong_dimensions(self):
        for values in [[], [0] * 5, [0] * 7, "000000"]:
            with self.assertRaises(SafetyError):
                vector6(values)

    def test_nonfinite_or_bool(self):
        for x in [float("nan"), float("inf"), True, "0"]:
            with self.assertRaises(SafetyError):
                vector6([x] * 6)

    def test_stale(self):
        with self.assertRaises(SafetyError):
            self.guard.validate([0] * 6, self.feedback, 10.2)

    def test_future(self):
        with self.assertRaises(SafetyError):
            self.guard.validate([0] * 6, self.feedback, 9.9)

    def test_excessive_speed(self):
        with self.assertRaises(SafetyError):
            self.guard.validate([0.002] * 6, self.feedback, 10.0, dt_s=0.02)

    def test_tracking_error(self):
        with self.assertRaises(SafetyError):
            self.guard.validate(
                [0.04] * 6, self.feedback, 10.0, previous_target=[0.04] * 6
            )

    def test_target_limits(self):
        with self.assertRaises(SafetyError):
            self.guard.validate([3] * 6, self.feedback, 10.0)

    def test_actual_limits_no_auto_recovery(self):
        with self.assertRaises(SafetyError):
            self.guard.validate([0] * 6, JointFeedback((3.0,) * 6, 10.0), 10.0)

    def test_invalid_time(self):
        for dt in [0, -0.02, float("nan")]:
            with self.assertRaises(SafetyError):
                self.guard.validate([0] * 6, self.feedback, 10.0, dt_s=dt)

    def test_invalid_guard_limits(self):
        with self.assertRaises(SafetyError):
            CommandGuard([(1, -1)] * 6)

    def test_dry_run_does_not_send(self):
        out = DryRunBackend().send_arm_positions([0] * 6)
        self.assertIs(out["sent_to_hardware"], False)

    def test_gripper_is_unimplemented(self):
        with self.assertRaises(SafetyError):
            DryRunBackend().set_gripper(0)

    def test_base_is_unimplemented(self):
        with self.assertRaises(SafetyError):
            DryRunBackend().send_base_velocity(0, 0, 0)

    def test_example_preflight_blocks(self):
        root = Path(__file__).resolve().parents[1]
        result = preflight(
            json.loads((root / "configs/real_robot.example.json").read_text())
        )
        self.assertFalse(result["ready_for_hardware_door_trial"])
        self.assertTrue(any("mismatch" in f for f in result["failures"]))

    def test_flags_cannot_enable_unimplemented_transport(self):
        config = {
            "mode": "dry_run",
            "robot": {"variant": "x", "gripper_force_limit_N": 1},
            "policy": {"robot_variant": "x"},
            "checks": {},
        }
        self.assertFalse(preflight(config)["ready_for_hardware_door_trial"])


class TransformTests(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(transform_point(IDENTITY, [1, 2, 3]), [1, 2, 3])

    def test_composition(self):
        a = copy.deepcopy(IDENTITY)
        a[0][3] = 0.1
        b = copy.deepcopy(IDENTITY)
        b[1][3] = 0.2
        self.assertEqual(transform_point(compose(a, b), [0, 0, 0]), [0.1, 0.2, 0])

    def test_bad_bottom(self):
        bad = copy.deepcopy(IDENTITY)
        bad[3][0] = 1
        with self.assertRaises(ValueError):
            rigid_transform(bad)

    def test_reflection(self):
        bad = copy.deepcopy(IDENTITY)
        bad[0][0] = -1
        with self.assertRaises(ValueError):
            rigid_transform(bad)

    def test_nonorthogonal(self):
        bad = copy.deepcopy(IDENTITY)
        bad[0][0] = 2
        with self.assertRaises(ValueError):
            rigid_transform(bad)

    def test_nominal_calibration_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            load_calibration(root / "configs/hand_eye.example.json")

    def test_wrong_units_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(
                json.dumps({"translation_units": "mm", "T_parent_child": IDENTITY})
            )
            with self.assertRaises(ValueError):
                load_calibration(p)


if __name__ == "__main__":
    unittest.main()
