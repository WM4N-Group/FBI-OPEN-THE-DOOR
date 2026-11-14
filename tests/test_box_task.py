import unittest
from fbi_door.box_task import PHASES, residual_mask, push_valid


class BoxContractTest(unittest.TestCase):
    def test_locked_arm_cannot_receive_residual(self):
        self.assertEqual(residual_mask("push"), (1, 1, 1, 0, 0, 0))
        for phase in ("stop", "retract", "success"):
            self.assertEqual(residual_mask(phase), (0,) * 6)
        for phase in PHASES:
            self.assertEqual(len(residual_mask(phase)), 6)
        with self.assertRaises(ValueError):
            residual_mask("door")

    def test_physical_contact_and_full_displacement_required(self):
        self.assertTrue(push_valid(0.52, 1, 0, 0.01, 0.02))
        for args in (
            (0.49, 1, 0, 0, 0),
            (0.6, 0, 0, 0, 0),
            (0.6, 1, 0.6, 0, 0),
            (0.6, 1, 0, 0.3, 0),
            (0.6, 1, 0, 0, 0.4),
        ):
            self.assertFalse(push_valid(*args))

    def test_invalid_evidence_fails_closed(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for index in range(5):
                args = [0.6, 1, 0, 0, 0]
                args[index] = value
                self.assertFalse(push_valid(*args))
        self.assertFalse(push_valid(0.6, 1, -1, 0, 0))

    def test_recorded_box_baseline_preserves_push_contract(self):
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[1]
            / "provenance/box_push_20261114/baseline_pose8.json"
        )
        data = json.loads(path.read_text())
        self.assertEqual(data["total"], 8)
        self.assertEqual(data["passed"], 8)
        self.assertEqual(
            [r["seed"] for r in data["results"]], list(range(71000, 71008))
        )
        for row in data["results"]:
            self.assertTrue(row["is_success"])
            self.assertEqual(row["phase_history"], list(PHASES))
            self.assertGreaterEqual(row["box_displacement_m"], 0.5)
            self.assertGreaterEqual(row["arm_box_contact_s"], 0.4)
            self.assertEqual(row["non_arm_box_impulse_Ns"], 0)
            self.assertEqual(row["push_lock_max_target_delta_rad"], 0)
            self.assertLess(row["retract_joint_error_rad"], 0.15)
