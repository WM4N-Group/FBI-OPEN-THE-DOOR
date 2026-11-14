import math
import unittest
from fbi_door.box_clearance import PHASES, mask, withdrawal_ready, ramp


class ClearanceContractTests(unittest.TestCase):
    def test_arm_and_base_residual_masks(self):
        for phase in ("touch", "push", "stop", "retreat", "retract", "success"):
            self.assertEqual(mask(phase)[3:], (0, 0, 0))
        for phase in ("settle", "touch", "stop", "retreat", "retract"):
            self.assertEqual(mask(phase), (0,) * 6)
        for phase in PHASES:
            self.assertEqual(len(mask(phase)), 6)
        with self.assertRaises(ValueError):
            mask("fold_through_box")

    def test_must_withdraw_and_unload_before_folding(self):
        self.assertTrue(withdrawal_ready(0.21, 0.1, 0, 0.31))
        for args in (
            (0.1, 0.2, 0, 1),
            (0.3, 0.01, 0, 1),
            (0.3, 0.2, 1, 1),
            (0.3, 0.2, 0, 0.1),
            (0.3, 0.2, -1, 1),
        ):
            self.assertFalse(withdrawal_ready(*args))
        for index in range(4):
            for value in (math.nan, math.inf, -math.inf):
                args = [0.3, 0.2, 0, 1]
                args[index] = value
                self.assertFalse(withdrawal_ready(*args))

    def test_base_acceleration_is_bounded(self):
        self.assertAlmostEqual(ramp(0, 0.2, 0.06, 0.02), 0.0012)
        self.assertAlmostEqual(ramp(0.075, -0.06, 0.06, 0.02), 0.0738)
        self.assertAlmostEqual(ramp(0, 0.001, 0.06, 0.02), 0.001)
        for args in ((0, 1, 0, 0.02), (0, 1, 0.1, 0), (0, math.nan, 0.1, 0.02)):
            with self.assertRaises(ValueError):
                ramp(*args)

    def test_paired_regression_records_the_missing_v1_safety_gate(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "provenance/box_clearance_20261114"
        old = json.loads((root / "legacy_audited8.json").read_text())["results"]
        new = json.loads((root / "v2_regression8.json").read_text())["results"]
        self.assertEqual([r["seed"] for r in old], list(range(71000, 71008)))
        self.assertEqual([r["seed"] for r in old], [r["seed"] for r in new])
        for row in old:
            self.assertTrue(row["is_success"])  # Task-only criterion was insufficient.
            self.assertLess(row["minimum_clearance_m"]["self"], 0)
            self.assertFalse(row["passes_v2_clearance_acceptance"])
        for row in new:
            self.assertTrue(row["is_success"])
            self.assertEqual(row["phase_history"], list(PHASES))
            self.assertGreaterEqual(row["minimum_clearance_m"]["self"], 0.02)
            self.assertEqual(row["self_contact_peak_N"], 0)
            self.assertEqual(row["non_pad_box_contact_peak_N"], 0)
            self.assertGreaterEqual(row["minimum_retract_pad_box_gap_m"], 0.08)
            withdraw = row["withdrawal_before_retract"]
            self.assertTrue(
                withdrawal_ready(
                    withdraw["base_retreat_m"],
                    withdraw["pad_gap_m"],
                    withdraw["force_N"],
                    withdraw["quiet_s"],
                )
            )

    def test_new_training_reuses_the_frozen_leg_actor_not_v1_high_level(self):
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "provenance"
        old = json.loads(
            (root / "box_push_20261114/training_snapshot.json").read_text()
        )
        new = json.loads(
            (root / "box_clearance_20261114/training_snapshot.json").read_text()
        )
        actor = "reference/models/policies/arm_moving_clock/exported/policy.onnx"
        original_actor = old["runs"][0]["source_and_asset_sha256"][actor]
        self.assertEqual(len(new["runs"]), 4)
        for run in new["runs"]:
            self.assertEqual(run["controller"], "clearance_v2")
            self.assertEqual(run["source_and_asset_sha256"][actor], original_actor)
            self.assertGreater(run["progress"]["ppo_updates"], 0)
            self.assertTrue(run["zip_crc_valid"])
            self.assertTrue(run["shared_checkpoint_matching"])
