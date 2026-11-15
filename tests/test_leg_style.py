import math
import unittest
from fbi_door.leg_skills.style_contracts import (
    STYLE_OBS_DIM,
    STYLE_SCHEMA,
    crouch_ready,
    style_acceptable,
    style_limits,
    target_increment,
)


class SmallStepContracts(unittest.TestCase):
    def test_versioned_observation(self):
        self.assertEqual(STYLE_OBS_DIM, 360)
        self.assertNotEqual(STYLE_SCHEMA, "leg-skills/v1")

    def test_slew_limits(self):
        self.assertAlmostEqual(target_increment("down"), 0.024)
        self.assertAlmostEqual(target_increment("climb"), 0.12)
        for bad in (0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                target_increment("down", bad)

    def test_crouch_before_crawl_requires_real_clearance_and_support(self):
        good = (0.14, 0.145, 0.49, 0.52, 0.02, 0.1, 4, 0.03)
        self.assertTrue(crouch_ready(*good))
        for index, bad in (
            (0, 0.23),
            (2, 0.515),
            (4, 0.08),
            (5, 0.5),
            (6, 2),
            (7, 0.2),
            (2, math.nan),
        ):
            values = list(good)
            values[index] = bad
            self.assertFalse(crouch_ready(*values))

    def test_style_acceptance_distinguishes_crossing_from_small_steps(self):
        self.assertTrue(style_acceptable("down", 0.10, 0.11, 0.05, 0.1))
        self.assertFalse(style_acceptable("down", 0.20, 0.11, 0.05, 0.1))
        self.assertFalse(style_acceptable("down", 0.10, 0.11, 0.05, 0.3))
        self.assertFalse(style_acceptable("down", 0.10, 0.11, math.nan, 0.1))
        self.assertTrue(style_acceptable("climb", 0.20, 0.18, 0.08, 0.2))
        self.assertFalse(style_acceptable("climb", 0.35, 0.18, 0.08, 0.2))

    def test_configs_are_copies_and_down_is_slower(self):
        a = style_limits("down")
        a["hold_s"] = 99
        self.assertEqual(style_limits("down")["hold_s"], 1.0)
        self.assertLess(
            style_limits("down")["command_speed_max"],
            style_limits("climb")["command_speed_max"],
        )
        with self.assertRaises(ValueError):
            style_limits("other")
