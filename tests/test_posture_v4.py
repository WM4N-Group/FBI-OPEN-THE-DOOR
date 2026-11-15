import math
import unittest
from fbi_door.leg_skills.posture_v4_contracts import (
    shallow_height,
    normal_stance,
    OBS_DIM,
    SCHEMA,
)


class PostureV4Contracts(unittest.TestCase):
    def test_only_needed_depth(self):
        self.assertAlmostEqual(shallow_height(0.30, 0.573, 0.383), 0.175)
        self.assertAlmostEqual(shallow_height(0.30, 0.650, 0.383), 0.252)
        self.assertAlmostEqual(shallow_height(0.30, 0.800, 0.383), 0.30)
        # Do not silently claim a too-low target is feasible by clamping the contract.
        self.assertLess(shallow_height(0.30, 0.500, 0.383), 0.13)

    def test_shallow_does_not_remove_margin(self):
        for opening in (0.52, 0.57, 0.62, 0.7):
            h = shallow_height(0.3, opening, 0.383)
            self.assertLessEqual(h + 0.383, opening - 0.015 + 1e-9)

    def test_normal_means_level_and_correct_support_height(self):
        self.assertTrue(normal_stance(0.30, 0.0, 0.0))
        self.assertTrue(normal_stance(0.29, math.radians(2), math.radians(2)))
        self.assertFalse(normal_stance(0.30, 0.0, math.radians(6)))
        self.assertFalse(normal_stance(0.25, 0.0, 0.0))
        self.assertFalse(normal_stance(0.30, math.nan, 0.0))
        self.assertEqual(OBS_DIM, 392)
        self.assertNotEqual(SCHEMA, "leg-skills/protected-curriculum-v3")

    def test_invalid_dimensions(self):
        for x in (math.nan, math.inf, -0.1):
            with self.assertRaises(ValueError):
                shallow_height(0.3, 0.6, x)
        with self.assertRaises(ValueError):
            shallow_height(0.3, 0.6, 0.38, 0.0)


class DwellRewardTests(unittest.TestCase):
    def test_progress_stops_at_required_duration(self):
        from fbi_door.leg_skills.posture_v4_contracts import bounded_dwell_progress as f

        self.assertAlmostEqual(f(1.0, 1.2, 2.0), 0.2)
        self.assertEqual(f(2.0, 30.0, 2.0), 0.0)
        self.assertEqual(f(30.0, 31.0, 2.0), 0.0)
        self.assertEqual(f(2.0, 0.0, 2.0), -2.0)
        self.assertEqual(f(0.0, 2.0, 2.0) + f(2.0, 0.0, 2.0), 0.0)
