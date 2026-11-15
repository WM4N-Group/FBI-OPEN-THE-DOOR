import math
import unittest
from fbi_door.leg_skills.curriculum_contracts import (
    SCHEMA,
    OBS_DIM,
    STAGES,
    LIMITS,
    next_stage,
    validate_stage,
    usable_style,
)


class CurriculumContracts(unittest.TestCase):
    def test_stage_order_and_real_gate(self):
        self.assertEqual(next_stage("climb", "settle", 23, 32), "settle")
        self.assertEqual(next_stage("climb", "settle", 24, 32), "full")
        self.assertEqual(next_stage("down", "crouch", 24, 32), "crawl")
        self.assertEqual(next_stage("down", "crawl", 24, 32), "full")
        self.assertEqual(next_stage("down", "full", 32, 32), "full")
        self.assertEqual(STAGES["down"], ("crouch", "crawl", "full"))

    def test_fail_closed(self):
        for task, stage in (("door", "full"), ("climb", "crawl"), ("down", "settle")):
            with self.assertRaises(ValueError):
                validate_stage(task, stage)
        for good, total, limit in (
            (1, 0, 0.75),
            (33, 32, 0.75),
            (-1, 32, 0.75),
            (1, 32, math.nan),
            (1, 32, 0),
        ):
            with self.assertRaises(ValueError):
                next_stage("down", "crouch", good, total, limit)

    def test_style_not_relaxed(self):
        self.assertTrue(usable_style("down", 0.13, 0.15, 0.08, 0.15))
        self.assertFalse(usable_style("down", 0.2, 0.15, 0.08, 0.15))
        self.assertFalse(usable_style("climb", 0.2, 0.2, 0.09, 0.3))
        self.assertFalse(usable_style("down", math.nan, 0.1, 0.05, 0.1))
        self.assertEqual(LIMITS["down"]["swing_m"], 0.14)
        self.assertEqual(LIMITS["climb"]["swing_m"], 0.23)
        self.assertEqual(OBS_DIM, 384)
        self.assertNotEqual(SCHEMA, "leg-skills/small-step-v2")
