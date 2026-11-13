import unittest
from fbi_door.experiments import curriculum_plan, maximum_requested_steps


class ExperimentBudgetTests(unittest.TestCase):
    def test_default_matches_previous_curriculum(self):
        self.assertEqual(maximum_requested_steps(curriculum_plan()), 1750000)

    def test_a800_base_and_retry_caps(self):
        self.assertEqual(4 * maximum_requested_steps(curriculum_plan(2)), 14000000)
        self.assertEqual(
            4 * maximum_requested_steps(curriculum_plan(2, retries=2)), 28000000
        )

    def test_start_stage_is_explicit(self):
        self.assertEqual(
            [s["stage"] for s in curriculum_plan(start_stage="push")], ["push", "full"]
        )

    def test_retries_are_extra_half_stage_not_gate_bypass(self):
        self.assertEqual(
            curriculum_plan(2, retries=2)[0]["attempt_steps"], [300000, 150000, 150000]
        )

    def test_invalid_scale(self):
        for scale in [0, 0.5, 21, float("inf"), float("nan"), True]:
            with self.assertRaises(ValueError):
                curriculum_plan(scale)

    def test_invalid_retry_count(self):
        for n in [-1, 6, 0.5, True]:
            with self.assertRaises(ValueError):
                curriculum_plan(retries=n)

    def test_invalid_stage(self):
        with self.assertRaises(ValueError):
            curriculum_plan(start_stage="skip_grasp")


if __name__ == "__main__":
    unittest.main()
