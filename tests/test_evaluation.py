import json
import unittest
from pathlib import Path
from fbi_door.evaluation import summarize_paired


def row(controller, seed, success=True, seconds=10.0):
    return {
        "controller": controller,
        "seed": seed,
        "full_success": success,
        "seconds": seconds,
        "last_phase": "success" if success else "press",
        "failure": None,
    }


class PairedEvaluationTests(unittest.TestCase):
    def test_success_means_and_matched_cohort_are_distinct(self):
        data = [
            row("rule_ik", 1, True, 10),
            row("rule_ik", 2, False, 55),
            row("rl_residual", 1, True, 8),
            row("rl_residual", 2, True, 9),
        ]
        s = summarize_paired(data)
        self.assertEqual(s["controllers"]["rule_ik"]["full_successes"], 1)
        self.assertEqual(
            s["controllers"]["rl_residual"]["mean_seconds_success_only"], 8.5
        )
        self.assertEqual(s["matched_success_pairs"]["rl_residual_mean_seconds"], 8)
        self.assertEqual(
            s["controllers"]["rule_ik"]["failure_counts"], {"timeout_at_press": 1}
        )

    def test_incomplete_pairs_rejected(self):
        with self.assertRaises(ValueError):
            summarize_paired([row("rule_ik", 1)])

    def test_duplicate_pair_rejected(self):
        with self.assertRaises(ValueError):
            summarize_paired(
                [row("rule_ik", 1), row("rule_ik", 1), row("rl_residual", 1)]
            )

    def test_nonfinite_duration_rejected(self):
        for n in [float("nan"), float("inf"), -1, True]:
            with self.assertRaises(ValueError):
                summarize_paired([row("rule_ik", 1, seconds=n), row("rl_residual", 1)])

    def test_no_success_does_not_invent_a_time(self):
        s = summarize_paired(
            [row("rule_ik", 1, False, 55), row("rl_residual", 1, False, 55)]
        )
        self.assertIsNone(s["controllers"]["rule_ik"]["mean_seconds_success_only"])
        self.assertIsNone(s["matched_success_pairs"]["rl_residual_mean_seconds"])

    def test_recorded_public_results_recompute(self):
        base = Path(__file__).resolve().parents[1] / "provenance/evaluation_20261114"
        actual = summarize_paired(json.loads((base / "results.json").read_text()))
        recorded = json.loads((base / "summary.json").read_text())
        for controller in ["rule_ik", "rl_residual"]:
            a = actual["controllers"][controller]
            b = recorded["controllers"][controller]
            self.assertEqual(a["full_successes"], b["full_successes"])
            self.assertEqual(a["episodes"], b["episodes"])
            self.assertAlmostEqual(
                a["mean_seconds_success_only"],
                b["mean_seconds_success_only"],
                places=10,
            )
        self.assertEqual(actual["controllers"]["rule_ik"]["full_successes"], 28)
        self.assertEqual(actual["controllers"]["rl_residual"]["full_successes"], 32)


if __name__ == "__main__":
    unittest.main()
