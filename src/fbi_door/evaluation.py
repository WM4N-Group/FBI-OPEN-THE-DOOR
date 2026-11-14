"""Dependency-free paired evaluation summaries; no hardware or simulator imports."""

import math
import statistics
from collections import Counter

CONTROLLERS = ("rule_ik", "rl_residual")


def summarize_paired(results):
    by_controller = {name: {} for name in CONTROLLERS}
    for row in results:
        controller = row["controller"]
        if controller not in by_controller:
            raise ValueError("Unknown controller")
        seed = row["seed"]
        if seed in by_controller[controller]:
            raise ValueError("Duplicate controller/seed pair")
        seconds = row["seconds"]
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError("Invalid episode duration")
        if type(row["full_success"]) is not bool:
            raise ValueError("Success must be an explicit boolean")
        by_controller[controller][seed] = row
    baseline, candidate = (by_controller[name] for name in CONTROLLERS)
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("Need complete, matching paired seeds for both controllers")
    summary = {}
    for name in CONTROLLERS:
        rows = list(by_controller[name].values())
        success = [row for row in rows if row["full_success"]]
        times = [row["seconds"] for row in success]
        failures = Counter(
            (row.get("failure") or "timeout_at_" + row.get("last_phase", "unknown"))
            for row in rows
            if not row["full_success"]
        )
        summary[name] = {
            "episodes": len(rows),
            "full_successes": len(success),
            "mean_seconds_success_only": statistics.mean(times) if times else None,
            "min_max_seconds_success_only": [min(times), max(times)] if times else None,
            "failure_counts": dict(failures),
            "external_contact_diagnostic_episodes": sum(
                bool(
                    row.get("diagnostic_external_nonfinger_contacts_over5N_sampled10Hz")
                )
                for row in rows
            ),
        }
    paired = [
        seed
        for seed in baseline
        if baseline[seed]["full_success"] and candidate[seed]["full_success"]
    ]
    common = {
        "episodes": len(paired),
        "rule_ik_mean_seconds": None,
        "rl_residual_mean_seconds": None,
    }
    if paired:
        common["rule_ik_mean_seconds"] = statistics.mean(
            baseline[s]["seconds"] for s in paired
        )
        common["rl_residual_mean_seconds"] = statistics.mean(
            candidate[s]["seconds"] for s in paired
        )
    return {
        "controllers": summary,
        "matched_success_pairs": common,
        "interpretation": "Per-controller means condition on that controller succeeding; matched-success means use identical successful seed pairs. Small same-asset tests are not universal reliability or hardware evidence.",
    }
