"""Dependency-free curriculum budgeting. Retry budgets never bypass the stage gate."""

import math

BASE_STAGE_STEPS = (
    ("grasp", 150000),
    ("press", 300000),
    ("push", 500000),
    ("full", 800000),
)


def curriculum_plan(scale=1.0, start_stage="grasp", retries=0):
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or not 1 <= scale <= 20
    ):
        raise ValueError("stage scale must be finite and between 1 and 20")
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 0 <= retries <= 5
    ):
        raise ValueError("gate retries must be an integer between 0 and 5")
    stages = [name for name, _ in BASE_STAGE_STEPS]
    if start_stage not in stages:
        raise ValueError("unknown starting stage")
    result = []
    for name, base in BASE_STAGE_STEPS[stages.index(start_stage) :]:
        steps = int(base * scale)
        result.append(
            {
                "mode": "teacher",
                "stage": name,
                "attempt_steps": [steps] + [max(1, steps // 2)] * retries,
            }
        )
    return result


def maximum_requested_steps(plan):
    return sum(sum(stage["attempt_steps"]) for stage in plan)


def with_initial_stage_steps(plan, steps):
    """Shorten only the first normal attempt; retain original bounded retry budgets."""
    import copy

    result = copy.deepcopy(plan)
    if steps is None:
        return result
    if (
        not result
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= result[0]["attempt_steps"][0]
    ):
        raise ValueError(
            "initial stage steps must be positive and no larger than its original budget"
        )
    result[0]["attempt_steps"][0] = steps
    return result
