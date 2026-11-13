import json, numpy as np, mujoco
from .core import Physics, ROOT

p = Physics()
m = p.model
d = p.data
assert m.neq == 2 and all(t == int(mujoco.mjtEq.mjEQ_JOINT) for t in m.eq_type), (
    "Only finger coupling and door latch; no grasp weld allowed."
)
assert len(p.ctrl.past_rl_actions) == 12
assert p.site == m.site("attachment_site").id
for pad in ["finger_left_pad", "finger_right_pad"]:
    g = m.geom(pad).id
    assert m.geom_contype[g] != 0 and m.geom_conaffinity[g] != 0
# Artificial depressed handle WITHOUT hand contact must remain latched: negative unit test.
p.reset()
d.qpos[:2] = [-3, -3]
d.qpos[p.handle.qposadr[0]] = -0.4
mujoco.mj_forward(m, d)
p.step_pd(p.home[7:19], p.home[p.arm_q], 60, 5)
assert d.eq_active[p.latch] and not p.unlocked and not p.grasp_valid and not p.bilateral
assert np.max(p.finger_forces) == 0
result = {
    "passed": True,
    "no_grasp_weld": True,
    "leg_action_dim": 12,
    "physical_finger_colliders": True,
    "depressed_handle_without_grasp_stays_locked": True,
    "gripper_force_limit_candidate_N": m.actuator_ctrlrange[18].tolist(),
    "warning": "Force range is a candidate simulator setting, not verified real hardware capability.",
}
(ROOT / "reports/grasp_contract_tests.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
