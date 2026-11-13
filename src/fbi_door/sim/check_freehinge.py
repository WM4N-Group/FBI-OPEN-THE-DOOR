"""Unit tests initialize hinge state; task rollouts never write door angles."""

import json
import mujoco
from .core import Physics, ROOT

p = Physics()
m = p.model
d = p.data
jid = m.joint("door_hinge").id
qi = int(m.jnt_qposadr[jid])
vi = int(m.jnt_dofadr[jid])
eq = p.latch
assert (
    m.dof_damping[vi] == 0 and m.dof_frictionloss[vi] == 0 and m.jnt_stiffness[jid] == 0
)
out = {
    "task_version": "door-grasp-v3",
    "compiled_damping": float(m.dof_damping[vi]),
    "compiled_frictionloss": float(m.dof_frictionloss[vi]),
    "compiled_stiffness": float(m.jnt_stiffness[jid]),
    "tests": [],
}


def init(angle, speed=0, unlocked=True):
    p.reset()
    d.qpos[0] = -3.0
    d.qpos[1] = -3.0
    d.qpos[qi] = angle
    d.qvel[vi] = speed
    d.eq_active[eq] = not unlocked
    mujoco.mj_forward(m, d)


def simulate(seconds):
    angles = []
    vel = []
    forces = []
    for _ in range(round(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
        angles.append(float(d.qpos[qi]))
        vel.append(float(d.qvel[vi]))
        forces.append(float(d.qfrc_passive[vi]))
    return angles, vel, forces


# Open leaf should not have a spring-driven return after release.
for angle in [0.4, 0.8, 1.2]:
    init(angle)
    aa, vv, ff = simulate(3.0)
    delta = float(d.qpos[qi] - angle)
    assert abs(delta) < 0.003, (angle, delta)
    assert max(map(abs, ff)) < 1e-10
    out["tests"].append(
        {
            "name": "released_at_rest",
            "initial_angle_rad": angle,
            "seconds": 3,
            "final_angle_rad": float(d.qpos[qi]),
            "drift_rad": delta,
            "max_abs_passive_hinge_torque_Nm": max(map(abs, ff)),
        }
    )
# Ideal undamped hinge coasts after an opening impulse; no passive braking.
init(0.3, 0.15)
aa, vv, ff = simulate(2.0)
assert abs(d.qvel[vi] - 0.15) < 0.003 and d.qpos[qi] > 0.58
out["tests"].append(
    {
        "name": "coasts_after_push",
        "initial_angle_rad": 0.3,
        "initial_speed_rad_s": 0.15,
        "seconds": 2,
        "final_angle_rad": float(d.qpos[qi]),
        "final_speed_rad_s": float(d.qvel[vi]),
        "max_abs_passive_hinge_torque_Nm": max(map(abs, ff)),
    }
)
# Closed latch remains intact under 8 Nm hinge loading; release still requires the handle.
init(0, unlocked=False)
for _ in range(500):
    d.qfrc_applied[vi] = 8
    mujoco.mj_step(m, d)
assert abs(d.qpos[qi]) < 0.015 and d.eq_active[eq]
out["tests"].append(
    {
        "name": "locked_until_handle_pressed",
        "external_hinge_torque_Nm": 8,
        "seconds": 1,
        "final_angle_rad": float(d.qpos[qi]),
        "latch_active": bool(d.eq_active[eq]),
    }
)
h = m.joint("handle_hinge").id
assert m.jnt_stiffness[h] == 1.0
out.update(
    passed=True,
    handle_return_spring_retained=True,
    note="Unit tests only; initialization changes qpos, task rollout physics is unchanged. No claim of completed door traversal.",
)
report = ROOT / "reports/freehinge_change_20261113/physics_tests.json"
report.write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
