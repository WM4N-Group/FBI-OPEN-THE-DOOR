"""Offline kinematic + static-load leg targets; never writes rollout qpos.

Reference is a controller baseline, NOT evidence of RL learning. Four nonnegative
vertical support forces balance floating-base gravity. PPO learns bounded joint
corrections around this reference/retargeted pretrained gait.
"""

import numpy as np
import mujoco
from scipy.optimize import least_squares, lsq_linear


def table(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    home = data.qpos.copy()
    mujoco.mj_forward(model, data)
    ids = [model.geom(n).id for n in ("FL", "FR", "RL", "RR")]
    feet = data.geom_xpos[ids].copy()
    feet[:, 2] = 0.022
    heights = np.linspace(0.13, 0.31, 37)
    poses = []
    torques = []
    errors = []
    for h in heights:
        data.qpos[:] = home
        data.qpos[2] = h
        data.qpos[25:27] = [0.006, -0.006]
        data.qvel[:] = 0

        def loss(q):
            data.qpos[7:19] = q
            mujoco.mj_forward(model, data)
            return (data.geom_xpos[ids] - feet).flatten()

        q = least_squares(
            loss,
            home[7:19],
            bounds=(model.jnt_range[1:13, 0] + 0.015, model.jnt_range[1:13, 1] - 0.015),
            max_nfev=100,
        ).x
        data.qpos[7:19] = q
        mujoco.mj_forward(model, data)
        jac = []
        for g in ids:
            jp = np.zeros((3, model.nv))
            jr = jp.copy()
            mujoco.mj_jac(model, data, jp, jr, data.geom_xpos[g], model.geom_bodyid[g])
            jac.append(jp[2])
        j = np.asarray(jac)
        support = lsq_linear(j[:, :6].T, data.qfrc_bias[:6], bounds=(0, 200.0)).x
        torque = data.qfrc_bias[6:18] - j[:, 6:18].T @ support
        error = float(np.max(np.abs(loss(q))))
        if error > 0.003:
            raise RuntimeError(("Infeasible stance reference", h, error))
        if np.any(np.abs(torque) > model.actuator_ctrlrange[:12, 1]):
            raise RuntimeError("Reference exceeds actuator torque limit")
        poses.append(q.copy())
        torques.append(torque)
        errors.append(error)
    return dict(
        heights=heights,
        poses=np.array(poses),
        torques=np.array(torques),
        fk_error_max_m=max(errors),
        scope="nominal static stance reference, not a verified real robot model or learned gait",
    )
