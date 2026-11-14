"""Pinned MuJoCo Warp 3.12.0 kernels. Audits execute at every 2ms substep."""

import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5


@wp.kernel
def pd(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    bias: wp.array2d[float],
    ctrl: wp.array2d[float],
    ctrlrange: wp.array2d[wp.vec2],
    target: wp.array2d[float],
    motor: wp.array[float],
    kp: wp.array[float],
    kd: wp.array[float],
    torque: wp.array2d[float],
):
    w, j = wp.tid()
    v = j + 6
    q = j + 7
    command = float(0.0)
    if j < 12:
        command = 20.0 * kp[w] * (target[w, j] - qpos[w, q]) - 1.5 * kd[w] * qvel[w, v]
    elif j < 18:
        command = 50.0 * (target[w, j] - qpos[w, q]) - 4.0 * qvel[w, v] + bias[w, v]
    else:
        command = 800.0 * (0.006 - qpos[w, 25]) - 5.0 * qvel[w, 24]
    if j < 12:
        command = command * motor[w]
    low = ctrlrange[0, j][0]
    high = ctrlrange[0, j][1]
    command = wp.clamp(command, low, high)
    ctrl[w, j] = command
    wp.atomic_max(torque, w, j, wp.abs(command))


@wp.kernel
def audit(
    cone: int,
    nacon: wp.array[int],
    worldid: wp.array[int],
    geom: wp.array[wp.vec2i],
    frame: wp.array[wp.mat33],
    friction: wp.array[vec5],
    dim: wp.array[int],
    efc_address: wp.array2d[int],
    force: wp.array2d[float],
    njmax: int,
    dist: wp.array[float],
    classes: wp.array[int],
    feet: wp.array[int],
    forbidden_self: wp.array2d[int],
    out: wp.array2d[float],
):
    c = wp.tid()
    if c < nacon[0]:
        w = worldid[c]
        g1 = geom[c][0]
        g2 = geom[c][1]
        if g1 >= 0 and g2 >= 0:
            f = contact_force_fn(
                cone,
                frame,
                friction,
                dim,
                efc_address,
                force,
                njmax,
                nacon,
                w,
                c,
                False,
            )
            normal = wp.max(0.0, f[0])
            a, b = classes[g1], classes[g2]
            # Classes: ground=0, foot=1, leg=2, trunk=3, arm=4, box=5, rail=6.
            if a == 5 or b == 5:
                other = g2
                if b == 5:
                    other = g1
                fi = feet[other]
                if fi >= 0:
                    wp.atomic_max(out, w, fi, normal)
                if classes[other] == 3 or classes[other] == 4:
                    wp.atomic_max(out, w, 4, normal)
            if (a == 6 and b > 0 and b < 5) or (b == 6 and a > 0 and a < 5):
                wp.atomic_max(out, w, 4, normal)
            if ((a == 3 or a == 4) and b == 0) or ((b == 3 or b == 4) and a == 0):
                wp.atomic_max(out, w, 4, normal)
            if forbidden_self[g1, g2] == 1:
                wp.atomic_max(out, w, 5, normal)
            wp.atomic_max(out, w, 6, normal)
            wp.atomic_max(out, w, 7, wp.max(0.0, -dist[c]))


@wp.kernel
def overflow_audit(flags: wp.array[int], total: wp.array[int]):
    w = wp.tid()
    wp.atomic_or(total, w, flags[w])
