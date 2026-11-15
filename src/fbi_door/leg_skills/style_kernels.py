"""Additional v2 foot-contact audit; existing forbidden-contact audit is retained."""

import warp as wp
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5


@wp.kernel
def foot_contacts(
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
    classes: wp.array[int],
    feet: wp.array[int],
    current: wp.array2d[float],
    peak: wp.array2d[float],
    impulse: wp.array2d[float],
):
    c = wp.tid()
    if c < nacon[0]:
        w = worldid[c]
        g1, g2 = geom[c][0], geom[c][1]
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
            for side in range(2):
                g, other = g1, g2
                if side == 1:
                    g, other = g2, g1
                fi = feet[g]
                if (
                    fi >= 0
                    and (classes[other] == 0 or classes[other] == 5)
                    and wp.abs(frame[c][0, 2]) > 0.65
                ):
                    wp.atomic_add(current, w, fi, normal)
                    wp.atomic_max(peak, w, fi, normal)
                    wp.atomic_add(impulse, w, fi, normal * 0.002)
