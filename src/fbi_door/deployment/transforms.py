"""Dependency-free rigid hand-eye transform checks. Translation is in metres."""

import json
import math


def rigid_transform(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        raise ValueError("Expected 4x4 matrix")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for row in value
        for v in row
    ):
        raise ValueError("Transform contains invalid numbers")
    if any(abs(value[3][i] - [0, 0, 0, 1][i]) > 1e-6 for i in range(4)):
        raise ValueError("Invalid homogeneous bottom row")
    r = value
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i] * r[k][j] for k in range(3)) - float(i == j)) > 1e-4:
                raise ValueError("Rotation must be orthonormal")
    det = (
        r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0])
    )
    if abs(det - 1) > 1e-4:
        raise ValueError("Rotation must have determinant +1")
    return value


def compose(a, b):
    rigid_transform(a)
    rigid_transform(b)
    return [
        [sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)
    ]


def transform_point(transform, xyz):
    rigid_transform(transform)
    if len(xyz) != 3 or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        for v in xyz
    ):
        raise ValueError("Point must contain three finite values")
    return [
        sum(transform[i][j] * xyz[j] for j in range(3)) + transform[i][3]
        for i in range(3)
    ]


def load_calibration(path, require_measured=True):
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    if (
        data.get("translation_units") != "m"
        or data.get("parent_frame") != "piper_link6"
        or data.get("child_frame") != "wrist_camera_optical"
    ):
        raise ValueError(
            "Calibration units/frame convention is missing or incompatible"
        )
    if require_measured and (
        data.get("source") != "measured_hand_eye" or not data.get("evidence")
    ):
        raise ValueError("Nominal simulation extrinsics are not measured calibration")
    return rigid_transform(data["T_parent_child"])
