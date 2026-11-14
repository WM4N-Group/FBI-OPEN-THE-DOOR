import math
import unittest
from fbi_door.leg_skills.contracts import (
    domain,
    climb_ready,
    down_ready,
    POLICY_TO_MJ,
    MJ_TO_POLICY,
    OBS_DIM,
)


class LegSkillsTests(unittest.TestCase):
    def test_joint_order_roundtrip(self):
        policy = list(range(12))
        mujoco = [policy[i] for i in POLICY_TO_MJ]
        self.assertEqual([mujoco[i] for i in MJ_TO_POLICY], policy)
        self.assertEqual(OBS_DIM, 320)

    def test_reproducible_disjoint_domains(self):
        for task in ("climb", "down"):
            self.assertEqual(domain(task, 12), domain(task, 12))
            self.assertNotEqual(domain(task, 12), domain(task, 12, split="heldout"))
            self.assertNotEqual(domain(task, 12), domain(task, 13))
        for seed in range(100):
            d = domain("climb", seed)
            self.assertTrue(0.12 <= d["height"] <= 0.24)
            self.assertGreaterEqual(d["length"], 0.85)
            self.assertGreaterEqual(d["mass"], 25.0)
            d = domain("down", seed)
            self.assertTrue(0.075 <= d["clearance_drop"] <= 0.155)
        for bad in (-0.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                domain("climb", 0, bad)
        with self.assertRaises(ValueError):
            domain("wrong", 0)
        with self.assertRaises(ValueError):
            domain("down", 0, split="wrong")

    def test_climb_requires_all_feet_and_stability(self):
        good = ([10.0, 10.0, 10.0, 10.0], True, 0.02, 0.1, 0.99, 0.02, 0.01, 0.01)
        self.assertTrue(climb_ready(*good))
        for index, replacement in (
            (0, [10.0, 10.0, 10.0, 0.0]),
            (1, False),
            (2, 0.2),
            (3, 0.5),
            (4, 0.8),
            (5, 0.2),
            (6, 0.3),
            (7, 0.1),
        ):
            values = list(good)
            values[index] = replacement
            self.assertFalse(climb_ready(*values))

    def test_down_cannot_skip_crouch_or_hit_rail(self):
        good = (2.0, 1.5, 0.18, 0.27, 0.27, 0.02, 0.1, 0.99, 0.02, 0.0, True)
        self.assertTrue(down_ready(*good))
        for index, replacement in (
            (0, 1.6),
            (2, 0.25),
            (4, 0.15),
            (5, 0.2),
            (6, 0.5),
            (7, 0.8),
            (8, 0.2),
            (9, 3.0),
            (10, False),
        ):
            values = list(good)
            values[index] = replacement
            self.assertFalse(down_ready(*values))


class SceneTests(unittest.TestCase):
    def test_build_preserves_robot_and_refuses_overwrite(self):
        from pathlib import Path
        import hashlib
        import tempfile
        from unittest.mock import patch
        import xml.etree.ElementTree as E
        from fbi_door.leg_skills.build_scene import build

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "source"
            assets = src / "assets"
            assets.mkdir(parents=True)
            checkpoint = (
                src / "reference/models/policies/arm_moving_clock/model_7999.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"trusted-test-fixture")
            q = " ".join(str(i) for i in range(34))
            original = (
                '<mujoco><worldbody><body name="base"><geom name="unchanged" size=".2 .1 .05" type="box"/></body></worldbody><keyframe><key name="home" qpos="'
                + q
                + '"/></keyframe></mujoco>'
            )
            (assets / "box_robot.xml").write_text(original)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            with patch("fbi_door.leg_skills.build_scene.LEG_SHA256", digest):
                for task in ("climb", "down"):
                    dst = root / task
                    report = build(src, dst, task)
                    self.assertFalse(report["hardware_enabled"])
                    self.assertEqual((assets / "box_robot.xml").read_text(), original)
                    robot = E.parse(dst / "assets/robot.xml")
                    self.assertEqual(
                        robot.find(".//geom").attrib,
                        {"name": "unchanged", "size": ".2 .1 .05", "type": "box"},
                    )
                    qnew = robot.find(".//key").get("qpos").split()
                    self.assertEqual(qnew[:27], [str(i) for i in range(27)])
                    self.assertEqual(len(qnew), 34 if task == "climb" else 27)
                    scene = E.parse(dst / "assets/scene.xml")
                    self.assertIsNotNone(scene.find('.//geom[@name="ground"]'))
                    self.assertEqual(
                        len(scene.findall(".//freejoint")), 1 if task == "climb" else 0
                    )
                    self.assertIsNone(scene.find(".//weld"))
                    with self.assertRaises(FileExistsError):
                        build(src, dst, task)
