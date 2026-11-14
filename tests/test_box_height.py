import math
import tempfile
import unittest
import xml.etree.ElementTree as E
from pathlib import Path
from fbi_door.box_height import contact_height, build_height_variant


class BoxHeightTests(unittest.TestCase):
    def test_geometry_only_contact_height(self):
        self.assertEqual(contact_height(0.4), 0.34)
        self.assertAlmostEqual(contact_height(0.3), 0.26)
        self.assertAlmostEqual(contact_height(0.2), 0.16)
        self.assertAlmostEqual(contact_height(0.15), 0.11)
        for h in (True, 0, -1, math.nan, math.inf, 0.03):
            with self.assertRaises(ValueError):
                contact_height(h)
        with self.assertRaises(ValueError):
            contact_height(0.2, 0.2)

    def test_scene_ablation_preserves_robot_and_physics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            src = root / "src"
            (src / "assets").mkdir(parents=True)
            (src / "reference").mkdir()
            robot = E.Element("mujoco")
            w = E.SubElement(robot, "worldbody")
            E.SubElement(w, "geom", name="robot", size=".1 .2 .3")
            keys = E.SubElement(robot, "keyframe")
            E.SubElement(
                keys,
                "key",
                name="home",
                qpos=" ".join(["0"] * 27 + ["1.15", "0", ".201", "1", "0", "0", "0"]),
            )
            E.ElementTree(robot).write(src / "assets/box_robot.xml")
            scene = E.Element("mujoco")
            E.SubElement(scene, "include", file="box_robot.xml")
            w = E.SubElement(scene, "worldbody")
            b = E.SubElement(w, "body", name="push_box", pos="1.15 0 .201")
            E.SubElement(b, "freejoint", name="box_free")
            E.SubElement(
                b,
                "geom",
                name="push_box_geom",
                type="box",
                size=".2 .25 .2",
                mass="2",
                friction=".35 .005 .0001",
            )
            E.ElementTree(scene).write(src / "assets/box_scene.xml")
            before = {p.name: p.read_bytes() for p in (src / "assets").iterdir()}
            manifest = build_height_variant(src, root / "out", 0.2)
            self.assertEqual(manifest["box_mass_kg"], 2)
            self.assertEqual(manifest["box_friction"][0], 0.35)
            self.assertEqual(manifest["box_size_m"], [0.4, 0.5, 0.2])
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in (src / "assets").iterdir()}
            )
            new = E.parse(root / "out/assets/box_robot.xml")
            self.assertEqual(new.find(".//geom").attrib, robot.find(".//geom").attrib)
            q = new.find(".//key").get("qpos").split()
            self.assertAlmostEqual(float(q[-5]), 0.101)
            with self.assertRaises(FileExistsError):
                build_height_variant(src, root / "out", 0.2)
            with self.assertRaises(ValueError):
                build_height_variant(src, root / "bad", 0.05)

    def test_recorded_height_ablation_keeps_weights_and_failure_boundaries(self):
        import json

        root = Path(__file__).resolve().parents[1] / "provenance/box_height_20261114"
        fingerprints = set()
        for name, height in (("h030", 0.3), ("h020", 0.2), ("h015", 0.15)):
            data = json.loads((root / (name + "_regression8.json")).read_text())
            self.assertEqual(data["height_m"], height)
            self.assertFalse(data["learning_performed"])
            self.assertEqual(
                data["policy_parameters_before"], data["policy_parameters_after"]
            )
            self.assertEqual(
                data["training_steps_before"], data["training_steps_after"]
            )
            fingerprints.add(data["policy_parameters_before"])
            self.assertEqual(data["summary"]["strict_ppo"]["front_push_passed"], 0)
            for mode in ("pose_ppo", "pose_ik"):
                expected = 0 if height == 0.15 else 8
                self.assertEqual(data["summary"][mode]["front_push_passed"], expected)
                rows = [r for r in data["results"] if r["mode"] == mode]
                self.assertEqual([r["seed"] for r in rows], list(range(920100, 920108)))
                for row in rows:
                    self.assertEqual(row["self_contact_peak_N"], 0)
                    self.assertEqual(row["non_pad_box_contact_peak_N"], 0)
                    self.assertEqual(row["push_lock_max_target_delta_rad"], 0)
                    if height == 0.15:
                        self.assertEqual(row["failure"], "arm_floor_clearance_stop")
                        self.assertFalse(row["is_success"])
                    else:
                        self.assertTrue(row["is_success"])
                        self.assertGreaterEqual(row["front_impulse_fraction"], 0.9)
                        self.assertGreaterEqual(
                            row["minimum_clearance_m"]["self"], 0.02
                        )
                        self.assertGreaterEqual(
                            row["minimum_clearance_m"]["arm_floor"], 0.07
                        )
        self.assertEqual(len(fingerprints), 1)
