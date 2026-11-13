import errno
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from fbi_door.storage import atomic_json, atomic_save_model, RuntimeMirror
from fbi_door.experiments import (
    curriculum_plan,
    with_initial_stage_steps,
    maximum_requested_steps,
)


class DummyModel:
    def save(self, path):
        with zipfile.ZipFile(path, "w") as z:
            for name in ["data", "policy.pth", "policy.optimizer.pth"]:
                z.writestr(name, "test-data")


class BrokenModel:
    def save(self, path):
        Path(path).write_bytes(b"incomplete")
        raise OSError(errno.ENOSPC, "simulated full device")


class StorageTests(unittest.TestCase):
    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "status.json"
            atomic_json(p, {"ok": True})
            self.assertEqual(json.loads(p.read_text()), {"ok": True})

    def test_atomic_checkpoint_crc(self):
        with tempfile.TemporaryDirectory() as d:
            p = atomic_save_model(DummyModel(), Path(d) / "policy")
            with zipfile.ZipFile(p) as z:
                self.assertIsNone(z.testzip())

    def test_failed_save_preserves_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "policy.zip"
            atomic_save_model(DummyModel(), p)
            old = p.read_bytes()
            with self.assertRaises(OSError):
                atomic_save_model(BrokenModel(), p)
            self.assertEqual(p.read_bytes(), old)
            self.assertEqual(len(list(Path(d).iterdir())), 1)

    def test_mirror_retries_without_deleting_local(self):
        with tempfile.TemporaryDirectory() as l, tempfile.TemporaryDirectory() as s:
            p = atomic_save_model(DummyModel(), Path(l) / "runs/example/policy")
            mirror = RuntimeMirror(l, s, ["runs/example"])
            with patch(
                "fbi_door.storage.atomic_bytes",
                side_effect=OSError(errno.ENOSPC, "injected"),
            ):
                result = mirror.sync_once()
                self.assertTrue(result["errors"])
                self.assertTrue(p.exists())
            result = mirror.sync_once()
            self.assertFalse(result["errors"])
            self.assertEqual(result["pending"], 0)
            self.assertEqual(
                p.read_bytes(), (Path(s) / "runs/example/policy.zip").read_bytes()
            )

    def test_hidden_and_partial_files_not_mirrored(self):
        with tempfile.TemporaryDirectory() as l, tempfile.TemporaryDirectory() as s:
            root = Path(l) / "runs/a"
            root.mkdir(parents=True)
            (root / ".half.tmp.zip").write_bytes(b"partial")
            (root / "status.tmp").write_text("{}")
            (root / "bad.zip").write_bytes(b"partial")
            mirror = RuntimeMirror(l, s, ["runs/a"])
            self.assertEqual(mirror.sync_once()["copied"], 0)

    def test_no_mirror_path_escape(self):
        with tempfile.TemporaryDirectory() as l, tempfile.TemporaryDirectory() as s:
            for bad in ["../other", "/absolute"]:
                with self.assertRaises(ValueError):
                    RuntimeMirror(l, s, [bad])

    def test_no_nested_roots(self):
        with tempfile.TemporaryDirectory() as l:
            with self.assertRaises(ValueError):
                RuntimeMirror(l, Path(l) / "nested", ["runs"])

    def test_unchanged_file_not_rewritten(self):
        with tempfile.TemporaryDirectory() as l, tempfile.TemporaryDirectory() as s:
            p = Path(l) / "runs/a/status.json"
            atomic_json(p, {"n": 1})
            m = RuntimeMirror(l, s, ["runs/a"])
            self.assertEqual(m.sync_once()["copied"], 1)
            self.assertEqual(m.sync_once()["copied"], 0)
            atomic_json(p, {"n": 2})
            self.assertEqual(m.sync_once()["copied"], 1)

    def test_remaining_budget_keeps_retry_limits(self):
        original = curriculum_plan(2, "push", 2)
        adjusted = with_initial_stage_steps(original, 319408)
        self.assertEqual(adjusted[0]["attempt_steps"], [319408, 500000, 500000])
        self.assertEqual(original[0]["attempt_steps"][0], 1000000)
        self.assertEqual(
            maximum_requested_steps(
                with_initial_stage_steps(curriculum_plan(2, "push"), 319408)
            ),
            1919408,
        )

    def test_bad_remaining_budget(self):
        for n in [0, -1, True, 1000001, 1.5]:
            with self.assertRaises(ValueError):
                with_initial_stage_steps(curriculum_plan(2, "push"), n)


if __name__ == "__main__":
    unittest.main()
