import importlib.util
import tarfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "asset_import", Path(__file__).resolve().parents[1] / "scripts/import_reference.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArchiveTests(unittest.TestCase):
    def test_safe_file(self):
        module.validate_member(tarfile.TarInfo("reference/models/a.xml"))

    def test_traversal(self):
        with self.assertRaises(ValueError):
            module.validate_member(tarfile.TarInfo("reference/../../a"))

    def test_absolute(self):
        with self.assertRaises(ValueError):
            module.validate_member(tarfile.TarInfo("/tmp/reference/a"))

    def test_other_root(self):
        with self.assertRaises(ValueError):
            module.validate_member(tarfile.TarInfo("other/a"))

    def test_symlink(self):
        info = tarfile.TarInfo("reference/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp"
        with self.assertRaises(ValueError):
            module.validate_member(info)

    def test_device(self):
        info = tarfile.TarInfo("reference/device")
        info.type = tarfile.CHRTYPE
        with self.assertRaises(ValueError):
            module.validate_member(info)


if __name__ == "__main__":
    unittest.main()
