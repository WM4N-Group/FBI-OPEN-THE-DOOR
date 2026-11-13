"""Guard against repeating the hosted-runner compatibility regression."""

import unittest
from pathlib import Path


class CIConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = (
            Path(__file__).resolve().parents[1] / ".github/workflows/tests.yml"
        ).read_text()

    def test_legacy_python_has_an_explicit_compatible_image(self):
        self.assertRegex(
            self.workflow,
            r"python:\s*['\"]3\.8['\"]\s*\n\s*os:\s*ubuntu-22\.04",
        )
        self.assertNotRegex(self.workflow, r"runs-on:\s*ubuntu-latest")

    def test_runtime_results_are_not_cancelled_by_fail_fast(self):
        self.assertRegex(self.workflow, r"fail-fast:\s*false")
        self.assertRegex(
            self.workflow,
            r"python:\s*['\"]3\.11['\"]\s*\n\s*os:\s*ubuntu-24\.04",
        )

    def test_motion_free_checks_remain_enabled(self):
        self.assertIn("python -m unittest discover -s tests -v", self.workflow)
        self.assertIn("python -m fbi_door.cli dry-run", self.workflow)
        self.assertIn("python -m fbi_door.cli doctor", self.workflow)
        self.assertIn('test "$result" -eq 2', self.workflow)


if __name__ == "__main__":
    unittest.main()
