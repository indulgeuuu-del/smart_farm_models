from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "common" / "check_requirements_profiles.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_requirements_profiles", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CheckRequirementsProfilesTest(unittest.TestCase):
    def test_current_profiles_are_consistent(self) -> None:
        module = load_module()

        report = module.validate_profiles(REPO_ROOT / "requirements")

        self.assertTrue(report.ok, report.issues)

    def test_opencv_five_with_numpy_one_is_rejected(self) -> None:
        module = load_module()

        issues = module.validate_packages(
            "base.txt", {"numpy": "1.26.4", "opencv-python": "5.0.0.93"}
        )

        self.assertTrue(any("requires NumPy 2" in issue for issue in issues))

    def test_unpinned_requirement_is_rejected(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "requirements.txt"
            path.write_text("numpy>=1.26\n", encoding="utf-8")

            _, issues = module.read_profile(path)

        self.assertTrue(any("exact package pin" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
