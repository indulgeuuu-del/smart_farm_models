from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class EnvironmentScriptsTest(unittest.TestCase):
    def test_paddledetection_installer_keeps_dependency_resolution_governed(self) -> None:
        text = (REPO_ROOT / "env" / "install_paddledetection.ps1").read_text(encoding="utf-8")

        self.assertIn("sklearn\\s*==\\s*0\\.0", text)
        self.assertIn("--constraint $RequirementsProfile", text)
        self.assertIn("-Encoding UTF8", text)
        self.assertIn("-m pip check", text)


if __name__ == "__main__":
    unittest.main()
