from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "common" / "check_supported_cli.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_supported_cli", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CheckSupportedCliTest(unittest.TestCase):
    def test_inventory_contains_only_existing_maintained_commands(self) -> None:
        module = load_module()
        paths = [REPO_ROOT / relative for relative in module.SUPPORTED_CLI]

        self.assertEqual(len(paths), len(set(paths)))
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertNotIn("scripts/common/target_det_model.py", module.SUPPORTED_CLI)
        self.assertNotIn("scripts/detection/_target_det_recipe.py", module.SUPPORTED_CLI)
        self.assertFalse(any("/legacy/" in relative for relative in module.SUPPORTED_CLI))

    def test_current_repository_help_checks_pass(self) -> None:
        module = load_module()
        issues = module.check_supported_cli(REPO_ROOT, python_executable=sys.executable, timeout=60)

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
