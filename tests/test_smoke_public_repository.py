from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "common" / "smoke_public_repository.py"


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_public_repository", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokePublicRepositoryTest(unittest.TestCase):
    def test_current_readme_satisfies_public_contract(self) -> None:
        module = load_module()

        self.assertEqual(module.check_readme_contract(REPO_ROOT), [])

    def test_readme_contract_reports_missing_sections(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# smart_farm_models\n", encoding="utf-8")

            issues = module.check_readme_contract(root)

        self.assertTrue(any("Windows" in issue for issue in issues))
        self.assertTrue(any("公开模型" in issue for issue in issues))

    def test_readme_documents_every_supported_cli(self) -> None:
        module = load_module()

        self.assertEqual(module.check_readme_cli_coverage(REPO_ROOT), [])

    def test_readme_cli_coverage_reports_missing_entry(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# smart_farm_models\n", encoding="utf-8")
            checker_path = root / "scripts" / "common" / "check_supported_cli.py"
            checker_path.parent.mkdir(parents=True)
            checker_path.write_text(
                "SUPPORTED_CLI = ('scripts/example.py',)\n",
                encoding="utf-8",
            )

            issues = module.check_readme_cli_coverage(root)

        self.assertEqual(issues, ["README.md does not document supported CLI: scripts/example.py"])

    def test_readme_cli_coverage_accepts_windows_path_separators(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("scripts\\example.py\n", encoding="utf-8")
            checker_path = root / "scripts" / "common" / "check_supported_cli.py"
            checker_path.parent.mkdir(parents=True)
            checker_path.write_text(
                "SUPPORTED_CLI = ('scripts/example.py',)\n",
                encoding="utf-8",
            )

            issues = module.check_readme_cli_coverage(root)

        self.assertEqual(issues, [])

    def test_python_source_check_compiles_maintained_files(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scripts").mkdir()
            (root / "tests").mkdir()
            (root / "scripts" / "ok.py").write_text("value = 1\n", encoding="utf-8")
            (root / "tests" / "ok_test.py").write_text("assert True\n", encoding="utf-8")

            issues = module.check_python_sources(root)

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
