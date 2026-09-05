from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release" / "verify_public_repository.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_public_repository", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_public_repo(root: Path) -> None:
    for relative in (
        "CONTRIBUTING.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "DATA_MODEL_PROVENANCE.md",
        ".github/CODEOWNERS",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/model_or_data_issue.yml",
        ".github/pull_request_template.md",
        ".github/workflows/ci.yml",
        ".github/workflows/codeql.yml",
        ".github/dependabot.yml",
        "requirements/base.txt",
        "requirements/ci.txt",
        "env/bootstrap_environment.ps1",
        "env/install_paddledetection.ps1",
        "scripts/release/build_public_data_release.ps1",
        "scripts/release/download_public_release.py",
        "scripts/release/verify_public_release.py",
        "scripts/common/smoke_public_repository.py",
        "scripts/common/target_det_model.py",
        "scripts/onboard/run_target_det_onboard.py",
        "scripts/onboard/check_target_det_baseline_compat.py",
        "scripts/onboard/eval_onboard_samples.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    (root / "README.md").write_text("readme\n", encoding="utf-8")
    (root / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (root / ".github/CODEOWNERS").write_text("* @owner\n", encoding="utf-8")
    (root / ".github/ISSUE_TEMPLATE/config.yml").write_text(
        "blank_issues_enabled: false\ncontact_links:\n", encoding="utf-8"
    )
    (root / ".github/ISSUE_TEMPLATE/bug_report.yml").write_text(
        "name: bug\ndescription: bug\nbody:\n  - id: summary\n  - id: reproduce\n", encoding="utf-8"
    )
    (root / ".github/ISSUE_TEMPLATE/model_or_data_issue.yml").write_text(
        "name: model\ndescription: model\nbody:\n  - id: area\n  - id: dataset\n", encoding="utf-8"
    )
    (root / ".github/pull_request_template.md").write_text(
        "## 这次改动解决什么问题\n## 改动范围\n## 验证记录\n## 风险与回滚\n",
        encoding="utf-8",
    )
    (root / ".github/workflows/ci.yml").write_text(
        "- uses: actions/checkout@" + "a" * 40 + "\n"
        "- uses: actions/setup-python@" + "b" * 40 + "\n"
        "python -m pip check\npython -m unittest discover\n"
        "scripts/release/verify_public_repository.py\n",
        encoding="utf-8",
    )
    (root / ".github/workflows/codeql.yml").write_text(
        "- uses: actions/checkout@" + "a" * 40 + "\n"
        "- uses: github/codeql-action/init@" + "b" * 40 + "\n"
        "- uses: github/codeql-action/analyze@" + "c" * 40 + "\n"
        "languages: python\n",
        encoding="utf-8",
    )
    (root / "RELEASE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": {"tag_name": "data-archives-20260824"},
                "archive": {
                    "part_count": 1,
                    "part_name_pattern": "part-{index:03d}.tar.gz",
                    "required_sidecars": ["SHA256SUMS.txt"],
                    "expected_assets": ["part-001.tar.gz", "SHA256SUMS.txt"],
                },
                "reproduction_tools": {
                    "builder": "scripts/release/build_public_data_release.ps1",
                    "downloader": "scripts/release/download_public_release.py",
                    "verifier": "scripts/release/verify_public_release.py",
                },
            }
        ),
        encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "upstream": "https://github.com/PaddlePaddle/PaddleDetection.git",
        "commit": "59d5f5ebebc2a380f5f07dd413a056b75af01f2a",
        "destination": "third_party/PaddleDetection",
    }
    lock_path = root / "third_party" / "paddledetection.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    models = root / "deploy" / "public_models"
    models.mkdir(parents=True, exist_ok=True)
    archive = models / "target_det_test.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("handoff/handoff_meta.json", "{}")
        package.writestr("handoff/target_det/infer_cfg.yml", "arch: YOLO")
        package.writestr("handoff/target_det/label_list.txt", "water\n")
        package.writestr("handoff/target_det/model.pdiparams", b"params")
        package.writestr("handoff/target_det/model.pdmodel", b"model")
    digest = __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")


class VerifyPublicRepositoryTest(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_accepts_complete_minimal_repository(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)

            report = module.verify_repository(root)

            self.assertTrue(report.ok, report.issues)

    def test_rejects_hash_mismatch_and_private_path(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            (root / "README.md").write_text("/home/aistudio/private/dataset\n", encoding="utf-8")
            (root / "deploy/public_models/target_det_test.zip.sha256").write_text("0" * 64 + "  target_det_test.zip\n", encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("SHA256 mismatch" in issue for issue in report.issues))
            self.assertTrue(any("obsolete path marker" in issue for issue in report.issues))
            self.assertTrue(any("/home/aistudio" in issue for issue in report.issues))

    def test_requires_codeql_workflow(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            (root / ".github/workflows/codeql.yml").unlink()

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertIn("Missing required public repository file: .github/workflows/codeql.yml", report.issues)

    def test_requires_repository_governance_files(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            (root / ".github/CODEOWNERS").unlink()

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertIn("Missing required public repository file: .github/CODEOWNERS", report.issues)

    def test_rejects_unstructured_issue_configuration(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            (root / ".github/ISSUE_TEMPLATE/config.yml").write_text(
                "blank_issues_enabled: true\n", encoding="utf-8"
            )

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("blank issues" in issue for issue in report.issues))

    def test_requires_structured_issue_forms_and_pull_request_template(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            (root / ".github/ISSUE_TEMPLATE/model_or_data_issue.yml").unlink()
            (root / ".github/pull_request_template.md").write_text("## 改动范围\n", encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertIn(
                "Missing required public repository file: .github/ISSUE_TEMPLATE/model_or_data_issue.yml",
                report.issues,
            )
            self.assertTrue(any("pull_request_template.md is missing required section" in issue for issue in report.issues))

    def test_rejects_unpinned_github_action(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            (root / ".github/workflows/ci.yml").write_text(
                "- uses: actions/checkout@v4\n"
                "- uses: actions/setup-python@" + "b" * 40 + "\n"
                "python -m pip check\npython -m unittest discover\n"
                "scripts/release/verify_public_repository.py\n",
                encoding="utf-8",
            )

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("unpinned GitHub Action" in issue for issue in report.issues))

    def test_rejects_release_tool_outside_repository(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            manifest["reproduction_tools"]["builder"] = "../build.ps1"
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("inside the repository" in issue for issue in report.issues))

    def test_requires_sidecars_to_be_listed_assets(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            manifest["archive"]["required_sidecars"] = ["missing.txt"]
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("required_sidecars" in issue for issue in report.issues))

    def test_requires_release_asset_contract(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            del manifest["archive"]["expected_assets"]
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertIn("archive.expected_assets", "\n".join(report.issues))

    def test_requires_declared_release_part_contract(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            del manifest["archive"]["part_count"]
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("part_count" in issue for issue in report.issues))

    def test_requires_all_declared_parts_in_release_assets(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
            manifest["archive"]["part_count"] = 2
            manifest["archive"]["expected_assets"] = ["SHA256SUMS.txt", "part-001.tar.gz"]
            (root / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("missing declared parts" in issue for issue in report.issues))

    def test_ignores_untracked_local_virtual_environment_in_git_worktree(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            local_file = root / ".venv/lib/site-packages/private.py"
            local_file.parent.mkdir(parents=True)
            local_file.write_text("/home/aistudio/private/dataset\n", encoding="utf-8")

            report = module.verify_repository(root)

            self.assertTrue(report.ok, report.issues)

    def test_rejects_private_marker_inside_model_zip(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            archive = root / "deploy/public_models/target_det_test.zip"
            marker = "".join(("Co", "dex"))
            with zipfile.ZipFile(archive, "a") as package:
                package.writestr("handoff/README.md", marker)
            digest = __import__("hashlib").sha256(archive.read_bytes()).hexdigest()
            archive.with_suffix(".zip.sha256").write_text(
                f"{digest}  {archive.name}\n", encoding="utf-8"
            )

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("ZIP member contains private marker" in issue for issue in report.issues))

    def test_validates_package_local_checksum_manifest(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            archive = root / "deploy/public_models/target_det_test.zip"
            model_digest = hashlib.sha256(b"model").hexdigest()
            with zipfile.ZipFile(archive, "a") as package:
                package.writestr(
                    "handoff/target_det/SHA256.txt",
                    f"{model_digest}  model.pdmodel\n",
                )
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            archive.with_suffix(".zip.sha256").write_text(
                f"{digest}  {archive.name}\n", encoding="utf-8"
            )

            report = module.verify_repository(root)

            self.assertTrue(report.ok, report.issues)

    def test_rejects_package_local_checksum_mismatch(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            make_public_repo(root)
            archive = root / "deploy/public_models/target_det_test.zip"
            with zipfile.ZipFile(archive, "a") as package:
                package.writestr("handoff/target_det/SHA256.txt", f"{'0' * 64}  model.pdmodel\n")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            archive.with_suffix(".zip.sha256").write_text(
                f"{digest}  {archive.name}\n", encoding="utf-8"
            )

            report = module.verify_repository(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("ZIP checksum mismatch" in issue for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
