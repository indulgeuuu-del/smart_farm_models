from __future__ import annotations

import importlib.util
import hashlib
import json
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
        ".github/workflows/ci.yml",
        ".github/dependabot.yml",
        "requirements/base.txt",
        "requirements/ci.txt",
        "env/bootstrap_environment.ps1",
        "env/install_paddledetection.ps1",
        "scripts/release/build_public_data_release.ps1",
        "scripts/release/verify_public_release.py",
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
    (root / "RELEASE_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": {"tag_name": "data-archives-20260824"},
                "archive": {"required_sidecars": ["SHA256SUMS.txt"]},
                "reproduction_tools": {
                    "builder": "scripts/release/build_public_data_release.ps1",
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
