"""Verify that the public repository has its required reproducibility artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "DATA_MODEL_PROVENANCE.md",
    "RELEASE_MANIFEST.json",
    ".github/workflows/ci.yml",
    ".github/dependabot.yml",
    "requirements/base.txt",
    "requirements/ci.txt",
    "third_party/paddledetection.lock.json",
    "env/bootstrap_environment.ps1",
    "env/install_paddledetection.ps1",
    "scripts/release/build_public_data_release.ps1",
    "scripts/release/verify_public_release.py",
    "scripts/common/target_det_model.py",
    "scripts/onboard/run_target_det_onboard.py",
    "scripts/onboard/check_target_det_baseline_compat.py",
    "scripts/onboard/eval_onboard_samples.py",
)
FORBIDDEN_TEXT = ("A CarCarCar", "/home/aistudio")
ZIP_TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}
ZIP_FORBIDDEN_TEXT = (
    "A CarCarCar",
    "/home/aistudio",
    "".join(("co", "dex")),
)
ZIP_DRIVE_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]+")
ZIP_HOME_PATH = re.compile(r"(?i)(?:/home/|/Users/)")
SHA256_LINE = re.compile(r"^([0-9A-Fa-f]{64})\s+[* ]?(.+)$")


@dataclass(frozen=True)
class VerificationReport:
    repo_root: Path
    checked_models: list[str]
    issues: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "repo_root": str(self.repo_root),
            "checked_models": self.checked_models,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root to verify.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sidecar_sha256(sidecar: Path, archive_name: str) -> str:
    lines = [line.strip() for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"SHA256 sidecar must contain exactly one non-empty line: {sidecar}")
    match = SHA256_LINE.fullmatch(lines[0])
    if not match:
        raise ValueError(f"Invalid SHA256 sidecar format: {sidecar}")
    digest, listed_name = match.groups()
    if Path(listed_name).name != archive_name:
        raise ValueError(f"SHA256 sidecar names {listed_name!r}, expected {archive_name!r}: {sidecar}")
    return digest.lower()


def read_zip_sha256_manifest(package: zipfile.ZipFile, manifest_name: str) -> list[str]:
    """Check one package-local checksum manifest and return any issues."""
    issues: list[str] = []
    try:
        lines = package.read(manifest_name).decode("utf-8-sig").splitlines()
    except (UnicodeDecodeError, KeyError, OSError, RuntimeError) as exc:
        return [f"Cannot read ZIP checksum manifest {manifest_name!r}: {exc}"]

    names = set(package.namelist())
    manifest_parent = Path(manifest_name).parent.as_posix()
    prefix = "" if manifest_parent == "." else f"{manifest_parent}/"
    listed_names: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = SHA256_LINE.fullmatch(line.strip())
        if not match:
            issues.append(f"Invalid ZIP checksum manifest line {manifest_name}:{line_number}.")
            continue
        expected, listed = match.groups()
        listed = listed.strip().replace("\\", "/")
        candidates = [listed]
        if prefix and not listed.startswith(prefix):
            candidates.append(prefix + listed)
        resolved = next((candidate for candidate in candidates if candidate in names), None)
        if resolved is None:
            issues.append(
                f"ZIP checksum manifest names missing member {listed!r}: "
                f"{manifest_name}"
            )
            continue
        if resolved in listed_names:
            issues.append(f"Duplicate ZIP checksum entry for {resolved!r}: {manifest_name}")
            continue
        listed_names.add(resolved)
        actual = hashlib.sha256(package.read(resolved)).hexdigest()
        if actual.casefold() != expected.casefold():
            issues.append(
                f"ZIP checksum mismatch for {resolved!r}: expected {expected}, got {actual}"
            )
    return issues


def check_zip_contents(archive: Path) -> list[str]:
    issues: list[str] = []
    try:
        with zipfile.ZipFile(archive) as package:
            bad_member = package.testzip()
            if bad_member is not None:
                return [f"Corrupt ZIP member {bad_member!r}: {archive}"]
            infos = package.infolist()
            names = {info.filename for info in infos}
            for manifest_name in sorted(
                name for name in names if name.endswith(("/SHA256SUMS.txt", "/SHA256.txt"))
            ):
                issues.extend(read_zip_sha256_manifest(package, manifest_name))
            for info in infos:
                member_path = Path(info.filename)
                if member_path.suffix.lower() not in ZIP_TEXT_SUFFIXES:
                    continue
                try:
                    text = package.read(info).decode("utf-8")
                except (UnicodeDecodeError, OSError, RuntimeError):
                    continue
                for marker in ZIP_FORBIDDEN_TEXT:
                    if marker.casefold() in text.casefold():
                        issues.append(
                            f"Public model ZIP member contains private marker {marker!r}: "
                            f"{archive.name}!/{info.filename}"
                        )
                if ZIP_DRIVE_PATH.search(text):
                    issues.append(
                        f"Public model ZIP member contains a drive-qualified path: "
                        f"{archive.name}!/{info.filename}"
                    )
                if ZIP_HOME_PATH.search(text):
                    issues.append(
                        f"Public model ZIP member contains a user home path: "
                        f"{archive.name}!/{info.filename}"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        return [f"Cannot read ZIP archive {archive}: {exc}"]

    if not any(name.endswith("/handoff_meta.json") for name in names):
        issues.append(f"Model package is missing handoff_meta.json: {archive}")
    if archive.name.startswith("target_det_"):
        required_suffixes = (
            "/target_det/infer_cfg.yml",
            "/target_det/label_list.txt",
            "/target_det/model.pdiparams",
        )
        for suffix in required_suffixes:
            if not any(name.endswith(suffix) for name in names):
                issues.append(f"Target detection package is missing {suffix.lstrip('/')}: {archive}")
        if not any(name.endswith("/target_det/model.pdmodel") or name.endswith("/target_det/model.json") for name in names):
            issues.append(f"Target detection package has no supported static model file: {archive}")
    if archive.name.startswith("nav_control_") and not any(name.endswith("/cnn_lane.nb") for name in names):
        issues.append(f"Navigation package is missing cnn_lane.nb: {archive}")
    return issues


def text_files(root: Path) -> list[Path]:
    suffixes = {".json", ".md", ".ps1", ".py", ".txt", ".yaml", ".yml"}
    paths: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if ".git" in relative.parts or relative.parts[0] == "tests":
            continue
        if relative == Path("scripts/release/verify_public_repository.py"):
            continue
        paths.append(path)
    return paths


def verify_repository(repo_root: Path) -> VerificationReport:
    root = repo_root.resolve()
    issues: list[str] = []
    warnings: list[str] = []
    checked_models: list[str] = []

    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            issues.append(f"Missing required public repository file: {relative}")

    lock_path = root / "third_party" / "paddledetection.lock.json"
    if lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            if lock.get("schema_version") != 1:
                issues.append("PaddleDetection lock schema_version must be 1.")
            if lock.get("upstream") != "https://github.com/PaddlePaddle/PaddleDetection.git":
                issues.append("PaddleDetection lock must use the official upstream URL.")
            if not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("commit", ""))):
                issues.append("PaddleDetection lock must contain a 40-character lowercase commit SHA.")
            if lock.get("destination") != "third_party/PaddleDetection":
                issues.append("PaddleDetection lock destination must be third_party/PaddleDetection.")
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"Cannot parse PaddleDetection lock: {exc}")

    release_manifest_path = root / "RELEASE_MANIFEST.json"
    if release_manifest_path.is_file():
        try:
            release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
            release = release_manifest.get("release")
            archive = release_manifest.get("archive")
            tools = release_manifest.get("reproduction_tools")
            if release_manifest.get("schema_version") != 1:
                issues.append("Release manifest schema_version must be 1.")
            if not isinstance(release, dict) or not isinstance(release.get("tag_name"), str) or not release["tag_name"]:
                issues.append("Release manifest must declare a non-empty release.tag_name.")
            if not isinstance(archive, dict) or not isinstance(archive.get("required_sidecars"), list):
                issues.append("Release manifest must declare archive.required_sidecars.")
            if not isinstance(tools, dict) or not all(isinstance(tools.get(key), str) and tools[key] for key in ("builder", "verifier")):
                issues.append("Release manifest must declare reproduction_tools.builder and reproduction_tools.verifier.")
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"Cannot parse release manifest: {exc}")

    models_dir = root / "deploy" / "public_models"
    archives = sorted(models_dir.glob("*.zip")) if models_dir.is_dir() else []
    if not archives:
        issues.append("No public model ZIP packages found under deploy/public_models.")
    for archive in archives:
        checked_models.append(archive.name)
        sidecar = archive.with_suffix(archive.suffix + ".sha256")
        if not sidecar.is_file():
            issues.append(f"Missing SHA256 sidecar for public model package: {archive.name}")
        else:
            try:
                expected = read_sidecar_sha256(sidecar, archive.name)
                actual = sha256_file(archive)
                if actual != expected:
                    issues.append(f"SHA256 mismatch for {archive.name}: expected {expected}, got {actual}")
            except (OSError, ValueError) as exc:
                issues.append(str(exc))
        issues.extend(check_zip_contents(archive))

    for path in text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            warnings.append(f"Skipped non-UTF-8 text file: {path.relative_to(root)}")
            continue
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in text:
                issues.append(f"Private or obsolete path marker {forbidden!r} remains in {path.relative_to(root)}")
        if path.suffix == ".py" and "NotImplementedError" in text:
            issues.append(f"Python entrypoint still contains NotImplementedError placeholder: {path.relative_to(root)}")

    return VerificationReport(root, checked_models, issues, warnings)


def main() -> int:
    args = build_parser().parse_args()
    report = verify_repository(Path(args.repo_root))
    if args.json:
        print(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print(f"STATUS={'OK' if report.ok else 'FAIL'}")
        print(f"CHECKED_MODELS={','.join(report.checked_models)}")
        for warning in report.warnings:
            print(f"WARNING: {warning}")
        for issue in report.issues:
            print(f"ISSUE: {issue}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
