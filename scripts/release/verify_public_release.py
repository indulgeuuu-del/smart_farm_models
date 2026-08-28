"""Verify a public data release before extraction or publication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_LINE = re.compile(r"^([0-9A-Fa-f]{64})\s+[* ]?(.+)$")
REQUIRED_FILES = (
    "SHA256SUMS.txt",
    "source_manifest.tsv",
    "part_plan.csv",
    "scope.json",
    "RELEASE_NOTES.md",
)


@dataclass(frozen=True)
class ReleaseReport:
    release_dir: Path
    checked_archives: list[str]
    manifest_files: int
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "release_dir": str(self.release_dir),
            "checked_archives": self.checked_archives,
            "manifest_files": self.manifest_files,
            "issues": self.issues,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", required=True, help="Directory containing release sidecars and part archives.")
    parser.add_argument(
        "--extract-dir",
        help="Optional empty directory. Safely extract and compare every file against source_manifest.tsv.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON report.")
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or not path.parts or path.is_absolute():
        raise ValueError(f"absolute or empty relative path: {value!r}")
    if len(path.parts[0]) >= 2 and path.parts[0][1] == ":":
        raise ValueError(f"drive-qualified path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return "/".join(path.parts)


def read_manifest(path: Path) -> tuple[dict[str, int], list[str]]:
    manifest: dict[str, int] = {}
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header != ["path", "bytes"]:
                issues.append("source_manifest.tsv must start with the header 'path<TAB>bytes'.")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != 2:
                    issues.append(f"source_manifest.tsv line {line_number} must contain path and bytes.")
                    continue
                try:
                    relative = safe_relative_path(row[0])
                    size = int(row[1])
                    if size < 0:
                        raise ValueError("negative size")
                except ValueError as exc:
                    issues.append(f"Invalid source_manifest.tsv line {line_number}: {exc}")
                    continue
                if relative in manifest:
                    issues.append(f"Duplicate source manifest path: {relative}")
                    continue
                manifest[relative] = size
    except (OSError, UnicodeError) as exc:
        issues.append(f"Cannot read source_manifest.tsv: {exc}")
    return manifest, issues


def read_sums(path: Path) -> tuple[dict[str, str], list[str]]:
    expected: dict[str, str] = {}
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return expected, [f"Cannot read SHA256SUMS.txt: {exc}"]
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = SHA256_LINE.fullmatch(line.strip())
        if not match:
            issues.append(f"Invalid SHA256SUMS.txt line {line_number}.")
            continue
        digest, name = match.groups()
        name = name.strip()
        if "/" in name or "\\" in name or not name:
            issues.append(f"SHA256SUMS.txt must contain archive basenames only: {name!r}")
            continue
        if name in expected:
            issues.append(f"Duplicate SHA256 entry: {name}")
        expected[name] = digest.lower()
    return expected, issues


def read_part_plan(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    plans: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Part", "Files", "Bytes"}
            if not required.issubset(set(reader.fieldnames or [])):
                issues.append("part_plan.csv is missing one of Part, Files, Bytes.")
            for row in reader:
                part = (row.get("Part") or "").strip()
                if not part:
                    issues.append("part_plan.csv contains an empty part name.")
                    continue
                if part in plans:
                    issues.append(f"Duplicate part plan entry: {part}")
                row["Archive"] = (row.get("Archive") or f"{part}.tar.gz").strip()
                plans[part] = row
    except (OSError, UnicodeError, csv.Error) as exc:
        issues.append(f"Cannot read part_plan.csv: {exc}")
    return plans, issues


def tar_file_stats(path: Path) -> tuple[set[str], int, int, list[str]]:
    names: set[str] = set()
    files = 0
    total_bytes = 0
    issues: list[str] = []
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            for member in archive.getmembers():
                try:
                    relative = safe_relative_path(member.name)
                except ValueError as exc:
                    issues.append(f"Unsafe archive member {member.name!r}: {exc}")
                    continue
                if member.isdir():
                    continue
                if not member.isfile():
                    issues.append(f"Archive contains non-regular member {member.name!r}; links are not allowed.")
                    continue
                if relative in names:
                    issues.append(f"Archive contains duplicate file member: {relative}")
                names.add(relative)
                files += 1
                total_bytes += member.size
    except (OSError, tarfile.TarError) as exc:
        issues.append(f"Cannot read archive {path.name}: {exc}")
    return names, files, total_bytes, issues


def compare_extracted(extract_dir: Path, manifest: dict[str, int]) -> list[str]:
    issues: list[str] = []
    actual: dict[str, int] = {}
    for path in extract_dir.rglob("*"):
        if path.is_file():
            relative = path.relative_to(extract_dir).as_posix()
            actual[relative] = path.stat().st_size
    for relative, expected_size in manifest.items():
        if relative not in actual:
            issues.append(f"Extracted file is missing: {relative}")
        elif actual[relative] != expected_size:
            issues.append(f"Extracted file size mismatch for {relative}: expected {expected_size}, got {actual[relative]}")
    extras = sorted(set(actual) - set(manifest))
    issues.extend(f"Extracted file is not in source_manifest.tsv: {relative}" for relative in extras)
    return issues


def verify_release(release_dir: Path, extract_dir: Path | None = None) -> ReleaseReport:
    root = release_dir.resolve()
    issues: list[str] = []
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            issues.append(f"Missing release sidecar: {name}")

    manifest, manifest_issues = read_manifest(root / "source_manifest.tsv")
    issues.extend(manifest_issues)
    expected_sums, sum_issues = read_sums(root / "SHA256SUMS.txt")
    issues.extend(sum_issues)
    plans, plan_issues = read_part_plan(root / "part_plan.csv")
    issues.extend(plan_issues)

    archives = sorted(root.glob("part-*.tar.gz"))
    checked_archives = [archive.name for archive in archives]
    all_archive_members: set[str] = set()
    if not archives:
        issues.append("No part-*.tar.gz archives found.")
    for archive in archives:
        expected = expected_sums.get(archive.name)
        if expected is None:
            issues.append(f"Archive is missing from SHA256SUMS.txt: {archive.name}")
        elif sha256_file(archive) != expected:
            issues.append(f"SHA256 mismatch for {archive.name}")
        names, files, total_bytes, archive_issues = tar_file_stats(archive)
        issues.extend(archive_issues)
        duplicate_members = sorted(names & all_archive_members)
        issues.extend(f"File is present in more than one archive: {name}" for name in duplicate_members)
        all_archive_members.update(names)
        part_name = archive.name.removesuffix(".tar.gz")
        plan = plans.get(part_name)
        if plan is None:
            issues.append(f"Archive is missing from part_plan.csv: {archive.name}")
            continue
        try:
            if int(plan.get("Files", "-1")) != files:
                issues.append(f"Archive file count disagrees with part_plan.csv: {archive.name}")
            if int(plan.get("Bytes", "-1")) != total_bytes:
                issues.append(f"Archive source bytes disagree with part_plan.csv: {archive.name}")
            if plan.get("Archive") != archive.name:
                issues.append(f"Archive name disagrees with part_plan.csv: {archive.name}")
        except ValueError:
            issues.append(f"Invalid Files or Bytes value in part_plan.csv: {part_name}")

    expected_archives = set(expected_sums)
    actual_archives = set(checked_archives)
    issues.extend(f"SHA256SUMS.txt names missing archive: {name}" for name in sorted(expected_archives - actual_archives))
    issues.extend(f"Archive is not listed in SHA256SUMS.txt: {name}" for name in sorted(actual_archives - expected_archives))
    expected_plan_parts = set(plans)
    actual_plan_parts = {name.removesuffix(".tar.gz") for name in actual_archives}
    issues.extend(f"part_plan.csv names missing archive: {name}.tar.gz" for name in sorted(expected_plan_parts - actual_plan_parts))
    issues.extend(f"Archive is missing from part_plan.csv: {name}.tar.gz" for name in sorted(actual_plan_parts - expected_plan_parts))
    issues.extend(f"Manifest file is missing from archives: {name}" for name in sorted(set(manifest) - all_archive_members))
    issues.extend(f"Archive file is missing from source_manifest.tsv: {name}" for name in sorted(all_archive_members - set(manifest)))

    try:
        scope = json.loads((root / "scope.json").read_text(encoding="utf-8-sig"))
        if scope.get("Repository") not in {None, "smart_farm_models"}:
            issues.append("scope.json must not expose a local repository path.")
        if scope.get("Files") is not None and int(scope["Files"]) != len(manifest):
            issues.append("scope.json Files does not match source_manifest.tsv.")
        if scope.get("Bytes") is not None and int(scope["Bytes"]) != sum(manifest.values()):
            issues.append("scope.json Bytes does not match source_manifest.tsv.")
        if scope.get("Parts") is not None and int(scope["Parts"]) != len(archives):
            issues.append("scope.json Parts does not match the archive count.")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        issues.append(f"Cannot validate scope.json: {exc}")

    if extract_dir is not None and not issues:
        extract_dir = extract_dir.resolve()
        if extract_dir.exists() and any(extract_dir.iterdir()):
            issues.append(f"Extraction directory must be empty before verification: {extract_dir}")
        else:
            extract_dir.mkdir(parents=True, exist_ok=True)
            for archive in archives:
                try:
                    with tarfile.open(archive, mode="r:gz") as source:
                        for member in source.getmembers():
                            if member.isdir():
                                (extract_dir / safe_relative_path(member.name)).mkdir(parents=True, exist_ok=True)
                            elif member.isfile():
                                destination = extract_dir / safe_relative_path(member.name)
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                source_file = source.extractfile(member)
                                if source_file is None:
                                    raise OSError(f"Cannot read archive member: {member.name}")
                                with destination.open("wb") as target:
                                    shutil.copyfileobj(source_file, target, length=1024 * 1024)
                except (OSError, tarfile.TarError, ValueError) as exc:
                    issues.append(f"Cannot extract {archive.name}: {exc}")
                    break
            if not issues:
                issues.extend(compare_extracted(extract_dir, manifest))

    return ReleaseReport(root, checked_archives, len(manifest), issues)


def main() -> int:
    args = build_parser().parse_args()
    report = verify_release(Path(args.release_dir), Path(args.extract_dir) if args.extract_dir else None)
    if args.json:
        import json as _json

        print(_json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print(f"STATUS={'OK' if report.ok else 'FAIL'}")
        print(f"CHECKED_ARCHIVES={','.join(report.checked_archives)}")
        print(f"MANIFEST_FILES={report.manifest_files}")
        for issue in report.issues:
            print(f"ISSUE: {issue}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
