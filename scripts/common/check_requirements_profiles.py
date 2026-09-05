"""Validate the repository's pinned Python dependency profiles without installing them."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


PROFILE_FILES = (
    "base.txt",
    "ci.txt",
    "training-windows-py310-gpu.txt",
    "export-windows-py310-paddle261.txt",
    "paddlelite-windows-py310.txt",
)
PIN_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9._+!-]*)$")
NAME_NORMALIZER = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class ProfileReport:
    profiles: dict[str, dict[str, str]]
    issues: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues


def normalized_name(value: str) -> str:
    return NAME_NORMALIZER.sub("-", value).lower()


def read_profile(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict[str, str], list[str]]:
    resolved = path.resolve()
    if resolved in stack:
        return {}, [f"Recursive requirements include: {path}"]
    packages: dict[str, str] = {}
    issues: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [f"Cannot read requirements profile {path}: {exc}"]
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            included = (path.parent / line[3:].strip()).resolve()
            if included.parent != path.parent.resolve():
                issues.append(f"{path}:{line_number} includes a file outside requirements/.")
                continue
            nested, nested_issues = read_profile(included, stack + (resolved,))
            issues.extend(nested_issues)
            for name, version in nested.items():
                if name in packages and packages[name] != version:
                    issues.append(f"{path}:{line_number} conflicts with included package {name}.")
                packages[name] = version
            continue
        match = PIN_PATTERN.fullmatch(line)
        if not match:
            issues.append(f"{path}:{line_number} must use an exact package pin: {raw_line!r}")
            continue
        name = normalized_name(match.group(1))
        version = match.group(2)
        if name in packages and packages[name] != version:
            issues.append(f"{path}:{line_number} repeats {name} with a conflicting version.")
        packages[name] = version
    return packages, issues


def major_version(version: str) -> int | None:
    match = re.match(r"^(\d+)", version)
    return int(match.group(1)) if match else None


def validate_packages(profile_name: str, packages: dict[str, str]) -> list[str]:
    issues: list[str] = []
    numpy_major = major_version(packages.get("numpy", ""))
    opencv_major = major_version(packages.get("opencv-python", ""))
    if opencv_major is not None and opencv_major >= 5 and (numpy_major is None or numpy_major < 2):
        issues.append(
            f"{profile_name}: opencv-python {packages['opencv-python']} requires NumPy 2 or newer, "
            f"but this profile resolves NumPy {packages.get('numpy', 'missing')}."
        )
    return issues


def validate_profiles(requirements_dir: Path) -> ProfileReport:
    profiles: dict[str, dict[str, str]] = {}
    issues: list[str] = []
    for name in PROFILE_FILES:
        packages, profile_issues = read_profile(requirements_dir / name)
        profiles[name] = packages
        issues.extend(profile_issues)
        issues.extend(validate_packages(name, packages))
    return ProfileReport(profiles=profiles, issues=issues)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements-dir", default="requirements", help="Directory containing the pinned profiles.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = validate_profiles(Path(args.requirements_dir))
    print(f"STATUS={'OK' if report.ok else 'FAIL'}")
    print(f"PROFILES={','.join(sorted(report.profiles))}")
    for issue in report.issues:
        print(f"ISSUE: {issue}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
