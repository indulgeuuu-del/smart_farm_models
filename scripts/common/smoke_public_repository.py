"""Run the dependency-light checks needed after cloning the public repository."""

from __future__ import annotations

import argparse
import importlib.util
import py_compile
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
README_MARKERS = (
    "# smart_farm_models",
    "## 当前保留的三条主线",
    "## Windows 环境准备",
    "## Linux 环境准备",
    "## 车道分割（00）",
    "## 目标检测（01）",
    "## 巡航控制（08）",
    "## 公开模型",
    "scripts\\common\\smoke_public_repository.py",
    "Jetson",
    "实车",
)


@dataclass(frozen=True)
class SmokeReport:
    checks: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def ok(self) -> bool:
        return all(not issues for _, issues in self.checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Public repository root to check.")
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compiling scripts/ and tests/. Use only after a separate compile check.",
    )
    return parser


def load_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load check script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def check_readme_contract(root: Path) -> list[str]:
    path = root / "README.md"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"Cannot read README.md: {exc}"]
    return [f"README.md is missing required contract text: {marker}" for marker in README_MARKERS if marker not in text]


def check_readme_cli_coverage(root: Path) -> list[str]:
    checker_path = root / "scripts" / "common" / "check_supported_cli.py"
    try:
        checker = load_script(checker_path, "smoke_supported_cli_inventory")
    except (OSError, RuntimeError, ImportError) as exc:
        return [f"Cannot load supported CLI inventory: {exc}"]
    readme_path = root / "README.md"
    try:
        text = readme_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"Cannot read README.md for CLI coverage: {exc}"]
    normalized_text = text.replace("\\", "/")
    return [
        f"README.md does not document supported CLI: {relative}"
        for relative in checker.SUPPORTED_CLI
        if relative not in normalized_text
    ]


def check_python_sources(root: Path) -> list[str]:
    issues: list[str] = []
    source_roots = (root / "scripts", root / "tests")
    for source_root in source_roots:
        if not source_root.is_dir():
            issues.append(f"Missing Python source directory: {source_root.relative_to(root)}")
            continue
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts or "legacy" in path.parts:
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except (OSError, py_compile.PyCompileError) as exc:
                issues.append(f"Python compile failed for {path.relative_to(root)}: {exc}")
    return issues


def run_smoke(root: Path, compile_sources: bool = True) -> SmokeReport:
    root = root.resolve()
    checks: list[tuple[str, tuple[str, ...]]] = []

    def add(name: str, check: Callable[[], list[str]]) -> None:
        try:
            issues = tuple(check())
        except Exception as exc:  # A check must report its failure without hiding it.
            issues = (f"{type(exc).__name__}: {exc}",)
        checks.append((name, issues))

    def load_for_check(path: Path, module_name: str, check_name: str):
        try:
            return load_script(path, module_name)
        except Exception as exc:
            checks.append((check_name, (f"{type(exc).__name__}: {exc}",)))
            return None

    add("README contract", lambda: check_readme_contract(root))
    add("README CLI coverage", lambda: check_readme_cli_coverage(root))

    requirements = load_for_check(
        root / "scripts/common/check_requirements_profiles.py",
        "smoke_requirements",
        "requirements profiles",
    )
    if requirements is not None:
        add(
            "requirements profiles",
            lambda: list(requirements.validate_profiles(root / "requirements").issues),
        )

    layout = load_for_check(
        root / "scripts/common/check_mandatory_data_layout.py",
        "smoke_layout",
        "retained dataset layout",
    )
    if layout is not None:
        add("retained dataset layout", lambda: layout.check_layout(root / "datasets"))

    repository = load_for_check(
        root / "scripts/release/verify_public_repository.py",
        "smoke_repository",
        "public repository artifacts",
    )
    if repository is not None:
        add("public repository artifacts", lambda: repository.verify_repository(root).issues)

    if compile_sources:
        add("Python sources", lambda: check_python_sources(root))

    return SmokeReport(tuple(checks))


def main() -> int:
    args = build_parser().parse_args()
    report = run_smoke(Path(args.repo_root), compile_sources=not args.skip_compile)
    for name, issues in report.checks:
        print(f"CHECK={name} STATUS={'OK' if not issues else 'FAIL'}")
        for issue in issues:
            print(f"ISSUE: {issue}")
    print(f"STATUS={'OK' if report.ok else 'FAIL'}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
