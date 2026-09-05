"""Run ``--help`` for the maintained command-line entry points."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep this inventory explicit. Internal import modules and the old notebook
# export under nav_control_optional/legacy are intentionally not commands.
SUPPORTED_CLI: tuple[str, ...] = (
    "scripts/common/check_mandatory_data_layout.py",
    "scripts/common/check_requirements_profiles.py",
    "scripts/common/check_supported_cli.py",
    "scripts/common/smoke_public_repository.py",
    "scripts/detection/analyze_target_det_dataset.py",
    "scripts/detection/augment_target_det_coco.py",
    "scripts/detection/build_nine_vegetable_capture_plan.py",
    "scripts/detection/check_det_coco_annotations.py",
    "scripts/detection/evaluate_target_det_hardcases.py",
    "scripts/detection/expand_target_det_checkpoint.py",
    "scripts/detection/export_target_det_coco.py",
    "scripts/detection/export_target_det_model.py",
    "scripts/detection/merge_target_det_coco_datasets.py",
    "scripts/detection/monitor_target_det_training.py",
    "scripts/detection/prepare_formal_target_det_dataset.py",
    "scripts/detection/summarize_target_det_hardcases.py",
    "scripts/detection/train_target_det.py",
    "scripts/detection/visualize_target_det_annotations.py",
    "scripts/detection/watch_export_target_det_best.py",
    "scripts/lane_seg/check_lane_seg_dataset.py",
    "scripts/lane_seg/export_seg_masks.py",
    "scripts/lane_seg/split_dataset.py",
    "scripts/nav_control_optional/check_nav_control_data.py",
    "scripts/nav_control_optional/check_nav_control_infer.py",
    "scripts/nav_control_optional/check_nav_control_sequence.py",
    "scripts/nav_control_optional/evaluate_nav_control_robustness.py",
    "scripts/nav_control_optional/export_nav_control_reg.py",
    "scripts/nav_control_optional/prepare_gamepad_nav_dataset.py",
    "scripts/nav_control_optional/prepare_nav_control_dataset.py",
    "scripts/nav_control_optional/train_nav_control_reg.py",
    "scripts/onboard/check_target_det_baseline_compat.py",
    "scripts/onboard/eval_onboard_samples.py",
    "scripts/onboard/run_target_det_onboard.py",
    "scripts/release/download_public_release.py",
    "scripts/release/verify_public_release.py",
    "scripts/release/verify_public_repository.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root to check.")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch each entry point. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Maximum seconds allowed for one --help invocation.",
    )
    parser.add_argument("--list", action="store_true", help="List the maintained entry points and exit.")
    return parser


def check_supported_cli(
    repo_root: str | Path,
    *,
    python_executable: str | Path = sys.executable,
    timeout: float = 60.0,
) -> list[str]:
    root = Path(repo_root).expanduser().resolve()
    executable = shutil.which(str(python_executable))
    issues: list[str] = []

    if timeout <= 0:
        return ["--timeout must be greater than zero"]
    if executable is None:
        return [f"Python executable was not found: {python_executable}"]

    for relative in SUPPORTED_CLI:
        script = root / relative
        if not script.is_file():
            issues.append(f"Missing supported CLI: {relative}")
            continue
        try:
            result = subprocess.run(
                [executable, str(script), "--help"],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            issues.append(f"--help failed for {relative}: {exc}")
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            suffix = f" ({detail[-1]})" if detail else ""
            issues.append(f"--help failed for {relative}: exit {result.returncode}{suffix}")
    return issues


def main() -> int:
    args = build_parser().parse_args()
    if args.list:
        print("\n".join(SUPPORTED_CLI))
        return 0

    issues = check_supported_cli(
        args.repo_root,
        python_executable=args.python_executable,
        timeout=args.timeout,
    )
    if issues:
        print("supported CLI check failed")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print(f"supported CLI check passed ({len(SUPPORTED_CLI)} entry points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
