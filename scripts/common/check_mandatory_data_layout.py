# 功能：静态检查当前三条模型主线的数据目录结构。
from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_DATASETS = {
    "00_lane_seg": ("metadata", "paddlex"),
    "01_target_det": ("annotations_coco", "paddlex"),
    "08_nav_control_optional": ("records", "paddle_custom"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        default="./datasets",
        help="Repository datasets root containing the retained model datasets.",
    )
    return parser


def check_layout(datasets_root: str | Path) -> list[str]:
    root = Path(datasets_root)
    errors: list[str] = []

    if not root.is_dir():
        return [f"Missing datasets root: {root}"]

    for dataset_name, required_dirs in EXPECTED_DATASETS.items():
        dataset_dir = root / dataset_name
        if not dataset_dir.is_dir():
            errors.append(f"Missing required dataset directory: {dataset_dir}")
            continue
        for child in required_dirs:
            child_dir = dataset_dir / child
            if not child_dir.is_dir():
                errors.append(f"Missing required directory: {child_dir}")

    for child in root.iterdir():
        if child.is_dir() and child.name not in EXPECTED_DATASETS:
            errors.append(f"Unexpected dataset directory: {child}")

    return errors


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    errors = check_layout(args.datasets_root)
    if errors:
        print("model data layout static check failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("model data layout static check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
