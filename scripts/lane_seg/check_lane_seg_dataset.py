"""Statically check the 00_lane_seg dataset layout before PaddleX checks.

This script does not train, convert, or modify data. It verifies the local
directory contract that must hold before running PaddleX ``check_dataset``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CLASSES = ("background", "road", "border", "cross_zone")
ALLOWED_SHAPES = {"polygon", "rectangle"}
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="./datasets/00_lane_seg",
        help="00_lane_seg dataset root containing raw, labelme_json, and paddlex.",
    )
    return parser


def check_dataset(dataset_dir: str | Path) -> list[str]:
    root = Path(dataset_dir)
    errors: list[str] = []
    raw_dir = root / "raw"
    labelme_dir = root / "labelme_json"
    paddlex_dir = root / "paddlex"

    for directory in (raw_dir, labelme_dir, paddlex_dir):
        if not directory.is_dir():
            errors.append(f"Missing directory: {directory}")

    if raw_dir.is_dir():
        _check_raw_dir(raw_dir, errors)
    if labelme_dir.is_dir():
        _check_labelme_dir(labelme_dir, raw_dir, errors)
    if paddlex_dir.is_dir():
        _check_paddlex_dir(paddlex_dir, errors)
    return errors


def _check_raw_dir(raw_dir: Path, errors: list[str]) -> None:
    image_files = [path for path in raw_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_files:
        errors.append(f"No raw images found: {raw_dir}")
    for path in sorted(path for path in raw_dir.iterdir() if path.is_file()):
        _check_safe_filename(path, errors)


def _check_labelme_dir(labelme_dir: Path, raw_dir: Path, errors: list[str]) -> None:
    json_paths = sorted(labelme_dir.glob("*.json"))
    if not json_paths:
        errors.append(f"No Labelme JSON files found: {labelme_dir}")
    for json_path in json_paths:
        _check_safe_filename(json_path, errors)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid Labelme JSON {json_path}: {exc}")
            continue
        image_name = Path(str(data.get("imagePath", ""))).name
        if not image_name:
            errors.append(f"Missing imagePath in {json_path}")
        else:
            if not SAFE_FILENAME_RE.fullmatch(image_name):
                errors.append(f"Unsafe filename in imagePath {json_path}: {image_name}")
            if raw_dir.is_dir() and not (raw_dir / image_name).is_file():
                errors.append(f"Labelme imagePath has no matching raw image {json_path}: {image_name}")
        _check_positive_int(data, "imageWidth", json_path, errors)
        _check_positive_int(data, "imageHeight", json_path, errors)
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list) or not shapes:
            errors.append(f"No Labelme shapes found: {json_path}")
            continue
        for index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                errors.append(f"Invalid shape entry {json_path}#{index}")
                continue
            label = shape.get("label")
            if label not in CLASSES:
                errors.append(f"Unknown label {json_path}#{index}: {label}")
            shape_type = shape.get("shape_type")
            if shape_type not in ALLOWED_SHAPES:
                errors.append(f"Unsupported shape_type {json_path}#{index}: {shape_type}")
            _check_shape_points(json_path, index, shape_type, shape.get("points"), errors)


def _check_positive_int(data: dict, key: str, json_path: Path, errors: list[str]) -> None:
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        errors.append(f"Invalid {key} in {json_path}: {value}")


def _check_shape_points(json_path: Path, index: int, shape_type: object, points: object, errors: list[str]) -> None:
    if not isinstance(points, list):
        errors.append(f"Invalid points {json_path}#{index}: {points}")
        return
    if shape_type == "rectangle" and len(points) != 2:
        errors.append(f"Rectangle requires 2 points {json_path}#{index}: {len(points)}")
    if shape_type == "polygon" and len(points) < 3:
        errors.append(f"Polygon requires at least 3 points {json_path}#{index}: {len(points)}")


def _check_paddlex_dir(paddlex_dir: Path, errors: list[str]) -> None:
    required_paths = [
        paddlex_dir / "images" / "train",
        paddlex_dir / "images" / "val",
        paddlex_dir / "annotations" / "train",
        paddlex_dir / "annotations" / "val",
        paddlex_dir / "class_names.txt",
        paddlex_dir / "train.txt",
        paddlex_dir / "val.txt",
    ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"Missing PaddleX path: {path}")
    class_names = paddlex_dir / "class_names.txt"
    if class_names.is_file():
        labels = class_names.read_text(encoding="utf-8").splitlines()
        if labels != list(CLASSES):
            errors.append(f"Invalid class_names.txt: {labels}")
    for split in ("train", "val"):
        _check_split_file(paddlex_dir, split, errors)


def _check_split_file(paddlex_dir: Path, split: str, errors: list[str]) -> None:
    split_file = paddlex_dir / f"{split}.txt"
    if not split_file.is_file():
        return
    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        errors.append(f"Empty PaddleX split file: {split_file}")
        return
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"Invalid PaddleX mapping {split_file}:{line_number}: {line}")
            continue
        image_rel, mask_rel = parts
        expected_image_prefix = f"images/{split}/"
        expected_mask_prefix = f"annotations/{split}/"
        if not image_rel.startswith(expected_image_prefix):
            errors.append(f"Invalid image path in {split_file}:{line_number}: {image_rel}")
        if not mask_rel.startswith(expected_mask_prefix):
            errors.append(f"Invalid mask path in {split_file}:{line_number}: {mask_rel}")
        for rel_path in (image_rel, mask_rel):
            basename = Path(rel_path).name
            if not SAFE_FILENAME_RE.fullmatch(basename):
                errors.append(f"Unsafe filename in PaddleX mapping {split_file}:{line_number}: {basename}")
            if not (paddlex_dir / rel_path).is_file():
                errors.append(f"Missing PaddleX mapped file {split_file}:{line_number}: {rel_path}")


def _check_safe_filename(path: Path, errors: list[str]) -> None:
    if not SAFE_FILENAME_RE.fullmatch(path.name):
        errors.append(f"Unsafe filename: {path}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    errors = check_dataset(args.dataset_dir)
    if errors:
        print("lane_seg dataset static check failed")
        for error in errors:
            print(f"- {error}")
        return 1
    print("lane_seg dataset static check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
