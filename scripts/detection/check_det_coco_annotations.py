# 功能：静态检查 01_target_det 的 COCO 标注结构是否可进入训练准备流程。
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_DET_CLASSES = (
    "animal",
    "cylinder_3",
    "cylinder_1",
    "cylinder_2",
    "water",
    "water_l3",
    "water_l2",
    "lable_yellow",
    "storage",
    "ball_yellow",
    "order",
    "name",
    "cargo",
    "ball_blue",
    "lable_blue",
)
DATASET_TASKS = {
    "01_target_det": "target_det",
}
TASK_DATASETS = {task: dataset for dataset, task in DATASET_TASKS.items()}


@dataclass(frozen=True)
class CocoCheckResult:
    task: str
    coco_json: Path
    image_dir: Path
    images: int
    annotations: int
    categories: int
    errors: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_DATASETS), help="Detection task name, currently only target_det.")
    parser.add_argument("--image-dir", type=Path, help="Raw image directory referenced by COCO file_name fields.")
    parser.add_argument("--coco-json", type=Path, action="append", default=[], help="COCO annotation JSON. Can be repeated.")
    parser.add_argument("--datasets-root", type=Path, default=Path("./datasets"), help="Repository datasets root for --all.")
    parser.add_argument("--all", action="store_true", help="Check JSON files under 01_target_det annotations_coco.")
    parser.add_argument("--allow-empty", action="store_true", help="Return success when --all finds no COCO JSON files.")
    parser.add_argument(
        "--allow-custom-classes",
        action="store_true",
        help="Do not require the historical/default 01_target_det class list.",
    )
    return parser


def check_all_datasets(datasets_root: str | Path, *, allow_custom_classes: bool = False) -> list[CocoCheckResult]:
    root = Path(datasets_root)
    results: list[CocoCheckResult] = []
    for dataset_name, task in DATASET_TASKS.items():
        dataset_dir = root / dataset_name
        image_dir = dataset_dir / "raw"
        annotation_dir = dataset_dir / "annotations_coco"
        if not annotation_dir.is_dir():
            continue
        for coco_json in sorted(annotation_dir.glob("*.json")):
            results.append(
                check_coco_annotation(
                    task=task,
                    image_dir=image_dir,
                    coco_json=coco_json,
                    expected_classes=None if allow_custom_classes else TARGET_DET_CLASSES,
                )
            )
    return results


def check_coco_annotation(
    *,
    task: str,
    image_dir: str | Path,
    coco_json: str | Path,
    expected_classes: tuple[str, ...] | None = None,
) -> CocoCheckResult:
    image_root = Path(image_dir)
    annotation_path = Path(coco_json)
    errors: list[str] = []
    payload = _load_coco_payload(annotation_path, errors)
    if payload is None:
        return CocoCheckResult(task, annotation_path, image_root, 0, 0, 0, errors)

    images = _get_list(payload, "images", annotation_path, errors)
    annotations = _get_list(payload, "annotations", annotation_path, errors)
    categories = _get_list(payload, "categories", annotation_path, errors)

    category_ids = _check_categories(categories, expected_classes, errors)
    image_sizes = _check_images(images, image_root, errors)
    _check_annotations(annotations, image_sizes, category_ids, errors)

    return CocoCheckResult(
        task=task,
        coco_json=annotation_path,
        image_dir=image_root,
        images=len(images),
        annotations=len(annotations),
        categories=len(categories),
        errors=errors,
    )


def _load_coco_payload(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"Missing COCO annotation file: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"Invalid COCO root {path}: expected object")
        return None
    return payload


def _get_list(payload: dict[str, Any], key: str, path: Path, errors: list[str]) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list):
        errors.append(f"COCO field must be a list: {path}:{key}")
        return []
    items: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"COCO {key}[{index}] must be an object")
            continue
        items.append(item)
    if key in {"images", "categories"} and not items:
        errors.append(f"COCO field must not be empty: {path}:{key}")
    return items


def _check_categories(
    categories: list[dict[str, Any]],
    expected_classes: tuple[str, ...] | None,
    errors: list[str],
) -> set[int]:
    ids: list[int] = []
    names: list[str] = []
    for index, category in enumerate(categories):
        category_id = category.get("id")
        name = category.get("name")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            errors.append(f"Invalid category id categories[{index}]: {category_id}")
        else:
            ids.append(category_id)
        if not isinstance(name, str) or not name.strip():
            errors.append(f"Invalid category name categories[{index}]: {name}")
        elif name != name.strip():
            errors.append(f"Category name has leading/trailing whitespace: {name!r}")
            names.append(name.strip())
        else:
            names.append(name)

    duplicate_ids = sorted({category_id for category_id in ids if ids.count(category_id) > 1})
    if duplicate_ids:
        errors.append("Duplicate category ids: " + ", ".join(str(item) for item in duplicate_ids))
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        errors.append("Duplicate category names: " + ", ".join(duplicate_names))

    if expected_classes is not None:
        actual = set(names)
        expected = set(expected_classes)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            parts = ["category names do not match expected classes"]
            if missing:
                parts.append("missing: " + ", ".join(missing))
            if unexpected:
                parts.append("unexpected: " + ", ".join(unexpected))
            errors.append("; ".join(parts))

    return set(ids)


def _check_images(
    images: list[dict[str, Any]],
    image_dir: Path,
    errors: list[str],
) -> dict[int, tuple[float, float]]:
    image_sizes: dict[int, tuple[float, float]] = {}
    ids: list[int] = []
    file_names: list[str] = []
    for index, image in enumerate(images):
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            errors.append(f"Invalid image id images[{index}]: {image_id}")
            continue
        ids.append(image_id)

        if not isinstance(file_name, str) or not file_name.strip():
            errors.append(f"Invalid image file_name images[{index}]: {file_name}")
        else:
            normalized = file_name.replace("\\", "/").strip()
            file_names.append(normalized)
            if _is_unsafe_relative_path(normalized):
                errors.append(f"Unsafe image file_name images[{index}]: {file_name}")
            elif not (image_dir / normalized).is_file():
                errors.append(f"Missing image images[{index}]: {normalized}")

        if not _is_positive_number(width) or not _is_positive_number(height):
            errors.append(f"Invalid image width/height images[{index}]: {width}x{height}")
            continue
        image_sizes[image_id] = (float(width), float(height))

    duplicate_ids = sorted({image_id for image_id in ids if ids.count(image_id) > 1})
    if duplicate_ids:
        errors.append("Duplicate image ids: " + ", ".join(str(item) for item in duplicate_ids))
    duplicate_files = sorted({file_name for file_name in file_names if file_names.count(file_name) > 1})
    if duplicate_files:
        errors.append("Duplicate image file_name values: " + ", ".join(duplicate_files[:5]))
    return image_sizes


def _check_annotations(
    annotations: list[dict[str, Any]],
    image_sizes: dict[int, tuple[float, float]],
    category_ids: set[int],
    errors: list[str],
) -> None:
    ids: list[int] = []
    for index, annotation in enumerate(annotations):
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool):
            errors.append(f"Invalid annotation id annotations[{index}]: {annotation_id}")
        else:
            ids.append(annotation_id)
        if image_id not in image_sizes:
            errors.append(f"Unknown image_id annotations[{index}]: {image_id}")
            image_size = None
        else:
            image_size = image_sizes[image_id]
        if category_id not in category_ids:
            errors.append(f"Unknown category_id annotations[{index}]: {category_id}")
        _check_bbox(index, bbox, image_size, errors)

        area = annotation.get("area")
        if area is not None and not _is_positive_number(area):
            errors.append(f"Invalid annotation area annotations[{index}]: {area}")
        iscrowd = annotation.get("iscrowd")
        if iscrowd is not None and iscrowd not in (0, 1):
            errors.append(f"Invalid iscrowd annotations[{index}]: {iscrowd}")

    duplicate_ids = sorted({annotation_id for annotation_id in ids if ids.count(annotation_id) > 1})
    if duplicate_ids:
        errors.append("Duplicate annotation ids: " + ", ".join(str(item) for item in duplicate_ids))


def _check_bbox(index: int, bbox: object, image_size: tuple[float, float] | None, errors: list[str]) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"Invalid bbox annotations[{index}]: expected [x, y, width, height]")
        return
    if any(not _is_finite_number(value) for value in bbox):
        errors.append(f"Invalid bbox annotations[{index}]: values must be finite numbers")
        return
    x, y, width, height = (float(value) for value in bbox)
    if x < 0 or y < 0:
        errors.append(f"Invalid bbox annotations[{index}]: x/y must be non-negative")
    if width <= 0 or height <= 0:
        errors.append(f"Invalid bbox annotations[{index}]: bbox width/height must be positive")
        return
    if image_size is None:
        return
    image_width, image_height = image_size
    tolerance = 1e-6
    if x + width > image_width + tolerance or y + height > image_height + tolerance:
        errors.append(f"Invalid bbox annotations[{index}]: bbox exceeds image bounds")


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_positive_number(value: object) -> bool:
    return _is_finite_number(value) and float(value) > 0


def _is_unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    parts = value.replace("\\", "/").split("/")
    return path.is_absolute() or ":" in value or any(part in {"", ".", ".."} for part in parts)


def _print_result(result: CocoCheckResult) -> None:
    print(f"{result.task}: {result.coco_json}")
    print(f"  images: {result.images}")
    print(f"  annotations: {result.annotations}")
    print(f"  categories: {result.categories}")
    if result.errors:
        for error in result.errors:
            print(f"  - {error}")


def main() -> int:
    args = build_parser().parse_args()
    if args.all:
        results = check_all_datasets(args.datasets_root, allow_custom_classes=args.allow_custom_classes)
        if not results:
            if args.allow_empty:
                print("det COCO annotation static check skipped: no COCO JSON files found")
                return 0
            print("det COCO annotation static check skipped: no COCO JSON files found")
            return 1
        failed = False
        for result in results:
            _print_result(result)
            failed = failed or bool(result.errors)
        return 1 if failed else 0

    if not args.task or not args.image_dir or not args.coco_json:
        raise SystemExit("--task, --image-dir and --coco-json are required unless --all is used")

    failed = False
    for coco_json in args.coco_json:
        result = check_coco_annotation(
            task=args.task,
            image_dir=args.image_dir,
            coco_json=coco_json,
            expected_classes=None if args.allow_custom_classes else TARGET_DET_CLASSES if args.task == "target_det" else None,
        )
        _print_result(result)
        failed = failed or bool(result.errors)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
