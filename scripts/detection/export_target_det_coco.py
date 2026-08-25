# 功能：把 Dataset_all 的 COCO 检测数据导入为 01_target_det 标准训练目录与 PaddleX 检测数据结构。
from __future__ import annotations

import argparse
import json
import shutil
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
SPLIT_TO_SOURCE_JSON = {
    "train": "train.json",
    "val": "valid.json",
    "test": "test.json",
}
SOURCE_IMAGES_DIR = "Images"


@dataclass(frozen=True)
class ImportResult:
    total_images: int
    train_images: int
    val_images: int
    test_images: int
    total_annotations: int
    train_annotations: int
    val_annotations: int
    test_annotations: int
    classes: int


def load_label_list(path: str | Path) -> tuple[str, ...]:
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label_list.txt: {label_path}")
    labels = tuple(line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    if not labels:
        raise ValueError(f"label_list.txt is empty: {label_path}")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError("Duplicate labels in label_list.txt: " + ", ".join(duplicates))
    return labels


def import_dataset_all_to_target_det(
    *,
    dataset_all_dir: str | Path,
    dataset_dir: str | Path,
) -> ImportResult:
    source_root = Path(dataset_all_dir).resolve()
    target_root = Path(dataset_dir).resolve()
    images_dir = source_root / SOURCE_IMAGES_DIR
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing source image directory: {images_dir}")

    expected_labels = load_label_list(source_root / "label_list.txt")
    _validate_exact_target_classes(expected_labels, source_root / "label_list.txt")

    split_payloads = {split: _load_coco(source_root / file_name) for split, file_name in SPLIT_TO_SOURCE_JSON.items()}
    for split, payload in split_payloads.items():
        _validate_payload_categories(payload, expected_labels, source_root / SPLIT_TO_SOURCE_JSON[split])

    referenced_images = _collect_referenced_images(split_payloads)
    for image_name in referenced_images:
        source_image = images_dir / image_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Image referenced by Dataset_all is missing: {source_image}")

    raw_dir = target_root / "raw"
    annotation_dir = target_root / "annotations_coco"
    paddlex_images_dir = target_root / "paddlex" / "images"
    paddlex_annotation_dir = target_root / "paddlex" / "annotations"
    for path in (raw_dir, annotation_dir, paddlex_images_dir, paddlex_annotation_dir):
        _reset_dir(path)

    for image_name in referenced_images:
        source_image = images_dir / image_name
        _copy_relative_file(source_image, images_dir, raw_dir)
        _copy_relative_file(source_image, images_dir, paddlex_images_dir)

    total_annotations = 0
    image_counts: dict[str, int] = {}
    annotation_counts: dict[str, int] = {}
    for split, payload in split_payloads.items():
        file_stem = f"instances_{split}.json"
        (annotation_dir / file_stem).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (paddlex_annotation_dir / f"instance_{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        image_counts[split] = len(payload["images"])
        annotation_counts[split] = len(payload["annotations"])
        total_annotations += len(payload["annotations"])

    shutil.copy2(source_root / "label_list.txt", annotation_dir / "label_list.txt")
    shutil.copy2(source_root / "data.yml", annotation_dir / "data.yml")
    (target_root / "paddlex" / "class_names.txt").write_text("\n".join(expected_labels) + "\n", encoding="utf-8")

    return ImportResult(
        total_images=len(referenced_images),
        train_images=image_counts["train"],
        val_images=image_counts["val"],
        test_images=image_counts["test"],
        total_annotations=total_annotations,
        train_annotations=annotation_counts["train"],
        val_annotations=annotation_counts["val"],
        test_annotations=annotation_counts["test"],
        classes=len(expected_labels),
    )


def export_target_det_coco(*, dataset_all_dir: str | Path, dataset_dir: str | Path) -> ImportResult:
    return import_dataset_all_to_target_det(dataset_all_dir=dataset_all_dir, dataset_dir=dataset_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Dataset_all into the standard 01_target_det repository layout.")
    parser.add_argument("--dataset-all-dir", type=Path, default=Path("./Dataset_all"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("./datasets/01_target_det"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = import_dataset_all_to_target_det(
        dataset_all_dir=args.dataset_all_dir,
        dataset_dir=args.dataset_dir,
    )
    print("target_det dataset import passed")
    print(f"classes: {result.classes}")
    print(f"total_images: {result.total_images}")
    print(f"train_images: {result.train_images}")
    print(f"val_images: {result.val_images}")
    print(f"test_images: {result.test_images}")
    print(f"total_annotations: {result.total_annotations}")
    return 0


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO root object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO field must be a list: {path}:{key}")
    return payload


def _validate_exact_target_classes(labels: tuple[str, ...], source: Path) -> None:
    if labels != TARGET_DET_CLASSES:
        missing = [label for label in TARGET_DET_CLASSES if label not in labels]
        unexpected = [label for label in labels if label not in TARGET_DET_CLASSES]
        message = [f"Target class list must exactly match the 15-class mainline: {source}"]
        if missing:
            message.append("missing: " + ", ".join(missing))
        if unexpected:
            message.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("; ".join(message))


def _validate_payload_categories(payload: dict[str, Any], expected_labels: tuple[str, ...], source: Path) -> None:
    names = tuple(category.get("name") for category in payload["categories"])
    invalid_names = [name for name in names if not isinstance(name, str) or not name]
    if invalid_names:
        raise ValueError(f"All category names must be non-empty strings: {source}")
    if names != expected_labels:
        missing = [label for label in expected_labels if label not in names]
        unexpected = [label for label in names if label not in expected_labels]
        message = [f"COCO categories do not match label_list.txt: {source}"]
        if missing:
            message.append("missing: " + ", ".join(missing))
        if unexpected:
            message.append("unexpected: " + ", ".join(unexpected))
        raise ValueError("; ".join(message))


def _collect_referenced_images(split_payloads: dict[str, dict[str, Any]]) -> list[str]:
    referenced: dict[str, None] = {}
    for payload in split_payloads.values():
        for image in payload["images"]:
            file_name = image.get("file_name")
            if not isinstance(file_name, str) or not file_name.strip():
                raise ValueError(f"Invalid image file_name in COCO payload: {file_name}")
            normalized = file_name.replace("\\", "/").strip()
            if _is_unsafe_relative_path(normalized):
                raise ValueError(f"Unsafe image file_name in COCO payload: {file_name}")
            referenced[normalized] = None
    return sorted(referenced)


def _copy_relative_file(source_file: Path, source_root: Path, target_root: Path) -> None:
    relative = source_file.relative_to(source_root)
    target_file = target_root / relative
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target_file)


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _is_unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    parts = value.replace("\\", "/").split("/")
    return path.is_absolute() or ":" in value or any(part in {"", ".", ".."} for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
