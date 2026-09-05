# 功能：把自采 COCO 检测数据集整理成 01_target_det 可训练目录，并按 4:1 切分训练/验证。
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_DIR = Path("./MyDataset/My_Formal_Target_0519")
DEFAULT_OUTPUT_DIR = Path("./datasets/01_target_det/paddlex_formal_0519")
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 20260519


@dataclass(frozen=True)
class PrepareResult:
    source_dir: Path
    output_dir: Path
    source_images_on_disk: int
    source_images_in_coco: int
    ignored_images_not_in_coco: int
    train_images: int
    val_images: int
    train_annotations: int
    val_annotations: int
    categories: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it already exists.")
    return parser


def prepare_dataset(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> PrepareResult:
    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    images_dir = source_root / "Images"
    annotation_path = source_root / "Annotations" / "coco_info.json"
    _validate_paths(source_root, images_dir, annotation_path, output_root, val_ratio)
    _prepare_output_dir(output_root, force=force)

    payload = _load_coco(annotation_path)
    categories = _validate_categories(payload)
    images = _validate_images(payload, images_dir)
    annotations = _validate_annotations(payload, images)

    disk_image_names = sorted(path.name for path in images_dir.glob("*.jpg"))
    coco_image_names = {str(image["file_name"]) for image in images}
    ignored_images = [name for name in disk_image_names if name not in coco_image_names]

    train_ids, val_ids = _split_image_ids(images=images, annotations=annotations, val_ratio=val_ratio, seed=seed)

    output_images_dir = output_root / "images"
    output_annotations_dir = output_root / "annotations"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    for image in images:
        image_name = str(image["file_name"])
        shutil.copy2(images_dir / image_name, output_images_dir / image_name)

    train_payload = _build_split_payload(payload, images, annotations, categories, train_ids, "train")
    val_payload = _build_split_payload(payload, images, annotations, categories, val_ids, "val")
    _write_json(output_annotations_dir / "instance_train.json", train_payload)
    _write_json(output_annotations_dir / "instance_val.json", val_payload)
    _write_json(output_annotations_dir / "instance_test.json", val_payload)

    class_names = [str(category["name"]) for category in categories]
    (output_root / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_root / "label_list.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_root / "data.yml").write_text(_build_data_yml(class_names), encoding="utf-8")
    (output_root / "ignored_images_not_in_coco.txt").write_text(
        "\n".join(ignored_images) + ("\n" if ignored_images else ""),
        encoding="utf-8",
    )
    _write_json(
        output_root / "split_meta.json",
        {
            "source_dir": str(source_root),
            "annotation_path": str(annotation_path),
            "seed": seed,
            "val_ratio": val_ratio,
            "source_images_on_disk": len(disk_image_names),
            "source_images_in_coco": len(images),
            "ignored_images_not_in_coco": len(ignored_images),
            "train_images": len(train_ids),
            "val_images": len(val_ids),
            "train_annotations": len(train_payload["annotations"]),
            "val_annotations": len(val_payload["annotations"]),
            "categories": class_names,
            "train_category_annotations": _category_annotation_counts(train_payload),
            "val_category_annotations": _category_annotation_counts(val_payload),
        },
    )

    return PrepareResult(
        source_dir=source_root,
        output_dir=output_root,
        source_images_on_disk=len(disk_image_names),
        source_images_in_coco=len(images),
        ignored_images_not_in_coco=len(ignored_images),
        train_images=len(train_ids),
        val_images=len(val_ids),
        train_annotations=len(train_payload["annotations"]),
        val_annotations=len(val_payload["annotations"]),
        categories=len(categories),
    )


def _validate_paths(source_root: Path, images_dir: Path, annotation_path: Path, output_root: Path, val_ratio: float) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source dataset directory: {source_root}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing source Images directory: {images_dir}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing source COCO json: {annotation_path}")
    if not 0.05 <= val_ratio <= 0.5:
        raise ValueError(f"val_ratio must be in [0.05, 0.5], got {val_ratio}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output_dir must not be the source dir or a child of source dir")
    if "datasets" not in output_root.parts or "01_target_det" not in output_root.parts:
        raise ValueError(f"Refusing output outside datasets/01_target_det: {output_root}")


def _prepare_output_dir(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output dir already exists, pass --force to overwrite: {output_root}")
        if "datasets" not in output_root.parts or "01_target_det" not in output_root.parts:
            raise ValueError(f"Refusing unsafe delete target: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _load_coco(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO root object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO field must be a list: {key}")
    return payload


def _validate_categories(payload: dict[str, Any]) -> list[dict[str, Any]]:
    categories = []
    ids: list[int] = []
    names: list[str] = []
    for category in payload["categories"]:
        category_id = category.get("id")
        name = category.get("name")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise ValueError(f"Invalid category id: {category_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid category name: {name}")
        ids.append(category_id)
        names.append(name.strip())
        categories.append({**category, "name": name.strip()})
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate category ids")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate category names")
    return categories


def _validate_images(payload: dict[str, Any], images_dir: Path) -> list[dict[str, Any]]:
    images = []
    ids: list[int] = []
    names: list[str] = []
    for image in payload["images"]:
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise ValueError(f"Invalid image id: {image_id}")
        if not isinstance(file_name, str) or _is_unsafe_relative_path(file_name):
            raise ValueError(f"Invalid image file_name: {file_name}")
        if not _is_positive_number(width) or not _is_positive_number(height):
            raise ValueError(f"Invalid image size for {file_name}: {width}x{height}")
        if not (images_dir / file_name).is_file():
            raise FileNotFoundError(f"Image referenced by COCO is missing: {images_dir / file_name}")
        ids.append(image_id)
        names.append(file_name)
        images.append({**image, "file_name": file_name, "width": int(width), "height": int(height)})
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image ids")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image file_name values")
    return images


def _validate_annotations(payload: dict[str, Any], images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_sizes = {int(image["id"]): (float(image["width"]), float(image["height"])) for image in images}
    category_ids = {int(category["id"]) for category in payload["categories"]}
    ids: list[int] = []
    annotations = []
    for annotation in payload["annotations"]:
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool):
            raise ValueError(f"Invalid annotation id: {annotation_id}")
        if image_id not in image_sizes:
            raise ValueError(f"Unknown image_id in annotation {annotation_id}: {image_id}")
        if category_id not in category_ids:
            raise ValueError(f"Unknown category_id in annotation {annotation_id}: {category_id}")
        clean_bbox = _validate_bbox(annotation_id, bbox, image_sizes[int(image_id)])
        ids.append(annotation_id)
        annotations.append(
            {
                **annotation,
                "bbox": clean_bbox,
                "area": _clean_number(float(clean_bbox[2]) * float(clean_bbox[3])),
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate annotation ids")
    return annotations


def _validate_bbox(annotation_id: int, bbox: object, image_size: tuple[float, float]) -> list[int | float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in annotation {annotation_id}: {bbox}")
    if any(not _is_finite_number(value) for value in bbox):
        raise ValueError(f"Non-finite bbox in annotation {annotation_id}: {bbox}")
    x, y, width, height = (float(value) for value in bbox)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox geometry in annotation {annotation_id}: {bbox}")
    image_width, image_height = image_size
    if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
        raise ValueError(f"Bbox exceeds image bounds in annotation {annotation_id}: {bbox}")
    return [_clean_number(value) for value in (x, y, width, height)]


def _split_image_ids(
    *,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    image_ids = [int(image["id"]) for image in images]
    target_val_count = max(1, int(round(len(image_ids) * val_ratio)))
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        anns_by_image[int(annotation["image_id"])].append(annotation)

    total_by_cat = Counter(int(annotation["category_id"]) for annotation in annotations)
    remaining_by_cat = Counter(
        {
            category_id: max(1, int(round(count * val_ratio)))
            for category_id, count in total_by_cat.items()
        }
    )
    candidates = image_ids[:]
    rng.shuffle(candidates)
    val_ids: set[int] = set()

    while len(val_ids) < target_val_count and candidates:
        best_index = 0
        best_score = -1.0
        for index, image_id in enumerate(candidates):
            image_counts = Counter(int(annotation["category_id"]) for annotation in anns_by_image.get(image_id, []))
            need_score = sum(min(count, max(0, remaining_by_cat[cat_id])) for cat_id, count in image_counts.items())
            score = need_score * 10.0 + 1.0 + rng.random() * 0.001
            if score > best_score:
                best_index = index
                best_score = score
        image_id = candidates.pop(best_index)
        val_ids.add(image_id)
        for annotation in anns_by_image.get(image_id, []):
            category_id = int(annotation["category_id"])
            if remaining_by_cat[category_id] > 0:
                remaining_by_cat[category_id] -= 1

    train_ids = set(image_ids) - val_ids
    return train_ids, val_ids


def _build_split_payload(
    source_payload: dict[str, Any],
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    image_ids: set[int],
    split_name: str,
) -> dict[str, Any]:
    split_images = [dict(image) for image in images if int(image["id"]) in image_ids]
    split_annotations = [dict(annotation) for annotation in annotations if int(annotation["image_id"]) in image_ids]
    return {
        "info": {
            "description": f"My_Formal_Target_0519 {split_name} split",
            "source_info": source_payload.get("info", ""),
        },
        "licenses": source_payload.get("licenses", []),
        "images": split_images,
        "annotations": split_annotations,
        "categories": [dict(category) for category in categories],
    }


def _category_annotation_counts(payload: dict[str, Any]) -> dict[str, int]:
    id_to_name = {int(category["id"]): str(category["name"]) for category in payload["categories"]}
    counts = Counter(int(annotation["category_id"]) for annotation in payload["annotations"])
    return {id_to_name[category_id]: counts[category_id] for category_id in sorted(id_to_name)}


def _build_data_yml(class_names: list[str]) -> str:
    lines = ["train: images", "val: images", "nc: " + str(len(class_names)), "names:"]
    lines.extend(f"  - {name}" for name in class_names)
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    parts = value.replace("\\", "/").split("/")
    return path.is_absolute() or ":" in value or any(part in {"", ".", ".."} for part in parts)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_positive_number(value: object) -> bool:
    return _is_finite_number(value) and float(value) > 0


def _clean_number(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 6)


def main() -> int:
    args = build_parser().parse_args()
    result = prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force=args.force,
    )
    print("formal target_det dataset prepared")
    print(f"source_dir: {result.source_dir}")
    print(f"output_dir: {result.output_dir}")
    print(f"source_images_on_disk: {result.source_images_on_disk}")
    print(f"source_images_in_coco: {result.source_images_in_coco}")
    print(f"ignored_images_not_in_coco: {result.ignored_images_not_in_coco}")
    print(f"train_images: {result.train_images}")
    print(f"val_images: {result.val_images}")
    print(f"train_annotations: {result.train_annotations}")
    print(f"val_annotations: {result.val_annotations}")
    print(f"categories: {result.categories}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
