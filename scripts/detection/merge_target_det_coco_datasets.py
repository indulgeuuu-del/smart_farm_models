# 功能：把多个 COCO 检测数据集合并成 01_target_det 可训练目录，并输出类别分布报告。
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIRS = (
    REPO_ROOT / "MyDataset" / "My_Formal_Target_0619",
    REPO_ROOT / "MyDataset" / "Target_name",
    REPO_ROOT / "MyDataset" / "Target_order and danyuan_0608",
    REPO_ROOT / "MyDataset" / "Target_storage0621",
    REPO_ROOT / "MyDataset" / "Target_shucai0622",
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "01_target_det" / "paddlex_fusion_0619_name_order_storage_shucai0622_26cls"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 20260624
DEFAULT_WEAK_THRESHOLD = 500
DEFAULT_SPLIT_GROUP_SIZE = 10

# 保留 2026-06-22 18 类大模型的类别顺序，再追加 8 个蔬菜类，降低上位机标签映射变动成本。
FUSION_CLASS_NAMES = (
    "water_l3",
    "water_l2",
    "water_l1",
    "water",
    "order",
    "cylinder_set",
    "cylinder_3",
    "cylinder_2",
    "cylinder_1",
    "ball_yellow",
    "ball_blue",
    "animal",
    "name",
    "danyuan_2",
    "danyuan_1",
    "storage",
    "lable_yellow",
    "lable_blue",
    "rape",
    "broccoli",
    "potato",
    "celery",
    "mushroom",
    "flammulina velutipes",
    "tomato",
    "green bean",
    "green pepper",
)

VEGETABLE_LABEL_ALIASES = {
    "h_you_cai": "rape",
    "h_xi_lan_hua": "broccoli",
    "h_tu_dou": "potato",
    "h_qin_cai": "celery",
    "h_mo_gu": "mushroom",
    "h_jin_zhen_gu": "flammulina velutipes",
    "h_fan_qie": "tomato",
    "h_dou_jiao": "green bean",
    "green_bean": "green bean",
    "flammulina_velutipes": "flammulina velutipes",
    "green_pepper": "green pepper",
    "h_qing_jiao": "green pepper",
}


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    dataset_dir: Path
    images_dir: Path
    annotation_path: Path
    slug: str


@dataclass(frozen=True)
class MergeResult:
    output_dir: Path
    images: int
    annotations: int
    categories: int
    train_images: int
    val_images: int
    test_images: int
    train_annotations: int
    val_annotations: int
    test_annotations: int
    weak_classes: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        default=[],
        help="Source dataset root. Can point to either the outer MyDataset folder or the inner EasyData export folder.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--split-group-size", type=int, default=DEFAULT_SPLIT_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--weak-threshold", type=int, default=DEFAULT_WEAK_THRESHOLD)
    parser.add_argument(
        "--drop-empty-categories",
        action="store_true",
        help="Exclude declared categories that have no annotations across all sources.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output dir if it already exists.")
    return parser


def merge_datasets(
    *,
    source_dirs: list[Path] | tuple[Path, ...] = DEFAULT_SOURCE_DIRS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    split_group_size: int = DEFAULT_SPLIT_GROUP_SIZE,
    seed: int = DEFAULT_SEED,
    weak_threshold: int = DEFAULT_WEAK_THRESHOLD,
    drop_empty_categories: bool = False,
    force: bool = False,
) -> MergeResult:
    if split_group_size < 1:
        raise ValueError(f"split_group_size must be >= 1, got {split_group_size}")
    sources = [_resolve_source_dataset(path, index) for index, path in enumerate(source_dirs)]
    output_root = output_dir.resolve()
    _validate_output_dir(output_root, force=force)
    _prepare_output_dir(output_root, force=force)

    images_dir = output_root / "images"
    annotations_dir = output_root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    class_to_new_id = {name: index + 1 for index, name in enumerate(FUSION_CLASS_NAMES)}
    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    annotated_class_names: set[str] = set()
    next_image_id = 1
    next_annotation_id = 1

    for source_index, source in enumerate(sources):
        payload = _load_coco(source.annotation_path)
        source_categories = _validate_categories(payload, source.annotation_path)
        source_images = sorted(
            _validate_images(payload, source.images_dir, source.annotation_path),
            key=lambda image: _natural_file_name_key(str(image["file_name"])),
        )
        source_annotations = _validate_annotations(payload, source_images, source_categories, source.annotation_path)
        source_counts = Counter(_canonical_class_name(source_categories[int(ann["category_id"])]) for ann in source_annotations)
        annotated_class_names.update(source_counts)
        source_image_name_by_id = {int(image["id"]): str(image["file_name"]) for image in source_images}
        source_new_image_id_by_old_id: dict[int, int] = {}
        source_new_name_by_old_id: dict[int, str] = {}

        for local_index, image_info in enumerate(source_images):
            old_image_id = int(image_info["id"])
            old_name = str(image_info["file_name"]).replace("\\", "/")
            new_name = _build_merged_image_name(source_index, source.slug, local_index, old_name)
            shutil.copy2(source.images_dir / old_name, images_dir / new_name)

            new_image_id = next_image_id
            next_image_id += 1
            source_new_image_id_by_old_id[old_image_id] = new_image_id
            source_new_name_by_old_id[old_image_id] = new_name
            merged_images.append(
                {
                    "id": new_image_id,
                    "file_name": new_name,
                    "width": int(image_info["width"]),
                    "height": int(image_info["height"]),
                    "_split_group": f"source-{source_index:02d}-group-{local_index // split_group_size:06d}",
                }
            )

        for annotation in source_annotations:
            old_category_name = _canonical_class_name(source_categories[int(annotation["category_id"])])
            if old_category_name not in class_to_new_id:
                raise ValueError(f"Unexpected class {old_category_name!r} in {source.annotation_path}")
            old_image_id = int(annotation["image_id"])
            new_image_id = source_new_image_id_by_old_id[old_image_id]
            bbox = _clean_bbox(annotation["bbox"])
            merged_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": new_image_id,
                    "category_id": class_to_new_id[old_category_name],
                    "bbox": bbox,
                    "area": _clean_number(float(bbox[2]) * float(bbox[3])),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )
            next_annotation_id += 1

        source_summaries.append(
            {
                "source_root": str(source.root),
                "dataset_dir": str(source.dataset_dir),
                "annotation_path": str(source.annotation_path),
                "images": len(source_images),
                "annotations": len(source_annotations),
                "categories": sorted(source_counts),
                "category_annotations": dict(sorted(source_counts.items())),
                "example_image_mapping": {
                    source_image_name_by_id[old_id]: source_new_name_by_old_id[old_id]
                    for old_id in list(source_new_name_by_old_id)[:5]
                },
            }
        )

    class_names = _select_output_class_names(
        annotated_class_names,
        drop_empty_categories=drop_empty_categories,
    )
    _remap_annotation_category_ids(merged_annotations, class_names)
    categories = [{"id": index + 1, "name": name} for index, name in enumerate(class_names)]
    train_ids, val_ids, test_ids = _split_image_ids(
        images=merged_images,
        annotations=merged_annotations,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    train_payload = _build_split_payload(merged_images, merged_annotations, categories, train_ids, "train")
    val_payload = _build_split_payload(merged_images, merged_annotations, categories, val_ids, "val")
    test_payload = _build_split_payload(merged_images, merged_annotations, categories, test_ids, "test")
    _write_json(annotations_dir / "instance_train.json", train_payload)
    _write_json(annotations_dir / "instance_val.json", val_payload)
    _write_json(annotations_dir / "instance_test.json", test_payload)

    (output_root / "class_names.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_root / "label_list.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    (output_root / "data.yml").write_text(_build_data_yml(class_names), encoding="utf-8")

    distribution = _build_distribution_report(
        output_root=output_root,
        sources=source_summaries,
        train_payload=train_payload,
        val_payload=val_payload,
        test_payload=test_payload,
        class_names=class_names,
        weak_threshold=weak_threshold,
    )
    _write_json(output_root / "class_distribution.json", distribution)
    _write_json(
        output_root / "merge_manifest.json",
        {
            "source_dirs": [str(Path(path).resolve()) for path in source_dirs],
            "output_dir": str(output_root),
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "split_group_size": split_group_size,
            "seed": seed,
            "drop_empty_categories": drop_empty_categories,
            "dropped_empty_categories": [name for name in FUSION_CLASS_NAMES if name not in class_names],
            "class_names": class_names,
            "sources": source_summaries,
            "images": len(merged_images),
            "annotations": len(merged_annotations),
            "train_images": len(train_payload["images"]),
            "val_images": len(val_payload["images"]),
            "test_images": len(test_payload["images"]),
            "train_annotations": len(train_payload["annotations"]),
            "val_annotations": len(val_payload["annotations"]),
            "test_annotations": len(test_payload["annotations"]),
        },
    )
    (output_root / "README.md").write_text(_render_dataset_readme(distribution), encoding="utf-8")
    (output_root / "class_distribution_report.md").write_text(_render_distribution_markdown(distribution), encoding="utf-8")

    return MergeResult(
        output_dir=output_root,
        images=len(merged_images),
        annotations=len(merged_annotations),
        categories=len(categories),
        train_images=len(train_payload["images"]),
        val_images=len(val_payload["images"]),
        test_images=len(test_payload["images"]),
        train_annotations=len(train_payload["annotations"]),
        val_annotations=len(val_payload["annotations"]),
        test_annotations=len(test_payload["annotations"]),
        weak_classes=distribution["weak_classes"],
    )


def _canonical_class_name(name: str) -> str:
    return VEGETABLE_LABEL_ALIASES.get(name, name)


def _select_output_class_names(
    annotated_class_names: set[str],
    *,
    drop_empty_categories: bool,
) -> list[str]:
    unknown = sorted(annotated_class_names - set(FUSION_CLASS_NAMES))
    if unknown:
        raise ValueError("Unexpected annotated classes: " + ", ".join(unknown))
    if not drop_empty_categories:
        return list(FUSION_CLASS_NAMES)
    selected = [name for name in FUSION_CLASS_NAMES if name in annotated_class_names]
    if not selected:
        raise ValueError("Cannot build a dataset with no annotated categories")
    return selected


def _remap_annotation_category_ids(
    annotations: list[dict[str, Any]],
    class_names: list[str],
) -> None:
    old_name_by_id = {index + 1: name for index, name in enumerate(FUSION_CLASS_NAMES)}
    new_id_by_name = {name: index + 1 for index, name in enumerate(class_names)}
    for annotation in annotations:
        old_category_id = int(annotation["category_id"])
        class_name = old_name_by_id.get(old_category_id)
        if class_name not in new_id_by_name:
            raise ValueError(f"Cannot drop annotated category id {old_category_id}: {class_name!r}")
        annotation["category_id"] = new_id_by_name[class_name]


def _resolve_source_dataset(path: Path, index: int) -> SourceDataset:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing source dataset: {root}")

    direct_annotation = root / "Annotations" / "coco_info.json"
    direct_images = root / "Images"
    if direct_annotation.is_file() and direct_images.is_dir():
        dataset_dir = root
    else:
        candidates = [
            child
            for child in sorted(root.iterdir())
            if child.is_dir() and (child / "Annotations" / "coco_info.json").is_file() and (child / "Images").is_dir()
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected exactly one EasyData dataset child under {root}, got {len(candidates)}")
        dataset_dir = candidates[0]

    return SourceDataset(
        root=root,
        dataset_dir=dataset_dir,
        images_dir=dataset_dir / "Images",
        annotation_path=dataset_dir / "Annotations" / "coco_info.json",
        slug=_slugify(root.name, index),
    )


def _validate_output_dir(output_root: Path, *, force: bool) -> None:
    if "datasets" not in output_root.parts or "01_target_det" not in output_root.parts:
        raise ValueError(f"Refusing output outside datasets/01_target_det: {output_root}")
    if output_root.exists() and not force:
        raise FileExistsError(f"Output dir already exists, pass --force to overwrite: {output_root}")


def _prepare_output_dir(output_root: Path, *, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output dir already exists: {output_root}")
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
            raise ValueError(f"COCO field must be a list: {path}:{key}")
    return payload


def _validate_categories(payload: dict[str, Any], path: Path) -> dict[int, str]:
    ids: list[int] = []
    names: list[str] = []
    mapping: dict[int, str] = {}
    for index, category in enumerate(payload["categories"]):
        category_id = category.get("id")
        name = category.get("name")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise ValueError(f"Invalid category id in {path}: categories[{index}]={category_id}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid category name in {path}: categories[{index}]={name}")
        clean_name = _canonical_class_name(name.strip())
        if clean_name not in FUSION_CLASS_NAMES:
            raise ValueError(f"Unexpected category {clean_name!r} in {path}")
        ids.append(category_id)
        names.append(clean_name)
        mapping[category_id] = clean_name
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate category ids in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate category names in {path}")
    return mapping


def _validate_images(payload: dict[str, Any], images_dir: Path, path: Path) -> list[dict[str, Any]]:
    ids: list[int] = []
    names: list[str] = []
    result: list[dict[str, Any]] = []
    for index, image in enumerate(payload["images"]):
        image_id = image.get("id")
        file_name = image.get("file_name")
        width = image.get("width")
        height = image.get("height")
        if not isinstance(image_id, int) or isinstance(image_id, bool):
            raise ValueError(f"Invalid image id in {path}: images[{index}]={image_id}")
        if not isinstance(file_name, str) or _is_unsafe_relative_path(file_name):
            raise ValueError(f"Invalid image file_name in {path}: images[{index}]={file_name}")
        clean_name = file_name.replace("\\", "/")
        if not (images_dir / clean_name).is_file():
            raise FileNotFoundError(f"Image referenced by COCO is missing: {images_dir / clean_name}")
        if not _is_positive_number(width) or not _is_positive_number(height):
            raise ValueError(f"Invalid image size in {path}: {file_name} {width}x{height}")
        ids.append(image_id)
        names.append(clean_name)
        result.append({**image, "file_name": clean_name, "width": int(width), "height": int(height)})
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image ids in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image file_name values in {path}")
    return result


def _validate_annotations(
    payload: dict[str, Any],
    images: list[dict[str, Any]],
    categories: dict[int, str],
    path: Path,
) -> list[dict[str, Any]]:
    image_sizes = {int(image["id"]): (float(image["width"]), float(image["height"])) for image in images}
    ids: list[int] = []
    result: list[dict[str, Any]] = []
    for index, annotation in enumerate(payload["annotations"]):
        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool):
            raise ValueError(f"Invalid annotation id in {path}: annotations[{index}]={annotation_id}")
        if image_id not in image_sizes:
            raise ValueError(f"Unknown image_id in {path}: annotations[{index}]={image_id}")
        if category_id not in categories:
            raise ValueError(f"Unknown category_id in {path}: annotations[{index}]={category_id}")
        clean_bbox = _validate_bbox(index, bbox, image_sizes[int(image_id)], path)
        ids.append(annotation_id)
        result.append({**annotation, "bbox": clean_bbox})
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate annotation ids in {path}")
    return result


def _validate_bbox(index: int, bbox: object, image_size: tuple[float, float], path: Path) -> list[int | float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox in {path}: annotations[{index}]={bbox}")
    if any(not _is_finite_number(value) for value in bbox):
        raise ValueError(f"Non-finite bbox in {path}: annotations[{index}]={bbox}")
    x, y, width, height = (float(value) for value in bbox)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid bbox geometry in {path}: annotations[{index}]={bbox}")
    image_width, image_height = image_size
    if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
        raise ValueError(f"Bbox exceeds image bounds in {path}: annotations[{index}]={bbox}")
    return [_clean_number(value) for value in (x, y, width, height)]


def _split_image_ids(
    *,
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[set[int], set[int], set[int]]:
    if not 0.05 <= val_ratio <= 0.4:
        raise ValueError(f"val_ratio must be in [0.05, 0.4], got {val_ratio}")
    if not 0.05 <= test_ratio <= 0.4:
        raise ValueError(f"test_ratio must be in [0.05, 0.4], got {test_ratio}")
    if val_ratio + test_ratio > 0.5:
        raise ValueError(f"val_ratio + test_ratio must be <= 0.5, got {val_ratio + test_ratio}")
    rng = random.Random(seed)
    image_ids = [int(image["id"]) for image in images]
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        anns_by_image[int(annotation["image_id"])].append(annotation)

    group_to_image_ids: dict[str, set[int]] = defaultdict(set)
    for image in images:
        image_id = int(image["id"])
        group_name = str(image.get("_split_group", f"image-{image_id}"))
        group_to_image_ids[group_name].add(image_id)

    all_groups = list(group_to_image_ids)
    val_groups = _select_split_groups(
        candidate_groups=all_groups,
        group_to_image_ids=group_to_image_ids,
        anns_by_image=anns_by_image,
        target_image_count=max(1, int(round(len(image_ids) * val_ratio))),
        category_ratio=val_ratio,
        rng=rng,
    )
    remaining_groups = [group for group in all_groups if group not in val_groups]
    test_groups = _select_split_groups(
        candidate_groups=remaining_groups,
        group_to_image_ids=group_to_image_ids,
        anns_by_image=anns_by_image,
        target_image_count=max(1, int(round(len(image_ids) * test_ratio))),
        category_ratio=test_ratio,
        rng=rng,
    )
    val_ids = set().union(*(group_to_image_ids[group] for group in val_groups)) if val_groups else set()
    test_ids = set().union(*(group_to_image_ids[group] for group in test_groups)) if test_groups else set()
    train_ids = set(image_ids) - val_ids - test_ids
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("Grouped split must produce non-empty train, val, and test sets")
    return train_ids, val_ids, test_ids


def _select_split_groups(
    *,
    candidate_groups: list[str],
    group_to_image_ids: dict[str, set[int]],
    anns_by_image: dict[int, list[dict[str, Any]]],
    target_image_count: int,
    category_ratio: float,
    rng: random.Random,
) -> set[str]:
    total_by_cat = Counter(
        int(annotation["category_id"])
        for group in candidate_groups
        for image_id in group_to_image_ids[group]
        for annotation in anns_by_image.get(image_id, [])
    )
    target_by_cat = {
        category_id: max(1, int(round(count * category_ratio)))
        for category_id, count in total_by_cat.items()
    }
    remaining_by_cat = Counter(target_by_cat)
    candidates = candidate_groups[:]
    rng.shuffle(candidates)
    selected: set[str] = set()
    selected_image_count = 0
    while selected_image_count < target_image_count and candidates:
        best_index = 0
        best_score = -math.inf
        for index, group in enumerate(candidates):
            group_image_ids = group_to_image_ids[group]
            group_counts = Counter(
                int(annotation["category_id"])
                for image_id in group_image_ids
                for annotation in anns_by_image.get(image_id, [])
            )
            need_score = sum(
                min(count, max(0, remaining_by_cat[category_id])) / target_by_cat[category_id]
                for category_id, count in group_counts.items()
                if remaining_by_cat[category_id] > 0
            )
            projected_count = selected_image_count + len(group_image_ids)
            size_penalty = abs(target_image_count - projected_count) / max(1, target_image_count)
            score = need_score * 10.0 - size_penalty + rng.random() * 0.001
            if score > best_score:
                best_index = index
                best_score = score
        group = candidates.pop(best_index)
        selected.add(group)
        selected_image_count += len(group_to_image_ids[group])
        for image_id in group_to_image_ids[group]:
            for annotation in anns_by_image.get(image_id, []):
                category_id = int(annotation["category_id"])
                if remaining_by_cat[category_id] > 0:
                    remaining_by_cat[category_id] -= 1
    return selected


def _build_split_payload(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    image_ids: set[int],
    split_name: str,
) -> dict[str, Any]:
    return {
        "info": {"description": f"target_det fusion {len(categories)}cls {split_name} split"},
        "licenses": [],
        "images": [
            {key: value for key, value in image.items() if not key.startswith("_")}
            for image in images
            if int(image["id"]) in image_ids
        ],
        "annotations": [dict(annotation) for annotation in annotations if int(annotation["image_id"]) in image_ids],
        "categories": [dict(category) for category in categories],
    }


def _build_distribution_report(
    *,
    output_root: Path,
    sources: list[dict[str, Any]],
    train_payload: dict[str, Any],
    val_payload: dict[str, Any],
    test_payload: dict[str, Any],
    class_names: list[str],
    weak_threshold: int,
) -> dict[str, Any]:
    train_counts = _category_counts(train_payload, class_names)
    val_counts = _category_counts(val_payload, class_names)
    test_counts = _category_counts(test_payload, class_names)
    total_counts = {name: train_counts[name] + val_counts[name] + test_counts[name] for name in class_names}
    weak_classes = [name for name in class_names if total_counts[name] < weak_threshold]
    return {
        "dataset_dir": str(output_root),
        "weak_threshold": weak_threshold,
        "sources": sources,
        "splits": {
            "train": {
                "images": len(train_payload["images"]),
                "annotations": len(train_payload["annotations"]),
                "class_counts": train_counts,
            },
            "val": {
                "images": len(val_payload["images"]),
                "annotations": len(val_payload["annotations"]),
                "class_counts": val_counts,
            },
            "test": {
                "images": len(test_payload["images"]),
                "annotations": len(test_payload["annotations"]),
                "class_counts": test_counts,
                "note": "test is an independent grouped holdout split.",
            },
        },
        "total_class_counts": total_counts,
        "weak_classes": weak_classes,
        "recommendations": _build_recommendations(total_counts, weak_classes),
    }


def _category_counts(payload: dict[str, Any], class_names: list[str]) -> dict[str, int]:
    id_to_name = {int(category["id"]): str(category["name"]) for category in payload["categories"]}
    counts = Counter(id_to_name[int(annotation["category_id"])] for annotation in payload["annotations"])
    return {name: int(counts[name]) for name in class_names}


def _build_recommendations(total_counts: dict[str, int], weak_classes: list[str]) -> list[str]:
    recommendations = []
    if weak_classes:
        recommendations.append("Weak classes need train-only extra augmentation: " + ", ".join(weak_classes))
    shucai = [name for name in total_counts if name.startswith("h_")]
    if shucai:
        recommendations.append("For h_* vegetable classes, keep flip/vflip/rotate180/object-level direction augmentations.")
    if any(name.startswith("water_l") for name in weak_classes):
        recommendations.append("For water_l1/water_l2/water_l3, keep stronger exposure/shadow/noise augmentation and inspect labels visually.")
    recommendations.append("Do not start formal training before augmented COCO static checks pass.")
    recommendations.append("Export final model as model.pdmodel + model.pdiparams + infer_cfg.yml for smartcar_baidu_21.")
    return recommendations


def _render_dataset_readme(distribution: dict[str, Any]) -> str:
    lines = [
        "# 01_target_det 26 类融合数据集",
        "",
        "## 定位",
        "",
        "本目录由多个自采 COCO 数据集合并生成，用于训练单个全类别检测模型，目标是替代 18 类大模型和 8 类蔬菜模型双开方案。",
        "",
        "## 类别顺序",
        "",
    ]
    for index, name in enumerate(distribution["total_class_counts"]):
        lines.append(f"{index} {name}")
    lines.extend(["", "## 分布报告", "", "详见 `class_distribution_report.md` 和 `class_distribution.json`。", ""])
    return "\n".join(lines)


def _render_distribution_markdown(distribution: dict[str, Any]) -> str:
    lines = [
        "# 26 类融合目标检测数据集分布报告",
        "",
        f"- dataset_dir: `{distribution['dataset_dir']}`",
        f"- weak_threshold: `{distribution['weak_threshold']}`",
        "",
        "## Split Summary",
        "",
        "| split | images | annotations |",
        "| --- | ---: | ---: |",
    ]
    for split, payload in distribution["splits"].items():
        lines.append(f"| {split} | {payload['images']} | {payload['annotations']} |")
    lines.extend(["", "## Class Counts", "", "| class | total | train | val | weak |", "| --- | ---: | ---: | ---: | --- |"])
    train_counts = distribution["splits"]["train"]["class_counts"]
    val_counts = distribution["splits"]["val"]["class_counts"]
    weak_classes = set(distribution["weak_classes"])
    for name, total in distribution["total_class_counts"].items():
        lines.append(f"| {name} | {total} | {train_counts[name]} | {val_counts[name]} | {'yes' if name in weak_classes else 'no'} |")
    lines.extend(["", "## Source Datasets", ""])
    for source in distribution["sources"]:
        lines.append(f"- `{source['dataset_dir']}`: {source['images']} images, {source['annotations']} annotations")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in distribution["recommendations"]:
        lines.append(f"- {recommendation}")
    return "\n".join(lines).strip() + "\n"


def _build_merged_image_name(source_index: int, slug: str, local_index: int, original_name: str) -> str:
    suffix = Path(original_name).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".jpg"
    return f"s{source_index:02d}_{slug}_{local_index:06d}{suffix}"


def _natural_file_name_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value.replace("\\", "/")))


def _slugify(value: str, index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or f"source_{index:02d}"


def _build_data_yml(class_names: list[str]) -> str:
    lines = ["train: images", "val: images", "nc: " + str(len(class_names)), "names:"]
    lines.extend(f"  - {name}" for name in class_names)
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clean_bbox(bbox: object) -> list[int | float]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    return [_clean_number(float(value)) for value in bbox]


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
    source_dirs = args.source_dir or list(DEFAULT_SOURCE_DIRS)
    result = merge_datasets(
        source_dirs=source_dirs,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_group_size=args.split_group_size,
        seed=args.seed,
        weak_threshold=args.weak_threshold,
        drop_empty_categories=args.drop_empty_categories,
        force=args.force,
    )
    print("target_det fusion dataset prepared")
    print(f"output_dir: {result.output_dir}")
    print(f"images: {result.images}")
    print(f"annotations: {result.annotations}")
    print(f"categories: {result.categories}")
    print(f"train_images: {result.train_images}")
    print(f"val_images: {result.val_images}")
    print(f"train_annotations: {result.train_annotations}")
    print(f"val_annotations: {result.val_annotations}")
    print("weak_classes: " + ", ".join(result.weak_classes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
