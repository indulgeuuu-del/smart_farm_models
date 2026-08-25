# Purpose: Build an offline augmented COCO dataset for 01_target_det without changing the source dataset.
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


DEFAULT_SOURCE_DIR = Path("./datasets/01_target_det/paddlex")
DEFAULT_OUTPUT_DIR = Path("./datasets/01_target_det/paddlex_aug")
SPLIT_FILES = {
    "train": "annotations/instance_train.json",
    "val": "annotations/instance_val.json",
    "test": "annotations/instance_test.json",
}
DEFAULT_AUGMENTATIONS = ("flip", "exposure", "blur", "noise", "shadow")
GEOMETRIC_AUGMENTATIONS = ("vflip", "affine", "perspective", "zoomout")
QUALITY_AUGMENTATIONS = ("lowres", "overexposure", "underexposure", "mixed_lighting")
OBJECT_AUGMENTATIONS = ("object_flip", "object_vflip", "object_rotate90", "object_rotate180", "object_rotate270")
SUPPORTED_AUGMENTATIONS = DEFAULT_AUGMENTATIONS + GEOMETRIC_AUGMENTATIONS + QUALITY_AUGMENTATIONS + OBJECT_AUGMENTATIONS
DEFAULT_HARD_CASE_LABELS = (
    "cylinder_3",
    "cylinder_1",
    "cylinder_2",
    "cylinder_set",
    "cargo",
    "ball_blue",
    "ball_yellow",
    "water",
    "water_l1",
    "water_l2",
    "water_l3",
)


@dataclass(frozen=True)
class AugmentResult:
    source_dir: Path
    output_dir: Path
    original_train_images: int
    augmented_train_images: int
    train_images: int
    train_annotations: int
    val_images: int
    test_images: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Source PaddleX COCO dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Augmented output dataset root.")
    parser.add_argument(
        "--augmentations",
        nargs="+",
        default=list(DEFAULT_AUGMENTATIONS),
        choices=sorted(SUPPORTED_AUGMENTATIONS),
        help="Offline augmentations used for generated train images.",
    )
    parser.add_argument("--variants-per-image", type=int, default=1, help="Generated variants for each train image.")
    parser.add_argument(
        "--hard-case-labels",
        nargs="*",
        default=list(DEFAULT_HARD_CASE_LABELS),
        help="Images containing these labels receive extra variants.",
    )
    parser.add_argument("--hard-case-extra", type=int, default=1, help="Extra variants for hard-case train images.")
    parser.add_argument(
        "--focus-labels",
        nargs="*",
        default=[],
        help="Images containing these labels receive another focus-specific set of variants.",
    )
    parser.add_argument("--focus-extra", type=int, default=0, help="Additional variants for focus-label train images.")
    parser.add_argument(
        "--focus-augmentations",
        nargs="*",
        default=[],
        choices=sorted(SUPPORTED_AUGMENTATIONS),
        help="Optional augmentation policy used only by focus-specific variants.",
    )
    parser.add_argument(
        "--zoomout-labels",
        nargs="*",
        default=[],
        help="Images containing these labels receive dedicated full-image zoom-out variants.",
    )
    parser.add_argument("--zoomout-extra", type=int, default=0, help="Zoom-out variants for matching train images.")
    parser.add_argument(
        "--max-augmented-train-images",
        type=int,
        default=None,
        help="Only generate variants for the first N train images. Original split is still copied completely.",
    )
    parser.add_argument("--seed", type=int, default=20260430, help="Deterministic random seed.")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory if it already exists.")
    return parser


def build_augmented_dataset(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    augmentations: list[str] | tuple[str, ...] = DEFAULT_AUGMENTATIONS,
    variants_per_image: int = 1,
    hard_case_labels: list[str] | tuple[str, ...] = DEFAULT_HARD_CASE_LABELS,
    hard_case_extra: int = 1,
    focus_labels: list[str] | tuple[str, ...] = (),
    focus_extra: int = 0,
    focus_augmentations: list[str] | tuple[str, ...] = (),
    zoomout_labels: list[str] | tuple[str, ...] = (),
    zoomout_extra: int = 0,
    max_augmented_train_images: int | None = None,
    seed: int = 20260430,
    force: bool = False,
) -> AugmentResult:
    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    _validate_args(
        source_root=source_root,
        output_root=output_root,
        augmentations=tuple(augmentations),
        variants_per_image=variants_per_image,
        hard_case_extra=hard_case_extra,
        focus_extra=focus_extra,
        zoomout_extra=zoomout_extra,
        max_augmented_train_images=max_augmented_train_images,
    )
    _prepare_output_dir(source_root=source_root, output_root=output_root, force=force)

    source_images_dir = source_root / "images"
    output_images_dir = output_root / "images"
    output_annotations_dir = output_root / "annotations"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    split_payloads = {split: _load_coco(source_root / relative_path) for split, relative_path in SPLIT_FILES.items()}
    for split, payload in split_payloads.items():
        _validate_coco_payload(payload, source_images_dir, split)

    train_payload = _augment_train_payload(
        payload=split_payloads["train"],
        source_images_dir=source_images_dir,
        output_images_dir=output_images_dir,
        augmentations=tuple(augmentations),
        variants_per_image=variants_per_image,
        hard_case_labels=set(hard_case_labels),
        hard_case_extra=hard_case_extra,
        focus_labels=set(focus_labels),
        focus_extra=focus_extra,
        focus_augmentations=tuple(focus_augmentations),
        zoomout_labels=set(zoomout_labels),
        zoomout_extra=zoomout_extra,
        max_augmented_train_images=max_augmented_train_images,
        seed=seed,
    )
    _write_json(output_annotations_dir / "instance_train.json", train_payload)

    for split in ("val", "test"):
        _copy_split_images(split_payloads[split], source_images_dir, output_images_dir)
        _write_json(output_annotations_dir / f"instance_{split}.json", split_payloads[split])

    for metadata_name in ("class_names.txt", "label_list.txt", "data.yml"):
        metadata_path = source_root / metadata_name
        if metadata_path.is_file():
            shutil.copy2(metadata_path, output_root / metadata_name)

    original_train_images = len(split_payloads["train"]["images"])
    augmented_train_images = len(train_payload["images"]) - original_train_images
    _write_json(
        output_root / "augmentation_meta.json",
        {
            "source_dir": str(source_root),
            "policy": "target_det_train_only_offline_augmentation",
            "augmentations": list(augmentations),
            "variants_per_image": variants_per_image,
            "hard_case_labels": list(hard_case_labels),
            "hard_case_extra": hard_case_extra,
            "focus_labels": list(focus_labels),
            "focus_extra": focus_extra,
            "focus_augmentations": list(focus_augmentations),
            "zoomout_labels": list(zoomout_labels),
            "zoomout_extra": zoomout_extra,
            "zoomout_scale_range": [0.55, 0.80],
            "seed": seed,
            "original_train_images": original_train_images,
            "augmented_train_images": augmented_train_images,
            "train_images": len(train_payload["images"]),
            "train_annotations": len(train_payload["annotations"]),
            "val_images": len(split_payloads["val"]["images"]),
            "test_images": len(split_payloads["test"]["images"]),
            "note": "Only train split is augmented; val/test are copied unchanged.",
        },
    )
    return AugmentResult(
        source_dir=source_root,
        output_dir=output_root,
        original_train_images=original_train_images,
        augmented_train_images=augmented_train_images,
        train_images=len(train_payload["images"]),
        train_annotations=len(train_payload["annotations"]),
        val_images=len(split_payloads["val"]["images"]),
        test_images=len(split_payloads["test"]["images"]),
    )


def flip_bbox_xywh(bbox: list[float] | tuple[float, float, float, float], image_width: int | float) -> list[int | float]:
    x, y, width, height = (float(value) for value in bbox)
    flipped_x = float(image_width) - x - width
    return [_clean_number(flipped_x), _clean_number(y), _clean_number(width), _clean_number(height)]


def vertical_flip_bbox_xywh(
    bbox: list[float] | tuple[float, float, float, float],
    image_height: int | float,
) -> list[int | float]:
    x, y, width, height = (float(value) for value in bbox)
    flipped_y = float(image_height) - y - height
    return [_clean_number(x), _clean_number(flipped_y), _clean_number(width), _clean_number(height)]


def _augment_train_payload(
    *,
    payload: dict[str, Any],
    source_images_dir: Path,
    output_images_dir: Path,
    augmentations: tuple[str, ...],
    variants_per_image: int,
    hard_case_labels: set[str],
    hard_case_extra: int,
    focus_labels: set[str],
    focus_extra: int,
    focus_augmentations: tuple[str, ...],
    zoomout_labels: set[str],
    zoomout_extra: int,
    max_augmented_train_images: int | None,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    output_payload = dict(payload)
    output_payload["images"] = [dict(image) for image in payload["images"]]
    output_payload["annotations"] = [dict(annotation) for annotation in payload["annotations"]]

    _copy_split_images(payload, source_images_dir, output_images_dir)

    id_to_name = {int(category["id"]): str(category["name"]) for category in payload["categories"]}
    anns_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        anns_by_image_id[int(annotation["image_id"])].append(annotation)

    next_image_id = _next_int_id(payload["images"])
    next_annotation_id = _next_int_id(payload["annotations"])
    generated = 0
    max_images = max_augmented_train_images if max_augmented_train_images is not None else len(payload["images"])

    for image_index, image_info in enumerate(payload["images"]):
        if image_index >= max_images:
            continue
        image_id = int(image_info["id"])
        image_annotations = anns_by_image_id.get(image_id, [])
        label_names = {id_to_name.get(int(annotation["category_id"]), "") for annotation in image_annotations}
        base_variant_count = variants_per_image
        if label_names & hard_case_labels:
            base_variant_count += hard_case_extra
        focus_variant_count = focus_extra if label_names & focus_labels else 0
        zoomout_variant_count = zoomout_extra if label_names & zoomout_labels else 0
        total_variants = base_variant_count + focus_variant_count + zoomout_variant_count
        if total_variants <= 0:
            continue

        original_name = str(image_info["file_name"]).replace("\\", "/")
        source_image = source_images_dir / original_name
        with Image.open(source_image) as opened:
            original_image = opened.convert("RGB")
            for variant_index in range(total_variants):
                focus_variant_index = variant_index - base_variant_count
                zoomout_variant_index = variant_index - base_variant_count - focus_variant_count
                if zoomout_variant_index >= 0:
                    aug_name = "zoomout"
                elif focus_variant_index >= 0 and focus_augmentations:
                    aug_name = focus_augmentations[(image_index + focus_variant_index) % len(focus_augmentations)]
                else:
                    aug_name = augmentations[(image_index + variant_index) % len(augmentations)]
                aug_rng = random.Random(rng.randint(0, 2**31 - 1))
                augmented_image, bbox_transform = _apply_augmentation(
                    original_image,
                    aug_name,
                    aug_rng,
                    annotations=image_annotations,
                    id_to_name=id_to_name,
                    object_labels=hard_case_labels | focus_labels | zoomout_labels,
                )
                augmented_name = _build_augmented_file_name(original_name, aug_name, output_images_dir)
                target_image = output_images_dir / augmented_name
                target_image.parent.mkdir(parents=True, exist_ok=True)
                augmented_image.save(target_image)

                new_image_id = next_image_id
                next_image_id += 1
                generated += 1
                output_payload["images"].append(
                    {
                        **dict(image_info),
                        "id": new_image_id,
                        "file_name": augmented_name,
                        "width": augmented_image.width,
                        "height": augmented_image.height,
                    }
                )
                for annotation in image_annotations:
                    new_bbox = bbox_transform(annotation["bbox"], augmented_image.width, augmented_image.height)
                    output_payload["annotations"].append(
                        {
                            **dict(annotation),
                            "id": next_annotation_id,
                            "image_id": new_image_id,
                            "bbox": new_bbox,
                            "area": _clean_number(float(new_bbox[2]) * float(new_bbox[3])),
                        }
                    )
                    next_annotation_id += 1

    if generated == 0 and variants_per_image > 0:
        raise ValueError("No augmented train images were generated")
    return output_payload


def _apply_augmentation(
    image: Image.Image,
    augmentation: str,
    rng: random.Random,
    *,
    annotations: list[dict[str, Any]] | None = None,
    id_to_name: dict[int, str] | None = None,
    object_labels: set[str] | None = None,
) -> tuple[Image.Image, Callable[[list[float], int, int], list[int | float]]]:
    if augmentation == "flip":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), lambda bbox, width, _height: flip_bbox_xywh(bbox, width)
    if augmentation == "vflip":
        return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), lambda bbox, _width, height: vertical_flip_bbox_xywh(
            bbox, height
        )
    if augmentation == "affine":
        return _augment_affine(image, rng)
    if augmentation == "perspective":
        return _augment_perspective(image, rng)
    if augmentation == "zoomout":
        return _augment_zoomout(image, rng)
    if augmentation in OBJECT_AUGMENTATIONS:
        return (
            _augment_objects_in_place(
                image,
                augmentation=augmentation,
                annotations=annotations or [],
                id_to_name=id_to_name or {},
                object_labels=object_labels or set(),
            ),
            _identity_bbox,
        )
    if augmentation == "exposure":
        return _augment_exposure(image, rng), _identity_bbox
    if augmentation == "blur":
        radius = rng.uniform(0.4, 1.2)
        return image.filter(ImageFilter.GaussianBlur(radius=radius)), _identity_bbox
    if augmentation == "lowres":
        width, height = image.size
        scale = rng.uniform(0.55, 0.80)
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
        return reduced.resize(image.size, Image.Resampling.BILINEAR), _identity_bbox
    if augmentation == "overexposure":
        return _augment_overexposure(image, rng), _identity_bbox
    if augmentation == "underexposure":
        return _augment_underexposure(image, rng), _identity_bbox
    if augmentation == "mixed_lighting":
        return _augment_mixed_lighting(image, rng), _identity_bbox
    if augmentation == "noise":
        return _augment_noise(image, rng), _identity_bbox
    if augmentation == "shadow":
        return _augment_shadow(image, rng), _identity_bbox
    raise ValueError(f"Unsupported augmentation: {augmentation}")


def _augment_zoomout(
    image: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Callable[[list[float], int, int], list[int | float]]]:
    width, height = image.size
    scale = rng.uniform(0.55, 0.80)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    offset_x = rng.randint(0, width - resized_width)
    offset_y = rng.randint(0, height - resized_height)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    transformed = Image.new("RGB", image.size, (123, 116, 103))
    transformed.paste(resized, (offset_x, offset_y))

    def transform_bbox(bbox: list[float], new_width: int, new_height: int) -> list[int | float]:
        return _transform_bbox_xywh(
            bbox,
            new_width,
            new_height,
            lambda x, y: (x * scale + offset_x, y * scale + offset_y),
        )

    return transformed, transform_bbox


def _augment_affine(
    image: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Callable[[list[float], int, int], list[int | float]]]:
    width, height = image.size
    angle = rng.uniform(-7.0, 7.0)
    translate_x = rng.uniform(-0.025, 0.025) * width
    translate_y = rng.uniform(-0.025, 0.025) * height
    center_x = width / 2.0
    center_y = height / 2.0
    radians = -angle * math.pi / 180.0
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)

    transformed = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        center=(center_x, center_y),
        translate=(translate_x, translate_y),
        fillcolor=(123, 116, 103),
    )

    def transform_bbox(bbox: list[float], new_width: int, new_height: int) -> list[int | float]:
        return _transform_bbox_xywh(
            bbox,
            new_width,
            new_height,
            lambda x, y: (
                cos_v * (x - center_x) - sin_v * (y - center_y) + center_x + translate_x,
                sin_v * (x - center_x) + cos_v * (y - center_y) + center_y + translate_y,
            ),
        )

    return transformed, transform_bbox


def _augment_perspective(
    image: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Callable[[list[float], int, int], list[int | float]]]:
    width, height = image.size
    max_x = width * 0.045
    max_y = height * 0.045
    source_points = [
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ]
    target_points = [
        (rng.uniform(0, max_x), rng.uniform(0, max_y)),
        (width - rng.uniform(0, max_x), rng.uniform(0, max_y)),
        (width - rng.uniform(0, max_x), height - rng.uniform(0, max_y)),
        (rng.uniform(0, max_x), height - rng.uniform(0, max_y)),
    ]
    coeffs = _find_perspective_coefficients(target_points, source_points)
    transformed = image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coeffs,
        resample=Image.Resampling.BILINEAR,
        fillcolor=(123, 116, 103),
    )

    def transform_bbox(bbox: list[float], new_width: int, new_height: int) -> list[int | float]:
        return _transform_bbox_xywh(
            bbox,
            new_width,
            new_height,
            lambda x, y: _perspective_point(x, y, coeffs),
        )

    return transformed, transform_bbox


def _find_perspective_coefficients(
    from_points: list[tuple[float, float]],
    to_points: list[tuple[float, float]],
) -> tuple[float, float, float, float, float, float, float, float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is part of the Paddle env.
        raise RuntimeError("numpy is required for perspective augmentation") from exc

    matrix = []
    vector = []
    for (x, y), (u, v) in zip(from_points, to_points):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        vector.extend([u, v])
    coeffs = np.linalg.solve(np.asarray(matrix, dtype="float64"), np.asarray(vector, dtype="float64"))
    return tuple(float(value) for value in coeffs)


def _perspective_point(
    x: float,
    y: float,
    coeffs: tuple[float, float, float, float, float, float, float, float],
) -> tuple[float, float]:
    a, b, c, d, e, f, g, h = coeffs
    denominator = g * x + h * y + 1.0
    if abs(denominator) < 1e-9:
        return x, y
    return (a * x + b * y + c) / denominator, (d * x + e * y + f) / denominator


def _augment_objects_in_place(
    image: Image.Image,
    *,
    augmentation: str,
    annotations: list[dict[str, Any]],
    id_to_name: dict[int, str],
    object_labels: set[str],
) -> Image.Image:
    result = image.copy()
    image_width, image_height = result.size
    for annotation in annotations:
        category_name = id_to_name.get(int(annotation.get("category_id", -1)), "")
        if category_name not in object_labels:
            continue
        crop_box = _bbox_xywh_to_int_crop(annotation.get("bbox", []), image_width, image_height)
        if crop_box is None:
            continue
        patch = result.crop(crop_box)
        if augmentation == "object_flip":
            patch = patch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif augmentation == "object_vflip":
            patch = patch.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        elif augmentation == "object_rotate90":
            patch = patch.rotate(90, resample=Image.Resampling.BILINEAR, expand=False)
        elif augmentation == "object_rotate180":
            patch = patch.transpose(Image.Transpose.ROTATE_180)
        elif augmentation == "object_rotate270":
            patch = patch.rotate(270, resample=Image.Resampling.BILINEAR, expand=False)
        else:
            raise ValueError(f"Unsupported object augmentation: {augmentation}")
        result.paste(patch, crop_box)
    return result


def _bbox_xywh_to_int_crop(bbox: Any, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x, y, width, height = (float(value) for value in bbox)
    left = max(0, int(math.floor(x)))
    top = max(0, int(math.floor(y)))
    right = min(image_width, int(math.ceil(x + width)))
    bottom = min(image_height, int(math.ceil(y + height)))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right, bottom


def _transform_bbox_xywh(
    bbox: list[float] | tuple[float, float, float, float],
    image_width: int | float,
    image_height: int | float,
    point_transform: Callable[[float, float], tuple[float, float]],
) -> list[int | float]:
    x, y, width, height = (float(value) for value in bbox)
    corners = [
        point_transform(x, y),
        point_transform(x + width, y),
        point_transform(x, y + height),
        point_transform(x + width, y + height),
    ]
    xs = [min(max(px, 0.0), float(image_width)) for px, _py in corners]
    ys = [min(max(py, 0.0), float(image_height)) for _px, py in corners]
    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)
    max_left = max(0.0, float(image_width) - 1.0)
    max_top = max(0.0, float(image_height) - 1.0)
    left = min(max(left, 0.0), max_left)
    top = min(max(top, 0.0), max_top)
    right = min(max(right, left), float(image_width))
    bottom = min(max(bottom, top), float(image_height))
    box_width = min(max(1.0, right - left), max(1.0, float(image_width) - left))
    box_height = min(max(1.0, bottom - top), max(1.0, float(image_height) - top))
    return [
        _clean_number(left),
        _clean_number(top),
        _clean_number(box_width),
        _clean_number(box_height),
    ]


def _augment_exposure(image: Image.Image, rng: random.Random) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.28))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.78, 1.25))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.82, 1.18))
    return ImageEnhance.Sharpness(result).enhance(rng.uniform(0.8, 1.4))


def _augment_overexposure(image: Image.Image, rng: random.Random) -> Image.Image:
    gamma = rng.uniform(0.45, 0.58)
    exposure_curve = [round(255.0 * ((value / 255.0) ** gamma)) for value in range(256)]
    result = image.convert("RGB").point(exposure_curve * 3)
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(1.02, 1.08))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.90, 1.05))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.88, 1.12))
    return ImageEnhance.Sharpness(result).enhance(rng.uniform(0.8, 1.3))


def _augment_underexposure(image: Image.Image, rng: random.Random) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(rng.uniform(0.35, 0.55))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.85, 1.15))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.85, 1.15))
    return ImageEnhance.Sharpness(result).enhance(rng.uniform(0.85, 1.25))


def _augment_mixed_lighting(image: Image.Image, rng: random.Random) -> Image.Image:
    bright = ImageEnhance.Brightness(image).enhance(rng.uniform(1.45, 1.80))
    bright = _apply_rgb_gains(bright, (1.08, 1.02, 0.90))
    dark = ImageEnhance.Brightness(image).enhance(rng.uniform(0.42, 0.65))
    dark = _apply_rgb_gains(dark, (0.90, 1.00, 1.08))

    gradient = Image.linear_gradient("L")
    if rng.random() < 0.5:
        gradient = gradient.transpose(Image.Transpose.ROTATE_90)
    gradient = gradient.resize(image.size, Image.Resampling.BILINEAR)
    if rng.random() < 0.5:
        gradient = gradient.point(lambda value: 255 - value)
    return Image.composite(bright, dark, gradient)


def _apply_rgb_gains(image: Image.Image, gains: tuple[float, float, float]) -> Image.Image:
    channels = image.convert("RGB").split()
    adjusted = [
        channel.point([min(255, round(value * gain)) for value in range(256)])
        for channel, gain in zip(channels, gains)
    ]
    return Image.merge("RGB", adjusted)


def _augment_noise(image: Image.Image, rng: random.Random) -> Image.Image:
    noise = Image.effect_noise(image.size, rng.uniform(8.0, 22.0)).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    return Image.blend(image, noise_rgb, alpha=rng.uniform(0.03, 0.08))


def _augment_shadow(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    overlay = Image.new("RGB", image.size, (0, 0, 0))
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    x1 = rng.randint(0, max(0, width // 3))
    x2 = rng.randint(max(x1 + 1, width // 2), width)
    y1 = rng.randint(0, max(0, height // 3))
    y2 = rng.randint(max(y1 + 1, height // 2), height)
    opacity = rng.randint(30, 85)
    draw.polygon([(x1, y1), (x2, max(0, y1 - height // 6)), (x2, y2), (x1, y2)], fill=opacity)
    return Image.composite(overlay, image, mask)


def _identity_bbox(bbox: list[float], _width: int, _height: int) -> list[int | float]:
    return [_clean_number(float(value)) for value in bbox]


def _load_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing COCO annotation file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO root object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_coco_payload(payload: dict[str, Any], image_dir: Path, split: str) -> None:
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO field must be a list: {split}:{key}")
    for image_info in payload["images"]:
        file_name = image_info.get("file_name")
        if not isinstance(file_name, str) or _is_unsafe_relative_path(file_name.replace("\\", "/").strip()):
            raise ValueError(f"Unsafe or invalid image file_name in {split}: {file_name}")
        if not (image_dir / file_name.replace("\\", "/").strip()).is_file():
            raise FileNotFoundError(f"Missing image referenced by {split}: {image_dir / file_name}")


def _copy_split_images(payload: dict[str, Any], source_images_dir: Path, output_images_dir: Path) -> None:
    for image_info in payload["images"]:
        relative_name = str(image_info["file_name"]).replace("\\", "/")
        source_image = source_images_dir / relative_name
        target_image = output_images_dir / relative_name
        if target_image.is_file():
            continue
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)


def _build_augmented_file_name(original_name: str, augmentation: str, output_images_dir: Path) -> str:
    original_path = Path(original_name.replace("\\", "/"))
    parent = original_path.parent if str(original_path.parent) != "." else Path()
    stem = original_path.stem
    suffix = original_path.suffix or ".jpg"
    base_name = parent / f"aug_{augmentation}_{stem}{suffix}"
    if not (output_images_dir / base_name).exists():
        return base_name.as_posix()
    for index in range(2, 10000):
        candidate = parent / f"aug_{augmentation}_{stem}_{index}{suffix}"
        if not (output_images_dir / candidate).exists():
            return candidate.as_posix()
    raise RuntimeError(f"Unable to build unique augmented file name for {original_name}")


def _validate_args(
    *,
    source_root: Path,
    output_root: Path,
    augmentations: tuple[str, ...],
    variants_per_image: int,
    hard_case_extra: int,
    focus_extra: int,
    zoomout_extra: int,
    max_augmented_train_images: int | None,
) -> None:
    if not source_root.is_dir():
        raise FileNotFoundError(f"Missing source dataset directory: {source_root}")
    if output_root == source_root:
        raise ValueError("source_dir and output_dir must be different")
    if source_root in output_root.parents:
        raise ValueError("output_dir must not be placed inside source_dir")
    if len(output_root.parts) < 3:
        raise ValueError(f"Refusing unsafe output_dir: {output_root}")
    unknown = [name for name in augmentations if name not in SUPPORTED_AUGMENTATIONS]
    if unknown:
        raise ValueError("Unsupported augmentations: " + ", ".join(unknown))
    if not augmentations:
        raise ValueError("At least one augmentation must be selected")
    if variants_per_image < 0:
        raise ValueError("variants_per_image must be >= 0")
    if hard_case_extra < 0:
        raise ValueError("hard_case_extra must be >= 0")
    if focus_extra < 0:
        raise ValueError("focus_extra must be >= 0")
    if zoomout_extra < 0:
        raise ValueError("zoomout_extra must be >= 0")
    if max_augmented_train_images is not None and max_augmented_train_images < 0:
        raise ValueError("max_augmented_train_images must be >= 0")


def _prepare_output_dir(*, source_root: Path, output_root: Path, force: bool) -> None:
    if output_root.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists, pass --force to overwrite: {output_root}")
        if output_root == source_root or len(output_root.parts) < 3:
            raise ValueError(f"Refusing unsafe output_dir: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _next_int_id(items: list[dict[str, Any]]) -> int:
    ids = [int(item["id"]) for item in items if isinstance(item.get("id"), int)]
    return (max(ids) + 1) if ids else 1


def _clean_number(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 6)


def _is_unsafe_relative_path(value: str) -> bool:
    path = Path(value)
    parts = value.replace("\\", "/").split("/")
    return path.is_absolute() or ":" in value or any(part in {"", ".", ".."} for part in parts)


def main() -> int:
    args = build_parser().parse_args()
    result = build_augmented_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        augmentations=args.augmentations,
        variants_per_image=args.variants_per_image,
        hard_case_labels=args.hard_case_labels,
        hard_case_extra=args.hard_case_extra,
        focus_labels=args.focus_labels,
        focus_extra=args.focus_extra,
        focus_augmentations=args.focus_augmentations,
        zoomout_labels=args.zoomout_labels,
        zoomout_extra=args.zoomout_extra,
        max_augmented_train_images=args.max_augmented_train_images,
        seed=args.seed,
        force=args.force,
    )
    print("target_det offline augmentation generated")
    print(f"source_dir: {result.source_dir}")
    print(f"output_dir: {result.output_dir}")
    print(f"original_train_images: {result.original_train_images}")
    print(f"augmented_train_images: {result.augmented_train_images}")
    print(f"train_images: {result.train_images}")
    print(f"train_annotations: {result.train_annotations}")
    print(f"val_images: {result.val_images}")
    print(f"test_images: {result.test_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
