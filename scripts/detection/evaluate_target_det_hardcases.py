# 功能：构建 01_target_det hard-case 子集，并评估导出模型推理 JSON 的类别混淆。
from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


SPLIT_FILES = {
    "train": "annotations/instance_train.json",
    "val": "annotations/instance_val.json",
    "test": "annotations/instance_test.json",
}
OBJECT_STRESS_AUGMENTATIONS = {"object_flip", "object_vflip", "object_rotate180"}
IMAGE_ZOOMOUT_SCALES = {
    "image_zoomout_75": 0.75,
    "image_zoomout_60": 0.60,
    "image_zoomout_50": 0.50,
}
IMAGE_STRESS_AUGMENTATIONS = {
    "image_dark",
    "image_bright",
    "image_blur",
    "image_lowres",
    *IMAGE_ZOOMOUT_SCALES,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="PaddleX style COCO dataset root.")
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="val", help="Dataset split to sample.")
    parser.add_argument("--labels", nargs="+", required=True, help="Target labels used to build hard-case subset.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for manifest and reports.")
    parser.add_argument("--max-images-per-label", type=int, default=None, help="Optional cap for each target label.")
    parser.add_argument("--copy-images", action="store_true", help="Copy selected images into output_dir/images.")
    parser.add_argument(
        "--stress-object-augmentations",
        nargs="*",
        default=[],
        choices=sorted(OBJECT_STRESS_AUGMENTATIONS),
        help="Create object-only orientation stress variants while preserving GT bboxes.",
    )
    parser.add_argument(
        "--stress-image-augmentations",
        nargs="*",
        default=[],
        choices=sorted(IMAGE_STRESS_AUGMENTATIONS),
        help="Create full-image illumination and blur stress variants while preserving GT bboxes.",
    )
    parser.add_argument(
        "--prediction-json",
        type=Path,
        help="Prediction JSON in onboard_result.json or PaddleDetection COCO bbox.json format.",
    )
    parser.add_argument("--model-name", default="candidate", help="Model name used in report filenames.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for matching predictions to GT.")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Prediction score threshold.")
    return parser


def iou_xywh(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, aw, ah = [float(value) for value in box_a]
    bx1, by1, bw, bh = [float(value) for value in box_b]
    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def build_manifest(
    dataset_dir: str | Path,
    split: str,
    labels: list[str],
    max_images_per_label: int | None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir)
    coco_path = dataset_root / SPLIT_FILES[split]
    payload = json.loads(coco_path.read_text(encoding="utf-8-sig"))
    category_by_id = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    label_set = set(labels)
    missing = sorted(label_set - set(category_by_id.values()))
    if missing:
        raise ValueError(f"Labels not found in {coco_path}: {', '.join(missing)}")

    image_by_id = {int(item["id"]): item for item in payload.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        label = category_by_id.get(int(ann["category_id"]))
        if label not in label_set:
            continue
        annotations_by_image[int(ann["image_id"])].append(
            {
                "annotation_id": int(ann.get("id", 0)),
                "category_id": int(ann["category_id"]),
                "label": label,
                "bbox_xywh": [float(value) for value in ann.get("bbox", [0, 0, 0, 0])],
            }
        )

    label_image_counts: Counter[str] = Counter()
    selected_images: list[dict[str, Any]] = []
    for image_id in sorted(annotations_by_image):
        image = image_by_id.get(image_id)
        if image is None:
            continue
        anns = annotations_by_image[image_id]
        image_labels = sorted({ann["label"] for ann in anns})
        if max_images_per_label is not None and all(label_image_counts[label] >= max_images_per_label for label in image_labels):
            continue
        selected_images.append(
            {
                "image_id": image_id,
                "image_name": str(image.get("file_name", "")),
                "image_path": str((dataset_root / "images" / str(image.get("file_name", ""))).resolve()),
                "width": int(image.get("width", 0) or 0),
                "height": int(image.get("height", 0) or 0),
                "labels": image_labels,
                "annotations": anns,
            }
        )
        for label in image_labels:
            label_image_counts[label] += 1

    return {
        "dataset_dir": str(dataset_root.resolve()),
        "split": split,
        "source_coco": str(coco_path.resolve()),
        "category_by_id": {str(category_id): label for category_id, label in category_by_id.items()},
        "labels": labels,
        "max_images_per_label": max_images_per_label,
        "image_count": len(selected_images),
        "label_image_counts": dict(label_image_counts),
        "images": selected_images,
    }


def normalize_prediction_payload(prediction: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    if isinstance(prediction, dict) and isinstance(prediction.get("images"), list):
        return prediction
    if not isinstance(prediction, list):
        raise ValueError("Prediction JSON must be onboard_result.json or a COCO bbox result list")

    manifest_images = [
        (int(item["image_id"]), str(item["image_name"]))
        for item in manifest.get("images", [])
    ]
    image_name_by_id: dict[int, str] = {}
    for image_id, image_name in manifest_images:
        image_name_by_id.setdefault(image_id, image_name)
    category_by_id = {
        int(category_id): str(label)
        for category_id, label in manifest.get("category_by_id", {}).items()
    }
    detections_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    detections_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prediction:
        if not isinstance(item, dict):
            continue
        image_id = int(item.get("image_id", -1))
        category_id = int(item.get("category_id", -1))
        file_name = str(item.get("file_name", ""))
        if category_id not in category_by_id:
            continue
        detection = {
            "label": category_by_id[category_id],
            "score": float(item.get("score", 0.0) or 0.0),
            "bbox_xywh": [float(value) for value in item.get("bbox", [0, 0, 0, 0])],
        }
        if file_name:
            detections_by_name[file_name].append(detection)
        elif image_id in image_name_by_id:
            detections_by_image[image_id].append(detection)

    return {
        "images": [
            {
                "image_name": image_name,
                "detections": detections_by_name.get(
                    image_name,
                    detections_by_image.get(image_id, []),
                ),
            }
            for image_id, image_name in manifest_images
        ]
    }


def evaluate_predictions(
    manifest: dict[str, Any],
    prediction: dict[str, Any],
    model_name: str,
    iou_threshold: float,
    score_threshold: float,
) -> dict[str, Any]:
    predictions_by_name = {str(item["image_name"]): item.get("detections", []) for item in prediction.get("images", [])}
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    per_stress_augmentation: dict[str, Counter[str]] = defaultdict(Counter)
    confusions: dict[str, Counter[str]] = defaultdict(Counter)
    cases: list[dict[str, Any]] = []
    overall = Counter({"gt_annotations": 0, "correct": 0, "confused": 0, "missed": 0})

    for image in manifest.get("images", []):
        stress_augmentation = str(image.get("stress_augmentation", "original"))
        detections = [
            det
            for det in predictions_by_name.get(str(image.get("image_name", "")), [])
            if float(det.get("score", 0.0) or 0.0) >= score_threshold
        ]
        for gt in image.get("annotations", []):
            gt_label = str(gt["label"])
            gt_bbox = [float(value) for value in gt["bbox_xywh"]]
            overall["gt_annotations"] += 1
            per_stress_augmentation[stress_augmentation]["gt_annotations"] += 1
            per_class[gt_label]["gt"] += 1

            best_det = None
            best_iou = 0.0
            for det in detections:
                overlap = iou_xywh(gt_bbox, [float(value) for value in det.get("bbox_xywh", [0, 0, 0, 0])])
                if overlap > best_iou:
                    best_iou = overlap
                    best_det = det

            if best_det is None or best_iou < iou_threshold:
                overall["missed"] += 1
                per_stress_augmentation[stress_augmentation]["missed"] += 1
                per_class[gt_label]["missed"] += 1
                cases.append(
                    {
                        "image_name": image.get("image_name"),
                        "gt_label": gt_label,
                        "pred_label": None,
                        "score": None,
                        "iou": round(best_iou, 4),
                        "status": "missed",
                    }
                )
                continue

            pred_label = str(best_det.get("label", ""))
            score = float(best_det.get("score", 0.0) or 0.0)
            if pred_label == gt_label:
                overall["correct"] += 1
                per_stress_augmentation[stress_augmentation]["correct"] += 1
                per_class[gt_label]["correct"] += 1
                status = "correct"
            else:
                overall["confused"] += 1
                per_stress_augmentation[stress_augmentation]["confused"] += 1
                per_class[gt_label]["confused"] += 1
                confusions[gt_label][pred_label] += 1
                status = "confused"

            cases.append(
                {
                    "image_name": image.get("image_name"),
                    "gt_label": gt_label,
                    "pred_label": pred_label,
                    "score": round(score, 6),
                    "iou": round(best_iou, 4),
                    "status": status,
                }
            )

    return {
        "model_name": model_name,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "manifest_image_count": int(manifest.get("image_count", len(manifest.get("images", [])))),
        "overall": dict(overall),
        "per_stress_augmentation": {
            name: {
                "gt_annotations": int(counts.get("gt_annotations", 0)),
                "correct": int(counts.get("correct", 0)),
                "confused": int(counts.get("confused", 0)),
                "missed": int(counts.get("missed", 0)),
            }
            for name, counts in sorted(per_stress_augmentation.items())
        },
        "per_class": {label: dict(counts) for label, counts in sorted(per_class.items())},
        "confusions": {label: dict(counts) for label, counts in sorted(confusions.items())},
        "cases": cases,
    }


def add_object_stress_variants(
    manifest: dict[str, Any],
    images_dir: str | Path,
    augmentations: list[str],
) -> dict[str, Any]:
    unknown = sorted(set(augmentations) - OBJECT_STRESS_AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unsupported stress augmentations: {', '.join(unknown)}")
    if not augmentations:
        return manifest

    output_images_dir = Path(images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    stressed = copy.deepcopy(manifest)
    stressed_images = list(stressed.get("images", []))

    for image in manifest.get("images", []):
        src = Path(str(image["image_path"]))
        if not src.is_file():
            continue
        original = Image.open(src).convert("RGB")
        for augmentation in augmentations:
            augmented = original.copy()
            for ann in image.get("annotations", []):
                crop_box = bbox_xywh_to_int_crop(ann.get("bbox_xywh", [0, 0, 0, 0]), augmented.size)
                if crop_box is None:
                    continue
                crop = augmented.crop(crop_box)
                transformed = transform_crop(crop, augmentation)
                augmented.paste(transformed, crop_box)
            out_name = f"stress_{augmentation}_{src.name}"
            out_path = output_images_dir / out_name
            augmented.save(out_path)
            new_image = copy.deepcopy(image)
            new_image["image_name"] = out_name
            new_image["image_path"] = str(out_path.resolve())
            new_image["stress_source"] = image.get("image_name")
            new_image["stress_augmentation"] = augmentation
            stressed_images.append(new_image)

    stressed["stress_object_augmentations"] = augmentations
    stressed["images"] = stressed_images
    stressed["image_count"] = len(stressed_images)
    return stressed


def add_image_stress_variants(
    manifest: dict[str, Any],
    images_dir: str | Path,
    augmentations: list[str],
) -> dict[str, Any]:
    unknown = sorted(set(augmentations) - IMAGE_STRESS_AUGMENTATIONS)
    if unknown:
        raise ValueError(f"Unsupported image stress augmentations: {', '.join(unknown)}")
    if not augmentations:
        return manifest

    output_images_dir = Path(images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    stressed = copy.deepcopy(manifest)
    stressed_images = list(stressed.get("images", []))

    for image in manifest.get("images", []):
        src = Path(str(image["image_path"]))
        if not src.is_file():
            continue
        original = Image.open(src).convert("RGB")
        for augmentation in augmentations:
            augmented = transform_image(original, augmentation)
            out_name = f"stress_{augmentation}_{src.name}"
            out_path = output_images_dir / out_name
            augmented.save(out_path)
            new_image = copy.deepcopy(image)
            new_image["image_name"] = out_name
            new_image["image_path"] = str(out_path.resolve())
            new_image["stress_source"] = image.get("image_name")
            new_image["stress_augmentation"] = augmentation
            if augmentation in IMAGE_ZOOMOUT_SCALES:
                scale = IMAGE_ZOOMOUT_SCALES[augmentation]
                offset_x = original.width * (1.0 - scale) / 2.0
                offset_y = original.height * (1.0 - scale) / 2.0
                for annotation in new_image.get("annotations", []):
                    annotation["bbox_xywh"] = zoomout_bbox_xywh(
                        annotation.get("bbox_xywh", [0, 0, 0, 0]),
                        scale,
                        offset_x,
                        offset_y,
                    )
                new_image["stress_scale"] = scale
            stressed_images.append(new_image)

    stressed["stress_image_augmentations"] = augmentations
    stressed["images"] = stressed_images
    stressed["image_count"] = len(stressed_images)
    return stressed


def bbox_xywh_to_int_crop(bbox: list[float], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = image_size
    x, y, w, h = [float(value) for value in bbox]
    left = max(0, int(round(x)))
    top = max(0, int(round(y)))
    right = min(width, int(round(x + w)))
    bottom = min(height, int(round(y + h)))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def transform_crop(crop: Image.Image, augmentation: str) -> Image.Image:
    if augmentation == "object_flip":
        return ImageOps.mirror(crop)
    if augmentation == "object_vflip":
        return ImageOps.flip(crop)
    if augmentation == "object_rotate180":
        return crop.rotate(180)
    raise ValueError(f"Unsupported stress augmentation: {augmentation}")


def transform_image(image: Image.Image, augmentation: str) -> Image.Image:
    if augmentation == "image_dark":
        return ImageEnhance.Brightness(image).enhance(0.35)
    if augmentation == "image_bright":
        return ImageEnhance.Brightness(image).enhance(1.70)
    if augmentation == "image_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2.4))
    if augmentation == "image_lowres":
        width, height = image.size
        reduced_size = (max(1, round(width * 0.32)), max(1, round(height * 0.32)))
        reduced = image.resize(reduced_size, Image.Resampling.BILINEAR)
        return reduced.resize(image.size, Image.Resampling.BILINEAR)
    if augmentation in IMAGE_ZOOMOUT_SCALES:
        return zoomout_image_centered(image, IMAGE_ZOOMOUT_SCALES[augmentation])
    raise ValueError(f"Unsupported image stress augmentation: {augmentation}")


def zoomout_image_centered(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    offset_x = round((width - resized_width) / 2.0)
    offset_y = round((height - resized_height) / 2.0)
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    transformed = Image.new("RGB", image.size, (123, 116, 103))
    transformed.paste(resized, (offset_x, offset_y))
    return transformed


def zoomout_bbox_xywh(bbox: list[float], scale: float, offset_x: float, offset_y: float) -> list[float]:
    x, y, width, height = [float(value) for value in bbox]
    return [
        round(x * scale + offset_x, 6),
        round(y * scale + offset_y, 6),
        round(width * scale, 6),
        round(height * scale, 6),
    ]


def render_markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    gt = int(overall.get("gt_annotations", 0))
    correct = int(overall.get("correct", 0))
    confused = int(overall.get("confused", 0))
    missed = int(overall.get("missed", 0))
    acc = correct / gt if gt else 0.0
    lines = [
        f"# Hard-case Report: {report['model_name']}",
        "",
        f"- IoU threshold: `{report['iou_threshold']}`",
        f"- Score threshold: `{report['score_threshold']}`",
        f"- GT annotations: `{gt}`",
        f"- Correct: `{correct}`",
        f"- Confused: `{confused}`",
        f"- Missed: `{missed}`",
        f"- Hard-case accuracy: `{acc:.4f}`",
        "",
        "## Per Stress Augmentation",
        "",
        "| augmentation | gt | correct | confused | missed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, counts in report.get("per_stress_augmentation", {}).items():
        lines.append(
            f"| {name} | {counts.get('gt_annotations', 0)} | {counts.get('correct', 0)} | "
            f"{counts.get('confused', 0)} | {counts.get('missed', 0)} |"
        )
    lines.extend(
        [
        "",
        "## Per Class",
        "",
        "| class | gt | correct | confused | missed |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, counts in report["per_class"].items():
        lines.append(
            f"| {label} | {counts.get('gt', 0)} | {counts.get('correct', 0)} | "
            f"{counts.get('confused', 0)} | {counts.get('missed', 0)} |"
        )
    lines.extend(["", "## Confusions", ""])
    if report["confusions"]:
        for gt_label, preds in report["confusions"].items():
            for pred_label, count in preds.items():
                lines.append(f"- `{gt_label}` -> `{pred_label}`: `{count}`")
    else:
        lines.append("- No class confusion above the matching threshold.")
    lines.extend(["", "## Problem Cases", ""])
    problem_cases = [case for case in report["cases"] if case["status"] != "correct"]
    if problem_cases:
        lines.extend(["| image | gt | pred | score | iou | status |", "| --- | --- | --- | ---: | ---: | --- |"])
        for case in problem_cases[:200]:
            lines.append(
                f"| {case['image_name']} | {case['gt_label']} | {case['pred_label']} | "
                f"{case['score']} | {case['iou']} | {case['status']} |"
            )
    else:
        lines.append("- No missed or confused GT annotations.")
    return "\n".join(lines).strip() + "\n"


def copy_manifest_images(manifest: dict[str, Any], output_dir: Path) -> Path:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for image in manifest.get("images", []):
        src = Path(str(image["image_path"]))
        dst = images_dir / src.name
        if src.is_file() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return images_dir


def write_manifest_outputs(manifest: dict[str, Any], output_dir: Path, copy_images: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "image_list.txt").write_text(
        "\n".join(str(item["image_path"]) for item in manifest.get("images", [])) + "\n",
        encoding="utf-8",
    )
    if copy_images:
        copy_manifest_images(manifest, output_dir)


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_manifest(args.dataset_dir, args.split, args.labels, args.max_images_per_label)
    if args.stress_object_augmentations:
        manifest = add_object_stress_variants(manifest, args.output_dir / "images", args.stress_object_augmentations)
    if args.stress_image_augmentations:
        manifest = add_image_stress_variants(manifest, args.output_dir / "images", args.stress_image_augmentations)
    write_manifest_outputs(manifest, args.output_dir, args.copy_images)
    print(f"MANIFEST={args.output_dir / 'manifest.json'}")
    print(f"IMAGE_COUNT={manifest['image_count']}")
    print(f"IMAGE_LIST={args.output_dir / 'image_list.txt'}")

    if args.prediction_json:
        raw_prediction = json.loads(args.prediction_json.read_text(encoding="utf-8"))
        prediction = normalize_prediction_payload(raw_prediction, manifest)
        report = evaluate_predictions(
            manifest,
            prediction,
            model_name=args.model_name,
            iou_threshold=args.iou_threshold,
            score_threshold=args.score_threshold,
        )
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in args.model_name)
        report_json = args.output_dir / f"{safe_name}_hardcase_report.json"
        report_md = args.output_dir / f"{safe_name}_hardcase_report.md"
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report_md.write_text(render_markdown(report), encoding="utf-8")
        print(f"REPORT_JSON={report_json}")
        print(f"REPORT_MD={report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
