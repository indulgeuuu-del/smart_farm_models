# 功能：把 COCO 检测标注画到图片上，辅助人工排查 01_target_det 错标。
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_DATASET_DIR = Path("./datasets/01_target_det/paddlex")
DEFAULT_OUTPUT_DIR = Path("./outputs/01_target_det/label_visual_review")
SPLIT_FILES = {
    "train": "annotations/instance_train.json",
    "val": "annotations/instance_val.json",
    "test": "annotations/instance_test.json",
}
PALETTE = (
    "#e6194B",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9A6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffe119",
    "#bfef45",
    "#42d4f4",
    "#dcbeff",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="COCO/PaddleDetection dataset root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for visual review files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_FILES),
        default=["train", "val", "test"],
        help="Dataset splits to visualize.",
    )
    parser.add_argument(
        "--duplicate-iou-threshold",
        type=float,
        default=0.92,
        help="Flag different-label boxes with IoU above this threshold.",
    )
    parser.add_argument(
        "--small-area-ratio",
        type=float,
        default=0.0006,
        help="Flag boxes smaller than this image-area ratio.",
    )
    parser.add_argument(
        "--large-area-ratio",
        type=float,
        default=0.65,
        help="Flag boxes larger than this image-area ratio.",
    )
    return parser


def visualize_dataset(
    *,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    splits: list[str] | tuple[str, ...] = ("train", "val", "test"),
    duplicate_iou_threshold: float = 0.92,
    small_area_ratio: float = 0.0006,
    large_area_ratio: float = 0.65,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir)
    output_root = Path(output_dir)
    visual_root = output_root / "visualized"
    output_root.mkdir(parents=True, exist_ok=True)
    visual_root.mkdir(parents=True, exist_ok=True)

    font = load_font()
    suspicious_rows: list[dict[str, str]] = []
    image_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_root.resolve()),
        "output_dir": str(output_root.resolve()),
        "splits": {},
        "total_images": 0,
        "total_annotations": 0,
        "suspicious_items": 0,
    }

    for split in splits:
        coco_path = dataset_root / SPLIT_FILES[split]
        if not coco_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotation file: {coco_path}")
        payload = load_coco(coco_path)
        category_names = category_name_map(payload)
        images = image_map(payload)
        annotations_by_image = annotations_map(payload)
        split_output = visual_root / split
        split_output.mkdir(parents=True, exist_ok=True)

        split_image_count = 0
        split_annotation_count = 0
        split_suspicious_count = 0
        for image_id, image_info in sorted(images.items(), key=lambda item: str(item[1].get("file_name", ""))):
            annotations = annotations_by_image.get(image_id, [])
            split_image_count += 1
            split_annotation_count += len(annotations)
            file_name = normalize_file_name(str(image_info.get("file_name", "")))
            source_path = dataset_root / "images" / file_name
            output_path = split_output / file_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            image_suspicious = inspect_image_annotations(
                split=split,
                image_info=image_info,
                annotations=annotations,
                category_names=category_names,
                duplicate_iou_threshold=duplicate_iou_threshold,
                small_area_ratio=small_area_ratio,
                large_area_ratio=large_area_ratio,
            )
            suspicious_rows.extend(image_suspicious)
            split_suspicious_count += len(image_suspicious)

            draw_annotation_image(
                source_path=source_path,
                output_path=output_path,
                image_info=image_info,
                annotations=annotations,
                category_names=category_names,
                font=font,
            )
            image_rows.append(
                {
                    "split": split,
                    "file_name": file_name,
                    "visual_path": output_path.relative_to(output_root).as_posix(),
                    "annotation_count": len(annotations),
                    "labels": sorted({category_names.get(int(ann.get("category_id", -1)), "unknown") for ann in annotations}),
                    "suspicious_count": len(image_suspicious),
                }
            )

        summary["splits"][split] = {
            "images": split_image_count,
            "annotations": split_annotation_count,
            "suspicious_items": split_suspicious_count,
            "coco_json": str(coco_path.resolve()),
        }
        summary["total_images"] += split_image_count
        summary["total_annotations"] += split_annotation_count
        summary["suspicious_items"] += split_suspicious_count

    write_suspicious_csv(output_root / "suspicious_annotations.csv", suspicious_rows)
    write_image_index_csv(output_root / "image_index.csv", image_rows)
    write_html_index(output_root / "index.html", summary, image_rows)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_coco(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO root object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO field must be a list: {path}:{key}")
    return payload


def category_name_map(payload: dict[str, Any]) -> dict[int, str]:
    return {int(category["id"]): str(category["name"]) for category in payload["categories"]}


def image_map(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(image["id"]): image for image in payload["images"]}


def annotations_map(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        grouped[int(annotation["image_id"])].append(annotation)
    return grouped


def normalize_file_name(file_name: str) -> str:
    normalized = file_name.replace("\\", "/").strip()
    if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"Unsafe image file_name: {file_name}")
    return normalized


def load_font() -> ImageFont.ImageFont:
    for font_path in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/consola.ttf"),
    ):
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), 14)
    return ImageFont.load_default()


def color_for_category(category_id: int) -> str:
    return PALETTE[(category_id - 1) % len(PALETTE)]


def draw_annotation_image(
    *,
    source_path: Path,
    output_path: Path,
    image_info: dict[str, Any],
    annotations: list[dict[str, Any]],
    category_names: dict[int, str],
    font: ImageFont.ImageFont,
) -> None:
    if source_path.is_file():
        image = Image.open(source_path).convert("RGB")
    else:
        width = int(image_info.get("width", 640) or 640)
        height = int(image_info.get("height", 480) or 480)
        image = Image.new("RGB", (width, height), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    width, height = image.size

    if not annotations:
        draw_label(draw, (8, 8), "NO_ANNOTATIONS", "#ff0000", font)

    for annotation in annotations:
        category_id = int(annotation.get("category_id", -1))
        label = category_names.get(category_id, f"unknown_{category_id}")
        bbox = [float(value) for value in annotation.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = bbox
        x1 = clamp(x, 0, width - 1)
        y1 = clamp(y, 0, height - 1)
        x2 = clamp(x + w, 0, width - 1)
        y2 = clamp(y + h, 0, height - 1)
        color = color_for_category(category_id)
        line_width = max(2, int(round(min(width, height) / 240)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        annotation_id = annotation.get("id", "")
        draw_label(draw, (int(x1), max(0, int(y1) - 18)), f"{label} #{annotation_id}", color, font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def draw_label(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str, color: str, font: ImageFont.ImageFont) -> None:
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
    draw.text((x, y), text, fill="white", font=font)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def inspect_image_annotations(
    *,
    split: str,
    image_info: dict[str, Any],
    annotations: list[dict[str, Any]],
    category_names: dict[int, str],
    duplicate_iou_threshold: float,
    small_area_ratio: float,
    large_area_ratio: float,
) -> list[dict[str, str]]:
    file_name = normalize_file_name(str(image_info.get("file_name", "")))
    width = float(image_info.get("width", 0) or 0)
    height = float(image_info.get("height", 0) or 0)
    image_area = max(1.0, width * height)
    rows: list[dict[str, str]] = []

    if not annotations:
        rows.append(make_issue(split, file_name, "", "", "image_has_no_annotations", "COCO image has no bbox."))
        return rows

    normalized_boxes: list[tuple[dict[str, Any], list[float], str]] = []
    for annotation in annotations:
        annotation_id = str(annotation.get("id", ""))
        category_id = int(annotation.get("category_id", -1))
        label = category_names.get(category_id, f"unknown_{category_id}")
        bbox = [float(value) for value in annotation.get("bbox", [0, 0, 0, 0])]
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            rows.append(make_issue(split, file_name, annotation_id, label, "invalid_bbox", f"bbox={bbox}"))
            continue
        if x < 0 or y < 0 or x + w > width or y + h > height:
            rows.append(make_issue(split, file_name, annotation_id, label, "bbox_out_of_bounds", f"bbox={bbox}, image={width}x{height}"))
        area_ratio = (w * h) / image_area
        if area_ratio < small_area_ratio:
            rows.append(make_issue(split, file_name, annotation_id, label, "very_small_bbox", f"area_ratio={area_ratio:.6f}, bbox={bbox}"))
        if area_ratio > large_area_ratio:
            rows.append(make_issue(split, file_name, annotation_id, label, "very_large_bbox", f"area_ratio={area_ratio:.6f}, bbox={bbox}"))
        edge_count = int(x <= 1) + int(y <= 1) + int(x + w >= width - 1) + int(y + h >= height - 1)
        if edge_count >= 2:
            rows.append(make_issue(split, file_name, annotation_id, label, "bbox_touches_multiple_edges", f"edges={edge_count}, bbox={bbox}"))
        normalized_boxes.append((annotation, bbox, label))

    for left_index in range(len(normalized_boxes)):
        left_ann, left_bbox, left_label = normalized_boxes[left_index]
        for right_ann, right_bbox, right_label in normalized_boxes[left_index + 1 :]:
            if left_label == right_label:
                continue
            overlap = iou_xywh(left_bbox, right_bbox)
            if overlap >= duplicate_iou_threshold:
                detail = (
                    f"iou={overlap:.4f}, left_ann={left_ann.get('id')}, left_label={left_label}, "
                    f"right_ann={right_ann.get('id')}, right_label={right_label}"
                )
                rows.append(
                    make_issue(
                        split,
                        file_name,
                        f"{left_ann.get('id')}|{right_ann.get('id')}",
                        f"{left_label}|{right_label}",
                        "duplicate_overlap_different_label",
                        detail,
                    )
                )
    return rows


def iou_xywh(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    lx2, ly2 = lx + lw, ly + lh
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    union = lw * lh + rw * rh - inter
    return 0.0 if union <= 0 else inter / union


def make_issue(split: str, file_name: str, annotation_id: str, label: str, issue: str, detail: str) -> dict[str, str]:
    return {
        "split": split,
        "file_name": file_name,
        "annotation_id": annotation_id,
        "label": label,
        "issue": issue,
        "detail": detail,
    }


def write_suspicious_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "file_name", "annotation_id", "label", "issue", "detail"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_image_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["split", "file_name", "visual_path", "annotation_count", "labels", "suspicious_count"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "labels": " ".join(row["labels"])})


def write_html_index(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "<!doctype html>",
        "<html lang=\"zh-CN\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>01_target_det 标注可视化检查</title>",
        "<style>",
        "body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:20px;background:#f7f7f7;color:#222}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}",
        ".card{background:white;border:1px solid #ddd;border-radius:6px;padding:8px}",
        ".card.bad{border-color:#d33;background:#fff6f6}",
        "img{width:100%;height:auto;display:block;border-radius:4px}",
        ".meta{font-size:12px;line-height:1.45;word-break:break-all}",
        ".split{margin:28px 0 12px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>01_target_det 标注可视化检查</h1>",
        f"<p>dataset: <code>{html.escape(summary['dataset_dir'])}</code></p>",
        f"<p>total images: <strong>{summary['total_images']}</strong>, annotations: <strong>{summary['total_annotations']}</strong>, suspicious items: <strong>{summary['suspicious_items']}</strong></p>",
        "<p>疑似问题详情见 <code>suspicious_annotations.csv</code>，图片索引见 <code>image_index.csv</code>。</p>",
    ]
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_split[str(row["split"])].append(row)
    for split, split_rows in rows_by_split.items():
        lines.append(f"<h2 class=\"split\">{html.escape(split)} ({len(split_rows)} images)</h2>")
        lines.append("<div class=\"grid\">")
        for row in split_rows:
            labels = ", ".join(row["labels"])
            card_class = "card bad" if int(row["suspicious_count"]) else "card"
            lines.extend(
                [
                    f"<div class=\"{card_class}\">",
                    f"<a href=\"{html.escape(row['visual_path'])}\"><img loading=\"lazy\" src=\"{html.escape(row['visual_path'])}\" alt=\"{html.escape(row['file_name'])}\"></a>",
                    "<div class=\"meta\">",
                    f"<div><strong>{html.escape(row['file_name'])}</strong></div>",
                    f"<div>boxes: {row['annotation_count']} | suspicious: {row['suspicious_count']}</div>",
                    f"<div>{html.escape(labels)}</div>",
                    "</div>",
                    "</div>",
                ]
            )
        lines.append("</div>")
    lines.extend(["</body>", "</html>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    summary = visualize_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        duplicate_iou_threshold=args.duplicate_iou_threshold,
        small_area_ratio=args.small_area_ratio,
        large_area_ratio=args.large_area_ratio,
    )
    print("target_det annotation visualization finished")
    print(f"dataset_dir: {summary['dataset_dir']}")
    print(f"output_dir: {summary['output_dir']}")
    print(f"total_images: {summary['total_images']}")
    print(f"total_annotations: {summary['total_annotations']}")
    print(f"suspicious_items: {summary['suspicious_items']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
