# Purpose: Generate dataset diagnostics for the 01_target_det COCO dataset.
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR = Path("./datasets/01_target_det/paddlex")
DEFAULT_OUTPUT_DIR = Path("./outputs/target_det_hq_diagnostics")
SPLIT_FILES = {
    "train": "annotations/instance_train.json",
    "val": "annotations/instance_val.json",
    "test": "annotations/instance_test.json",
}


@dataclass(frozen=True)
class SplitSummary:
    split: str
    category_ids: tuple[int, ...]
    class_names: tuple[str, ...]
    images: int
    annotations: int
    class_counts: dict[str, int]
    size_counts: dict[str, int]
    class_size_counts: dict[str, dict[str, int]]
    weak_classes: list[str]
    sample_images_by_class: dict[str, list[str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="01_target_det paddlex dataset root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for diagnostic reports.")
    parser.add_argument(
        "--weak-threshold",
        type=non_negative_int,
        default=100,
        help="Classes below this train count are treated as weak.",
    )
    parser.add_argument(
        "--sample-limit",
        type=non_negative_int,
        default=12,
        help="Max sample image names kept per weak class.",
    )
    return parser


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def analyze_dataset(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    weak_threshold: int = 100,
    sample_limit: int = 12,
) -> dict[str, Any]:
    dataset_root = Path(dataset_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_split(split, dataset_root / relative_path, weak_threshold, sample_limit)
        for split, relative_path in SPLIT_FILES.items()
    ]
    class_names = validate_split_category_contract(summaries)
    payload = {
        "dataset_dir": str(dataset_root),
        "class_names": list(class_names),
        "num_classes": len(class_names),
        "weak_threshold": weak_threshold,
        "splits": {summary.split: split_summary_to_dict(summary) for summary in summaries},
        "recommendations": build_recommendations(summaries),
    }

    (output_root / "class_distribution.json").write_text(
        json.dumps(build_class_distribution(summaries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "box_size_distribution.json").write_text(
        json.dumps(build_box_size_distribution(summaries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "weak_class_samples.json").write_text(
        json.dumps(build_weak_class_samples(summaries), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "diagnostics_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "diagnostics_report.md").write_text(render_markdown_report(payload), encoding="utf-8")
    return payload


def summarize_split(coco_json: str | Path, split: str, weak_threshold: int = 100, sample_limit: int = 12) -> SplitSummary:
    # Backward-compatible call style: summarize_split("train", path, ...).
    if isinstance(coco_json, str) and coco_json in SPLIT_FILES:
        split_name = coco_json
        coco_path = Path(split)
    else:
        split_name = str(split)
        coco_path = Path(coco_json)

    payload = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = validate_categories(payload.get("categories", []), coco_path)
    category_ids = tuple(categories)
    class_names = tuple(categories.values())
    image_by_id = {int(image["id"]): image for image in payload.get("images", [])}
    class_counts = Counter({name: 0 for name in class_names})
    size_counts = Counter({"small": 0, "medium": 0, "large": 0})
    class_size_counts: dict[str, Counter[str]] = {
        name: Counter({"small": 0, "medium": 0, "large": 0}) for name in class_names
    }
    sample_images: dict[str, list[str]] = defaultdict(list)

    for ann in payload.get("annotations", []):
        category_id = int(ann["category_id"])
        if category_id not in categories:
            raise ValueError(f"Annotation references unknown category_id {category_id} in {coco_path}")
        class_name = categories[category_id]
        image = image_by_id.get(int(ann["image_id"]))
        bbox = ann.get("bbox", [0, 0, 0, 0])
        area = float(ann.get("area") or float(bbox[2]) * float(bbox[3]))
        size_name = classify_box_size(area)
        class_counts[class_name] += 1
        size_counts[size_name] += 1
        class_size_counts[class_name][size_name] += 1
        if image is not None and len(sample_images[class_name]) < sample_limit:
            sample_images[class_name].append(str(image.get("file_name", "")))

    weak_classes = [name for name in class_names if class_counts[name] < weak_threshold]
    return SplitSummary(
        split=split_name,
        category_ids=category_ids,
        class_names=class_names,
        images=len(payload.get("images", [])),
        annotations=len(payload.get("annotations", [])),
        class_counts={name: int(class_counts[name]) for name in class_names},
        size_counts={name: int(size_counts[name]) for name in ("small", "medium", "large")},
        class_size_counts={
            class_name: {size: int(counts[size]) for size in ("small", "medium", "large")}
            for class_name, counts in class_size_counts.items()
        },
        weak_classes=weak_classes,
        sample_images_by_class={name: sample_images.get(name, []) for name in class_names},
    )


def validate_categories(categories: object, coco_path: Path) -> dict[int, str]:
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"Categories must be a non-empty list in {coco_path}")

    id_to_name: dict[int, str] = {}
    names_seen: set[str] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict):
            raise ValueError(f"Category at index {index} is not an object in {coco_path}")
        category_id = category.get("id")
        if not isinstance(category_id, int) or isinstance(category_id, bool):
            raise ValueError(f"Category at index {index} has an invalid id in {coco_path}")
        if category_id <= 0:
            raise ValueError(f"Category id must be positive in {coco_path}: {category_id}")
        if category_id in id_to_name:
            raise ValueError(f"Duplicate category id in {coco_path}: {category_id}")

        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Category at index {index} has an empty or invalid name in {coco_path}")
        if name != name.strip():
            raise ValueError(f"Category name has leading/trailing whitespace in {coco_path}: {name!r}")
        if name in names_seen:
            raise ValueError(f"Duplicate category name in {coco_path}: {name}")
        names_seen.add(name)
        id_to_name[category_id] = name
    return id_to_name


def validate_split_category_contract(summaries: list[SplitSummary]) -> tuple[str, ...]:
    if not summaries:
        raise ValueError("At least one split is required")
    expected_ids = summaries[0].category_ids
    expected = summaries[0].class_names
    for summary in summaries[1:]:
        if summary.category_ids != expected_ids or summary.class_names != expected:
            raise ValueError(
                "Category contract differs between splits: "
                f"{summaries[0].split}=ids {list(expected_ids)}, names {list(expected)}; "
                f"{summary.split}=ids {list(summary.category_ids)}, names {list(summary.class_names)}"
            )
    return expected


def classify_box_size(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def split_summary_to_dict(summary: SplitSummary) -> dict[str, Any]:
    return {
        "category_ids": list(summary.category_ids),
        "class_names": list(summary.class_names),
        "images": summary.images,
        "annotations": summary.annotations,
        "class_counts": summary.class_counts,
        "size_counts": summary.size_counts,
        "class_size_counts": summary.class_size_counts,
        "weak_classes": summary.weak_classes,
        "sample_images_by_class": summary.sample_images_by_class,
    }


def build_class_distribution(summaries: list[SplitSummary]) -> dict[str, Any]:
    return {
        summary.split: {
            "images": summary.images,
            "annotations": summary.annotations,
            "class_counts": summary.class_counts,
            "weak_classes": summary.weak_classes,
        }
        for summary in summaries
    }


def build_box_size_distribution(summaries: list[SplitSummary]) -> dict[str, Any]:
    return {
        summary.split: {
            "size_counts": summary.size_counts,
            "class_size_counts": summary.class_size_counts,
        }
        for summary in summaries
    }


def build_weak_class_samples(summaries: list[SplitSummary]) -> dict[str, Any]:
    train = next((summary for summary in summaries if summary.split == "train"), None)
    if train is None:
        return {}
    return {class_name: train.sample_images_by_class[class_name] for class_name in train.weak_classes}


def build_recommendations(summaries: list[SplitSummary]) -> list[str]:
    train = next((summary for summary in summaries if summary.split == "train"), None)
    if train is None:
        return ["Missing train split; cannot make training recommendations."]
    recommendations = []
    if train.weak_classes:
        recommendations.append("Inspect weak classes before long training: " + ", ".join(train.weak_classes))
    if train.size_counts["small"] < max(30, int(train.annotations * 0.03)):
        recommendations.append("Small objects are rare; prioritize box precision and class balance over tiny-object tricks.")
    recommendations.append("Keep the current full-NMS model as the replacement baseline until a new export beats it on AP and visual checks.")
    return recommendations


def render_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# 01_target_det Dataset Diagnostics",
        "",
        f"- dataset_dir: `{payload['dataset_dir']}`",
        f"- weak_threshold: `{payload['weak_threshold']}`",
        "",
        "## Split Summary",
        "",
        "| split | images | annotations | small | medium | large |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, summary in payload["splits"].items():
        size_counts = summary["size_counts"]
        lines.append(
            f"| {split} | {summary['images']} | {summary['annotations']} | "
            f"{size_counts['small']} | {size_counts['medium']} | {size_counts['large']} |"
        )
    lines.extend(["", "## Train Class Counts", "", "| class | count | weak |", "| --- | ---: | --- |"])
    train = payload["splits"].get("train", {})
    train_counts = train.get("class_counts", {})
    weak_classes = set(train.get("weak_classes", []))
    for class_name in payload.get("class_names", train_counts):
        lines.append(f"| {class_name} | {train_counts.get(class_name, 0)} | {'yes' if class_name in weak_classes else 'no'} |")
    lines.extend(["", "## Recommendations", ""])
    for item in payload["recommendations"]:
        lines.append(f"- {item}")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = build_parser().parse_args()
    result = analyze_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        weak_threshold=args.weak_threshold,
        sample_limit=args.sample_limit,
    )
    print("target_det dataset diagnostics generated")
    print(f"dataset_dir: {result['dataset_dir']}")
    print(f"output_dir: {Path(args.output_dir)}")
    print(f"train weak classes: {', '.join(result['splits']['train']['weak_classes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
