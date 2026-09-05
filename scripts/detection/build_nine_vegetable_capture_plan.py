# Purpose: Build and validate the approved nine-vegetable capture checklist.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "datasets"
    / "01_target_det"
    / "capture_plans"
    / "nine_vegetable_20260722"
)

FUSION_CLASS_NAMES_26 = (
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
)
FUSION_CLASS_NAMES_27 = FUSION_CLASS_NAMES_26 + ("green pepper",)

VEGETABLES = (
    ("RAPE", "rape"),
    ("BRO", "broccoli"),
    ("POTA", "potato"),
    ("CELE", "celery"),
    ("MUSH", "mushroom"),
    ("ENOKI", "flammulina velutipes"),
    ("TOMA", "tomato"),
    ("BEAN", "green bean"),
    ("PEPPER", "green pepper"),
)
VEGETABLE_LABELS = tuple(label for _, label in VEGETABLES)
VEGETABLE_CODES = {label: code for code, label in VEGETABLES}

POSITIONS = ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")
MATRIX_OFFSETS = {
    "train_a": (1, 2, 3, 4, 5, 6, 7, 8),
    "train_b": (1, 3, 5, 7, 2, 4, 6, 8),
    "val": (2, 5, 8, 3, 6, 1, 4, 7),
    "test": (4, 1, 7, 3, 8, 5, 2, 6),
}
MATRIX_SPLITS = {
    "train_a": "train",
    "train_b": "train",
    "val": "val",
    "test": "test",
}

CSV_COLUMNS = (
    ("sequence", "序号"),
    ("phase", "采集类型"),
    ("split", "数据划分"),
    ("matrix", "布局组"),
    ("layout_id", "布局编号"),
    ("missing_class", "缺席类别"),
    ("target_class", "目标类别"),
    ("target_position", "目标位置"),
    ("L1", "左1"),
    ("L2", "左2"),
    ("L3", "左3"),
    ("L4", "左4"),
    ("R1", "右1"),
    ("R2", "右2"),
    ("R3", "右3"),
    ("R4", "右4"),
    ("min_images", "最少图片数"),
    ("max_images", "最多图片数"),
    ("session_rule", "采集轮次要求"),
    ("card_orientation", "卡片方向"),
    ("status", "状态"),
    ("valid_images", "有效图片数"),
    ("notes", "备注"),
)
CSV_HEADERS = tuple(header for _, header in CSV_COLUMNS)

VEGETABLE_DISPLAY_NAMES = {
    "rape": "油菜",
    "broccoli": "西兰花",
    "potato": "土豆",
    "celery": "芹菜",
    "mushroom": "蘑菇",
    "flammulina velutipes": "金针菇",
    "tomato": "番茄",
    "green bean": "四季豆",
    "green pepper": "青椒",
}
POSITION_DISPLAY_NAMES = {
    "L1": "左1",
    "L2": "左2",
    "L3": "左3",
    "L4": "左4",
    "R1": "右1",
    "R2": "右2",
    "R3": "右3",
    "R4": "右4",
}
MATRIX_DISPLAY_NAMES = {
    "single": "单类别",
    "train_a": "训练矩阵甲",
    "train_b": "训练矩阵乙",
    "val": "验证矩阵",
    "test": "测试矩阵",
}
CSV_DISPLAY_VALUES = {
    "single": "单类别",
    "mixed": "混合",
    "train": "训练集",
    "val": "验证集",
    "test": "测试集",
    "at_least_two_independent_rounds": "至少两轮独立采集",
    "independent_vehicle_offsets": "独立车辆位置偏移",
    "separate_validation_session": "独立验证采集轮次",
    "separate_test_session": "独立测试采集轮次",
    "fixed_competition": "正式比赛固定方向",
    "pending": "待采集",
    "completed": "已完成",
    "invalid": "无效",
    "BLANK": "空白卡",
}


def _base_row(sequence: int, **values: Any) -> dict[str, Any]:
    row = {
        "sequence": sequence,
        "phase": "",
        "split": "",
        "matrix": "",
        "layout_id": "",
        "missing_class": "",
        "target_class": "",
        "target_position": "",
        **{position: "" for position in POSITIONS},
        "min_images": 0,
        "max_images": 0,
        "session_rule": "",
        "card_orientation": "fixed_competition",
        "status": "pending",
        "valid_images": 0,
        "notes": "",
    }
    row.update(values)
    return row


def _build_single_rows(start_sequence: int = 1) -> list[dict[str, Any]]:
    rows = []
    sequence = start_sequence
    for label in VEGETABLE_LABELS:
        code = VEGETABLE_CODES[label].lower()
        for position in POSITIONS:
            slots = {slot: "BLANK" for slot in POSITIONS}
            slots[position] = label
            rows.append(
                _base_row(
                    sequence,
                    phase="single",
                    split="train",
                    matrix="single",
                    layout_id=f"single_{code}_{position.lower()}",
                    target_class=label,
                    target_position=position,
                    min_images=6,
                    max_images=10,
                    session_rule="at_least_two_independent_rounds",
                    **slots,
                )
            )
            sequence += 1
    return rows


def _build_mixed_rows(start_sequence: int) -> list[dict[str, Any]]:
    rows = []
    sequence = start_sequence
    label_count = len(VEGETABLE_LABELS)
    for matrix_name, offsets in MATRIX_OFFSETS.items():
        split = MATRIX_SPLITS[matrix_name]
        min_images, max_images = (10, 20) if split == "train" else (10, 10)
        session_rule = {
            "train": "independent_vehicle_offsets",
            "val": "separate_validation_session",
            "test": "separate_test_session",
        }[split]
        for omitted_index, missing_class in enumerate(VEGETABLE_LABELS):
            slot_values = {
                position: VEGETABLE_LABELS[(omitted_index + offset) % label_count]
                for position, offset in zip(POSITIONS, offsets)
            }
            missing_code = VEGETABLE_CODES[missing_class].lower()
            rows.append(
                _base_row(
                    sequence,
                    phase="mixed",
                    split=split,
                    matrix=matrix_name,
                    layout_id=f"{matrix_name}_omit_{missing_code}",
                    missing_class=missing_class,
                    min_images=min_images,
                    max_images=max_images,
                    session_rule=session_rule,
                    **slot_values,
                )
            )
            sequence += 1
    return rows


def build_capture_plan() -> dict[str, Any]:
    """Return metadata and 108 placement rows from the approved design."""
    single_rows = _build_single_rows()
    mixed_rows = _build_mixed_rows(len(single_rows) + 1)
    rows = single_rows + mixed_rows
    return {
        "schema_version": 1,
        "plan_name": "nine_vegetable_20260722",
        "fusion_class_names": list(FUSION_CLASS_NAMES_27),
        "vegetables": [
            {"code": code, "label": label}
            for code, label in VEGETABLES
        ],
        "positions": list(POSITIONS),
        "matrix_offsets": {
            name: list(offsets)
            for name, offsets in MATRIX_OFFSETS.items()
        },
        "summary": {
            "single_placement_rows": len(single_rows),
            "mixed_placement_rows": len(mixed_rows),
            "placement_rows": len(rows),
            "expected_min_images": sum(row["min_images"] for row in rows),
            "expected_max_images": sum(row["max_images"] for row in rows),
        },
        "rows": rows,
    }


def validate_capture_plan(plan: dict[str, Any]) -> list[str]:
    """Return concrete invariant violations; return an empty list when valid."""
    errors = []
    rows = plan.get("rows")
    if not isinstance(rows, list):
        return ["rows must be a list"]

    if tuple(plan.get("fusion_class_names", ())) != FUSION_CLASS_NAMES_27:
        errors.append("fusion class order must preserve 26 classes and append green pepper")

    single_rows = [row for row in rows if row.get("phase") == "single"]
    expected_single_pairs = {
        (label, position)
        for label in VEGETABLE_LABELS
        for position in POSITIONS
    }
    actual_single_pairs = {
        (row.get("target_class"), row.get("target_position"))
        for row in single_rows
    }
    if len(single_rows) != 72 or actual_single_pairs != expected_single_pairs:
        errors.append("single rows must cover all 72 class-position pairs exactly once")

    mixed_rows = [row for row in rows if row.get("phase") == "mixed"]
    layouts = [tuple(row.get(position) for position in POSITIONS) for row in mixed_rows]
    if len(mixed_rows) != 36:
        errors.append("mixed rows must contain 36 layouts")
    if len(layouts) != len(set(layouts)):
        errors.append("duplicate mixed layout detected")
    expected_labels = set(VEGETABLE_LABELS)
    for matrix_name in MATRIX_OFFSETS:
        matrix_rows = [row for row in mixed_rows if row.get("matrix") == matrix_name]
        if len(matrix_rows) != 9:
            errors.append(f"matrix {matrix_name} must contain 9 layouts")
            continue
        if {row.get("missing_class") for row in matrix_rows} != expected_labels:
            errors.append(f"matrix {matrix_name} must omit each class once")
        for position in POSITIONS:
            if {row.get(position) for row in matrix_rows} != expected_labels:
                errors.append(f"matrix {matrix_name} position {position} must cover each class once")
        for row in matrix_rows:
            slot_values = [row.get(position) for position in POSITIONS]
            if row.get("missing_class") in slot_values:
                errors.append(f"layout {row.get('layout_id')} includes its missing class")
            if len(set(slot_values)) != len(POSITIONS):
                errors.append(f"layout {row.get('layout_id')} repeats a vegetable class")

    if len(rows) != 108:
        errors.append("capture plan must contain 108 placement rows")
    return errors


def _render_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    if row["phase"] == "single":
        layout_id = "_".join(
            (
                "单类别",
                VEGETABLE_DISPLAY_NAMES[row["target_class"]],
                POSITION_DISPLAY_NAMES[row["target_position"]],
            )
        )
    else:
        layout_id = "_".join(
            (
                MATRIX_DISPLAY_NAMES[row["matrix"]],
                "缺少",
                VEGETABLE_DISPLAY_NAMES[row["missing_class"]],
            )
        )

    rendered = {}
    for field, header in CSV_COLUMNS:
        value = row[field]
        if field == "layout_id":
            value = layout_id
        elif field == "matrix":
            value = MATRIX_DISPLAY_NAMES[value]
        elif field in {"missing_class", "target_class", *POSITIONS}:
            value = VEGETABLE_DISPLAY_NAMES.get(value, CSV_DISPLAY_VALUES.get(value, value))
        elif field == "target_position":
            value = POSITION_DISPLAY_NAMES.get(value, value)
        else:
            value = CSV_DISPLAY_VALUES.get(value, value)
        rendered[header] = value
    return rendered


def _render_readme(plan: dict[str, Any]) -> str:
    summary = plan["summary"]
    vegetable_rows = "\n".join(
        f"| {item['code']} | `{item['label']}` |"
        for item in plan["vegetables"]
    )
    return f"""# 九类蔬菜现场采集清单

本目录由 `scripts/detection/build_nine_vegetable_capture_plan.py` 生成，对应已经批准的九类蔬菜、八位置采集设计。

## 文件

- `capture_checklist.csv`：现场逐项执行和填写的 108 行摆放清单。
- `capture_plan.json`：供后续数据导入和自动检查使用的机器可读计划。
- `README.md`：本操作说明。

CSV 仅用于中文现场执行；JSON、生成器内部常量和后续训练类别仍使用下表中的英文模型标签。

## 类别

| 短码 | 模型标签 |
|---|---|
{vegetable_rows}

完整融合类别数为 `{len(plan['fusion_class_names'])}`，`green pepper` 必须追加在现有 26 类之后。

## 数量

- 单类别八位置：`{summary['single_placement_rows']}` 行。
- 八卡混合布局：`{summary['mixed_placement_rows']}` 行。
- 总摆放项：`{summary['placement_rows']}` 行。
- 预计有效图片：`{summary['expected_min_images']}～{summary['expected_max_images']}` 张。

## 现场执行

1. 按 CSV 的“序号”顺序执行，先核对“布局编号”和左1～右4。
2. 所有蔬菜卡保持正式比赛使用的固定方向，不在位置之间旋转。
3. 单类别行的其他七个位置使用同尺寸、同底色空白卡。
4. 每行采够“最少图片数”前不得标记完成；图片必须来自真实距离或左右偏移变化，不能用静止连拍充数。
5. 在“有效图片数”填写数量；完成后将“状态”改为“已完成”，废弃摆放改为“无效”并在“备注”写明原因。
6. 验证集和测试集必须使用各自独立采集轮次，不能复制训练图片或连续帧。

物理图片采集、筛选和标注完成前，不得把这份计划写成 COCO 数据已完成，也不得启动 27 类正式训练。
"""


def write_capture_plan(plan: dict[str, Any], output_dir: Path) -> None:
    errors = validate_capture_plan(plan)
    if errors:
        raise ValueError("Invalid capture plan:\n" + "\n".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "capture_plan.json").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "capture_checklist.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(_render_csv_row(row) for row in plan["rows"])
    with (output_dir / "README.md").open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(_render_readme(plan))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the approved nine-vegetable capture checklist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for capture_plan.json, capture_checklist.csv, and README.md.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = build_capture_plan()
    write_capture_plan(plan, args.output_dir)
    print("CAPTURE_PLAN_STATUS=OK")
    print(f"PLACEMENT_ROWS={plan['summary']['placement_rows']}")
    print(f"OUTPUT_DIR={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
