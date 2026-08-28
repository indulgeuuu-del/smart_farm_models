# 功能：汇总多个 01_target_det hard-case report，便于 checkpoint 横向对比。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Hard-case report mapping, for example: epoch9=path/to/report.json.",
    )
    parser.add_argument(
        "--confusion-pair",
        nargs="*",
        default=[],
        metavar="GT:PRED",
        help="Optional named confusion pairs to extract, for example: water_l1:water_l2.",
    )
    parser.add_argument("--output-json", type=Path, required=True, help="Summary JSON output path.")
    parser.add_argument("--output-md", type=Path, help="Optional Markdown summary output path.")
    parser.add_argument("--title", default="Hard-case Summary", help="Markdown title when --output-md is used.")
    return parser


def parse_report_args(items: list[str]) -> dict[str, Path]:
    reports: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Report must use NAME=PATH format: {item}")
        name, raw_path = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Report name is empty: {item}")
        reports[name] = Path(raw_path)
    return reports


def parse_confusion_pairs(items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Confusion pair must use GT:PRED format: {item}")
        gt_label, pred_label = item.split(":", 1)
        pairs.append((gt_label.strip(), pred_label.strip()))
    return pairs


def build_summary(reports: dict[str, Path], confusion_pairs: list[tuple[str, str]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, path in reports.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        overall = payload.get("overall", {})
        gt = int(overall.get("gt_annotations", 0) or 0)
        correct = int(overall.get("correct", 0) or 0)
        confused = int(overall.get("confused", 0) or 0)
        missed = int(overall.get("missed", 0) or 0)
        confusions = payload.get("confusions", {})
        row: dict[str, Any] = {
            "gt": gt,
            "correct": correct,
            "confused": confused,
            "missed": missed,
            "accuracy": round(correct / gt, 4) if gt else None,
        }
        for gt_label, pred_label in confusion_pairs:
            key = f"{gt_label}_to_{pred_label}"
            row[key] = int(confusions.get(gt_label, {}).get(pred_label, 0) or 0)
        summary[name] = row
    return summary


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(summary: dict[str, Any], title: str) -> str:
    base_columns = ["model", "gt", "correct", "confused", "missed", "accuracy"]
    extra_columns: list[str] = []
    for row in summary.values():
        for key in row:
            if key not in base_columns and key not in extra_columns:
                extra_columns.append(key)
    columns = base_columns + extra_columns

    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")
    for name, row in summary.items():
        values = [name]
        values.extend(format_cell(row.get(column)) for column in columns[1:])
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    reports = parse_report_args(args.report)
    confusion_pairs = parse_confusion_pairs(args.confusion_pair)
    summary = build_summary(reports, confusion_pairs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY_JSON={args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary, args.title), encoding="utf-8")
        print(f"SUMMARY_MD={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
