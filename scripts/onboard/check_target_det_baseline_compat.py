"""Check the portable structure and label contract of an exported target_det model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_det_model import resolve_infer_model_files


@dataclass(frozen=True)
class CompatReport:
    model_dir: Path
    model_file: Path | None
    params_file: Path | None
    infer_cfg: Path | None
    arch: str | None
    metric: str | None
    draw_threshold: float | None
    target_size: list[int] | None
    label_count: int
    labels: list[str]
    labels_file: Path | None
    issues: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_dir": str(self.model_dir),
            "model_file": str(self.model_file) if self.model_file else None,
            "params_file": str(self.params_file) if self.params_file else None,
            "infer_cfg": str(self.infer_cfg) if self.infer_cfg else None,
            "arch": self.arch,
            "metric": self.metric,
            "draw_threshold": self.draw_threshold,
            "target_size": self.target_size,
            "label_count": self.label_count,
            "labels": self.labels,
            "labels_file": str(self.labels_file) if self.labels_file else None,
            "issues": self.issues,
            "warnings": self.warnings,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Exported target_det model directory to check.")
    parser.add_argument(
        "--expected-labels-file",
        default=None,
        help="Optional UTF-8 text file whose non-empty lines must exactly match infer_cfg.yml label_list.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - only happens in incomplete envs.
        raise RuntimeError("PyYAML is required to parse infer_cfg.yml.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"infer_cfg.yml must be a YAML mapping: {path}")
    return data


def extract_target_size(config: dict[str, Any]) -> list[int] | None:
    preprocess = config.get("Preprocess")
    if not isinstance(preprocess, list):
        return None
    for item in preprocess:
        if isinstance(item, dict) and item.get("type") == "Resize":
            target_size = item.get("target_size")
            if isinstance(target_size, list) and len(target_size) == 2:
                return [int(target_size[0]), int(target_size[1])]
    return None


def load_labels_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_model_dir(
    model_dir: Path,
    *,
    expected_labels_file: Path | None = None,
) -> CompatReport:
    model_dir = model_dir.resolve()
    issues: list[str] = []
    warnings: list[str] = []

    if not model_dir.is_dir():
        issues.append(f"Missing model directory: {model_dir}")
        return CompatReport(model_dir, None, None, None, None, None, None, None, 0, [], None, issues, warnings)

    model_file: Path | None = None
    params_file: Path | None = None
    try:
        model_file, params_file = resolve_infer_model_files(model_dir)
    except FileNotFoundError as exc:
        issues.append(str(exc))

    infer_cfg = model_dir / "infer_cfg.yml"
    config: dict[str, Any] = {}
    if not infer_cfg.is_file():
        issues.append("Missing infer_cfg.yml.")
        infer_cfg_path: Path | None = None
    else:
        infer_cfg_path = infer_cfg
        try:
            config = load_yaml(infer_cfg)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(str(exc))

    arch = config.get("arch") if config else None
    metric = config.get("metric") if config else None
    draw_threshold = config.get("draw_threshold") if config else None
    target_size = extract_target_size(config) if config else None
    labels = config.get("label_list", []) if config else []
    if labels is None:
        labels = []

    if config:
        if arch != "YOLO":
            issues.append(f"infer_cfg.yml arch must be YOLO for the repository target_det runner, got {arch!r}.")
        if metric != "COCO":
            issues.append(f"infer_cfg.yml metric should be COCO, got {metric!r}.")
        if target_size is None:
            issues.append("infer_cfg.yml Preprocess must include Resize.target_size.")
        elif any(value <= 0 for value in target_size):
            issues.append(f"Resize.target_size must contain positive integers, got {target_size}.")
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            issues.append("infer_cfg.yml label_list must be a list of strings.")
            labels = []
        else:
            if not labels:
                issues.append("infer_cfg.yml label_list must not be empty.")
            elif len(set(labels)) != len(labels):
                issues.append("infer_cfg.yml label_list contains duplicate labels.")

    labels_file: Path | None = None
    bundled_labels = model_dir / "label_list.txt"
    if bundled_labels.is_file():
        labels_file = bundled_labels
        try:
            if load_labels_file(bundled_labels) != labels:
                issues.append("label_list.txt does not exactly match infer_cfg.yml label_list.")
        except OSError as exc:
            issues.append(f"Unable to read bundled label_list.txt: {exc}")

    if expected_labels_file is not None:
        expected_labels_file = expected_labels_file.resolve()
        if not expected_labels_file.is_file():
            issues.append(f"Missing expected labels file: {expected_labels_file}")
        else:
            labels_file = expected_labels_file
            try:
                if load_labels_file(expected_labels_file) != labels:
                    issues.append(f"Expected labels file does not match infer_cfg.yml label_list: {expected_labels_file}")
            except OSError as exc:
                issues.append(f"Unable to read expected labels file: {exc}")

    threshold_value = float(draw_threshold) if isinstance(draw_threshold, (int, float)) else None
    if config and (threshold_value is None or not 0.0 <= threshold_value <= 1.0):
        issues.append(f"infer_cfg.yml draw_threshold must be a number in [0, 1], got {draw_threshold!r}.")
    return CompatReport(
        model_dir=model_dir,
        model_file=model_file,
        params_file=params_file,
        infer_cfg=infer_cfg_path,
        arch=str(arch) if arch is not None else None,
        metric=str(metric) if metric is not None else None,
        draw_threshold=threshold_value,
        target_size=target_size,
        label_count=len(labels),
        labels=labels,
        labels_file=labels_file,
        issues=issues,
        warnings=warnings,
    )


def print_report(report: CompatReport) -> None:
    status = "OK" if report.ok else "FAIL"
    print(f"STATUS={status}")
    print(f"MODEL_DIR={report.model_dir}")
    print(f"MODEL_FILE={report.model_file}")
    print(f"PARAMS_FILE={report.params_file}")
    print(f"INFER_CFG={report.infer_cfg}")
    print(f"ARCH={report.arch}")
    print(f"METRIC={report.metric}")
    print(f"DRAW_THRESHOLD={report.draw_threshold}")
    print(f"TARGET_SIZE={report.target_size}")
    print(f"LABEL_COUNT={report.label_count}")
    print(f"LABELS_FILE={report.labels_file}")
    if report.warnings:
        print("WARNINGS:")
        for item in report.warnings:
            print(f"- {item}")
    if report.issues:
        print("ISSUES:")
        for item in report.issues:
            print(f"- {item}")


def main() -> int:
    args = build_parser().parse_args()
    report = check_model_dir(
        Path(args.model_dir),
        expected_labels_file=Path(args.expected_labels_file) if args.expected_labels_file else None,
    )
    if args.json:
        print(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
