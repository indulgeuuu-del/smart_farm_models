"""Check whether a target_det handoff directory matches the official baseline loader shape."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = (
    REPO_ROOT
    / "deploy"
    / "onboard_handoff"
    / "jetson_models_20260604_hardfix_epoch9_candidate"
    / "target_det"
)

EXPECTED_TARGET_DET_LABELS = [
    "water_l3",
    "water_l2",
    "water_l1",
    "water",
    "storage",
    "lable_yellow",
    "lable_blue",
    "ball_yellow",
    "ball_blue",
    "animal",
    "cylinder_set",
    "cylinder_3",
    "cylinder_2",
    "cylinder_1",
    "rape",
    "broccoli",
    "potato",
    "h_qing_jiao",
    "celery",
    "mushroom",
    "flammulina velutipes",
    "tomato",
    "green bean",
]


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
            "issues": self.issues,
            "warnings": self.warnings,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="target_det handoff directory to check.")
    parser.add_argument(
        "--allow-label-superset",
        action="store_true",
        help="Allow infer_cfg.yml labels to contain the repo labels plus extra classes.",
    )
    parser.add_argument(
        "--allow-custom-labels",
        action="store_true",
        help="Allow a custom label_list while still checking model files, infer_cfg shape, arch, metric, and target size.",
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


def resolve_baseline_model_files(model_dir: Path) -> tuple[Path | None, Path | None, list[str]]:
    issues: list[str] = []
    candidates = [
        (model_dir / "model.pdmodel", model_dir / "model.pdiparams"),
        (model_dir / "inference.pdmodel", model_dir / "inference.pdiparams"),
    ]
    for model_file, params_file in candidates:
        if model_file.is_file() and params_file.is_file():
            return model_file, params_file, issues

    if (model_dir / "model.json").is_file() and (model_dir / "model.pdiparams").is_file():
        issues.append(
            "Found model.json + model.pdiparams only. Official baseline YoloeInfer expects "
            "model.pdmodel + model.pdiparams or inference.pdmodel + inference.pdiparams."
        )
    else:
        issues.append(
            "Missing supported inference model files. Expected model.pdmodel + model.pdiparams "
            "or inference.pdmodel + inference.pdiparams."
        )
    return None, None, issues


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


def check_labels(labels: list[str], allow_superset: bool, allow_custom: bool) -> list[str]:
    if allow_custom:
        if not labels:
            return ["label_list must not be empty when custom labels are allowed."]
        return []
    if allow_superset:
        missing = [label for label in EXPECTED_TARGET_DET_LABELS if label not in labels]
        if missing:
            return [f"label_list is missing required repo labels: {missing}"]
        return []
    if labels != EXPECTED_TARGET_DET_LABELS:
        return [
            "label_list must exactly match the current repo label order. "
            f"Expected {EXPECTED_TARGET_DET_LABELS}, got {labels}"
        ]
    return []


def check_model_dir(
    model_dir: Path,
    *,
    allow_label_superset: bool = False,
    allow_custom_labels: bool = False,
) -> CompatReport:
    model_dir = model_dir.resolve()
    issues: list[str] = []
    warnings: list[str] = []

    if not model_dir.is_dir():
        issues.append(f"Missing model directory: {model_dir}")
        return CompatReport(model_dir, None, None, None, None, None, None, None, 0, [], issues, warnings)

    model_file, params_file, model_issues = resolve_baseline_model_files(model_dir)
    issues.extend(model_issues)

    infer_cfg = model_dir / "infer_cfg.yml"
    config: dict[str, Any] = {}
    if not infer_cfg.is_file():
        issues.append("Missing infer_cfg.yml.")
        infer_cfg_path: Path | None = None
    else:
        infer_cfg_path = infer_cfg
        config = load_yaml(infer_cfg)

    arch = config.get("arch") if config else None
    metric = config.get("metric") if config else None
    draw_threshold = config.get("draw_threshold") if config else None
    target_size = extract_target_size(config) if config else None
    labels = config.get("label_list", []) if config else []
    if labels is None:
        labels = []

    if config:
        if arch != "YOLO":
            issues.append(f"infer_cfg.yml arch must be YOLO for official baseline YoloeInfer, got {arch!r}.")
        if metric != "COCO":
            issues.append(f"infer_cfg.yml metric should be COCO, got {metric!r}.")
        if target_size is None:
            issues.append("infer_cfg.yml Preprocess must include Resize.target_size.")
        elif target_size not in ([640, 640], [416, 416], [512, 512]):
            warnings.append(f"Uncommon target_size {target_size}; verify FPS and accuracy on Jetson.")
        if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
            issues.append("infer_cfg.yml label_list must be a list of strings.")
            labels = []
        else:
            issues.extend(check_labels(labels, allow_label_superset, allow_custom_labels))

    threshold_value = float(draw_threshold) if isinstance(draw_threshold, (int, float)) else None
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
        allow_label_superset=args.allow_label_superset,
        allow_custom_labels=args.allow_custom_labels,
    )
    if args.json:
        print(json.dumps(report.to_jsonable(), ensure_ascii=False, indent=2))
    else:
        print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
