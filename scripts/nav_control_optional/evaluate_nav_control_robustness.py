# 功能：按 session 评估巡航模型的分段误差、欠转和连续帧输出稳定性。
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import paddle


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_nav_control_reg import (  # noqa: E402
    DEFAULT_INPUT_SIZE,
    _load_rgb_image,
    _normalize_to_chw,
    load_model_from_params,
    summarize_deviation_predictions,
)


ILLUMINATION_VARIANTS = (
    "original",
    "low_light",
    "overexposure",
    "low_contrast",
    "left_shadow",
    "right_shadow",
    "center_glare",
    "warm_cast",
    "cool_cast",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-format", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--model-path", required=True, help="Dynamic .pdparams or static model prefix.")
    parser.add_argument("--data-dir", required=True, help="Prepared split directory containing images/ and data.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON, Markdown, and CSV reports.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--spike-threshold", type=float, default=0.25)
    parser.add_argument("--illumination-variant", choices=ILLUMINATION_VARIANTS, default="original")
    return parser


def summarize_ordered_predictions(
    records: list[dict[str, Any]],
    predictions: np.ndarray,
    spike_threshold: float = 0.25,
) -> dict[str, Any]:
    prediction_values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if len(records) != len(prediction_values):
        raise ValueError(f"prediction count mismatch: records={len(records)} predictions={len(prediction_values)}")
    if not records:
        raise ValueError("records must not be empty")
    if not math.isfinite(spike_threshold) or spike_threshold <= 0:
        raise ValueError("spike_threshold must be a positive finite value")

    labels = np.asarray([_record_deviation(record) for record in records], dtype=np.float64)
    sessions = [_record_session(record) for record in records]
    robustness = summarize_deviation_predictions(prediction_values, labels)
    prediction_deltas: list[float] = []
    spike_count = 0
    session_indices: dict[int, list[int]] = {}
    for index, session in enumerate(sessions):
        session_indices.setdefault(session, []).append(index)
        if index == 0 or session != sessions[index - 1]:
            continue
        delta = abs(float(prediction_values[index] - prediction_values[index - 1]))
        prediction_deltas.append(delta)
        if delta >= spike_threshold:
            spike_count += 1

    per_session: dict[str, dict[str, Any]] = {}
    for session in sorted(session_indices):
        indices = session_indices[session]
        session_predictions = prediction_values[indices]
        session_labels = labels[indices]
        session_deltas = np.abs(np.diff(session_predictions))
        per_session[str(session)] = {
            "records": len(indices),
            "robustness": summarize_deviation_predictions(session_predictions, session_labels),
            "prediction_delta_p95": _percentile(session_deltas, 95.0),
            "prediction_delta_max": float(session_deltas.max()) if session_deltas.size else None,
            "spike_count": int((session_deltas >= spike_threshold).sum()),
        }

    return {
        "records": len(records),
        "sessions": sorted(session_indices),
        "spike_threshold": float(spike_threshold),
        "continuous_frame_pairs": len(prediction_deltas),
        "prediction_delta_p95": _percentile(np.asarray(prediction_deltas, dtype=np.float64), 95.0),
        "prediction_delta_max": max(prediction_deltas) if prediction_deltas else None,
        "spike_count": spike_count,
        "robustness": robustness,
        "per_session": per_session,
    }


def evaluate_model(
    model_format: str,
    model_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    batch_size: int = 256,
    device: str = "gpu:0",
    spike_threshold: float = 0.25,
    illumination_variant: str = "original",
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    data_root = Path(data_dir).resolve()
    records = _load_records(data_root)
    paddle.device.set_device(device)
    model = _load_model(model_format, model_path, device)
    predictions = _predict(model, data_root, records, batch_size, illumination_variant=illumination_variant)
    summary = summarize_ordered_predictions(records, predictions, spike_threshold=spike_threshold)
    summary.update(
        {
            "model_format": model_format,
            "model_path": str(Path(model_path).resolve()),
            "data_dir": str(data_root),
            "device": paddle.get_device(),
            "illumination_variant": illumination_variant,
        }
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "robustness_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "robustness_report.md").write_text(_render_markdown(summary), encoding="utf-8")
    _write_frame_csv(output / "frame_predictions.csv", records, predictions, spike_threshold)
    return summary


def _load_records(data_dir: Path) -> list[dict[str, Any]]:
    data_path = data_dir / "data.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data.jsonl: {data_path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record {data_path}:{line_number}")
        image_path = data_dir / _record_image_rel(record)
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image {data_path}:{line_number}: {image_path}")
        _record_session(record)
        _record_deviation(record)
        records.append(record)
    if not records:
        raise ValueError(f"Empty data.jsonl: {data_path}")
    return records


def _load_model(model_format: str, model_path: str | Path, device: str) -> paddle.nn.Layer:
    if model_format == "dynamic":
        return load_model_from_params(model_path, device=device)
    if model_format == "static":
        model = paddle.jit.load(str(model_path))
        model.eval()
        return model
    raise ValueError(f"Unsupported model format: {model_format}")


def _predict(
    model: paddle.nn.Layer,
    data_dir: Path,
    records: list[dict[str, Any]],
    batch_size: int,
    illumination_variant: str = "original",
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with paddle.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            images = []
            for record in batch_records:
                image = _load_rgb_image(data_dir / _record_image_rel(record), DEFAULT_INPUT_SIZE)
                image = apply_illumination_variant(image, illumination_variant)
                images.append(_normalize_to_chw(image))
            output = model(paddle.to_tensor(np.stack(images), dtype="float32"))
            predictions.append(output[:, 0].numpy().reshape(-1))
    return np.concatenate(predictions).astype(np.float64)


def apply_illumination_variant(image: np.ndarray, variant: str) -> np.ndarray:
    source = image.astype(np.float32, copy=True)
    if variant == "original":
        result = source
    elif variant == "low_light":
        normalized = np.clip(source, 0.0, 255.0) / 255.0
        result = np.power(normalized, 1.5) * 255.0 * 0.7 - 5.0
    elif variant == "overexposure":
        normalized = np.clip(source, 0.0, 255.0) / 255.0
        result = np.power(normalized, 0.7) * 255.0 * 1.12 + 18.0
    elif variant == "low_contrast":
        mean = source.mean(axis=(0, 1), keepdims=True)
        result = mean + (source - mean) * 0.55
    elif variant in {"left_shadow", "right_shadow"}:
        width = source.shape[1]
        ramp = np.linspace(0.35, 1.0, width, dtype=np.float32)
        if variant == "right_shadow":
            ramp = ramp[::-1].copy()
        result = source * ramp.reshape(1, width, 1)
    elif variant == "center_glare":
        height, width = source.shape[:2]
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        mask = np.exp(-(x**2 + (y + 0.2) ** 2) / 0.35)
        result = source + 85.0 * mask[:, :, None]
    elif variant in {"warm_cast", "cool_cast"}:
        gains = np.array([1.18, 1.0, 0.82] if variant == "warm_cast" else [0.82, 1.0, 1.18], dtype=np.float32)
        result = source * gains.reshape(1, 1, 3)
    else:
        raise ValueError(f"Unsupported illumination variant: {variant}")
    return np.clip(result, 0.0, 255.0).astype(np.float32)


def _write_frame_csv(
    path: Path,
    records: list[dict[str, Any]],
    predictions: np.ndarray,
    spike_threshold: float,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "session", "img_path", "label", "prediction", "abs_error", "prediction_delta", "spike"])
        previous_prediction: float | None = None
        previous_session: int | None = None
        for index, (record, prediction) in enumerate(zip(records, predictions)):
            session = _record_session(record)
            label = _record_deviation(record)
            delta = None if previous_session != session or previous_prediction is None else abs(float(prediction) - previous_prediction)
            writer.writerow(
                [
                    index,
                    session,
                    _record_image_rel(record),
                    label,
                    float(prediction),
                    abs(float(prediction) - label),
                    "" if delta is None else delta,
                    False if delta is None else delta >= spike_threshold,
                ]
            )
            previous_prediction = float(prediction)
            previous_session = session


def _render_markdown(summary: dict[str, Any]) -> str:
    robustness = summary["robustness"]
    lines = [
        "# 巡航模型鲁棒性离线评估",
        "",
        f"- 模型：`{summary['model_path']}`",
        f"- 数据：`{summary['data_dir']}`",
        f"- 光照压力变体：`{summary.get('illumination_variant', 'original')}`",
        f"- 样本：`{summary['records']}`",
        f"- 全局 MAE：`{robustness['global_mae']:.6f}`",
        f"- 方向错误率：`{_format_optional(robustness['direction_error_rate'])}`",
        f"- 欠转率：`{_format_optional(robustness['understeer_rate'])}`",
        f"- 输出变化 P95：`{_format_optional(summary['prediction_delta_p95'])}`",
        f"- 尖峰数：`{summary['spike_count']}`",
        "",
        "| session | records | global MAE | corner score | delta P95 | spikes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for session, item in summary["per_session"].items():
        session_robustness = item["robustness"]
        lines.append(
            f"| {session} | {item['records']} | {session_robustness['global_mae']:.6f} | "
            f"{_format_optional(session_robustness['corner_score'])} | "
            f"{_format_optional(item['prediction_delta_p95'])} | {item['spike_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _record_session(record: dict[str, Any]) -> int:
    value = record.get("session")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != float(value):
        raise ValueError(f"Invalid session value: {value}")
    return int(value)


def _record_deviation(record: dict[str, Any]) -> float:
    value = record.get("deviation")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid deviation value: {value}")
    return float(value)


def _record_image_rel(record: dict[str, Any]) -> str:
    value = record.get("img_path")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid img_path: {value}")
    normalized = value.replace("\\", "/").strip()
    path = Path(normalized)
    if path.is_absolute() or ":" in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"Unsafe img_path: {value}")
    return normalized


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.percentile(values, percentile))


def _format_optional(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def main() -> int:
    args = build_parser().parse_args()
    summary = evaluate_model(
        model_format=args.model_format,
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        spike_threshold=args.spike_threshold,
        illumination_variant=args.illumination_variant,
    )
    print("nav control robustness evaluation passed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
