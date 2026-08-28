# 功能：离线分析 08_nav_control_optional 连续帧推理曲线，检查抖动、尖峰和限幅/平滑建议。
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import paddle
from paddle.io import DataLoader

from train_nav_control_reg import (
    DEFAULT_EVAL_SOURCES,
    DEFAULT_TRAIN_SOURCES,
    MODEL_BASENAME,
    PaddleLaneDataset,
    load_model_from_params,
    load_samples,
    _parse_names,
)


DEFAULT_DATASET_DIR = Path("./datasets/08_nav_control_optional")
DEFAULT_OUTPUT_ROOT = Path("./deploy/infer_test/nav_control_optional")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-format", default="dynamic", choices=("dynamic", "static"), help="dynamic loads .pdparams; static loads exported jit model.")
    parser.add_argument("--model-path", required=True, help="Path to .pdparams or exported model prefix/.json.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="08_nav_control_optional dataset root.")
    parser.add_argument("--train-sources", default=",".join(DEFAULT_TRAIN_SOURCES), help="Comma-separated raw image-set names for training split.")
    parser.add_argument("--eval-sources", default=",".join(DEFAULT_EVAL_SOURCES), help="Comma-separated raw image-set names for eval split.")
    parser.add_argument("--split", default="eval", choices=("train", "eval", "val"), help="Split to inspect.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for offline inference.")
    parser.add_argument("--device", default="gpu:0", help="Paddle device.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Centered moving-average window for smoothing preview.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for summary and frame csv.")
    return parser


def run_sequence_analysis(
    model_format: str,
    model_path: str | Path,
    dataset_dir: str | Path,
    split: str = "eval",
    batch_size: int = 256,
    device: str = "gpu:0",
    smooth_window: int = 5,
    output_dir: str | Path | None = None,
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES,
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> dict[str, Any]:
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")
    samples = load_samples(dataset_dir, split, train_sources, eval_sources)
    model, resolved_model_path = _load_inference_model(model_format, model_path, device)
    records = _collect_sequence_records(model, samples, batch_size, device)
    source_groups = _group_records_by_source(records)
    source_summaries = []
    frame_rows: list[dict[str, Any]] = []
    for source_name, source_records in source_groups.items():
        labels = np.array([row["label"] for row in source_records], dtype=np.float32)
        preds = np.array([row["pred"] for row in source_records], dtype=np.float32)
        source_summary = summarize_sequence_metrics(source_name, labels, preds, smooth_window=smooth_window)
        source_summaries.append(source_summary)
        smooth_preds = np.array(source_summary["smooth_preds"], dtype=np.float32)
        for index, row in enumerate(source_records):
            frame_rows.append(
                {
                    "global_index": row["global_index"],
                    "source_name": source_name,
                    "source_frame_index": index,
                    "image_rel": row["image_rel"],
                    "label_0": float(labels[index, 0]),
                    "label_1": float(labels[index, 1]),
                    "pred_0": float(preds[index, 0]),
                    "pred_1": float(preds[index, 1]),
                    "smooth_pred_0": float(smooth_preds[index, 0]),
                    "smooth_pred_1": float(smooth_preds[index, 1]),
                    "abs_error_0": float(abs(preds[index, 0] - labels[index, 0])),
                    "abs_error_1": float(abs(preds[index, 1] - labels[index, 1])),
                }
            )

    result = {
        "model_format": model_format,
        "model_path": str(resolved_model_path),
        "split": "eval" if split == "val" else split,
        "device": paddle.get_device(),
        "checked_samples": len(records),
        "smooth_window": smooth_window,
        "source_count": len(source_summaries),
        "sources": source_summaries,
        "overall": _build_overall_summary(source_summaries),
    }

    target_dir = _resolve_output_dir(output_dir, model_path, result["split"])
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_path = target_dir / "summary.json"
    frames_path = target_dir / "frames.csv"
    markdown_path = target_dir / "summary.md"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_frame_rows(frames_path, frame_rows)
    markdown_path.write_text(_build_summary_markdown(result), encoding="utf-8")
    result["output_dir"] = str(target_dir)
    result["summary_path"] = str(summary_path)
    result["frames_path"] = str(frames_path)
    result["markdown_path"] = str(markdown_path)
    return result


def summarize_sequence_metrics(
    source_name: str,
    labels: np.ndarray,
    preds: np.ndarray,
    smooth_window: int = 5,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float32)
    preds = np.asarray(preds, dtype=np.float32)
    if labels.shape != preds.shape:
        raise ValueError(f"labels/preds shape mismatch: {labels.shape} vs {preds.shape}")
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] arrays, got {labels.shape}")

    smooth_preds = np.stack(
        [_moving_average(preds[:, head_index], smooth_window) for head_index in range(preds.shape[1])],
        axis=1,
    )
    outputs: dict[str, dict[str, Any]] = {}
    for head_index in range(preds.shape[1]):
        head_name = f"output_{head_index}"
        outputs[head_name] = _summarize_output_series(labels[:, head_index], preds[:, head_index], smooth_preds[:, head_index])

    return {
        "source_name": source_name,
        "frame_count": int(labels.shape[0]),
        "smooth_window": smooth_window,
        "outputs": outputs,
        "smooth_preds": smooth_preds.tolist(),
    }


def _summarize_output_series(labels: np.ndarray, preds: np.ndarray, smooth_preds: np.ndarray) -> dict[str, Any]:
    label_min = float(np.min(labels))
    label_max = float(np.max(labels))
    pred_min = float(np.min(preds))
    pred_max = float(np.max(preds))
    label_range = float(label_max - label_min)
    margin = max(label_range * 0.1, 0.02)

    raw_mae = float(np.mean(np.abs(preds - labels)))
    smooth_mae = float(np.mean(np.abs(smooth_preds - labels)))

    pred_delta = np.diff(preds)
    label_delta = np.diff(labels)
    pred_second = np.diff(preds, n=2)
    label_second = np.diff(labels, n=2)
    smooth_second = np.diff(smooth_preds, n=2)

    pred_step_abs = np.abs(pred_delta)
    label_step_abs = np.abs(label_delta)
    pred_spike = np.abs(preds[1:-1] - 0.5 * (preds[:-2] + preds[2:])) if len(preds) >= 3 else np.array([], dtype=np.float32)
    spike_threshold = max(
        float(np.percentile(np.abs(label_second), 95)) * 2.0 if label_second.size else 0.0,
        float(np.percentile(label_step_abs, 95)) * 2.0 if label_step_abs.size else 0.0,
        label_range * 0.15,
        0.03,
    )
    spike_indices = [int(index + 1) for index, value in enumerate(pred_spike) if float(value) > spike_threshold]
    out_of_range_mask = (preds < (label_min - margin)) | (preds > (label_max + margin))
    out_of_range_count = int(np.count_nonzero(out_of_range_mask))

    pred_jitter = float(np.mean(np.abs(pred_second))) if pred_second.size else 0.0
    smooth_jitter = float(np.mean(np.abs(smooth_second))) if smooth_second.size else 0.0
    label_jitter = float(np.mean(np.abs(label_second))) if label_second.size else 0.0
    suggest_smoothing = (
        (len(spike_indices) > 0 or pred_jitter > max(label_jitter * 1.5, 0.01))
        and smooth_jitter <= pred_jitter * 0.8 + 1e-8
        and smooth_mae <= raw_mae + max(label_range * 0.05, 0.01)
    )
    suggest_clamp = out_of_range_count > 0

    return {
        "label_min": label_min,
        "label_max": label_max,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "raw_mae": raw_mae,
        "smooth_mae": smooth_mae,
        "pred_step_abs_mean": _mean_or_zero(pred_step_abs),
        "pred_step_abs_p95": _percentile_or_zero(pred_step_abs, 95),
        "pred_step_abs_max": _max_or_zero(pred_step_abs),
        "label_step_abs_mean": _mean_or_zero(label_step_abs),
        "label_step_abs_p95": _percentile_or_zero(label_step_abs, 95),
        "label_step_abs_max": _max_or_zero(label_step_abs),
        "pred_jitter_mean": pred_jitter,
        "label_jitter_mean": label_jitter,
        "smooth_jitter_mean": smooth_jitter,
        "spike_threshold": spike_threshold,
        "spike_count": len(spike_indices),
        "spike_indices": spike_indices,
        "out_of_range_count": out_of_range_count,
        "clamp_margin": margin,
        "suggest_clamp": suggest_clamp,
        "suggested_clamp_min": label_min,
        "suggested_clamp_max": label_max,
        "suggest_smoothing": bool(suggest_smoothing),
    }


def _load_inference_model(model_format: str, model_path: str | Path, device: str) -> tuple[paddle.nn.Layer, Path]:
    if model_format == "dynamic":
        resolved = Path(model_path)
        model = load_model_from_params(resolved, device=device)
        return model, resolved
    paddle.device.set_device(device)
    resolved = _normalize_static_prefix(model_path)
    model = paddle.jit.load(str(resolved))
    model.eval()
    return model, resolved


def _collect_sequence_records(
    model: paddle.nn.Layer,
    samples: list[Any],
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    paddle.device.set_device(device)
    dataset = PaddleLaneDataset(samples, training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    records: list[dict[str, Any]] = []
    sample_cursor = 0
    with paddle.no_grad():
        for images, _ in loader:
            preds = model(images).numpy()
            batch_size_now = int(preds.shape[0])
            batch_samples = samples[sample_cursor : sample_cursor + batch_size_now]
            for local_index, sample in enumerate(batch_samples):
                records.append(
                    {
                        "global_index": sample_cursor + local_index,
                        "source_name": sample.source_name,
                        "image_rel": sample.image_rel,
                        "label": [float(sample.label[0]), float(sample.label[1])],
                        "pred": [float(preds[local_index, 0]), float(preds[local_index, 1])],
                    }
                )
            sample_cursor += batch_size_now
    if sample_cursor != len(samples):
        raise RuntimeError(f"Inference sample count mismatch: {sample_cursor} vs {len(samples)}")
    return records


def _group_records_by_source(records: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    groups: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
    for row in records:
        groups.setdefault(row["source_name"], []).append(row)
    return groups


def _build_overall_summary(source_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    smoothing_heads: list[str] = []
    clamp_heads: list[str] = []
    spike_count = 0
    out_of_range_count = 0
    for source in source_summaries:
        for head_name, metrics in source["outputs"].items():
            scoped_name = f"{source['source_name']}:{head_name}"
            if metrics["suggest_smoothing"]:
                smoothing_heads.append(scoped_name)
            if metrics["suggest_clamp"]:
                clamp_heads.append(scoped_name)
            spike_count += int(metrics["spike_count"])
            out_of_range_count += int(metrics["out_of_range_count"])
    return {
        "recommend_smoothing_targets": smoothing_heads,
        "recommend_clamp_targets": clamp_heads,
        "total_spike_count": spike_count,
        "total_out_of_range_count": out_of_range_count,
        "need_smoothing": bool(smoothing_heads),
        "need_clamp": bool(clamp_heads),
    }


def _build_summary_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# nav_control_optional sequence check",
        "",
        f"- model_format: {result['model_format']}",
        f"- model_path: {result['model_path']}",
        f"- split: {result['split']}",
        f"- checked_samples: {result['checked_samples']}",
        f"- smooth_window: {result['smooth_window']}",
        f"- need_smoothing: {result['overall']['need_smoothing']}",
        f"- need_clamp: {result['overall']['need_clamp']}",
        "",
    ]
    for source in result["sources"]:
        lines.append(f"## {source['source_name']}")
        lines.append("")
        lines.append(f"- frame_count: {source['frame_count']}")
        for head_name, metrics in source["outputs"].items():
            lines.append(f"- {head_name}: raw_mae={metrics['raw_mae']:.6f}, smooth_mae={metrics['smooth_mae']:.6f}, spike_count={metrics['spike_count']}, suggest_smoothing={metrics['suggest_smoothing']}, suggest_clamp={metrics['suggest_clamp']}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _write_frame_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "global_index",
        "source_name",
        "source_frame_index",
        "image_rel",
        "label_0",
        "label_1",
        "pred_0",
        "pred_1",
        "smooth_pred_0",
        "smooth_pred_1",
        "abs_error_0",
        "abs_error_1",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_output_dir(output_dir: str | Path | None, model_path: str | Path, split: str) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    path = Path(model_path)
    if path.suffix in {".pdparams", ".pdopt", ".pkl", ".pdmodel", ".json", ".pdiparams"}:
        base_name = path.stem
    else:
        base_name = path.name
    safe_name = base_name.replace(".", "_")
    return DEFAULT_OUTPUT_ROOT / f"{safe_name}_{split}_sequence_check"


def _normalize_static_prefix(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.suffix in {".pdmodel", ".json", ".pdiparams"}:
        return path.with_suffix("")
    if path.name == MODEL_BASENAME and (path.with_suffix(".pdmodel").is_file() or path.with_suffix(".json").is_file()):
        return path
    if path.is_dir():
        candidate = path / MODEL_BASENAME
        if candidate.with_suffix(".pdmodel").is_file() or candidate.with_suffix(".json").is_file():
            return candidate
    if path.with_suffix(".pdmodel").is_file() or path.with_suffix(".json").is_file():
        return path
    raise FileNotFoundError(f"Cannot resolve exported model prefix from: {model_path}")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if window <= 1 or values.size <= 1:
        return values.copy()
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _max_or_zero(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def main() -> int:
    args = build_parser().parse_args()
    result = run_sequence_analysis(
        model_format=args.model_format,
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        smooth_window=args.smooth_window,
        output_dir=args.output_dir,
        train_sources=_parse_names(args.train_sources),
        eval_sources=_parse_names(args.eval_sources),
    )
    print("nav_control_optional sequence check passed")
    for key, value in result.items():
        if key == "sources":
            print(f"{key}: {len(value)} source summaries")
            continue
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
