# 功能：离线检查 08_nav_control_optional 模型回载后的推理输出是否稳定为 [N, 2]。
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-format", default="dynamic", choices=("dynamic", "static"), help="dynamic loads .pdparams; static loads exported jit model.")
    parser.add_argument("--model-path", required=True, help="Path to .pdparams or exported model prefix/.pdmodel.")
    parser.add_argument("--dataset-dir", default="./datasets/08_nav_control_optional", help="08_nav_control_optional dataset root.")
    parser.add_argument("--train-sources", default=",".join(DEFAULT_TRAIN_SOURCES), help="Comma-separated raw image-set names for training split.")
    parser.add_argument("--eval-sources", default=",".join(DEFAULT_EVAL_SOURCES), help="Comma-separated raw image-set names for eval split.")
    parser.add_argument("--split", default="eval", choices=("train", "eval", "val"), help="Split to inspect.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for offline inference.")
    parser.add_argument("--device", default="gpu:0", help="Paddle device.")
    parser.add_argument("--limit-batches", type=int, default=None, help="Optional batch cap for smoke checks.")
    return parser


def run_dynamic_inference_check(
    params_path: str | Path,
    dataset_dir: str | Path,
    split: str = "eval",
    batch_size: int = 256,
    device: str = "gpu:0",
    limit_batches: int | None = None,
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES,
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> dict[str, Any]:
    model = load_model_from_params(params_path, device=device)
    return _run_inference_check(
        model,
        dataset_dir,
        split,
        batch_size,
        device,
        limit_batches,
        train_sources,
        eval_sources,
        model_format="dynamic",
        model_path=params_path,
    )


def run_static_inference_check(
    model_path: str | Path,
    dataset_dir: str | Path,
    split: str = "eval",
    batch_size: int = 256,
    device: str = "gpu:0",
    limit_batches: int | None = None,
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES,
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> dict[str, Any]:
    paddle.device.set_device(device)
    model_prefix = _normalize_static_prefix(model_path)
    model = paddle.jit.load(str(model_prefix))
    model.eval()
    return _run_inference_check(
        model,
        dataset_dir,
        split,
        batch_size,
        device,
        limit_batches,
        train_sources,
        eval_sources,
        model_format="static",
        model_path=model_prefix,
    )


def _run_inference_check(
    model: paddle.nn.Layer,
    dataset_dir: str | Path,
    split: str,
    batch_size: int,
    device: str,
    limit_batches: int | None,
    train_sources: tuple[str, ...],
    eval_sources: tuple[str, ...],
    model_format: str,
    model_path: str | Path,
) -> dict[str, Any]:
    paddle.device.set_device(device)
    samples = load_samples(dataset_dir, split, train_sources, eval_sources)
    dataset = PaddleLaneDataset(samples, training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)

    checked_batches = 0
    checked_samples = 0
    pred_widths: set[int] = set()
    last_pred_shape: list[int] | None = None
    with paddle.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            if limit_batches is not None and batch_index >= limit_batches:
                break
            preds = model(images)
            pred_shape = list(preds.shape)
            if len(pred_shape) != 2 or pred_shape[1] != 2:
                raise ValueError(f"Unexpected prediction shape at batch {batch_index}: {pred_shape}")
            pred_widths.add(pred_shape[1])
            last_pred_shape = pred_shape
            checked_batches += 1
            checked_samples += int(pred_shape[0])
    if checked_batches == 0 or last_pred_shape is None:
        raise RuntimeError("No inference batches were processed")
    return {
        "model_format": model_format,
        "model_path": str(model_path),
        "split": "eval" if split == "val" else split,
        "device": paddle.get_device(),
        "checked_batches": checked_batches,
        "checked_samples": checked_samples,
        "pred_widths": sorted(pred_widths),
        "last_pred_shape": last_pred_shape,
    }


def _normalize_static_prefix(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.suffix == ".pdmodel":
        return path.with_suffix("")
    if path.suffix == ".json":
        return path.with_suffix("")
    if path.suffix == ".pdiparams":
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


def main() -> int:
    args = build_parser().parse_args()
    if args.model_format == "dynamic":
        result = run_dynamic_inference_check(
            params_path=args.model_path,
            dataset_dir=args.dataset_dir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            limit_batches=args.limit_batches,
            train_sources=_parse_names(args.train_sources),
            eval_sources=_parse_names(args.eval_sources),
        )
    else:
        result = run_static_inference_check(
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            limit_batches=args.limit_batches,
            train_sources=_parse_names(args.train_sources),
            eval_sources=_parse_names(args.eval_sources),
        )
    print("nav_control_optional offline inference check passed")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
