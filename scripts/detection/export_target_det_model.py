# 功能：导出 01_target_det 的默认 PP-YOLOE+_s 416 PaddleDetection 推理模型。
from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _target_det_recipe import (  # type: ignore
    DEFAULT_CONFIG_PATH,
    build_export_command,
    format_command,
    get_repo_root,
    resolve_paddledet_root,
    resolve_repo_path,
    run_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export 01_target_det default PP-YOLOE+_s 416 PaddleDetection inference model.")
    parser.add_argument("--paddledet-root", default=None, help="PaddleDetection source root. Defaults to detected local source.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Target detection PaddleDetection config path.")
    parser.add_argument("--weights", required=True, help="Required weights path passed to PaddleDetection export_model.")
    parser.add_argument("--output-dir", required=True, help="Required export output directory.")
    parser.add_argument("--dataset-dir", default=None, help="Optional COCO dataset root used to export the correct label_list.")
    parser.add_argument("--trt", action="store_true", help="Append trt=True for TensorRT export compatibility.")
    parser.add_argument("--exclude-nms", action="store_true", help="Append exclude_nms=True for benchmark-style export.")
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", dest="use_gpu", action="store_true", help="Force PaddleDetection export with use_gpu=true.")
    gpu_group.add_argument("--no-use-gpu", dest="use_gpu", action="store_false", help="Force PaddleDetection export with use_gpu=false.")
    parser.set_defaults(use_gpu=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command only.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = get_repo_root()
    paddledet_root = resolve_paddledet_root(repo_root, args.paddledet_root)
    config_path = resolve_repo_path(repo_root, args.config).resolve()
    weights_path = resolve_repo_path(repo_root, args.weights).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir).resolve()
    dataset_dir = resolve_repo_path(repo_root, args.dataset_dir).resolve() if args.dataset_dir else None

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing target_det config: {config_path}")
    if dataset_dir is not None and not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing target_det dataset dir: {dataset_dir}")

    command = build_export_command(
        python_executable=sys.executable,
        paddledet_root=paddledet_root,
        config_path=config_path,
        weights_path=weights_path,
        output_dir=output_dir,
        trt=args.trt,
        exclude_nms=args.exclude_nms,
        dataset_dir=dataset_dir,
        use_gpu=args.use_gpu,
    )
    print(format_command(command))
    if args.dry_run:
        return 0
    return run_command(command, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
