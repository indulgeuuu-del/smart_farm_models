# 功能：运行 01_target_det 的默认 PP-YOLOE+_s 416 PaddleDetection 主训练入口。
from __future__ import annotations

import argparse
import sys
from pathlib import Path


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _target_det_recipe import (  # type: ignore
    DEFAULT_CONFIG_PATH,
    build_train_command,
    format_command,
    get_repo_root,
    resolve_paddledet_root,
    resolve_repo_path,
    run_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 01_target_det with the default PP-YOLOE+_s 416 PaddleDetection recipe.")
    parser.add_argument("--paddledet-root", default=None, help="PaddleDetection source root. Defaults to detected local source.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Target detection PaddleDetection config path.")
    parser.add_argument("--dataset-dir", required=True, help="Required 01_target_det COCO dataset root for this training run.")
    parser.add_argument("--output-dir", required=True, help="Required training save_dir for this run.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint directory or params path for resume.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional TrainReader.batch_size override.")
    parser.add_argument("--base-lr", type=float, default=None, help="Optional LearningRate.base_lr override.")
    parser.add_argument("--worker-num", type=int, default=None, help="Optional top-level worker_num override.")
    parser.add_argument("--use-vdl", action="store_true", help="Enable VisualDL logging.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training.")
    parser.add_argument(
        "--allow-non-augmented-dataset",
        action="store_true",
        help="Explicitly allow training from a non-augmented dataset. Use only for smoke tests or approved ablations.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command only.")
    return parser


def is_augmented_dataset(dataset_dir: str | Path) -> bool:
    path = Path(dataset_dir)
    if (path / "augmentation_meta.json").is_file():
        return True
    return "aug" in path.name.lower()


def main() -> int:
    args = build_parser().parse_args()
    repo_root = get_repo_root()
    paddledet_root = resolve_paddledet_root(repo_root, args.paddledet_root)
    config_path = resolve_repo_path(repo_root, args.config).resolve()
    dataset_dir = resolve_repo_path(repo_root, args.dataset_dir).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir).resolve()
    resume_path = resolve_repo_path(repo_root, args.resume).resolve() if args.resume else None

    if not config_path.is_file():
        raise FileNotFoundError(f"Missing target_det config: {config_path}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Missing target_det dataset dir: {dataset_dir}")
    if not args.allow_non_augmented_dataset and not is_augmented_dataset(dataset_dir):
        raise ValueError(
            "Target detection formal training must use a train-only offline augmented dataset. "
            "Run scripts/detection/augment_target_det_coco.py first, or pass "
            "--allow-non-augmented-dataset only for an approved smoke test/ablation."
        )

    command = build_train_command(
        python_executable=sys.executable,
        paddledet_root=paddledet_root,
        config_path=config_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        use_vdl=args.use_vdl,
        amp=not args.no_amp,
        resume_path=resume_path,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        worker_num=args.worker_num,
    )
    print(format_command(command))
    if args.dry_run:
        return 0
    return run_command(command, cwd=repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
