"""Create deterministic train/val manifests for 00_lane_seg Labelme JSON.

This script only writes split manifests. It does not copy images, convert
masks, run PaddleX, train, export, or validate a model.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


class SplitResult:
    def __init__(self, total: int, train_count: int, val_count: int, train_file: Path, val_file: Path) -> None:
        self.total = total
        self.train_count = train_count
        self.val_count = val_count
        self.train_file = train_file
        self.val_file = val_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="./datasets/00_lane_seg",
        help="00_lane_seg dataset root. Used to infer labelme_json and splits paths.",
    )
    parser.add_argument("--labelme-dir", default=None, help="Labelme JSON directory. Overrides --dataset-dir.")
    parser.add_argument("--output-dir", default=None, help="Split manifest output directory. Defaults to <dataset-dir>/splits.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio, exclusive range (0, 1).")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic split.")
    return parser


def split_labelme_jsons(
    labelme_dir: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.2,
    seed: int = 2026,
) -> SplitResult:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in the exclusive range (0, 1): {val_ratio}")

    labelme_path = Path(labelme_dir)
    if not labelme_path.is_dir():
        raise FileNotFoundError(f"Labelme JSON directory not found: {labelme_path}")

    json_names = sorted(path.name for path in labelme_path.glob("*.json") if path.is_file())
    if not json_names:
        raise ValueError(f"No Labelme JSON files found: {labelme_path}")

    shuffled = list(json_names)
    random.Random(seed).shuffle(shuffled)
    val_count = _val_count(len(shuffled), val_ratio)
    val_names = sorted(shuffled[:val_count])
    train_names = sorted(shuffled[val_count:])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_file = output_path / "train.txt"
    val_file = output_path / "val.txt"
    _write_lines(train_file, train_names)
    _write_lines(val_file, val_names)
    return SplitResult(
        total=len(json_names),
        train_count=len(train_names),
        val_count=len(val_names),
        train_file=train_file,
        val_file=val_file,
    )


def _val_count(total: int, val_ratio: float) -> int:
    if total == 1:
        return 0
    count = round(total * val_ratio)
    return min(max(int(count), 1), total - 1)


def _write_lines(path: Path, lines: list[str]) -> None:
    content = "\n".join(lines)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    labelme_dir = Path(args.labelme_dir) if args.labelme_dir else dataset_dir / "labelme_json"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "splits"
    try:
        result = split_labelme_jsons(labelme_dir, output_dir, val_ratio=args.val_ratio, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"split_dataset failed: {exc}")
        return 1
    print(f"split {result.total} Labelme JSON files")
    print(f"train: {result.train_count} -> {result.train_file}")
    print(f"val: {result.val_count} -> {result.val_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
