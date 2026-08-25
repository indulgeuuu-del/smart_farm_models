# 功能：登记 08_nav_control_optional 正式外部数据源并生成顺序切分清单。
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from check_nav_control_data import check_nav_control_data  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    r"D:\A CarCarCar\26th BaiDu\Online Competition\2.cv\PaddleLane_v30\PaddleLane_v30\data_set\image_set_l"
)
DEFAULT_DATASET_DIR = Path("./datasets/08_nav_control_optional")
SPLIT_METHOD = "sequential_by_data_json_order"


@dataclass(frozen=True)
class PrepareResult:
    source_name: str
    total_records: int
    train_count: int
    val_count: int
    test_count: int
    sources_path: Path
    split_dir: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Official image_set style source directory.")
    parser.add_argument("--train-source-dir", default=None, help="Explicit train source directory.")
    parser.add_argument("--val-source-dir", default=None, help="Explicit val source directory.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="08_nav_control_optional dataset root.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Sequential train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Sequential val split ratio.")
    return parser


def prepare_nav_control_train_val_dataset(
    train_source_dir: str | Path,
    val_source_dir: str | Path,
    dataset_dir: str | Path,
) -> PrepareResult:
    train_source_path = Path(train_source_dir).resolve()
    val_source_path = Path(val_source_dir).resolve()
    dataset_root = Path(dataset_dir)
    train_source_name = train_source_path.name
    val_source_name = val_source_path.name

    if train_source_name == val_source_name:
        raise ValueError(f"Train and val source names must differ: {train_source_name}")
    train_check_result = check_nav_control_data([train_source_path])
    val_check_result = check_nav_control_data([val_source_path])
    errors = train_check_result.errors + val_check_result.errors
    if errors:
        raise ValueError("Official train/val sources failed static check:\n" + "\n".join(errors))

    train_records = _load_records(train_source_path)
    val_records = _load_records(val_source_path)
    if not train_records:
        raise ValueError("Train source must contain records")
    if not val_records:
        raise ValueError("Val source must contain records")

    train_items = [f"{train_source_name}/{_normalize_img_path(record['img_path'])}" for record in train_records]
    val_items = [f"{val_source_name}/{_normalize_img_path(record['img_path'])}" for record in val_records]

    records_dir = dataset_root / "records"
    split_dir = dataset_root / "paddle_custom" / "splits"
    records_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    sources_path = records_dir / "sources.json"
    _write_json(
        sources_path,
        {
            "official_source": f"train={train_source_name},val={val_source_name}",
            "official_splits": {"train": train_source_name, "val": val_source_name},
            "source_policy": _source_policy(dataset_root, [train_source_path, val_source_path]),
            "sources": [
                _source_manifest_entry(train_source_name, train_source_path, len(train_records), "official_train", train_check_result),
                _source_manifest_entry(val_source_name, val_source_path, len(val_records), "official_val", val_check_result),
            ],
        },
    )

    _write_lines(split_dir / "train.txt", train_items)
    _write_lines(split_dir / "val.txt", val_items)
    test_path = split_dir / "test.txt"
    if test_path.exists():
        test_path.unlink()
    _write_json(
        split_dir / "split_meta.json",
        {
            "split_method": "source_directory_train_val",
            "train_source_name": train_source_name,
            "train_source_path": str(train_source_path),
            "val_source_name": val_source_name,
            "val_source_path": str(val_source_path),
            "total_records": len(train_records) + len(val_records),
            "train_count": len(train_records),
            "val_count": len(val_records),
            "test_count": 0,
        },
    )

    return PrepareResult(
        source_name=f"train={train_source_name},val={val_source_name}",
        total_records=len(train_records) + len(val_records),
        train_count=len(train_records),
        val_count=len(val_records),
        test_count=0,
        sources_path=sources_path,
        split_dir=split_dir,
    )


def prepare_nav_control_dataset(
    source_dir: str | Path,
    dataset_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> PrepareResult:
    source_path = Path(source_dir).resolve()
    dataset_root = Path(dataset_dir)
    source_name = source_path.name

    check_result = check_nav_control_data([source_path])
    if check_result.errors:
        raise ValueError("Official source failed static check:\n" + "\n".join(check_result.errors))

    records = _load_records(source_path)
    if len(records) < 3:
        raise ValueError("Official source must contain at least 3 records to create train/val/test splits")

    train_count, val_count, test_count = _split_counts(len(records), train_ratio, val_ratio)
    identifiers = [f"{source_name}/{_normalize_img_path(record['img_path'])}" for record in records]
    train_items = identifiers[:train_count]
    val_items = identifiers[train_count : train_count + val_count]
    test_items = identifiers[train_count + val_count :]

    records_dir = dataset_root / "records"
    split_dir = dataset_root / "paddle_custom" / "splits"
    records_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    sources_path = records_dir / "sources.json"
    _write_json(
        sources_path,
        {
            "official_source": source_name,
            "source_policy": _source_policy(dataset_root, [source_path]),
            "sources": [
                _source_manifest_entry(source_name, source_path, len(records), "official", check_result)
            ],
        },
    )

    _write_lines(split_dir / "train.txt", train_items)
    _write_lines(split_dir / "val.txt", val_items)
    _write_lines(split_dir / "test.txt", test_items)
    _write_json(
        split_dir / "split_meta.json",
        {
            "source_name": source_name,
            "source_path": str(source_path),
            "split_method": SPLIT_METHOD,
            "total_records": len(records),
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
        },
    )

    return PrepareResult(
        source_name=source_name,
        total_records=len(records),
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        sources_path=sources_path,
        split_dir=split_dir,
    )


def _load_records(source_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((source_dir / "data.json").read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"data.json root must be a list: {source_dir / 'data.json'}")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"data.json record must be an object: {source_dir / 'data.json'}#{index}")
        records.append(item)
    return records


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("Split ratios must leave at least 1 record for train, val, and test")
    return train_count, val_count, test_count


def _normalize_img_path(value: str) -> str:
    return value.replace("\\", "/").strip()


def _source_manifest_entry(
    source_name: str,
    source_path: Path,
    records: int,
    status: str,
    check_result: Any,
) -> dict[str, Any]:
    return {
        "name": source_name,
        "path": str(source_path),
        "data_json": str(source_path / "data.json"),
        "records": records,
        "status": status,
        "state0_values": check_result.state0_values,
        "state_ranges": check_result.state_ranges,
    }


def _source_policy(dataset_root: Path, source_paths: list[Path]) -> str:
    raw_root = (dataset_root / "raw").resolve()
    for source_path in source_paths:
        try:
            source_path.resolve().relative_to(raw_root)
        except ValueError:
            return "external_read_only"
    return "workspace_copy"


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.train_source_dir or args.val_source_dir:
        if not args.train_source_dir or not args.val_source_dir:
            parser.error("--train-source-dir and --val-source-dir must be provided together")
        result = prepare_nav_control_train_val_dataset(args.train_source_dir, args.val_source_dir, args.dataset_dir)
    else:
        result = prepare_nav_control_dataset(args.source_dir, args.dataset_dir, args.train_ratio, args.val_ratio)
    print("nav_control_optional official dataset prepared")
    print(f"source: {result.source_name}")
    print(f"total_records: {result.total_records}")
    print(f"train/val/test: {result.train_count}/{result.val_count}/{result.test_count}")
    print(f"sources: {result.sources_path}")
    print(f"splits: {result.split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
