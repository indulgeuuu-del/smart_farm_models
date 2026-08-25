# 功能：静态检查 08_nav_control_optional 的 image_set 风格巡航控制数据。
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_SPLITS = ("train", "val", "test")
DEFAULT_RAW_SOURCES = ("image_set1208", "image_set_l", "image_set_r")


@dataclass(frozen=True)
class CheckResult:
    total_records: int
    source_counts: dict[str, int]
    state0_values: list[float]
    state_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
    errors: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="./datasets/08_nav_control_optional",
        help="08_nav_control_optional root used to auto-discover records/*/data.json.",
    )
    parser.add_argument(
        "--record-dir",
        action="append",
        default=[],
        help="Image-set style source directory containing data.json and image files. Can be repeated.",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Optional directory containing train.txt, val.txt, and test.txt split manifests.",
    )
    parser.add_argument(
        "--required-splits",
        default="train,val,test",
        help="Comma-separated split names to require. Use train,val for train/val-only datasets.",
    )
    return parser


def discover_record_dirs(dataset_dir: str | Path) -> list[Path]:
    root = Path(dataset_dir)
    records_dir = root / "records"
    discovered: list[Path] = []
    if records_dir.is_dir():
        discovered.extend(path.parent for path in records_dir.glob("*/data.json"))
        sources_json = records_dir / "sources.json"
        if sources_json.is_file():
            discovered.extend(_load_source_manifest_dirs(sources_json))
    raw_dir = root / "raw"
    if raw_dir.is_dir():
        discovered.extend(path for name in DEFAULT_RAW_SOURCES if (path := raw_dir / name).joinpath("data.json").is_file())
    return _dedupe_sorted_paths(discovered)


def _load_source_manifest_dirs(sources_json: Path) -> list[Path]:
    payload = json.loads(sources_json.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sources.json {sources_json}: root must be an object")
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError(f"Invalid sources.json {sources_json}: sources must be a list")
    record_dirs: list[Path] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Invalid sources.json {sources_json}: sources[{index}] must be an object")
        path_value = source.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"Invalid sources.json {sources_json}: sources[{index}].path must be a non-empty string")
        record_dirs.append(Path(path_value))
    return record_dirs


def _dedupe_sorted_paths(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def check_nav_control_data(
    record_dirs: list[str | Path],
    split_dir: str | Path | None = None,
    required_splits: tuple[str, ...] = REQUIRED_SPLITS,
) -> CheckResult:
    errors: list[str] = []
    source_counts: dict[str, int] = {}
    identifiers: set[str] = set()
    state_columns: list[list[float]] = [[], [], []]

    source_names: list[str] = []
    for raw_dir in record_dirs:
        source_dir = Path(raw_dir)
        source_name = source_dir.name
        source_names.append(source_name)
        if source_names.count(source_name) > 1:
            errors.append(f"Duplicate source name: {source_name}")
            continue
        records = _load_records(source_dir, errors)
        if records is None:
            continue
        source_counts[source_name] = len(records)
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                errors.append(f"Invalid record {source_name}#{index}: not an object")
                continue
            normalized_img_path = _normalize_img_path(record.get("img_path"), source_name, index, errors)
            if normalized_img_path is not None:
                identifier = f"{source_name}/{normalized_img_path}"
                if identifier in identifiers:
                    errors.append(f"Duplicate record image: {identifier}")
                identifiers.add(identifier)
                if not (source_dir / normalized_img_path).is_file():
                    errors.append(f"Missing image {source_name}#{index}: {normalized_img_path}")
            state = _validate_state(record.get("state"), source_name, index, errors)
            if state is not None:
                for column_index, value in enumerate(state):
                    state_columns[column_index].append(value)

    if split_dir is not None:
        _check_split_dir(Path(split_dir), identifiers, errors, required_splits)

    return CheckResult(
        total_records=sum(source_counts.values()),
        source_counts=source_counts,
        state0_values=_unique_sorted_values(state_columns[0]),
        state_ranges=_state_ranges(state_columns),
        errors=errors,
    )


def _load_records(source_dir: Path, errors: list[str]) -> list[dict[str, Any]] | None:
    data_json = source_dir / "data.json"
    if not source_dir.is_dir():
        errors.append(f"Missing source directory: {source_dir}")
        return None
    if not data_json.is_file():
        errors.append(f"Missing data.json: {data_json}")
        return None
    try:
        payload = json.loads(data_json.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid data.json {data_json}: {exc}")
        return None
    if not isinstance(payload, list):
        errors.append(f"Invalid data.json {data_json}: root must be a list")
        return None
    return payload


def _normalize_img_path(value: object, source_name: str, index: int, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"Invalid img_path {source_name}#{index}: {value}")
        return None
    normalized = value.replace("\\", "/").strip()
    path = Path(normalized)
    if path.is_absolute() or ":" in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        errors.append(f"Unsafe img_path {source_name}#{index}: {value}")
        return None
    return normalized


def _validate_state(value: object, source_name: str, index: int, errors: list[str]) -> tuple[float, float, float] | None:
    if not isinstance(value, list) or len(value) != 3:
        errors.append(f"Invalid state {source_name}#{index}: expected 3 numeric values")
        return None
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            errors.append(f"Invalid state {source_name}#{index}: expected finite numeric values")
            return None
        numbers.append(float(item))
    return numbers[0], numbers[1], numbers[2]


def _check_split_dir(split_dir: Path, identifiers: set[str], errors: list[str], required_splits: tuple[str, ...]) -> None:
    split_entries: dict[str, set[str]] = {}
    for split in required_splits:
        split_path = split_dir / f"{split}.txt"
        if not split_path.is_file():
            errors.append(f"Missing split file: {split_path}")
            split_entries[split] = set()
            continue
        entries = _read_split_entries(split_path, split, identifiers, errors)
        split_entries[split] = entries

    for left_index, left_split in enumerate(required_splits):
        for right_split in required_splits[left_index + 1 :]:
            overlap = sorted(split_entries[left_split] & split_entries[right_split])
            if overlap:
                errors.append(f"Split overlap {left_split}/{right_split}: {', '.join(overlap[:5])}")

    covered = set().union(*split_entries.values()) if split_entries else set()
    missing = sorted(identifiers - covered)
    if missing:
        errors.append(f"Records missing from split files: {', '.join(missing[:5])}")


def _read_split_entries(split_path: Path, split: str, identifiers: set[str], errors: list[str]) -> set[str]:
    lines = [line.strip().replace("\\", "/") for line in split_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        errors.append(f"Empty split file: {split_path}")
        return set()
    entries: set[str] = set()
    for line_number, entry in enumerate(lines, start=1):
        if entry in entries:
            errors.append(f"Duplicate split entry {split_path}:{line_number}: {entry}")
        entries.add(entry)
        if entry.startswith("/") or ":" in entry or any(part in {"", ".", ".."} for part in entry.split("/")):
            errors.append(f"Unsafe split entry {split_path}:{line_number}: {entry}")
        elif entry not in identifiers:
            errors.append(f"Unknown split entry {split_path}:{line_number}: {entry}")
    return entries


def _unique_sorted_values(values: list[float]) -> list[float]:
    return sorted({round(value, 8) for value in values})


def _state_ranges(state_columns: list[list[float]]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    if any(not column for column in state_columns):
        return None
    return tuple((round(min(column), 8), round(max(column), 8)) for column in state_columns)  # type: ignore[return-value]


def _print_summary(result: CheckResult) -> None:
    print(f"total_records: {result.total_records}")
    for source_name, count in sorted(result.source_counts.items()):
        print(f"source {source_name}: {count}")
    print(f"state0_values: {result.state0_values}")
    if result.state_ranges is not None:
        print(f"state_ranges: {result.state_ranges}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    record_dirs = [Path(item) for item in args.record_dir] or discover_record_dirs(args.dataset_dir)
    if not record_dirs:
        print("nav_control_optional data static check failed")
        print("- No record directories provided and no records/*/data.json found")
        return 1
    required_splits = tuple(item.strip() for item in args.required_splits.split(",") if item.strip())
    result = check_nav_control_data(record_dirs, args.split_dir, required_splits=required_splits)
    if result.errors:
        print("nav_control_optional data static check failed")
        _print_summary(result)
        for error in result.errors:
            print(f"- {error}")
        return 1
    print("nav_control_optional data static check passed")
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
