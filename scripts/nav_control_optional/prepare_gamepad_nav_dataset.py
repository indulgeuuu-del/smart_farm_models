# 功能：把手柄巡航采集目录过滤并按连续时间块拆分为训练集和验证集。
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any


DEFAULT_BLOCK_SIZE = 150
DEFAULT_VAL_EVERY = 5
SPLIT_MODES = ("blocks", "sessions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Source directory containing images/ and data.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Output root containing train/ and val/.")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE, help="Consecutive records per split block.")
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY, help="Use every Nth block for validation.")
    parser.add_argument("--split-mode", choices=SPLIT_MODES, default="blocks", help="Split by consecutive blocks or sessions.")
    parser.add_argument("--val-sessions", default="", help="Comma-separated validation session ids for session mode.")
    parser.add_argument("--drop-sessions", default="", help="Comma-separated session ids to remove before splitting.")
    parser.add_argument("--drop-zero-speed", action="store_true", help="Remove records whose source command_speed is zero.")
    parser.add_argument("--drop-tail-frames", type=int, default=0, help="Remove this many trailing records from each session.")
    parser.add_argument("--max-speed", type=float, default=None, help="Normalize retained speed fields to this value.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser


def load_valid_records(source_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(source_dir)
    data_path = source / "data.jsonl"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing data.jsonl: {data_path}")

    records: list[dict[str, Any]] = []
    invalid_count = 0
    for line_number, line in enumerate(data_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL {data_path}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record {data_path}:{line_number}: expected object")
        if record.get("valid_for_training") is not True:
            invalid_count += 1
            continue
        image_rel = _normalize_image_rel(record.get("img_path"), data_path, line_number)
        image_path = source / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image {data_path}:{line_number}: {image_path}")
        deviation = record.get("deviation")
        if isinstance(deviation, bool) or not isinstance(deviation, (int, float)):
            raise ValueError(f"Invalid deviation {data_path}:{line_number}: {deviation}")
        normalized = dict(record)
        normalized["img_path"] = image_rel
        records.append(normalized)

    if not records:
        raise ValueError(f"No valid_for_training=true records found: {data_path}")
    return records, {"input_records": len(records) + invalid_count, "valid_records": len(records), "invalid_records": invalid_count}


def split_records_by_blocks(
    records: list[dict[str, Any]],
    block_size: int = DEFAULT_BLOCK_SIZE,
    val_every: int = DEFAULT_VAL_EVERY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if val_every < 2:
        raise ValueError("val_every must be at least 2")

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    for block_index, start in enumerate(range(0, len(records), block_size)):
        block = records[start : start + block_size]
        split = "val" if block_index % val_every == val_every - 1 else "train"
        (val if split == "val" else train).extend(block)
        blocks.append(
            {
                "block_index": block_index,
                "split": split,
                "start_record": start,
                "end_record_exclusive": start + len(block),
                "count": len(block),
                "first_img_path": block[0]["img_path"],
                "last_img_path": block[-1]["img_path"],
            }
        )
    if not train or not val:
        raise ValueError(f"Split produced an empty set: train={len(train)}, val={len(val)}")
    return train, val, blocks


def split_records_by_sessions(
    records: list[dict[str, Any]],
    val_sessions: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[int]]]:
    if not val_sessions:
        raise ValueError("val_sessions is required for session split mode")
    available_sessions = sorted({_record_session(record) for record in records})
    val_session_set = set(val_sessions)
    unknown_sessions = sorted(val_session_set - set(available_sessions))
    if unknown_sessions:
        raise ValueError(f"validation sessions not found: {unknown_sessions}")
    train_sessions = [session for session in available_sessions if session not in val_session_set]
    if not train_sessions:
        raise ValueError("Session split produced no training sessions")
    train = [record for record in records if _record_session(record) in train_sessions]
    val = [record for record in records if _record_session(record) in val_session_set]
    if not train or not val:
        raise ValueError(f"Session split produced an empty set: train={len(train)}, val={len(val)}")
    return train, val, {"train_sessions": train_sessions, "val_sessions": sorted(val_session_set)}


def prepare_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    block_size: int = DEFAULT_BLOCK_SIZE,
    val_every: int = DEFAULT_VAL_EVERY,
    split_mode: str = "blocks",
    val_sessions: tuple[int, ...] = (),
    drop_sessions: tuple[int, ...] = (),
    drop_zero_speed: bool = False,
    drop_tail_frames: int = 0,
    max_speed: float | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    records, filter_summary = load_valid_records(source)
    if split_mode == "blocks":
        train, val, blocks = split_records_by_blocks(records, block_size=block_size, val_every=val_every)
        split_metadata: dict[str, Any] = {
            "strategy": "consecutive_blocks",
            "block_size": int(block_size),
            "val_every": int(val_every),
            "blocks": blocks,
        }
    elif split_mode == "sessions":
        records, session_filter_metadata = _prepare_session_records(
            records,
            drop_sessions=drop_sessions,
            drop_zero_speed=drop_zero_speed,
            drop_tail_frames=drop_tail_frames,
            max_speed=max_speed,
        )
        train, val, session_split_metadata = split_records_by_sessions(records, val_sessions=val_sessions)
        split_metadata = {
            "strategy": "sessions",
            **session_filter_metadata,
            **session_split_metadata,
            "admission_stats": _build_admission_stats(records),
        }
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    output.mkdir(parents=True)
    _write_split(source, output / "train", train)
    _write_split(source, output / "val", val)

    metadata = {
        "source_dir": str(source),
        "output_dir": str(output),
        **filter_summary,
        **split_metadata,
        "train_records": len(train),
        "val_records": len(val),
        "note": "Only valid_for_training=true records are included; source dataset is unchanged.",
    }
    (output / "split_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    source_meta = source / "session_meta.json"
    if source_meta.is_file():
        shutil.copy2(source_meta, output / "source_session_meta.json")
    return metadata


def _prepare_session_records(
    records: list[dict[str, Any]],
    drop_sessions: tuple[int, ...],
    drop_zero_speed: bool,
    drop_tail_frames: int,
    max_speed: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if drop_tail_frames < 0:
        raise ValueError("drop_tail_frames must be non-negative")
    if max_speed is not None and (not math.isfinite(max_speed) or max_speed <= 0):
        raise ValueError("max_speed must be a positive finite value")

    removed_by_reason = {"dropped_session": 0, "zero_speed": 0, "tail_trim": 0}
    drop_session_set = set(drop_sessions)
    after_session_drop: list[dict[str, Any]] = []
    for record in records:
        session = _record_session(record)
        if session in drop_session_set:
            removed_by_reason["dropped_session"] += 1
            continue
        after_session_drop.append(dict(record))

    speed_before_values = [_record_speed(record, "command_speed") for record in after_session_drop]
    after_zero_speed: list[dict[str, Any]] = []
    for record in after_session_drop:
        if drop_zero_speed and abs(_record_speed(record, "command_speed")) < 1e-12:
            removed_by_reason["zero_speed"] += 1
            continue
        after_zero_speed.append(record)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in after_zero_speed:
        grouped.setdefault(_record_session(record), []).append(record)
    trimmed: list[dict[str, Any]] = []
    for session in sorted(grouped):
        session_records = grouped[session]
        if drop_tail_frames >= len(session_records):
            raise ValueError(
                f"drop_tail_frames removes entire session {session}: "
                f"records={len(session_records)} drop={drop_tail_frames}"
            )
        keep_count = len(session_records) - drop_tail_frames
        trimmed.extend(session_records[:keep_count])
        removed_by_reason["tail_trim"] += drop_tail_frames

    if not trimmed:
        raise ValueError("Session filtering removed every record")
    if max_speed is not None:
        for record in trimmed:
            if "command_speed" in record:
                record["source_command_speed"] = _record_speed(record, "command_speed")
                record["command_speed"] = float(max_speed)
            if "gamepad_speed_target" in record:
                record["source_gamepad_speed_target"] = _record_speed(record, "gamepad_speed_target")
                record["gamepad_speed_target"] = float(max_speed)
            state = record.get("state")
            if isinstance(state, list) and len(state) == 3:
                record["source_state0"] = float(state[0])
                record["state"] = [float(max_speed), state[1], state[2]]

    speed_after_values = [_record_speed(record, "command_speed") for record in trimmed]
    return trimmed, {
        "drop_sessions": sorted(drop_session_set),
        "drop_zero_speed": bool(drop_zero_speed),
        "drop_tail_frames": int(drop_tail_frames),
        "max_speed": max_speed,
        "removed_by_reason": removed_by_reason,
        "speed_before": _range_summary(speed_before_values),
        "speed_after": _range_summary(speed_after_values),
    }


def _record_session(record: dict[str, Any]) -> int:
    value = record.get("session")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid session value: {value}")
    session = int(value)
    if float(value) != float(session):
        raise ValueError(f"Session must be an integer: {value}")
    return session


def _record_speed(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid {field} value: {value}")
    return float(value)


def _range_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty numeric sequence")
    return {"min": min(values), "max": max(values)}


def _build_admission_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    deviations = [float(record["deviation"]) for record in records]
    frame_intervals_ms = [
        (float(right["frame_receive_monotonic_ns"]) - float(left["frame_receive_monotonic_ns"])) / 1_000_000.0
        for left, right in zip(records, records[1:])
        if _record_session(left) == _record_session(right)
        and "frame_receive_monotonic_ns" in left
        and "frame_receive_monotonic_ns" in right
    ]
    deviation_deltas = [
        abs(float(right["deviation"]) - float(left["deviation"]))
        for left, right in zip(records, records[1:])
        if _record_session(left) == _record_session(right)
    ]
    turn_runs = _turn_run_lengths(records, threshold=0.3)
    return {
        "records": len(records),
        "frame_interval_ms": _distribution_summary(frame_intervals_ms),
        "deviation": {
            "min": min(deviations),
            "max": max(deviations),
            "zero_ratio": sum(abs(value) < 1e-12 for value in deviations) / len(deviations),
            "adjacent_delta": _distribution_summary(deviation_deltas),
            "abs_ge_ratio": {
                str(threshold): sum(abs(value) >= threshold for value in deviations) / len(deviations)
                for threshold in (0.3, 0.5, 0.8)
            },
        },
        "turn_runs_abs_ge_0.3": _distribution_summary([float(value) for value in turn_runs]),
    }


def _turn_run_lengths(records: list[dict[str, Any]], threshold: float) -> list[int]:
    lengths: list[int] = []
    current_length = 0
    previous_session: int | None = None
    for record in records:
        session = _record_session(record)
        if previous_session is not None and session != previous_session and current_length:
            lengths.append(current_length)
            current_length = 0
        if abs(float(record["deviation"])) >= threshold:
            current_length += 1
        elif current_length:
            lengths.append(current_length)
            current_length = 0
        previous_session = session
    if current_length:
        lengths.append(current_length)
    return lengths


def _distribution_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    return ordered[math.floor((len(ordered) - 1) * fraction)]


def _write_split(source: Path, split_dir: Path, records: list[dict[str, Any]]) -> None:
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True)
    output_records: list[dict[str, Any]] = []
    for output_index, record in enumerate(records):
        source_image = source / str(record["img_path"])
        suffix = source_image.suffix.lower() or ".jpg"
        image_name = f"{output_index:06d}{suffix}"
        shutil.copy2(source_image, image_dir / image_name)
        output_record = dict(record)
        output_record["source_img_path"] = str(record["img_path"])
        output_record["img_path"] = f"images/{image_name}"
        output_records.append(output_record)
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in output_records]
    (split_dir / "data.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_image_rel(value: object, data_path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid img_path {data_path}:{line_number}: {value}")
    normalized = value.replace("\\", "/").strip()
    path = Path(normalized)
    if path.is_absolute() or ":" in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"Unsafe img_path {data_path}:{line_number}: {value}")
    return normalized


def main() -> int:
    args = build_parser().parse_args()
    metadata = prepare_dataset(
        args.source_dir,
        args.output_dir,
        block_size=args.block_size,
        val_every=args.val_every,
        split_mode=args.split_mode,
        val_sessions=_parse_int_tuple(args.val_sessions, "val_sessions"),
        drop_sessions=_parse_int_tuple(args.drop_sessions, "drop_sessions"),
        drop_zero_speed=args.drop_zero_speed,
        drop_tail_frames=args.drop_tail_frames,
        max_speed=args.max_speed,
        overwrite=args.overwrite,
    )
    print("gamepad nav dataset preparation passed")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


def _parse_int_tuple(value: str, field: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value}") from exc
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate {field}: {value}")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
