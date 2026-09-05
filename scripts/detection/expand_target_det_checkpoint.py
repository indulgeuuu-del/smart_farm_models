# 功能：把 PP-YOLOE 检测 checkpoint 的分类头从旧类别数安全扩展到新类别数。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import paddle


CLASSIFICATION_KEYS = tuple(
    f"yolo_head.pred_cls.{level}.{suffix}"
    for level in range(3)
    for suffix in ("bias", "weight")
)
DEFAULT_INITIALIZATION_CLASS_INDICES = tuple(range(18, 26))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source .pdparams checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Expanded .pdparams checkpoint.")
    parser.add_argument("--source-classes", type=int, default=26)
    parser.add_argument("--target-classes", type=int, default=27)
    parser.add_argument(
        "--initialization-class-indices",
        type=int,
        nargs="+",
        default=list(DEFAULT_INITIALIZATION_CLASS_INDICES),
        help="Zero-based source class rows averaged to initialize each appended class row.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output checkpoint.")
    return parser


def expand_state_dict(
    state_dict: Mapping[str, Any],
    *,
    source_classes: int,
    target_classes: int,
    initialization_class_indices: tuple[int, ...],
) -> tuple[dict[str, Any], list[str]]:
    if target_classes <= source_classes:
        raise ValueError("target_classes must be greater than source_classes")
    if not initialization_class_indices:
        raise ValueError("initialization_class_indices must not be empty")
    if min(initialization_class_indices) < 0 or max(initialization_class_indices) >= source_classes:
        raise ValueError("initialization_class_indices must refer to source class rows")

    missing = sorted(set(CLASSIFICATION_KEYS) - set(state_dict))
    if missing:
        raise ValueError("Missing PP-YOLOE classification tensors: " + ", ".join(missing))

    expanded = dict(state_dict)
    changed: list[str] = []
    appended_classes = target_classes - source_classes
    for key in CLASSIFICATION_KEYS:
        tensor = state_dict[key]
        if not isinstance(tensor, paddle.Tensor):
            raise TypeError(f"Checkpoint value is not a paddle.Tensor: {key}")
        if not tensor.shape or tensor.shape[0] != source_classes:
            raise ValueError(f"Unexpected source shape for {key}: {tuple(tensor.shape)}")
        seed_row = paddle.mean(tensor[list(initialization_class_indices)], axis=0, keepdim=True)
        appended_rows = paddle.tile(seed_row, repeat_times=[appended_classes] + [1] * (tensor.ndim - 1))
        expanded[key] = paddle.concat([tensor, appended_rows], axis=0)
        changed.append(key)
    return expanded, sorted(changed)


def expand_checkpoint(
    *,
    input_path: Path,
    output_path: Path,
    source_classes: int,
    target_classes: int,
    initialization_class_indices: tuple[int, ...],
    force: bool,
) -> dict[str, Any]:
    source = input_path.resolve()
    target = output_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {source}")
    if target.exists() and not force:
        raise FileExistsError(f"Output checkpoint already exists, pass --force: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    state_dict = paddle.load(str(source))
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping: {type(state_dict)!r}")
    expanded, changed = expand_state_dict(
        state_dict,
        source_classes=source_classes,
        target_classes=target_classes,
        initialization_class_indices=initialization_class_indices,
    )
    paddle.save(expanded, str(target))
    manifest = {
        "source_checkpoint": str(source),
        "output_checkpoint": str(target),
        "source_classes": source_classes,
        "target_classes": target_classes,
        "initialization_class_indices": list(initialization_class_indices),
        "tensor_count": len(expanded),
        "changed_tensors": changed,
    }
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    args = build_parser().parse_args()
    manifest = expand_checkpoint(
        input_path=args.input,
        output_path=args.output,
        source_classes=args.source_classes,
        target_classes=args.target_classes,
        initialization_class_indices=tuple(args.initialization_class_indices),
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
