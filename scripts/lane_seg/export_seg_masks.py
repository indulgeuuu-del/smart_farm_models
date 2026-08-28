"""Convert Labelme segmentation annotations to PaddleX segmentation masks.

This script is intentionally scoped to ``00_lane_seg``. It converts standard
Labelme ``polygon`` and ``rectangle`` shapes into single-channel PNG masks
using the fixed project classes:

0 background
1 road
2 border
3 cross_zone
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import zlib
from pathlib import Path


CLASSES = ("background", "road", "border", "cross_zone")
LABEL_TO_ID = {label: index for index, label in enumerate(CLASSES)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="./datasets/00_lane_seg/raw", help="Raw lane image directory.")
    parser.add_argument("--labelme-dir", default="./datasets/00_lane_seg/labelme_json", help="Labelme JSON directory.")
    parser.add_argument("--output-dir", default="./datasets/00_lane_seg/paddlex", help="PaddleX segmentation directory.")
    parser.add_argument("--split", choices=("train", "val"), default="train", help="Target PaddleX split.")
    return parser


def convert_dataset(raw_dir: str | Path, labelme_dir: str | Path, output_dir: str | Path, split: str = "train") -> int:
    raw_path = Path(raw_dir)
    labelme_path = Path(labelme_dir)
    output_path = Path(output_dir)
    _validate_input_dirs(raw_path, labelme_path)

    image_output = output_path / "images" / split
    annotation_output = output_path / "annotations" / split
    image_output.mkdir(parents=True, exist_ok=True)
    annotation_output.mkdir(parents=True, exist_ok=True)
    _write_class_names(output_path / "class_names.txt")

    converted = 0
    split_entries: list[str] = []
    for json_path in sorted(labelme_path.glob("*.json")):
        with json_path.open("r", encoding="utf-8-sig") as file:
            labelme_data = json.load(file)
        image_width = int(labelme_data["imageWidth"])
        image_height = int(labelme_data["imageHeight"])
        image_name = labelme_data.get("imagePath") or f"{json_path.stem}.jpg"
        source_image = raw_path / Path(image_name).name
        if not source_image.is_file():
            raise FileNotFoundError(f"Raw image not found for {json_path.name}: {source_image}")

        mask = [[0 for _ in range(image_width)] for _ in range(image_height)]
        for shape in labelme_data.get("shapes", []):
            label = shape.get("label")
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown label in {json_path.name}: {label}")
            points = _shape_points(shape)
            _fill_polygon(mask, points, LABEL_TO_ID[label])

        shutil.copy2(source_image, image_output / source_image.name)
        mask_name = f"{json_path.stem}.png"
        _write_grayscale_png(annotation_output / mask_name, mask, image_width, image_height)
        split_entries.append(f"images/{split}/{source_image.name} annotations/{split}/{mask_name}")
        converted += 1

    _write_split_file(output_path / f"{split}.txt", split_entries)
    return converted


def _validate_input_dirs(raw_dir: Path, labelme_dir: Path) -> None:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw image directory not found: {raw_dir}")
    if not labelme_dir.is_dir():
        raise FileNotFoundError(f"Labelme JSON directory not found: {labelme_dir}")


def _write_class_names(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def _write_split_file(path: Path, entries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(entries)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _shape_points(shape: dict) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y in shape.get("points", [])]
    if shape.get("shape_type") == "rectangle":
        if len(points) != 2:
            raise ValueError("Rectangle shape requires exactly 2 points.")
        (x1, y1), (x2, y2) = points
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        return [(left, top), (right, top), (right, bottom), (left, bottom)]
    if len(points) < 3:
        raise ValueError("Polygon shape requires at least 3 points.")
    return points


def _fill_polygon(mask: list[list[int]], points: list[tuple[float, float]], value: int) -> None:
    height = len(mask)
    width = len(mask[0]) if height else 0
    for y in range(height):
        for x in range(width):
            if _point_in_polygon(x + 0.5, y + 0.5, points):
                mask[y][x] = value


def _point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside = False
    previous_x, previous_y = points[-1]
    for current_x, current_y in points:
        crosses = (current_y > y) != (previous_y > y)
        if crosses:
            x_at_y = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
            if x < x_at_y:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _write_grayscale_png(path: Path, mask: list[list[int]], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_rows = bytearray()
    for row in mask:
        if len(row) != width:
            raise ValueError("Mask row width does not match imageWidth.")
        raw_rows.append(0)
        raw_rows.extend(row)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw_rows))))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    converted = convert_dataset(args.raw_dir, args.labelme_dir, args.output_dir, args.split)
    print(f"converted {converted} Labelme files to PaddleX segmentation masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
