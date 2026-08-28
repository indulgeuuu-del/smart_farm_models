from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lane_seg" / "export_seg_masks.py"
SPEC = importlib.util.spec_from_file_location("export_seg_masks", SCRIPT_PATH)
export_seg_masks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_seg_masks)


def read_grayscale_png(path: Path) -> tuple[int, int, list[list[int]]]:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = height = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
            assert bit_depth == 8
            assert color_type == 0
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    assert width is not None
    assert height is not None
    raw = zlib.decompress(bytes(compressed))
    rows: list[list[int]] = []
    stride = width + 1
    for y in range(height):
        row = raw[y * stride : (y + 1) * stride]
        assert row[0] == 0
        rows.append(list(row[1:]))
    return width, height, rows


class ExportSegMasksTest(unittest.TestCase):
    def test_converts_labelme_shapes_to_paddlex_train_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            labelme_dir = root / "labelme_json"
            output_dir = root / "paddlex"
            raw_dir.mkdir()
            labelme_dir.mkdir()
            (raw_dir / "lane.jpg").write_bytes(b"fake-image")
            (labelme_dir / "lane.json").write_text(
                json.dumps(
                    {
                        "imagePath": "lane.jpg",
                        "imageHeight": 4,
                        "imageWidth": 5,
                        "shapes": [
                            {
                                "label": "road",
                                "shape_type": "polygon",
                                "points": [[1, 1], [3, 1], [3, 2], [1, 2]],
                            },
                            {
                                "label": "border",
                                "shape_type": "rectangle",
                                "points": [[0, 0], [1, 1]],
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            converted = export_seg_masks.convert_dataset(raw_dir, labelme_dir, output_dir, split="train")

            self.assertEqual(converted, 1)
            self.assertEqual((output_dir / "images" / "train" / "lane.jpg").read_bytes(), b"fake-image")
            self.assertEqual(
                (output_dir / "train.txt").read_text(encoding="utf-8").splitlines(),
                ["images/train/lane.jpg annotations/train/lane.png"],
            )
            self.assertEqual(
                (output_dir / "class_names.txt").read_text(encoding="utf-8").splitlines(),
                ["background", "road", "border", "cross_zone"],
            )
            width, height, mask = read_grayscale_png(output_dir / "annotations" / "train" / "lane.png")
            self.assertEqual((width, height), (5, 4))
            self.assertEqual(mask[0][0], 2)
            self.assertEqual(mask[1][1], 1)
            self.assertEqual(mask[1][2], 1)
            self.assertEqual(mask[2][3], 0)
            self.assertEqual(mask[3][4], 0)

    def test_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            labelme_dir = root / "labelme_json"
            output_dir = root / "paddlex"
            raw_dir.mkdir()
            labelme_dir.mkdir()
            (raw_dir / "lane.jpg").write_bytes(b"fake-image")
            (labelme_dir / "lane.json").write_text(
                json.dumps(
                    {
                        "imagePath": "lane.jpg",
                        "imageHeight": 2,
                        "imageWidth": 2,
                        "shapes": [{"label": "tree", "points": [[0, 0], [1, 0], [1, 1]]}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unknown label"):
                export_seg_masks.convert_dataset(raw_dir, labelme_dir, output_dir, split="train")

    def test_accepts_utf8_bom_labelme_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            labelme_dir = root / "labelme_json"
            output_dir = root / "paddlex"
            raw_dir.mkdir()
            labelme_dir.mkdir()
            (raw_dir / "lane.jpg").write_bytes(b"fake-image")
            payload = json.dumps(
                {
                    "imagePath": "lane.jpg",
                    "imageHeight": 2,
                    "imageWidth": 2,
                    "shapes": [{"label": "road", "shape_type": "rectangle", "points": [[0, 0], [1, 1]]}],
                },
                ensure_ascii=False,
            )
            (labelme_dir / "lane.json").write_text(payload, encoding="utf-8-sig")

            converted = export_seg_masks.convert_dataset(raw_dir, labelme_dir, output_dir, split="train")

            self.assertEqual(converted, 1)
            self.assertTrue((output_dir / "annotations" / "train" / "lane.png").is_file())


if __name__ == "__main__":
    unittest.main()
