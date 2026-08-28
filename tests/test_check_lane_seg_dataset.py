from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lane_seg" / "check_lane_seg_dataset.py"
SPEC = importlib.util.spec_from_file_location("check_lane_seg_dataset", SCRIPT_PATH)
check_lane_seg_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_lane_seg_dataset)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_valid_dataset(root: Path) -> None:
    raw_dir = root / "raw"
    labelme_dir = root / "labelme_json"
    paddlex_dir = root / "paddlex"
    raw_dir.mkdir(parents=True)
    labelme_dir.mkdir(parents=True)
    for path in [
        paddlex_dir / "images" / "train",
        paddlex_dir / "images" / "val",
        paddlex_dir / "annotations" / "train",
        paddlex_dir / "annotations" / "val",
    ]:
        path.mkdir(parents=True)

    (raw_dir / "lane_001.png").write_bytes(b"image")
    write_json(
        labelme_dir / "lane_001.json",
        {
            "imagePath": "lane_001.png",
            "imageHeight": 10,
            "imageWidth": 12,
            "shapes": [{"label": "road", "shape_type": "polygon", "points": [[1, 1], [5, 1], [5, 5]]}],
        },
    )
    (paddlex_dir / "class_names.txt").write_text("background\nroad\nborder\ncross_zone\n", encoding="utf-8")
    (paddlex_dir / "images" / "train" / "lane_001.png").write_bytes(b"image")
    (paddlex_dir / "images" / "val" / "lane_001.png").write_bytes(b"image")
    (paddlex_dir / "annotations" / "train" / "lane_001.png").write_bytes(b"mask")
    (paddlex_dir / "annotations" / "val" / "lane_001.png").write_bytes(b"mask")
    (paddlex_dir / "train.txt").write_text("images/train/lane_001.png annotations/train/lane_001.png\n", encoding="utf-8")
    (paddlex_dir / "val.txt").write_text("images/val/lane_001.png annotations/val/lane_001.png\n", encoding="utf-8")


class CheckLaneSegDatasetTest(unittest.TestCase):
    def test_valid_lane_seg_dataset_has_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            make_valid_dataset(root)

            errors = check_lane_seg_dataset.check_dataset(root)

            self.assertEqual(errors, [])

    def test_reports_label_filename_and_paddlex_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            labelme_dir = root / "labelme_json"
            paddlex_dir = root / "paddlex"
            raw_dir.mkdir(parents=True)
            labelme_dir.mkdir(parents=True)
            paddlex_dir.mkdir(parents=True)
            (raw_dir / "bad name.png").write_bytes(b"image")
            write_json(
                labelme_dir / "bad name.json",
                {
                    "imagePath": "bad name.png",
                    "imageHeight": 10,
                    "imageWidth": 12,
                    "shapes": [{"label": "tree", "shape_type": "circle", "points": [[1, 1], [5, 5]]}],
                },
            )

            errors = check_lane_seg_dataset.check_dataset(root)

            joined = "\n".join(errors)
            self.assertIn("Unsafe filename", joined)
            self.assertIn("Unknown label", joined)
            self.assertIn("Unsupported shape_type", joined)
            self.assertIn("Missing PaddleX path", joined)


if __name__ == "__main__":
    unittest.main()
