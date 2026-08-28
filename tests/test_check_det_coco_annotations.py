from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "check_det_coco_annotations.py"
SPEC = importlib.util.spec_from_file_location("check_det_coco_annotations", SCRIPT_PATH)
check_det_coco_annotations = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_det_coco_annotations
SPEC.loader.exec_module(check_det_coco_annotations)


TARGET_CLASSES = [
    "animal",
    "cylinder_3",
    "cylinder_1",
    "cylinder_2",
    "water",
    "water_l3",
    "water_l2",
    "lable_yellow",
    "storage",
    "ball_yellow",
    "order",
    "name",
    "cargo",
    "ball_blue",
    "lable_blue",
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-image")


def valid_target_coco() -> dict[str, object]:
    return {
        "images": [
            {"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "nested/0002.jpg", "width": 320, "height": 240},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80], "area": 8000, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 15, "bbox": [0, 0, 20, 30], "area": 600, "iscrowd": 0},
        ],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(TARGET_CLASSES)],
    }


class CheckDetCocoAnnotationsTest(unittest.TestCase):
    def test_valid_target_det_coco_has_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "datasets" / "01_target_det" / "raw"
            coco_json = root / "datasets" / "01_target_det" / "annotations_coco" / "instances_train.json"
            make_image(image_dir / "0001.jpg")
            make_image(image_dir / "nested" / "0002.jpg")
            write_json(coco_json, valid_target_coco())

            result = check_det_coco_annotations.check_coco_annotation(
                task="target_det",
                image_dir=image_dir,
                coco_json=coco_json,
                expected_classes=tuple(TARGET_CLASSES),
            )

            self.assertEqual(result.errors, [])
            self.assertEqual(result.images, 2)
            self.assertEqual(result.annotations, 2)
            self.assertEqual(result.categories, 15)

    def test_target_det_reports_category_image_reference_and_bbox_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "raw"
            coco_json = root / "annotations_coco" / "bad.json"
            make_image(image_dir / "0001.jpg")
            coco = valid_target_coco()
            coco["images"] = [
                {"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "missing.jpg", "width": 640, "height": 480},
            ]
            coco["categories"] = [{"id": index + 1, "name": name} for index, name in enumerate(TARGET_CLASSES[:-1])] + [
                {"id": 99, "name": "weed"}
            ]
            coco["annotations"] = [
                {"id": 1, "image_id": 1, "category_id": 999, "bbox": [1, 2, 3, 4]},
                {"id": 2, "image_id": 99, "category_id": 1, "bbox": [1, 2, 3, 4]},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [630, 10, 30, 20]},
                {"id": 4, "image_id": 1, "category_id": 1, "bbox": [0, 0, 0, 5]},
            ]
            write_json(coco_json, coco)

            result = check_det_coco_annotations.check_coco_annotation(
                task="target_det",
                image_dir=image_dir,
                coco_json=coco_json,
                expected_classes=tuple(TARGET_CLASSES),
            )
            joined = "\n".join(result.errors)

            self.assertIn("missing: lable_blue", joined)
            self.assertIn("unexpected: weed", joined)
            self.assertIn("Missing image", joined)
            self.assertIn("Unknown category_id", joined)
            self.assertIn("Unknown image_id", joined)
            self.assertIn("bbox exceeds image bounds", joined)
            self.assertIn("bbox width/height must be positive", joined)

    def test_check_all_only_discovers_new_target_det_mainline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "datasets" / "01_target_det" / "raw"
            make_image(image_dir / "0001.jpg")
            coco = valid_target_coco()
            coco["images"] = [{"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480}]
            coco["annotations"] = [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80]}]
            write_json(root / "datasets" / "01_target_det" / "annotations_coco" / "instances_train.json", coco)

            results = check_det_coco_annotations.check_all_datasets(root / "datasets")

            self.assertEqual([result.task for result in results], ["target_det"])
            self.assertTrue(all(not result.errors for result in results))

    def test_check_all_allows_custom_classes_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "datasets" / "01_target_det" / "raw"
            make_image(image_dir / "0001.jpg")
            coco = {
                "images": [{"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80]}],
                "categories": [{"id": 1, "name": "water_l3"}, {"id": 2, "name": "h_qing_jiao"}],
            }
            write_json(root / "datasets" / "01_target_det" / "annotations_coco" / "instances_train.json", coco)

            results = check_det_coco_annotations.check_all_datasets(root / "datasets", allow_custom_classes=True)

            self.assertEqual([result.task for result in results], ["target_det"])
            self.assertEqual(results[0].errors, [])


if __name__ == "__main__":
    unittest.main()
