from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "export_target_det_coco.py"
SPEC = importlib.util.spec_from_file_location("export_target_det_coco", SCRIPT_PATH)
export_target_det = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = export_target_det
SPEC.loader.exec_module(export_target_det)


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
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_dataset_all(root: Path) -> Path:
    dataset_dir = root / "Dataset_all"
    images_dir = dataset_dir / "Images"
    images_dir.mkdir(parents=True)
    for name in ("0001.jpg", "0002.jpg", "0003.jpg"):
        (images_dir / name).write_bytes(b"fake-jpg")

    categories = [{"id": index + 1, "name": name} for index, name in enumerate(TARGET_CLASSES)]
    write_json(
        dataset_dir / "train.json",
        {
            "images": [{"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 30, 40], "area": 1200, "iscrowd": 0}],
            "categories": categories,
        },
    )
    write_json(
        dataset_dir / "valid.json",
        {
            "images": [{"id": 2, "file_name": "0002.jpg", "width": 640, "height": 480}],
            "annotations": [{"id": 2, "image_id": 2, "category_id": 5, "bbox": [5, 6, 10, 20], "area": 200, "iscrowd": 0}],
            "categories": categories,
        },
    )
    write_json(
        dataset_dir / "test.json",
        {
            "images": [{"id": 3, "file_name": "0003.jpg", "width": 640, "height": 480}],
            "annotations": [],
            "categories": categories,
        },
    )
    (dataset_dir / "label_list.txt").write_text("\n".join(TARGET_CLASSES) + "\n", encoding="utf-8")
    (dataset_dir / "data.yml").write_text("num_classes: 15\nmetric: COCO\n", encoding="utf-8")
    return dataset_dir


class ExportTargetDetCocoTest(unittest.TestCase):
    def test_imports_dataset_all_into_standard_target_det_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_all_dir = make_dataset_all(root)
            dataset_dir = root / "datasets" / "01_target_det"

            result = export_target_det.import_dataset_all_to_target_det(
                dataset_all_dir=dataset_all_dir,
                dataset_dir=dataset_dir,
            )

            self.assertEqual(result.train_images, 1)
            self.assertEqual(result.val_images, 1)
            self.assertEqual(result.test_images, 1)
            self.assertEqual(result.classes, 15)
            self.assertTrue((dataset_dir / "raw" / "0001.jpg").is_file())
            self.assertTrue((dataset_dir / "annotations_coco" / "instances_train.json").is_file())
            self.assertTrue((dataset_dir / "annotations_coco" / "instances_val.json").is_file())
            self.assertTrue((dataset_dir / "annotations_coco" / "instances_test.json").is_file())
            self.assertTrue((dataset_dir / "annotations_coco" / "label_list.txt").is_file())
            self.assertTrue((dataset_dir / "annotations_coco" / "data.yml").is_file())
            self.assertTrue((dataset_dir / "paddlex" / "images" / "0001.jpg").is_file())
            self.assertTrue((dataset_dir / "paddlex" / "annotations" / "instance_train.json").is_file())
            self.assertTrue((dataset_dir / "paddlex" / "annotations" / "instance_val.json").is_file())
            self.assertTrue((dataset_dir / "paddlex" / "annotations" / "instance_test.json").is_file())
            class_names = (dataset_dir / "paddlex" / "class_names.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(class_names, TARGET_CLASSES)

    def test_rejects_category_mismatch_between_label_list_and_coco(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_all_dir = make_dataset_all(root)
            payload = json.loads((dataset_all_dir / "valid.json").read_text(encoding="utf-8"))
            payload["categories"][-1]["name"] = "wrong_label"
            write_json(dataset_all_dir / "valid.json", payload)

            with self.assertRaisesRegex(ValueError, "wrong_label"):
                export_target_det.import_dataset_all_to_target_det(
                    dataset_all_dir=dataset_all_dir,
                    dataset_dir=root / "datasets" / "01_target_det",
                )


if __name__ == "__main__":
    unittest.main()
