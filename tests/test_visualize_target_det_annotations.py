from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "visualize_target_det_annotations.py"
SPEC = importlib.util.spec_from_file_location("visualize_target_det_annotations", SCRIPT_PATH)
visualize_target_det_annotations = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = visualize_target_det_annotations
SPEC.loader.exec_module(visualize_target_det_annotations)


class VisualizeTargetDetAnnotationsTest(unittest.TestCase):
    def test_visualize_dataset_writes_images_index_and_suspicious_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            images_dir = dataset_dir / "images"
            annotations_dir = dataset_dir / "annotations"
            output_dir = root / "out"
            images_dir.mkdir(parents=True)
            annotations_dir.mkdir(parents=True)

            Image.new("RGB", (100, 80), "white").save(images_dir / "0001.jpg")
            Image.new("RGB", (100, 80), "white").save(images_dir / "0002.jpg")

            payload = {
                "images": [
                    {"id": 1, "file_name": "0001.jpg", "width": 100, "height": 80},
                    {"id": 2, "file_name": "0002.jpg", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 20], "area": 600},
                    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [10, 10, 30, 20], "area": 600},
                ],
                "categories": [
                    {"id": 1, "name": "water_l1"},
                    {"id": 2, "name": "water_l2"},
                ],
            }
            (annotations_dir / "instance_train.json").write_text(json.dumps(payload), encoding="utf-8")

            summary = visualize_target_det_annotations.visualize_dataset(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                splits=["train"],
            )

            self.assertEqual(summary["total_images"], 2)
            self.assertEqual(summary["total_annotations"], 2)
            self.assertTrue((output_dir / "visualized" / "train" / "0001.jpg").is_file())
            self.assertTrue((output_dir / "visualized" / "train" / "0002.jpg").is_file())
            self.assertTrue((output_dir / "index.html").is_file())
            suspicious_text = (output_dir / "suspicious_annotations.csv").read_text(encoding="utf-8")
            self.assertIn("duplicate_overlap_different_label", suspicious_text)
            self.assertIn("image_has_no_annotations", suspicious_text)


if __name__ == "__main__":
    unittest.main()
