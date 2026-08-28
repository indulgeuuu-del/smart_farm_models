from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "analyze_target_det_dataset.py"
SPEC = importlib.util.spec_from_file_location("analyze_target_det_dataset", SCRIPT_PATH)
analyze_target_det_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analyze_target_det_dataset
SPEC.loader.exec_module(analyze_target_det_dataset)


def write_coco(path: Path, annotations: list[dict[str, object]]) -> None:
    payload = {
        "images": [
            {"id": 1, "file_name": "0001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "0002.jpg", "width": 640, "height": 480},
        ],
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(analyze_target_det_dataset.TARGET_CLASSES)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class AnalyzeTargetDetDatasetTest(unittest.TestCase):
    def test_summarize_split_counts_classes_and_box_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            coco_path = Path(temp_dir) / "instance_train.json"
            write_coco(
                coco_path,
                [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "area": 400},
                    {"id": 2, "image_id": 2, "category_id": 2, "bbox": [0, 0, 50, 50], "area": 2500},
                    {"id": 3, "image_id": 2, "category_id": 2, "bbox": [0, 0, 120, 120], "area": 14400},
                ],
            )

            summary = analyze_target_det_dataset.summarize_split("train", coco_path, weak_threshold=2)

            self.assertEqual(summary.images, 2)
            self.assertEqual(summary.annotations, 3)
            self.assertEqual(summary.class_counts["animal"], 1)
            self.assertEqual(summary.class_counts["cylinder_3"], 2)
            self.assertEqual(summary.size_counts, {"small": 1, "medium": 1, "large": 1})
            self.assertIn("animal", summary.weak_classes)
            self.assertNotIn("cylinder_3", summary.weak_classes)

    def test_analyze_dataset_writes_diagnostic_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            output_dir = root / "out"
            annotations = [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50], "area": 2500},
            ]
            for split in ("train", "val", "test"):
                write_coco(dataset_dir / "annotations" / f"instance_{split}.json", annotations)

            result = analyze_target_det_dataset.analyze_dataset(dataset_dir, output_dir, weak_threshold=2)

            self.assertEqual(result["splits"]["train"]["annotations"], 1)
            self.assertTrue((output_dir / "class_distribution.json").is_file())
            self.assertTrue((output_dir / "box_size_distribution.json").is_file())
            self.assertTrue((output_dir / "weak_class_samples.json").is_file())
            self.assertTrue((output_dir / "diagnostics_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
