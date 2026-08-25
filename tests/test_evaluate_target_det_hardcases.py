from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "evaluate_target_det_hardcases.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_target_det_hardcases", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvaluateTargetDetHardcasesTest(unittest.TestCase):
    def test_iou_xywh_returns_overlap_ratio(self) -> None:
        module = load_module()

        self.assertAlmostEqual(
            module.iou_xywh([0, 0, 10, 10], [5, 5, 10, 10]),
            25 / 175,
        )

    def test_build_manifest_selects_images_containing_requested_labels(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            (dataset_dir / "annotations").mkdir(parents=True)
            (dataset_dir / "images").mkdir()
            (dataset_dir / "images" / "water.jpg").write_bytes(b"fake")
            (dataset_dir / "images" / "ball.jpg").write_bytes(b"fake")
            (dataset_dir / "annotations" / "instance_val.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "water.jpg", "width": 100, "height": 80},
                            {"id": 2, "file_name": "ball.jpg", "width": 100, "height": 80},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                            {"id": 2, "image_id": 2, "category_id": 2, "bbox": [30, 30, 20, 20]},
                        ],
                        "categories": [
                            {"id": 1, "name": "water_l1"},
                            {"id": 2, "name": "ball_blue"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = module.build_manifest(dataset_dir, "val", ["water_l1"], None)

            self.assertEqual([item["image_name"] for item in manifest["images"]], ["water.jpg"])
            self.assertEqual(manifest["labels"], ["water_l1"])
            self.assertEqual(manifest["category_by_id"], {"1": "water_l1", "2": "ball_blue"})

    def test_evaluate_predictions_counts_class_confusion(self) -> None:
        module = load_module()
        manifest = {
            "images": [
                {
                    "image_name": "sample.jpg",
                    "annotations": [
                        {
                            "category_id": 1,
                            "label": "water_l1",
                            "bbox_xywh": [10, 10, 20, 20],
                        }
                    ],
                }
            ]
        }
        prediction = {
            "images": [
                {
                    "image_name": "sample.jpg",
                    "detections": [
                        {
                            "label": "water_l2",
                            "score": 0.92,
                            "bbox_xywh": [11, 11, 19, 19],
                        }
                    ],
                }
            ]
        }

        report = module.evaluate_predictions(manifest, prediction, model_name="candidate", iou_threshold=0.5, score_threshold=0.3)

        self.assertEqual(report["overall"]["gt_annotations"], 1)
        self.assertEqual(report["overall"]["correct"], 0)
        self.assertEqual(report["overall"]["confused"], 1)
        self.assertEqual(report["confusions"]["water_l1"]["water_l2"], 1)
        self.assertEqual(
            report["per_stress_augmentation"]["original"],
            {"gt_annotations": 1, "correct": 0, "confused": 1, "missed": 0},
        )

    def test_normalize_prediction_payload_converts_coco_bbox_results(self) -> None:
        module = load_module()
        manifest = {
            "category_by_id": {"1": "water_l1", "2": "water_l2"},
            "images": [
                {
                    "image_id": 7,
                    "image_name": "sample.jpg",
                    "annotations": [],
                }
            ],
        }
        coco_predictions = [
            {
                "image_id": 7,
                "category_id": 2,
                "bbox": [11, 12, 19, 18],
                "score": 0.92,
            }
        ]

        prediction = module.normalize_prediction_payload(coco_predictions, manifest)

        self.assertEqual(
            prediction,
            {
                "images": [
                    {
                        "image_name": "sample.jpg",
                        "detections": [
                            {
                                "label": "water_l2",
                                "score": 0.92,
                                "bbox_xywh": [11.0, 12.0, 19.0, 18.0],
                            }
                        ],
                    }
                ]
            },
        )

    def test_normalize_prediction_payload_prefers_coco_file_name_for_sequential_ids(self) -> None:
        module = load_module()
        manifest = {
            "category_by_id": {"1": "water_l1", "2": "water_l2"},
            "images": [
                {
                    "image_id": 42,
                    "image_name": "sample.jpg",
                    "annotations": [],
                }
            ],
        }
        onboard_bbox_predictions = [
            {
                "image_id": 0,
                "file_name": "sample.jpg",
                "category_id": 2,
                "bbox": [11, 12, 19, 18],
                "score": 0.92,
            }
        ]

        prediction = module.normalize_prediction_payload(onboard_bbox_predictions, manifest)

        self.assertEqual(prediction["images"][0]["image_name"], "sample.jpg")
        self.assertEqual(prediction["images"][0]["detections"][0]["label"], "water_l2")

    def test_normalize_prediction_payload_preserves_stress_variants_with_duplicate_image_ids(self) -> None:
        module = load_module()
        manifest = {
            "category_by_id": {"1": "name"},
            "images": [
                {"image_id": 42, "image_name": "sample.jpg", "annotations": []},
                {"image_id": 42, "image_name": "stress_image_zoomout_50_sample.jpg", "annotations": []},
            ],
        }
        predictions = [
            {"image_id": 0, "file_name": "sample.jpg", "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9},
            {
                "image_id": 1,
                "file_name": "stress_image_zoomout_50_sample.jpg",
                "category_id": 1,
                "bbox": [5, 6, 7, 8],
                "score": 0.8,
            },
        ]

        normalized = module.normalize_prediction_payload(predictions, manifest)

        self.assertEqual([item["image_name"] for item in normalized["images"]], ["sample.jpg", "stress_image_zoomout_50_sample.jpg"])
        self.assertEqual(normalized["images"][0]["detections"][0]["bbox_xywh"], [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(normalized["images"][1]["detections"][0]["bbox_xywh"], [5.0, 6.0, 7.0, 8.0])

    def test_add_object_stress_variants_preserves_bbox(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (40, 30), (10, 20, 30)).save(image_path)
            manifest = {
                "images": [
                    {
                        "image_name": "sample.jpg",
                        "image_path": str(image_path),
                        "annotations": [
                            {
                                "category_id": 1,
                                "label": "water_l1",
                                "bbox_xywh": [5, 6, 10, 8],
                            }
                        ],
                    }
                ]
            }

            stressed = module.add_object_stress_variants(manifest, root / "stress_images", ["object_rotate180"])

            self.assertEqual(stressed["image_count"], 2)
            self.assertTrue((root / "stress_images" / "stress_object_rotate180_sample.jpg").is_file())
            stress_item = stressed["images"][1]
            self.assertEqual(stress_item["image_name"], "stress_object_rotate180_sample.jpg")
            self.assertEqual(stress_item["annotations"][0]["bbox_xywh"], [5, 6, 10, 8])

    def test_add_image_stress_variants_changes_image_without_changing_bbox(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (40, 30), (120, 120, 120)).save(image_path)
            manifest = {
                "images": [
                    {
                        "image_name": "sample.jpg",
                        "image_path": str(image_path),
                        "annotations": [
                            {
                                "category_id": 1,
                                "label": "water_l1",
                                "bbox_xywh": [5, 6, 10, 8],
                            }
                        ],
                    }
                ]
            }

            stressed = module.add_image_stress_variants(manifest, root / "stress_images", ["image_dark"])

            output_path = root / "stress_images" / "stress_image_dark_sample.jpg"
            self.assertTrue(output_path.is_file())
            self.assertLess(Image.open(output_path).convert("L").getpixel((0, 0)), 100)
            stress_item = stressed["images"][1]
            self.assertEqual(stress_item["image_name"], "stress_image_dark_sample.jpg")
            self.assertEqual(stress_item["annotations"][0]["bbox_xywh"], [5, 6, 10, 8])

    def test_transform_image_lowres_preserves_canvas_size(self) -> None:
        module = load_module()
        image = Image.new("RGB", (80, 60), (120, 120, 120))

        transformed = module.transform_image(image, "image_lowres")

        self.assertEqual(transformed.size, image.size)

    def test_add_zoomout_stress_variants_updates_bbox(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sample.jpg"
            Image.new("RGB", (80, 60), (120, 120, 120)).save(image_path)
            manifest = {
                "images": [
                    {
                        "image_name": "sample.jpg",
                        "image_path": str(image_path),
                        "width": 80,
                        "height": 60,
                        "annotations": [
                            {
                                "category_id": 1,
                                "label": "name",
                                "bbox_xywh": [20, 10, 16, 12],
                            }
                        ],
                    }
                ]
            }

            stressed = module.add_image_stress_variants(
                manifest,
                root / "stress_images",
                ["image_zoomout_75", "image_zoomout_60", "image_zoomout_50"],
            )

            self.assertEqual(stressed["image_count"], 4)
            expected = {
                "image_zoomout_75": [25.0, 15.0, 12.0, 9.0],
                "image_zoomout_60": [28.0, 18.0, 9.6, 7.2],
                "image_zoomout_50": [30.0, 20.0, 8.0, 6.0],
            }
            for item in stressed["images"][1:]:
                augmentation = item["stress_augmentation"]
                self.assertEqual(item["annotations"][0]["bbox_xywh"], expected[augmentation])
                self.assertTrue((root / "stress_images" / item["image_name"]).is_file())


if __name__ == "__main__":
    unittest.main()
