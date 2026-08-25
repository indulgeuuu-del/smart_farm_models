from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "onboard" / "run_target_det_onboard.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_target_det_onboard", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunTargetDetOnboardScriptTest(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing onboard entrypoint: {SCRIPT_PATH}")

    def test_default_model_dir_points_to_fullnms_export(self) -> None:
        module = load_module()
        self.assertIn("onboard_handoff", str(module.DEFAULT_MODEL_DIR))
        self.assertIn("jetson_models_20260604_hardfix_epoch9_candidate", str(module.DEFAULT_MODEL_DIR))
        self.assertIn("target_det", str(module.DEFAULT_MODEL_DIR))

    def test_default_tensorrt_opt_shape_matches_current_416_model(self) -> None:
        module = load_module()
        args = module.build_parser().parse_args(["--image-file", "demo.jpg"])

        self.assertEqual(args.trt_opt_shape, 416)

    def test_parser_exposes_tensorrt_benchmark_options(self) -> None:
        module = load_module()
        args = module.build_parser().parse_args(
            [
                "--image-file",
                "demo.jpg",
                "--run-mode",
                "trt_fp16",
                "--trt-opt-shape",
                "416",
                "--run-benchmark",
                "--repeats",
                "20",
                "--no-save-images",
            ]
        )

        self.assertEqual(args.run_mode, "trt_fp16")
        self.assertEqual(args.trt_opt_shape, 416)
        self.assertTrue(args.run_benchmark)
        self.assertEqual(args.repeats, 20)
        self.assertTrue(args.no_save_images)

    def test_build_detection_records_converts_boxes_to_repo_schema(self) -> None:
        module = load_module()
        image_list = ["demo.jpg"]
        merged = {
            "boxes_num": [2],
            "boxes": [
                [0, 0.95, 10.0, 20.0, 30.0, 40.0],
                [11, 0.88, 1.0, 2.0, 5.0, 8.0],
            ],
        }
        labels = ["animal"] + [f"class_{i}" for i in range(1, 11)] + ["name"] + [f"class_{i}" for i in range(12, 15)]
        records = module.build_detection_records(image_list, merged, labels, 0.3)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["image_name"], "demo.jpg")
        self.assertEqual(records[0]["detections"][0]["label"], "animal")
        self.assertEqual(records[0]["detections"][0]["category_id"], 1)
        self.assertEqual(records[0]["detections"][0]["bbox_xywh"], [10.0, 20.0, 20.0, 20.0])
        self.assertEqual(records[0]["detections"][1]["label"], "name")
        self.assertEqual(records[0]["detections"][1]["category_id"], 12)

    def test_build_timing_summary_reports_latency_and_fps(self) -> None:
        module = load_module()

        class FakeTimer:
            def report(self, average: bool = False):
                self.average = average
                return {
                    "img_num": 2,
                    "preprocess_time_s": 0.01,
                    "inference_time_s": 0.04,
                    "postprocess_time_s": 0.01,
                }

        summary = module.build_timing_summary(FakeTimer())

        self.assertEqual(summary["img_num"], 2)
        self.assertEqual(summary["latency_ms"], 60.0)
        self.assertAlmostEqual(summary["fps"], 16.667, places=3)


if __name__ == "__main__":
    unittest.main()
