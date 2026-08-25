# 功能：测试巡航模型分 session 鲁棒性评估汇总。
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "evaluate_nav_control_robustness.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nav_control_robustness", SCRIPT_PATH)
evaluate_nav_control_robustness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = evaluate_nav_control_robustness
SPEC.loader.exec_module(evaluate_nav_control_robustness)


class EvaluateNavControlRobustnessTest(unittest.TestCase):
    def test_illumination_stress_variants_are_deterministic_and_bounded(self) -> None:
        image = np.linspace(20.0, 220.0, 24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3)

        for variant in evaluate_nav_control_robustness.ILLUMINATION_VARIANTS:
            first = evaluate_nav_control_robustness.apply_illumination_variant(image, variant)
            second = evaluate_nav_control_robustness.apply_illumination_variant(image, variant)

            self.assertEqual(first.shape, image.shape)
            self.assertEqual(first.dtype, np.float32)
            self.assertGreaterEqual(float(first.min()), 0.0)
            self.assertLessEqual(float(first.max()), 255.0)
            np.testing.assert_allclose(first, second)

    def test_unknown_illumination_variant_is_rejected(self) -> None:
        image = np.full((24, 32, 3), 100.0, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "illumination variant"):
            evaluate_nav_control_robustness.apply_illumination_variant(image, "not-a-profile")

    def test_summary_keeps_continuous_deltas_inside_each_session(self) -> None:
        records = [
            {"img_path": "images/0000.jpg", "session": 2, "deviation": 0.0},
            {"img_path": "images/0001.jpg", "session": 2, "deviation": 0.4},
            {"img_path": "images/0002.jpg", "session": 2, "deviation": 0.8},
            {"img_path": "images/0003.jpg", "session": 3, "deviation": -0.4},
            {"img_path": "images/0004.jpg", "session": 3, "deviation": -0.8},
        ]
        predictions = np.array([0.1, 0.2, 0.5, 0.2, -0.2], dtype=np.float32)

        summary = evaluate_nav_control_robustness.summarize_ordered_predictions(
            records,
            predictions,
            spike_threshold=0.25,
        )

        self.assertEqual(summary["records"], 5)
        self.assertEqual(summary["sessions"], [2, 3])
        self.assertEqual(summary["continuous_frame_pairs"], 3)
        self.assertEqual(summary["spike_count"], 2)
        self.assertEqual(summary["per_session"]["2"]["records"], 3)
        self.assertEqual(summary["per_session"]["3"]["records"], 2)
        self.assertAlmostEqual(summary["per_session"]["2"]["prediction_delta_max"], 0.3)
        self.assertAlmostEqual(summary["per_session"]["3"]["prediction_delta_max"], 0.4)
        self.assertGreater(summary["robustness"]["direction_error_rate"], 0.0)

    def test_summary_rejects_prediction_count_mismatch(self) -> None:
        records = [{"img_path": "images/0000.jpg", "session": 2, "deviation": 0.0}]

        with self.assertRaisesRegex(ValueError, "prediction count"):
            evaluate_nav_control_robustness.summarize_ordered_predictions(records, np.array([], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
