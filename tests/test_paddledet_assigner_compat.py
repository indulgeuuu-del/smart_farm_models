from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLEDET_ROOT = Path(
    os.environ.get("SMART_FARM_PADDLEDET_ROOT", REPO_ROOT / "third_party" / "PaddleDetection")
).resolve()
RUN_INTEGRATION = os.environ.get("SMART_FARM_RUN_PADDLEDET_INTEGRATION") == "1"
PADDLEDET_READY = (PADDLEDET_ROOT / "ppdet" / "__init__.py").is_file()


@unittest.skipUnless(
    RUN_INTEGRATION and PADDLEDET_READY,
    "Set SMART_FARM_RUN_PADDLEDET_INTEGRATION=1 after installing the pinned PaddleDetection checkout.",
)
class PaddleDetAssignerCompatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if str(PADDLEDET_ROOT) not in sys.path:
            sys.path.insert(0, str(PADDLEDET_ROOT))
        import paddle
        from ppdet.modeling.assigners.atss_assigner import ATSSAssigner
        from ppdet.modeling.assigners.task_aligned_assigner import TaskAlignedAssigner

        cls.paddle = paddle
        cls.ATSSAssigner = ATSSAssigner
        cls.TaskAlignedAssigner = TaskAlignedAssigner

    def test_atss_assigner_accepts_int32_gt_labels(self) -> None:
        paddle = self.paddle
        assigner = self.ATSSAssigner(topk=9, num_classes=15)
        anchor_bboxes = paddle.to_tensor(
            [
                [0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0], [20.0, 0.0, 30.0, 10.0],
                [0.0, 10.0, 10.0, 20.0], [10.0, 10.0, 20.0, 20.0], [20.0, 10.0, 30.0, 20.0],
                [0.0, 20.0, 10.0, 30.0], [10.0, 20.0, 20.0, 30.0], [20.0, 20.0, 30.0, 30.0],
            ],
            dtype="float32",
        )
        assigned_labels, assigned_bboxes, assigned_scores = assigner(
            anchor_bboxes,
            [9],
            paddle.to_tensor([[[1]]], dtype="int32"),
            paddle.to_tensor([[[5.0, 5.0, 15.0, 15.0]]], dtype="float32"),
            paddle.to_tensor([[[1.0]]], dtype="float32"),
            bg_index=15,
        )
        self.assertEqual(list(assigned_labels.shape), [1, 9])
        self.assertEqual(list(assigned_bboxes.shape), [1, 9, 4])
        self.assertEqual(list(assigned_scores.shape), [1, 9, 15])

    def test_task_aligned_assigner_accepts_int32_gt_labels(self) -> None:
        paddle = self.paddle
        assigner = self.TaskAlignedAssigner(topk=3, alpha=1.0, beta=6.0)
        pred_bboxes = paddle.to_tensor(
            [[
                [0.0, 0.0, 10.0, 10.0], [10.0, 0.0, 20.0, 10.0], [20.0, 0.0, 30.0, 10.0],
                [0.0, 10.0, 10.0, 20.0], [10.0, 10.0, 20.0, 20.0], [20.0, 10.0, 30.0, 20.0],
                [0.0, 20.0, 10.0, 30.0], [10.0, 20.0, 20.0, 30.0], [20.0, 20.0, 30.0, 30.0],
            ]],
            dtype="float32",
        )
        anchor_points = paddle.to_tensor(
            [[5.0, 5.0], [15.0, 5.0], [25.0, 5.0], [5.0, 15.0], [15.0, 15.0], [25.0, 15.0], [5.0, 25.0], [15.0, 25.0], [25.0, 25.0]],
            dtype="float32",
        )
        assigned_labels, assigned_bboxes, assigned_scores = assigner(
            paddle.full([1, 9, 15], 0.5, dtype="float32"),
            pred_bboxes,
            anchor_points,
            [9],
            paddle.to_tensor([[[1]]], dtype="int32"),
            paddle.to_tensor([[[5.0, 5.0, 15.0, 15.0]]], dtype="float32"),
            paddle.to_tensor([[[1.0]]], dtype="float32"),
            bg_index=15,
        )
        self.assertEqual(list(assigned_labels.shape), [1, 9])
        self.assertEqual(list(assigned_bboxes.shape), [1, 9, 4])
        self.assertEqual(list(assigned_scores.shape), [1, 9, 15])


if __name__ == "__main__":
    unittest.main()
