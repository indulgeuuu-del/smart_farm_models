from __future__ import annotations

import sys
import unittest
from pathlib import Path

import paddle


REPO_ROOT = Path(__file__).resolve().parents[1]
PADDLEDET_ROOT = REPO_ROOT / "ai studio Fork" / "PaddleDetection" / "PaddleDetection"

if str(PADDLEDET_ROOT) not in sys.path:
    sys.path.insert(0, str(PADDLEDET_ROOT))

from ppdet.modeling.assigners.atss_assigner import ATSSAssigner
from ppdet.modeling.assigners.task_aligned_assigner import TaskAlignedAssigner


class PaddleDetAssignerCompatTest(unittest.TestCase):
    def test_atss_assigner_accepts_int32_gt_labels(self) -> None:
        assigner = ATSSAssigner(topk=9, num_classes=15)
        anchor_bboxes = paddle.to_tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 0.0, 20.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
                [0.0, 10.0, 10.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 10.0, 30.0, 20.0],
                [0.0, 20.0, 10.0, 30.0],
                [10.0, 20.0, 20.0, 30.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            dtype="float32",
        )
        num_anchors_list = [9]
        gt_labels = paddle.to_tensor([[[1]]], dtype="int32")
        gt_bboxes = paddle.to_tensor([[[5.0, 5.0, 15.0, 15.0]]], dtype="float32")
        pad_gt_mask = paddle.to_tensor([[[1.0]]], dtype="float32")

        assigned_labels, assigned_bboxes, assigned_scores = assigner(
            anchor_bboxes,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            pad_gt_mask,
            bg_index=15,
        )

        self.assertEqual(list(assigned_labels.shape), [1, 9])
        self.assertEqual(list(assigned_bboxes.shape), [1, 9, 4])
        self.assertEqual(list(assigned_scores.shape), [1, 9, 15])

    def test_task_aligned_assigner_accepts_int32_gt_labels(self) -> None:
        assigner = TaskAlignedAssigner(topk=3, alpha=1.0, beta=6.0)
        pred_scores = paddle.full([1, 9, 15], 0.5, dtype="float32")
        pred_bboxes = paddle.to_tensor(
            [[
                [0.0, 0.0, 10.0, 10.0],
                [10.0, 0.0, 20.0, 10.0],
                [20.0, 0.0, 30.0, 10.0],
                [0.0, 10.0, 10.0, 20.0],
                [10.0, 10.0, 20.0, 20.0],
                [20.0, 10.0, 30.0, 20.0],
                [0.0, 20.0, 10.0, 30.0],
                [10.0, 20.0, 20.0, 30.0],
                [20.0, 20.0, 30.0, 30.0],
            ]],
            dtype="float32",
        )
        anchor_points = paddle.to_tensor(
            [
                [5.0, 5.0],
                [15.0, 5.0],
                [25.0, 5.0],
                [5.0, 15.0],
                [15.0, 15.0],
                [25.0, 15.0],
                [5.0, 25.0],
                [15.0, 25.0],
                [25.0, 25.0],
            ],
            dtype="float32",
        )
        num_anchors_list = [9]
        gt_labels = paddle.to_tensor([[[1]]], dtype="int32")
        gt_bboxes = paddle.to_tensor([[[5.0, 5.0, 15.0, 15.0]]], dtype="float32")
        pad_gt_mask = paddle.to_tensor([[[1.0]]], dtype="float32")

        assigned_labels, assigned_bboxes, assigned_scores = assigner(
            pred_scores,
            pred_bboxes,
            anchor_points,
            num_anchors_list,
            gt_labels,
            gt_bboxes,
            pad_gt_mask,
            bg_index=15,
        )

        self.assertEqual(list(assigned_labels.shape), [1, 9])
        self.assertEqual(list(assigned_bboxes.shape), [1, 9, 4])
        self.assertEqual(list(assigned_scores.shape), [1, 9, 15])


if __name__ == "__main__":
    unittest.main()
