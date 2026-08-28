from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from scripts.detection.monitor_target_det_training import parse_log, render_html


class MonitorTargetDetTrainingTest(unittest.TestCase):
    def test_parse_log_and_render_responsive_monitor(self) -> None:
        line = (
            "[07/23 00:27:42] ppdet.engine.callbacks INFO: Epoch: [0] [  50/1876] "
            "learning_rate: 0.000042 loss: 0.824391 loss_cls: 0.386098 "
            "loss_iou: 0.060501 loss_dfl: 0.563638 loss_l1: 0.090797 "
            "eta: 8:10:56 batch_cost: 0.2588 data_cost: 0.0950 "
            "ips: 61.8343 images/s"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "train.log"
            log_path.write_text(line + "\n", encoding="utf-8")
            metrics = parse_log(log_path, max_epochs=50)

        self.assertEqual(metrics["latest"]["step"], 50)
        self.assertAlmostEqual(metrics["latest"]["loss"], 0.824391)
        self.assertAlmostEqual(metrics["latest"]["ips"], 61.8343)

        page = render_html(metrics)
        self.assertIn('<link rel="icon" href="data:,">', page)
        self.assertIn("@media (max-width: 640px)", page)
        self.assertIn("grid-template-columns: minmax(0, 1fr)", page)

    def test_final_epoch_with_completed_eval_is_marked_complete(self) -> None:
        lines = [
            (
                "[07/23 10:40:47] ppdet.engine.callbacks INFO: Epoch: [49] [1850/1876] "
                "learning_rate: 0.000000 loss: 0.612722 loss_cls: 0.295080 "
                "loss_iou: 0.023000 loss_dfl: 0.515703 loss_l1: 0.037072 "
                "eta: 0:00:08 batch_cost: 0.4203 data_cost: 0.1586 "
                "ips: 38.0706 images/s"
            ),
            "[07/23 10:40:58] ppdet.engine.callbacks INFO: Eval iter: 0",
            " Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.947",
            "[07/23 10:41:23] ppdet.engine INFO: Best test bbox ap is 0.951.",
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "train.log"
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            metrics = parse_log(log_path, max_epochs=50)

        self.assertEqual(metrics["status"], "completed")
        self.assertEqual(metrics["latest"]["progress_percent"], 100.0)
        self.assertEqual(metrics["remaining_steps"], 0)

    def test_fatal_stderr_marks_training_as_failed(self) -> None:
        line = (
            "[07/31 23:16:22] ppdet.engine.callbacks INFO: Epoch: [0] [ 100/3070] "
            "learning_rate: 0.000007 loss: 4.616380 loss_cls: 3.344438 "
            "loss_iou: 0.280502 loss_dfl: 1.143943 loss_l1: 0.735959 "
            "eta: 1 day, 16:15:27 batch_cost: 0.5663 data_cost: 0.3791 "
            "ips: 28.2539 images/s"
        )
        self.assertIn(
            "error_log_path",
            inspect.signature(parse_log).parameters,
            "parse_log must accept a separate stderr log for fatal-error detection",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "train.stdout.log"
            error_log_path = Path(temp_dir) / "train.stderr.log"
            log_path.write_text(line + "\n", encoding="utf-8")
            error_log_path.write_text(
                "Traceback (most recent call last):\n"
                "RuntimeError: CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED\n",
                encoding="utf-8",
            )
            metrics = parse_log(log_path, max_epochs=80, error_log_path=error_log_path)

        self.assertEqual(metrics["status"], "failed")
        self.assertEqual(metrics["failure"]["source"], "stderr")
        self.assertIn("CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED", metrics["failure"]["message"])

    def test_failure_page_has_prominent_single_popup_alert(self) -> None:
        metrics = {
            "status": "failed",
            "failure": {
                "id": "stderr:2:RuntimeError",
                "source": "stderr",
                "line_number": 2,
                "message": "RuntimeError: CUDA out of memory",
            },
            "max_epochs": 80,
            "latest": None,
            "train_points": [],
            "evals": [],
            "warnings": [],
        }

        page = render_html(metrics)

        self.assertIn('id="failureBanner"', page)
        self.assertIn('role="alert"', page)
        self.assertIn("sessionStorage", page)
        self.assertIn("window.alert", page)
        self.assertIn("训练失败", page)


if __name__ == "__main__":
    unittest.main()
