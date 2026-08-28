# 功能：测试手柄巡航数据过滤和连续块划分。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "prepare_gamepad_nav_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_gamepad_nav_dataset", SCRIPT_PATH)
prepare_gamepad_nav_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare_gamepad_nav_dataset
SPEC.loader.exec_module(prepare_gamepad_nav_dataset)


class PrepareGamepadNavDatasetTest(unittest.TestCase):
    def test_filters_invalid_records_and_splits_consecutive_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            images = source / "images"
            images.mkdir(parents=True)
            records = []
            for index in range(11):
                image_name = f"{index:04d}.jpg"
                Image.new("RGB", (16, 12), (index, 20, 30)).save(images / image_name)
                records.append(
                    {
                        "img_path": f"images/{image_name}",
                        "deviation": index / 10,
                        "valid_for_training": index != 0,
                    }
                )
            (source / "data.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            result = prepare_gamepad_nav_dataset.prepare_dataset(
                source,
                root / "prepared",
                block_size=2,
                val_every=3,
            )

            self.assertEqual(result["input_records"], 11)
            self.assertEqual(result["valid_records"], 10)
            self.assertEqual(result["invalid_records"], 1)
            self.assertEqual(result["train_records"], 8)
            self.assertEqual(result["val_records"], 2)
            self.assertTrue((root / "prepared" / "train" / "data.jsonl").is_file())
            self.assertTrue((root / "prepared" / "val" / "images" / "000000.jpg").is_file())

    def test_session_split_filters_zero_speed_and_normalizes_max_speed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            images = source / "images"
            images.mkdir(parents=True)
            records = []
            record_index = 0
            for session, count in ((1, 2), (2, 5), (3, 5), (4, 5)):
                for session_index in range(count):
                    image_name = f"{record_index:04d}.jpg"
                    Image.new("RGB", (16, 12), (record_index, 20, 30)).save(images / image_name)
                    speed = 0.0 if session_index == 0 else session_index / 10
                    records.append(
                        {
                            "img_path": f"images/{image_name}",
                            "deviation": (session_index - 2) / 10,
                            "command_speed": speed,
                            "gamepad_speed_target": speed,
                            "frame_receive_monotonic_ns": record_index * 30_000_000,
                            "session": session,
                            "valid_for_training": True,
                        }
                    )
                    record_index += 1
            (source / "data.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            result = prepare_gamepad_nav_dataset.prepare_dataset(
                source,
                root / "prepared",
                split_mode="sessions",
                val_sessions=(4,),
                drop_sessions=(1,),
                drop_zero_speed=True,
                drop_tail_frames=2,
                max_speed=0.6,
            )

            self.assertEqual(result["train_sessions"], [2, 3])
            self.assertEqual(result["val_sessions"], [4])
            self.assertEqual(result["train_records"], 4)
            self.assertEqual(result["val_records"], 2)
            self.assertEqual(result["speed_after"], {"min": 0.6, "max": 0.6})
            self.assertEqual(result["removed_by_reason"]["dropped_session"], 2)
            self.assertEqual(result["removed_by_reason"]["zero_speed"], 3)
            self.assertEqual(result["removed_by_reason"]["tail_trim"], 6)

            output_record = json.loads(
                (root / "prepared" / "train" / "data.jsonl").read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(output_record["command_speed"], 0.6)
            self.assertEqual(output_record["gamepad_speed_target"], 0.6)
            self.assertEqual(output_record["source_command_speed"], 0.1)
            self.assertEqual(output_record["source_gamepad_speed_target"], 0.1)

    def test_session_split_rejects_unknown_validation_sessions(self) -> None:
        records = [
            {"img_path": "images/0000.jpg", "deviation": 0.0, "session": 4},
            {"img_path": "images/0001.jpg", "deviation": 0.1, "session": 5},
        ]

        with self.assertRaisesRegex(ValueError, "validation sessions not found"):
            prepare_gamepad_nav_dataset.split_records_by_sessions(records, val_sessions=(9,))

    def test_admission_stats_do_not_measure_deviation_jumps_across_sessions(self) -> None:
        records = [
            {"deviation": 0.0, "session": 2},
            {"deviation": 0.1, "session": 2},
            {"deviation": -1.0, "session": 3},
            {"deviation": -0.9, "session": 3},
        ]

        stats = prepare_gamepad_nav_dataset._build_admission_stats(records)

        self.assertAlmostEqual(stats["deviation"]["adjacent_delta"]["max"], 0.1)


if __name__ == "__main__":
    unittest.main()
