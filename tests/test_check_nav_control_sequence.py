# 功能：测试 08_nav_control_optional 连续帧离线推理曲线检查脚本。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import paddle
from PIL import Image


TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "train_nav_control_reg.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_nav_control_reg", TRAIN_SCRIPT_PATH)
train_nav_control_reg = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC.loader is not None
sys.modules[TRAIN_SPEC.name] = train_nav_control_reg
TRAIN_SPEC.loader.exec_module(train_nav_control_reg)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "check_nav_control_sequence.py"
SPEC = importlib.util.spec_from_file_location("check_nav_control_sequence", SCRIPT_PATH)
check_nav_control_sequence = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_nav_control_sequence
SPEC.loader.exec_module(check_nav_control_sequence)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_image_set(root: Path, name: str, count: int) -> Path:
    source_dir = root / "raw" / name
    source_dir.mkdir(parents=True)
    records = []
    for index in range(count):
        image_name = f"{index:04d}.jpg"
        Image.new("RGB", (32, 24), (index * 30 % 255, 20, 100)).save(source_dir / image_name)
        records.append({"img_path": image_name, "state": [0.15, index / 10, -index / 20]})
    write_json(source_dir / "data.json", records)
    return source_dir


def make_dataset(root: Path) -> Path:
    dataset_dir = root / "datasets" / "08_nav_control_optional"
    make_image_set(dataset_dir, "image_set_r", 2)
    make_image_set(dataset_dir, "image_set_l", 2)
    make_image_set(dataset_dir, "image_set1208", 3)
    return dataset_dir


class CheckNavControlSequenceTest(unittest.TestCase):
    def test_default_dataset_dir_uses_final_optional_numbering(self) -> None:
        args = check_nav_control_sequence.build_parser().parse_args(["--model-path", "dummy.pdparams"])

        self.assertEqual(Path(args.dataset_dir), Path("datasets/08_nav_control_optional"))

    def test_sequence_metrics_flag_smoothing_and_clamp_when_spike_is_large(self) -> None:
        labels = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
        preds = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.75],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )

        metrics = check_nav_control_sequence.summarize_sequence_metrics(
            source_name="image_set1208",
            labels=labels,
            preds=preds,
            smooth_window=3,
        )

        output_1 = metrics["outputs"]["output_1"]
        self.assertGreaterEqual(output_1["spike_count"], 1)
        self.assertTrue(output_1["suggest_smoothing"])
        self.assertTrue(output_1["suggest_clamp"])

    def test_sequence_metrics_keep_smooth_series_unflagged(self) -> None:
        labels = np.array(
            [
                [0.00, -0.10],
                [0.02, -0.08],
                [0.04, -0.06],
                [0.06, -0.04],
                [0.08, -0.02],
            ],
            dtype=np.float32,
        )
        preds = labels.copy()

        metrics = check_nav_control_sequence.summarize_sequence_metrics(
            source_name="image_set1208",
            labels=labels,
            preds=preds,
            smooth_window=3,
        )

        output_0 = metrics["outputs"]["output_0"]
        output_1 = metrics["outputs"]["output_1"]
        self.assertEqual(output_0["spike_count"], 0)
        self.assertFalse(output_0["suggest_smoothing"])
        self.assertFalse(output_0["suggest_clamp"])
        self.assertEqual(output_1["spike_count"], 0)
        self.assertFalse(output_1["suggest_smoothing"])
        self.assertFalse(output_1["suggest_clamp"])

    def test_sequence_analysis_honors_custom_eval_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            make_image_set(dataset_dir, "official_train", 4)
            make_image_set(dataset_dir, "official_eval", 2)
            params_dir = root / "dynamic"
            params_dir.mkdir()
            params_path = params_dir / train_nav_control_reg.MODEL_PARAMS_NAME

            paddle.device.set_device("cpu")
            model = train_nav_control_reg.PaddleLaneCnnModel()
            paddle.save(model.state_dict(), str(params_path))

            result = check_nav_control_sequence.run_sequence_analysis(
                model_format="dynamic",
                model_path=params_path,
                dataset_dir=dataset_dir,
                split="eval",
                batch_size=2,
                device="cpu",
                output_dir=root / "sequence_check",
                train_sources=("official_train",),
                eval_sources=("official_eval",),
            )

            self.assertEqual(result["checked_samples"], 2)
            self.assertEqual(result["source_count"], 1)
            self.assertEqual(result["sources"][0]["source_name"], "official_eval")


if __name__ == "__main__":
    unittest.main()
