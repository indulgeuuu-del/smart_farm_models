# 功能：测试 08_nav_control_optional 离线推理检查脚本。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import paddle
from PIL import Image


TRAIN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "train_nav_control_reg.py"
TRAIN_SPEC = importlib.util.spec_from_file_location("train_nav_control_reg", TRAIN_SCRIPT_PATH)
train_nav_control_reg = importlib.util.module_from_spec(TRAIN_SPEC)
assert TRAIN_SPEC.loader is not None
sys.modules[TRAIN_SPEC.name] = train_nav_control_reg
TRAIN_SPEC.loader.exec_module(train_nav_control_reg)

EXPORT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "export_nav_control_reg.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_nav_control_reg", EXPORT_SCRIPT_PATH)
export_nav_control_reg = importlib.util.module_from_spec(EXPORT_SPEC)
assert EXPORT_SPEC.loader is not None
sys.modules[EXPORT_SPEC.name] = export_nav_control_reg
EXPORT_SPEC.loader.exec_module(export_nav_control_reg)

INFER_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "check_nav_control_infer.py"
INFER_SPEC = importlib.util.spec_from_file_location("check_nav_control_infer", INFER_SCRIPT_PATH)
check_nav_control_infer = importlib.util.module_from_spec(INFER_SPEC)
assert INFER_SPEC.loader is not None
sys.modules[INFER_SPEC.name] = check_nav_control_infer
INFER_SPEC.loader.exec_module(check_nav_control_infer)


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


class CheckNavControlInferTest(unittest.TestCase):
    def test_default_dataset_dir_uses_final_optional_numbering(self) -> None:
        args = check_nav_control_infer.build_parser().parse_args(["--model-path", "dummy.pdparams"])

        self.assertEqual(Path(args.dataset_dir), Path("datasets/08_nav_control_optional"))

    def test_dynamic_inference_check_reports_width_two_for_eval_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            params_dir = root / "dynamic"
            params_dir.mkdir()
            params_path = params_dir / train_nav_control_reg.MODEL_PARAMS_NAME

            paddle.device.set_device("cpu")
            model = train_nav_control_reg.PaddleLaneCnnModel()
            paddle.save(model.state_dict(), str(params_path))

            result = check_nav_control_infer.run_dynamic_inference_check(
                params_path=params_path,
                dataset_dir=dataset_dir,
                split="eval",
                batch_size=2,
                device="cpu",
            )

            self.assertEqual(result["checked_samples"], 3)
            self.assertEqual(result["pred_widths"], [2])
            self.assertEqual(result["last_pred_shape"], [1, 2])

    def test_dynamic_inference_check_honors_custom_eval_source(self) -> None:
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

            result = check_nav_control_infer.run_dynamic_inference_check(
                params_path=params_path,
                dataset_dir=dataset_dir,
                split="eval",
                batch_size=2,
                device="cpu",
                train_sources=("official_train",),
                eval_sources=("official_eval",),
            )

            self.assertEqual(result["checked_samples"], 2)
            self.assertEqual(result["pred_widths"], [2])

    def test_static_inference_check_reports_width_two_for_eval_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            params_dir = root / "dynamic"
            params_dir.mkdir()
            params_path = params_dir / train_nav_control_reg.MODEL_PARAMS_NAME

            paddle.device.set_device("cpu")
            model = train_nav_control_reg.PaddleLaneCnnModel()
            paddle.save(model.state_dict(), str(params_path))
            export_result = export_nav_control_reg.export_model(params_path, root / "exported", device="cpu")

            result = check_nav_control_infer.run_static_inference_check(
                model_path=export_result["export_prefix"],
                dataset_dir=dataset_dir,
                split="eval",
                batch_size=2,
                device="cpu",
            )

            self.assertEqual(result["checked_samples"], 3)
            self.assertEqual(result["pred_widths"], [2])
            self.assertEqual(result["last_pred_shape"], [1, 2])


if __name__ == "__main__":
    unittest.main()
