# 功能：测试 08_nav_control_optional 回归模型导出脚本。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import paddle


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


class ExportNavControlRegTest(unittest.TestCase):
    def test_swap_output_layer_returns_second_column_first(self) -> None:
        class DummyModel(paddle.nn.Layer):
            def forward(self, x: paddle.Tensor) -> paddle.Tensor:
                return paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

        wrapped = export_nav_control_reg.SwapOutputLayer(DummyModel())
        pred = wrapped(paddle.zeros([2, 3, 128, 128], dtype="float32"))

        self.assertEqual(list(pred.shape), [2, 2])
        self.assertEqual(pred.numpy().tolist(), [[2.0, 1.0], [4.0, 3.0]])

    def test_export_model_creates_static_files_and_meta(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            params_dir = root / "dynamic"
            params_dir.mkdir()
            params_path = params_dir / train_nav_control_reg.MODEL_PARAMS_NAME

            paddle.device.set_device("cpu")
            model = train_nav_control_reg.PaddleLaneCnnModel()
            paddle.save(model.state_dict(), str(params_path))

            result = export_nav_control_reg.export_model(params_path, root / "exported", device="cpu")

            self.assertEqual(result["reload_pred_shape"], [1, 2])
            self.assertTrue((root / "exported" / "cnn_lane.pdiparams").is_file())
            self.assertTrue(Path(result["program_path"]).is_file())
            meta = json.loads((root / "exported" / "export_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["reload_pred_shape"], [1, 2])
            self.assertEqual(meta["input_shape"], [None, 3, 128, 128])
            self.assertIn(Path(meta["program_path"]).suffix, {".pdmodel", ".json"})


if __name__ == "__main__":
    unittest.main()
