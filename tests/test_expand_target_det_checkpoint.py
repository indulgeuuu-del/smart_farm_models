from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import paddle


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "detection" / "expand_target_det_checkpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("expand_target_det_checkpoint", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExpandTargetDetCheckpointTest(unittest.TestCase):
    def test_expand_state_dict_preserves_old_channels_and_uses_vegetable_mean(self) -> None:
        module = load_module()
        state = {"backbone.weight": paddle.arange(6, dtype="float32").reshape([2, 3])}
        for level, channels in enumerate((4, 3, 2)):
            weight = paddle.arange(26 * channels, dtype="float32").reshape([26, channels, 1, 1])
            bias = paddle.arange(26, dtype="float32") + level
            state[f"yolo_head.pred_cls.{level}.weight"] = weight
            state[f"yolo_head.pred_cls.{level}.bias"] = bias

        expanded, changed = module.expand_state_dict(
            state,
            source_classes=26,
            target_classes=27,
            initialization_class_indices=tuple(range(18, 26)),
        )

        self.assertEqual(
            changed,
            [
                "yolo_head.pred_cls.0.bias",
                "yolo_head.pred_cls.0.weight",
                "yolo_head.pred_cls.1.bias",
                "yolo_head.pred_cls.1.weight",
                "yolo_head.pred_cls.2.bias",
                "yolo_head.pred_cls.2.weight",
            ],
        )
        self.assertTrue(paddle.equal_all(expanded["backbone.weight"], state["backbone.weight"]))
        for level in range(3):
            for suffix in ("weight", "bias"):
                key = f"yolo_head.pred_cls.{level}.{suffix}"
                self.assertEqual(expanded[key].shape[0], 27)
                self.assertTrue(paddle.equal_all(expanded[key][:26], state[key]))
                expected = paddle.mean(state[key][18:26], axis=0)
                self.assertTrue(paddle.allclose(expanded[key][26], expected))

    def test_expand_state_dict_rejects_missing_classification_tensor(self) -> None:
        module = load_module()
        state = {}
        for level in range(3):
            state[f"yolo_head.pred_cls.{level}.weight"] = paddle.zeros([26, 2, 1, 1])
            if level != 2:
                state[f"yolo_head.pred_cls.{level}.bias"] = paddle.zeros([26])

        with self.assertRaisesRegex(ValueError, "classification tensors"):
            module.expand_state_dict(
                state,
                source_classes=26,
                target_classes=27,
                initialization_class_indices=tuple(range(18, 26)),
            )


if __name__ == "__main__":
    unittest.main()
