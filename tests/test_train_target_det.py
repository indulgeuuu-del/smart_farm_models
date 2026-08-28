from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "detection" / "train_target_det.py"


def load_module():
    spec = importlib.util.spec_from_file_location("train_target_det", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainTargetDetScriptTest(unittest.TestCase):
    def test_train_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing training entrypoint: {SCRIPT_PATH}")

    def test_build_train_command_points_to_paddledetection_recipe(self) -> None:
        module = load_module()
        command = module.build_train_command(
            python_executable="python",
            paddledet_root=Path(r"E:\PaddleDetection"),
            config_path=REPO_ROOT / "configs" / "target_det.yaml",
            dataset_dir=REPO_ROOT / "datasets" / "01_target_det" / "paddlex_formal_0519_aug",
            output_dir=REPO_ROOT / "outputs" / "my_target_det_run",
            use_vdl=False,
            amp=True,
            resume_path=None,
        )

        self.assertEqual(command[:3], ["python", str(Path(r"E:\PaddleDetection") / "tools" / "train.py"), "-c"])
        self.assertIn("--eval", command)
        self.assertIn("--amp", command)
        self.assertIn("use_gpu=true", command)
        self.assertIn(
            f"save_dir={REPO_ROOT / 'outputs' / 'my_target_det_run'}",
            command,
        )
        self.assertIn(
            f"TrainDataset.dataset_dir={REPO_ROOT / 'datasets' / '01_target_det' / 'paddlex_formal_0519_aug'}",
            command,
        )
        self.assertIn(
            f"EvalDataset.dataset_dir={REPO_ROOT / 'datasets' / '01_target_det' / 'paddlex_formal_0519_aug'}",
            command,
        )
        self.assertIn(
            f"TestDataset.dataset_dir={REPO_ROOT / 'datasets' / '01_target_det' / 'paddlex_formal_0519_aug'}",
            command,
        )

    def test_augmented_dataset_guard_accepts_aug_name_or_meta(self) -> None:
        module = load_module()

        self.assertTrue(module.is_augmented_dataset(Path("datasets/01_target_det/paddlex_formal_0519_aug")))

    def test_augmented_dataset_guard_rejects_plain_name_without_meta(self) -> None:
        module = load_module()

        self.assertFalse(module.is_augmented_dataset(Path("datasets/01_target_det/paddlex_formal_0519")))


if __name__ == "__main__":
    unittest.main()
