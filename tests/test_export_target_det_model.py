from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "detection" / "export_target_det_model.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_target_det_model", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExportTargetDetModelScriptTest(unittest.TestCase):
    def test_export_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing export entrypoint: {SCRIPT_PATH}")

    def test_build_export_command_points_to_repo_recipe(self) -> None:
        module = load_module()
        command = module.build_export_command(
            python_executable="python",
            paddledet_root=Path(r"E:\PaddleDetection"),
            config_path=REPO_ROOT / "configs" / "target_det.yaml",
            weights_path=REPO_ROOT / "outputs" / "my_target_det_run" / "best_model",
            output_dir=REPO_ROOT / "deploy" / "exported_models" / "01_target_det" / "my_target_det_run",
            trt=False,
            exclude_nms=False,
            dataset_dir=REPO_ROOT / "datasets" / "01_target_det" / "paddlex_formal_0519",
        )

        self.assertEqual(command[:3], ["python", str(Path(r"E:\PaddleDetection") / "tools" / "export_model.py"), "-c"])
        self.assertIn("--output_dir", command)
        self.assertIn(
            f"weights={REPO_ROOT / 'outputs' / 'my_target_det_run' / 'best_model'}",
            command,
        )
        self.assertIn(str(REPO_ROOT / "deploy" / "exported_models" / "01_target_det" / "my_target_det_run"), command)
        self.assertIn(
            f"EvalDataset.dataset_dir={REPO_ROOT / 'datasets' / '01_target_det' / 'paddlex_formal_0519'}",
            command,
        )
        self.assertIn(
            f"TestDataset.dataset_dir={REPO_ROOT / 'datasets' / '01_target_det' / 'paddlex_formal_0519'}",
            command,
        )

    def test_build_export_command_can_force_cpu_export(self) -> None:
        module = load_module()
        command = module.build_export_command(
            python_executable="python",
            paddledet_root=Path(r"E:\PaddleDetection"),
            config_path=REPO_ROOT / "configs" / "target_det.yaml",
            weights_path=REPO_ROOT / "outputs" / "my_target_det_run" / "best_model",
            output_dir=REPO_ROOT / "deploy" / "exported_models" / "01_target_det" / "my_target_det_run",
            trt=False,
            exclude_nms=False,
            dataset_dir=None,
            use_gpu=False,
        )

        self.assertIn("use_gpu=false", command)


if __name__ == "__main__":
    unittest.main()
