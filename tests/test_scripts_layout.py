# 功能：测试 scripts 目录瘦身后的职责分组布局。
from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


class ScriptsLayoutTest(unittest.TestCase):
    def test_root_scripts_directory_has_no_python_entrypoint_clutter(self) -> None:
        top_level_py_files = sorted(path.name for path in SCRIPTS_DIR.glob("*.py"))

        self.assertEqual(top_level_py_files, [])

    def test_expected_script_groups_exist(self) -> None:
        expected_files = {
            "common/check_mandatory_data_layout.py",
            "lane_seg/split_dataset.py",
            "lane_seg/export_seg_masks.py",
            "lane_seg/check_lane_seg_dataset.py",
            "detection/check_det_coco_annotations.py",
            "detection/analyze_target_det_dataset.py",
            "detection/augment_target_det_coco.py",
            "detection/export_target_det_coco.py",
            "detection/train_target_det.py",
            "detection/export_target_det_model.py",
            "nav_control_optional/prepare_nav_control_dataset.py",
            "nav_control_optional/check_nav_control_data.py",
            "nav_control_optional/train_nav_control_reg.py",
            "nav_control_optional/export_nav_control_reg.py",
            "nav_control_optional/check_nav_control_infer.py",
            "nav_control_optional/check_nav_control_sequence.py",
            "onboard/eval_onboard_samples.py",
            "onboard/check_target_det_baseline_compat.py",
        }

        missing = sorted(relative for relative in expected_files if not (SCRIPTS_DIR / relative).is_file())

        self.assertEqual(missing, [])

    def test_root_readme_documents_retained_model_workflows(self) -> None:
        text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## 车道分割", text)
        self.assertIn("## 目标检测", text)
        self.assertIn("## 巡航控制", text)


if __name__ == "__main__":
    unittest.main()
