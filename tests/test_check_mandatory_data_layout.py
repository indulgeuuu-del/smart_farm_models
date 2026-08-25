from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "common" / "check_mandatory_data_layout.py"
SPEC = importlib.util.spec_from_file_location("check_mandatory_data_layout", SCRIPT_PATH)
check_mandatory_data_layout = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_mandatory_data_layout
SPEC.loader.exec_module(check_mandatory_data_layout)


def make_required_layout(root: Path) -> None:
    required_dirs = {
        "00_lane_seg": ("metadata", "paddlex"),
        "01_target_det": ("annotations_coco", "paddlex"),
        "08_nav_control_optional": ("records", "paddle_custom"),
    }
    for dataset_name, children in required_dirs.items():
        for child in children:
            (root / dataset_name / child).mkdir(parents=True)


class CheckMandatoryDataLayoutTest(unittest.TestCase):
    def test_valid_mandatory_layout_has_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_root = Path(temp_dir) / "datasets"
            make_required_layout(datasets_root)

            errors = check_mandatory_data_layout.check_layout(datasets_root)

            self.assertEqual(errors, [])

    def test_reports_missing_required_dirs_and_unexpected_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datasets_root = Path(temp_dir) / "datasets"
            make_required_layout(datasets_root)
            (datasets_root / "01_target_det" / "annotations_coco").rmdir()
            (datasets_root / "unexpected_module").mkdir()

            errors = check_mandatory_data_layout.check_layout(datasets_root)
            joined = "\n".join(errors)

            self.assertIn("Missing required directory", joined)
            self.assertIn("Unexpected dataset directory", joined)


if __name__ == "__main__":
    unittest.main()
