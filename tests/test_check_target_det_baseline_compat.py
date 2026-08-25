from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "onboard" / "check_target_det_baseline_compat.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_target_det_baseline_compat", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_infer_cfg(path: Path, labels: list[str], target_size: int = 416) -> None:
    label_lines = "\n".join(f"- {label}" for label in labels)
    path.write_text(
        "\n".join(
            [
                "mode: paddle",
                "draw_threshold: 0.5",
                "metric: COCO",
                "use_dynamic_shape: false",
                "arch: YOLO",
                "min_subgraph_size: 3",
                "Preprocess:",
                "- interp: 2",
                "  keep_ratio: false",
                "  target_size:",
                f"  - {target_size}",
                f"  - {target_size}",
                "  type: Resize",
                "- mean:",
                "  - 0.0",
                "  - 0.0",
                "  - 0.0",
                "  norm_type: none",
                "  std:",
                "  - 1.0",
                "  - 1.0",
                "  - 1.0",
                "  type: NormalizeImage",
                "- type: Permute",
                "label_list:",
                label_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


class CheckTargetDetBaselineCompatTest(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file(), f"Missing script: {SCRIPT_PATH}")

    def test_default_model_dir_points_to_current_hardfix_candidate(self) -> None:
        module = load_module()

        self.assertIn("jetson_models_20260604_hardfix_epoch9_candidate", str(module.DEFAULT_MODEL_DIR))

    def test_expected_labels_are_current_23_class_hardfix_labels(self) -> None:
        module = load_module()

        self.assertEqual(len(module.EXPECTED_TARGET_DET_LABELS), 23)
        self.assertEqual(module.EXPECTED_TARGET_DET_LABELS[0], "water_l3")
        self.assertIn("h_qing_jiao", module.EXPECTED_TARGET_DET_LABELS)
        self.assertNotIn("order", module.EXPECTED_TARGET_DET_LABELS)
        self.assertNotIn("name", module.EXPECTED_TARGET_DET_LABELS)

    def test_accepts_current_hardfix_pdmodel_shape(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            write_infer_cfg(root / "infer_cfg.yml", module.EXPECTED_TARGET_DET_LABELS)

            report = module.check_model_dir(root)

            self.assertTrue(report.ok, report.issues)
            self.assertEqual(report.model_file.name, "model.pdmodel")
            self.assertEqual(report.params_file.name, "model.pdiparams")
            self.assertEqual(report.target_size, [416, 416])
            self.assertEqual(report.label_count, 23)

    def test_rejects_model_json_without_pdmodel(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.json").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            write_infer_cfg(root / "infer_cfg.yml", module.EXPECTED_TARGET_DET_LABELS)

            report = module.check_model_dir(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("model.json" in issue for issue in report.issues))

    def test_rejects_label_order_mismatch(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            labels = list(module.EXPECTED_TARGET_DET_LABELS)
            labels[0], labels[1] = labels[1], labels[0]
            write_infer_cfg(root / "infer_cfg.yml", labels)

            report = module.check_model_dir(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("current repo label order" in issue for issue in report.issues))

    def test_accepts_custom_labels_when_explicitly_allowed(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            write_infer_cfg(root / "infer_cfg.yml", ["water", "ball_blue", "animal"])

            report = module.check_model_dir(root, allow_custom_labels=True)

            self.assertTrue(report.ok, report.issues)
            self.assertEqual(report.label_count, 3)


if __name__ == "__main__":
    unittest.main()
