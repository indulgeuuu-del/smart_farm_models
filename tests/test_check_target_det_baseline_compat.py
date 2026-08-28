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
                "arch: YOLO",
                "Preprocess:",
                "- interp: 2",
                "  keep_ratio: false",
                "  target_size:",
                f"  - {target_size}",
                f"  - {target_size}",
                "  type: Resize",
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

    def test_model_dir_is_required(self) -> None:
        module = load_module()
        with self.assertRaises(SystemExit):
            module.build_parser().parse_args([])

    def test_accepts_static_model_and_matching_bundled_labels(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.json").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            labels = ["water", "storage", "green pepper"]
            write_infer_cfg(root / "infer_cfg.yml", labels)
            (root / "label_list.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

            report = module.check_model_dir(root)

            self.assertTrue(report.ok, report.issues)
            self.assertEqual(report.model_file.name, "model.json")
            self.assertEqual(report.params_file.name, "model.pdiparams")
            self.assertEqual(report.target_size, [416, 416])
            self.assertEqual(report.label_count, 3)
            self.assertEqual(report.labels_file, root / "label_list.txt")

    def test_rejects_label_file_mismatch(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            write_infer_cfg(root / "infer_cfg.yml", ["water", "storage"])
            (root / "label_list.txt").write_text("storage\nwater\n", encoding="utf-8")

            report = module.check_model_dir(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("label_list.txt" in issue for issue in report.issues))

    def test_expected_labels_file_enforces_exact_order(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            expected = root / "expected.txt"
            (root / "inference.pdmodel").write_text("", encoding="utf-8")
            (root / "inference.pdiparams").write_text("", encoding="utf-8")
            write_infer_cfg(root / "infer_cfg.yml", ["water", "storage"])
            expected.write_text("storage\nwater\n", encoding="utf-8")

            report = module.check_model_dir(root, expected_labels_file=expected)

            self.assertFalse(report.ok)
            self.assertTrue(any("Expected labels file" in issue for issue in report.issues))

    def test_rejects_duplicate_or_empty_labels(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            write_infer_cfg(root / "infer_cfg.yml", ["water", "water"])

            report = module.check_model_dir(root)

            self.assertFalse(report.ok)
            self.assertTrue(any("duplicate" in issue for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
