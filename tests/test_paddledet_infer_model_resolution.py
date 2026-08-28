from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.common.target_det_model import resolve_infer_model_files


class TargetDetModelResolutionTest(unittest.TestCase):
    def test_prefers_pdmodel_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            (root / "model.json").write_text("", encoding="utf-8")

            infer_model, infer_params = resolve_infer_model_files(root)

            self.assertEqual(infer_model.name, "model.pdmodel")
            self.assertEqual(infer_params.name, "model.pdiparams")

    def test_falls_back_to_model_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.json").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")

            infer_model, infer_params = resolve_infer_model_files(root)

            self.assertEqual(infer_model.name, "model.json")
            self.assertEqual(infer_params.name, "model.pdiparams")

    def test_supports_inference_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "inference.json").write_text("", encoding="utf-8")
            (root / "inference.pdiparams").write_text("", encoding="utf-8")

            infer_model, infer_params = resolve_infer_model_files(root)

            self.assertEqual(infer_model.name, "inference.json")
            self.assertEqual(infer_params.name, "inference.pdiparams")

    def test_raises_when_no_supported_model_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                resolve_infer_model_files(tmpdir)


if __name__ == "__main__":
    unittest.main()
