from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_PATH = (
    REPO_ROOT
    / "ai studio Fork"
    / "PaddleDetection"
    / "PaddleDetection"
    / "deploy"
    / "python"
    / "infer.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("paddledet_deploy_infer", INFER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load infer module: {INFER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaddleDetInferModelResolutionTest(unittest.TestCase):
    def test_prefers_pdmodel_when_present(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.pdmodel").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            (root / "model.json").write_text("", encoding="utf-8")
            infer_model, infer_params = module.resolve_infer_model_files(str(root))
            self.assertEqual(Path(infer_model).name, "model.pdmodel")
            self.assertEqual(Path(infer_params).name, "model.pdiparams")

    def test_falls_back_to_model_json(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "model.json").write_text("", encoding="utf-8")
            (root / "model.pdiparams").write_text("", encoding="utf-8")
            infer_model, infer_params = module.resolve_infer_model_files(str(root))
            self.assertEqual(Path(infer_model).name, "model.json")
            self.assertEqual(Path(infer_params).name, "model.pdiparams")

    def test_raises_when_no_supported_model_exists(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                module.resolve_infer_model_files(tmpdir)


if __name__ == "__main__":
    unittest.main()
