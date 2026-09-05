from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "onboard" / "eval_onboard_samples.py"


def load_module():
    spec = importlib.util.spec_from_file_location("eval_onboard_samples", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvalOnboardSamplesTest(unittest.TestCase):
    def test_script_exists(self) -> None:
        self.assertTrue(SCRIPT_PATH.is_file())

    def test_samples_and_model_are_required(self) -> None:
        module = load_module()
        with self.assertRaises(SystemExit):
            module.build_parser().parse_args([])

    def test_discover_images_finds_supported_files_and_rejects_empty_or_invalid_inputs(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "nested").mkdir()
            (root / "nested" / "frame.JPG").write_bytes(b"image")
            (root / "note.txt").write_text("ignore", encoding="utf-8")

            images = module.discover_images(root)
            self.assertEqual([image.name for image in images], ["frame.JPG"])
            self.assertEqual(module.discover_images(root, max_images=1), images)
            with self.assertRaises(ValueError):
                module.discover_images(root, max_images=0)
            with self.assertRaises(FileNotFoundError):
                module.discover_images(root / "missing")

    def test_main_writes_unlabeled_preflight_manifest_without_inference(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            samples = root / "samples"
            model = root / "model"
            output = root / "output"
            samples.mkdir()
            model.mkdir()
            (samples / "frame.jpg").write_bytes(b"image")
            (model / "model.pdmodel").write_bytes(b"model")
            (model / "model.pdiparams").write_bytes(b"params")

            original_argv = sys.argv
            try:
                sys.argv = [
                    str(SCRIPT_PATH),
                    "--samples-dir",
                    str(samples),
                    "--model-dir",
                    str(model),
                    "--output-dir",
                    str(output),
                ]
                self.assertEqual(module.main(), 0)
            finally:
                sys.argv = original_argv

            manifest = json.loads((output / "sample_preflight.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["image_count"], 1)
            self.assertFalse(manifest["ground_truth_evaluation"])
            self.assertFalse(manifest["inference_requested"])
            self.assertEqual(manifest["images"], ["frame.jpg"])


if __name__ == "__main__":
    unittest.main()
