from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "detection" / "watch_export_target_det_best.py"


def load_module():
    spec = importlib.util.spec_from_file_location("watch_export_target_det_best", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WatchExportTargetDetBestTest(unittest.TestCase):
    def test_readme_uses_actual_label_count_without_historical_epoch_text(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            handoff_dir = root / "handoff"
            target_det_dir = handoff_dir / "target_det"
            target_det_dir.mkdir(parents=True)
            labels = [f"class_{index}" for index in range(27)]
            (target_det_dir / "label_list.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

            module.write_readme(
                handoff_dir,
                target_det_dir,
                root / "run",
                root / "dataset",
                root / "config.yaml",
                {"epoch": 1, "step": 100, "steps_per_epoch": 3200},
                0.5,
            )

            text = (handoff_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("# 27 类融合检测 current best 上车候选包", text)
            self.assertIn("类别数：27", text)
            self.assertNotIn("26 类", text)
            self.assertNotIn("120 epoch", text)


if __name__ == "__main__":
    unittest.main()
