from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "summarize_target_det_hardcases.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_target_det_hardcases", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SummarizeTargetDetHardcasesTest(unittest.TestCase):
    def test_build_summary_counts_accuracy_and_named_confusion(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "candidate_hardcase_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "overall": {
                            "gt_annotations": 4,
                            "correct": 3,
                            "confused": 1,
                            "missed": 0,
                        },
                        "confusions": {
                            "water_l1": {
                                "water_l2": 1,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = module.build_summary(
                {"candidate": report_path},
                confusion_pairs=[("water_l1", "water_l2")],
            )

            self.assertEqual(summary["candidate"]["gt"], 4)
            self.assertEqual(summary["candidate"]["correct"], 3)
            self.assertEqual(summary["candidate"]["confused"], 1)
            self.assertEqual(summary["candidate"]["missed"], 0)
            self.assertEqual(summary["candidate"]["accuracy"], 0.75)
            self.assertEqual(summary["candidate"]["water_l1_to_water_l2"], 1)

    def test_render_markdown_outputs_comparable_table(self) -> None:
        module = load_module()
        summary = {
            "candidate_a": {
                "gt": 4,
                "correct": 3,
                "confused": 1,
                "missed": 0,
                "accuracy": 0.75,
                "water_l1_to_water_l2": 1,
            },
            "candidate_b": {
                "gt": 4,
                "correct": 4,
                "confused": 0,
                "missed": 0,
                "accuracy": 1.0,
                "water_l1_to_water_l2": 0,
            },
        }

        markdown = module.render_markdown(summary, title="Hard-case Summary")

        self.assertIn("# Hard-case Summary", markdown)
        self.assertIn("| model | gt | correct | confused | missed | accuracy | water_l1_to_water_l2 |", markdown)
        self.assertIn("| candidate_a | 4 | 3 | 1 | 0 | 0.7500 | 1 |", markdown)
        self.assertIn("| candidate_b | 4 | 4 | 0 | 0 | 1.0000 | 0 |", markdown)


if __name__ == "__main__":
    unittest.main()
