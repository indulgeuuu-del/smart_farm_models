from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "detection"
    / "build_nine_vegetable_capture_plan.py"
)


def load_module():
    if not SCRIPT_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "build_nine_vegetable_capture_plan",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = load_module()


class CapturePlanModelTest(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(module, f"Missing capture-plan generator: {SCRIPT_PATH}")
        return module

    def test_appends_green_pepper_as_class_27(self) -> None:
        target = self.require_module()

        self.assertEqual(len(target.FUSION_CLASS_NAMES_27), 27)
        self.assertEqual(target.FUSION_CLASS_NAMES_27[-1], "green pepper")
        self.assertEqual(
            target.FUSION_CLASS_NAMES_27[:-1],
            target.FUSION_CLASS_NAMES_26,
        )

    def test_builds_every_single_class_position_once(self) -> None:
        target = self.require_module()
        plan = target.build_capture_plan()
        rows = [row for row in plan["rows"] if row["phase"] == "single"]

        self.assertEqual(len(rows), 72)
        self.assertEqual(
            {(row["target_class"], row["target_position"]) for row in rows},
            {
                (label, position)
                for label in target.VEGETABLE_LABELS
                for position in target.POSITIONS
            },
        )
        for row in rows:
            self.assertEqual(row["split"], "train")
            self.assertEqual((row["min_images"], row["max_images"]), (6, 10))
            self.assertEqual(row[row["target_position"]], row["target_class"])
            for position in target.POSITIONS:
                if position != row["target_position"]:
                    self.assertEqual(row[position], "BLANK")

    def test_each_mixed_matrix_has_complete_slot_and_omission_coverage(self) -> None:
        target = self.require_module()
        plan = target.build_capture_plan()
        mixed = [row for row in plan["rows"] if row["phase"] == "mixed"]

        self.assertEqual(len(mixed), 36)
        self.assertEqual(
            len({tuple(row[position] for position in target.POSITIONS) for row in mixed}),
            36,
        )
        expected_split_counts = {"train": 18, "val": 9, "test": 9}
        self.assertEqual(
            {
                split: sum(row["split"] == split for row in mixed)
                for split in expected_split_counts
            },
            expected_split_counts,
        )
        for matrix_name in target.MATRIX_OFFSETS:
            rows = [row for row in mixed if row["matrix"] == matrix_name]
            self.assertEqual(len(rows), 9)
            self.assertEqual(
                {row["missing_class"] for row in rows},
                set(target.VEGETABLE_LABELS),
            )
            for position in target.POSITIONS:
                self.assertEqual(
                    {row[position] for row in rows},
                    set(target.VEGETABLE_LABELS),
                )

    def test_validator_rejects_duplicate_mixed_layout(self) -> None:
        target = self.require_module()
        plan = target.build_capture_plan()
        mixed = [row for row in plan["rows"] if row["phase"] == "mixed"]
        for position in target.POSITIONS:
            mixed[1][position] = mixed[0][position]

        errors = target.validate_capture_plan(plan)

        self.assertTrue(
            any("duplicate mixed layout" in error for error in errors),
            errors,
        )

    def test_write_capture_plan_creates_json_csv_and_readme(self) -> None:
        target = self.require_module()
        self.assertTrue(
            hasattr(target, "write_capture_plan"),
            "write_capture_plan is not implemented",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            target.write_capture_plan(target.build_capture_plan(), output_dir)

            payload = json.loads(
                (output_dir / "capture_plan.json").read_text(encoding="utf-8")
            )
            with (output_dir / "capture_checklist.csv").open(
                encoding="utf-8-sig",
                newline="",
            ) as file:
                csv_rows = list(csv.DictReader(file))
            readme = (output_dir / "README.md").read_text(encoding="utf-8")

            self.assertEqual(payload["summary"]["placement_rows"], 108)
            self.assertEqual(payload["fusion_class_names"][-1], "green pepper")
            self.assertEqual(payload["rows"][0]["target_class"], "rape")
            self.assertEqual(payload["rows"][0]["target_position"], "L1")
            self.assertEqual(len(csv_rows), 108)
            self.assertEqual(csv_rows[0]["状态"], "待采集")
            self.assertEqual(csv_rows[0]["目标类别"], "油菜")
            self.assertEqual(csv_rows[0]["目标位置"], "左1")
            self.assertEqual(csv_rows[90]["布局组"], "验证矩阵")
            self.assertEqual(csv_rows[90]["布局编号"], "验证矩阵_缺少_油菜")
            self.assertEqual(csv_rows[99]["布局组"], "测试矩阵")
            csv_text = "\n".join(
                [*csv_rows[0].keys()]
                + [value for row in csv_rows for value in row.values()]
            )
            self.assertIsNone(re.search(r"[A-Za-z]", csv_text), csv_text)
            self.assertIn("green pepper", readme)
            self.assertIn("CSV 仅用于中文现场执行", readme)
            self.assertNotIn(
                b"\r\n",
                (output_dir / "capture_plan.json").read_bytes(),
            )
            self.assertNotIn(
                b"\r\n",
                (output_dir / "README.md").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
