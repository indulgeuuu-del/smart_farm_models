from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "lane_seg" / "split_dataset.py"
SPEC = importlib.util.spec_from_file_location("split_dataset", SCRIPT_PATH)
split_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(split_dataset)


def make_labelme_jsons(labelme_dir: Path, count: int) -> None:
    labelme_dir.mkdir(parents=True)
    for index in range(count):
        (labelme_dir / f"lane_{index:03d}.json").write_text("{}", encoding="utf-8")


class SplitDatasetTest(unittest.TestCase):
    def test_splits_labelme_jsons_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labelme_dir = root / "labelme_json"
            first_output = root / "splits_first"
            second_output = root / "splits_second"
            make_labelme_jsons(labelme_dir, 5)

            first = split_dataset.split_labelme_jsons(labelme_dir, first_output, val_ratio=0.4, seed=2026)
            second = split_dataset.split_labelme_jsons(labelme_dir, second_output, val_ratio=0.4, seed=2026)

            first_train = (first_output / "train.txt").read_text(encoding="utf-8").splitlines()
            first_val = (first_output / "val.txt").read_text(encoding="utf-8").splitlines()
            second_train = (second_output / "train.txt").read_text(encoding="utf-8").splitlines()
            second_val = (second_output / "val.txt").read_text(encoding="utf-8").splitlines()

            self.assertEqual(first.total, 5)
            self.assertEqual(first.train_count, 3)
            self.assertEqual(first.val_count, 2)
            self.assertEqual(first_train, second_train)
            self.assertEqual(first_val, second_val)
            self.assertEqual(set(first_train).intersection(first_val), set())
            self.assertEqual(set(first_train + first_val), {f"lane_{index:03d}.json" for index in range(5)})

    def test_rejects_empty_labelme_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labelme_dir = root / "labelme_json"
            labelme_dir.mkdir()

            with self.assertRaisesRegex(ValueError, "No Labelme JSON files"):
                split_dataset.split_labelme_jsons(labelme_dir, root / "splits", val_ratio=0.2, seed=2026)

    def test_rejects_invalid_val_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            labelme_dir = root / "labelme_json"
            make_labelme_jsons(labelme_dir, 2)

            with self.assertRaisesRegex(ValueError, "val_ratio"):
                split_dataset.split_labelme_jsons(labelme_dir, root / "splits", val_ratio=1.0, seed=2026)


if __name__ == "__main__":
    unittest.main()
