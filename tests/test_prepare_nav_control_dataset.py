# 功能：测试 08_nav_control_optional 正式数据源登记和切分清单生成脚本。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "prepare_nav_control_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_nav_control_dataset", SCRIPT_PATH)
prepare_nav_control_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = prepare_nav_control_dataset
SPEC.loader.exec_module(prepare_nav_control_dataset)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_source(root: Path, count: int) -> Path:
    source_dir = root / "image_set_l"
    source_dir.mkdir(parents=True)
    records = []
    for index in range(count):
        image_name = f"{index:04d}.jpg"
        (source_dir / image_name).write_bytes(b"image")
        records.append({"img_path": image_name, "state": [0.15, index / 1000, -index / 1000]})
    write_json(source_dir / "data.json", records)
    return source_dir


class PrepareNavControlDatasetTest(unittest.TestCase):
    def test_default_dataset_dir_uses_final_optional_numbering(self) -> None:
        args = prepare_nav_control_dataset.build_parser().parse_args([])

        self.assertEqual(Path(args.dataset_dir), Path("datasets/08_nav_control_optional"))

    def test_registers_official_source_and_writes_sequential_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_source(root, 10)
            dataset_dir = root / "datasets" / "08_nav_control_optional"

            result = prepare_nav_control_dataset.prepare_nav_control_dataset(
                source_dir=source_dir,
                dataset_dir=dataset_dir,
                train_ratio=0.6,
                val_ratio=0.2,
            )

            self.assertEqual(result.total_records, 10)
            self.assertEqual(result.train_count, 6)
            self.assertEqual(result.val_count, 2)
            self.assertEqual(result.test_count, 2)

            sources = json.loads((dataset_dir / "records" / "sources.json").read_text(encoding="utf-8"))
            self.assertEqual(sources["official_source"], "image_set_l")
            self.assertEqual(sources["sources"][0]["status"], "official")
            self.assertEqual(sources["sources"][0]["records"], 10)

            split_dir = dataset_dir / "paddle_custom" / "splits"
            self.assertEqual((split_dir / "train.txt").read_text(encoding="utf-8").splitlines()[0], "image_set_l/0000.jpg")
            self.assertEqual((split_dir / "val.txt").read_text(encoding="utf-8").splitlines()[0], "image_set_l/0006.jpg")
            self.assertEqual((split_dir / "test.txt").read_text(encoding="utf-8").splitlines()[0], "image_set_l/0008.jpg")

            meta = json.loads((split_dir / "split_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["split_method"], "sequential_by_data_json_order")
            self.assertEqual(meta["source_name"], "image_set_l")

    def test_rejects_source_with_too_few_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_source(root, 2)

            with self.assertRaisesRegex(ValueError, "at least 3 records"):
                prepare_nav_control_dataset.prepare_nav_control_dataset(
                    source_dir=source_dir,
                    dataset_dir=root / "dataset",
                )

    def test_registers_explicit_train_and_val_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_source = make_source(root / "train_parent", 3)
            train_source = train_source.rename(root / "train_parent" / "image_set_r")
            val_source = make_source(root / "val_parent", 4)
            dataset_dir = root / "datasets" / "08_nav_control_optional"

            result = prepare_nav_control_dataset.prepare_nav_control_train_val_dataset(
                train_source_dir=train_source,
                val_source_dir=val_source,
                dataset_dir=dataset_dir,
            )

            self.assertEqual(result.train_count, 3)
            self.assertEqual(result.val_count, 4)
            self.assertEqual(result.test_count, 0)
            sources = json.loads((dataset_dir / "records" / "sources.json").read_text(encoding="utf-8"))
            self.assertEqual(sources["official_splits"], {"train": "image_set_r", "val": "image_set_l"})
            self.assertEqual(sources["sources"][0]["state_ranges"][1], [0.0, 0.002])
            self.assertEqual(sources["sources"][1]["state_ranges"][1], [0.0, 0.003])

    def test_marks_sources_under_dataset_raw_as_workspace_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets" / "08_nav_control_optional"
            (dataset_dir / "raw").mkdir(parents=True)
            train_source = make_source(dataset_dir / "raw_train", 3)
            train_source = train_source.rename(dataset_dir / "raw" / "image_set_r")
            val_source = make_source(dataset_dir / "raw_val", 4)
            val_source = val_source.rename(dataset_dir / "raw" / "image_set_l")

            prepare_nav_control_dataset.prepare_nav_control_train_val_dataset(
                train_source_dir=train_source,
                val_source_dir=val_source,
                dataset_dir=dataset_dir,
            )

            sources = json.loads((dataset_dir / "records" / "sources.json").read_text(encoding="utf-8"))
            self.assertEqual(sources["source_policy"], "workspace_copy")

            split_dir = dataset_dir / "paddle_custom" / "splits"
            self.assertEqual((split_dir / "train.txt").read_text(encoding="utf-8").splitlines()[0], "image_set_r/0000.jpg")
            self.assertEqual((split_dir / "val.txt").read_text(encoding="utf-8").splitlines()[0], "image_set_l/0000.jpg")
            self.assertFalse((split_dir / "test.txt").exists())


if __name__ == "__main__":
    unittest.main()
