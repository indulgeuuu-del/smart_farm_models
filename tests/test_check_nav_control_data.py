# 功能：测试 08_nav_control_optional 巡航控制数据静态检查脚本。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "check_nav_control_data.py"
SPEC = importlib.util.spec_from_file_location("check_nav_control_data", SCRIPT_PATH)
check_nav_control_data = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_nav_control_data
SPEC.loader.exec_module(check_nav_control_data)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_source(root: Path, name: str, records: list[dict]) -> Path:
    source_dir = root / name
    source_dir.mkdir(parents=True)
    for record in records:
        img_path = record.get("img_path")
        if isinstance(img_path, str):
            (source_dir / img_path).parent.mkdir(parents=True, exist_ok=True)
            (source_dir / img_path).write_bytes(b"image")
    write_json(source_dir / "data.json", records)
    return source_dir


class CheckNavControlDataTest(unittest.TestCase):
    def test_default_dataset_dir_uses_final_optional_numbering(self) -> None:
        args = check_nav_control_data.build_parser().parse_args([])

        self.assertEqual(Path(args.dataset_dir), Path("datasets/08_nav_control_optional"))

    def test_valid_sources_and_splits_have_no_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_l = make_source(
                root,
                "image_set_l",
                [
                    {"img_path": "0000.jpg", "state": [0.15, 0.0, 0.0]},
                    {"img_path": "0001.jpg", "state": [0.15, -0.01, 0.2]},
                ],
            )
            source_r = make_source(
                root,
                "image_set_r",
                [{"img_path": "0000.jpg", "state": [0.15, 0.02, -0.3]}],
            )
            split_dir = root / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("image_set_l/0000.jpg\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("image_set_l/0001.jpg\n", encoding="utf-8")
            (split_dir / "test.txt").write_text("image_set_r/0000.jpg\n", encoding="utf-8")

            result = check_nav_control_data.check_nav_control_data([source_l, source_r], split_dir)

            self.assertEqual(result.errors, [])
            self.assertEqual(result.total_records, 3)
            self.assertEqual(result.source_counts, {"image_set_l": 2, "image_set_r": 1})
            self.assertEqual(result.state0_values, [0.15])

    def test_reports_bad_records_and_split_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "image_set1208"
            source_dir.mkdir(parents=True)
            (source_dir / "0000.jpg").write_bytes(b"image")
            write_json(
                source_dir / "data.json",
                [
                    {"img_path": "0000.jpg", "state": [0.15, 0.0, 0.0]},
                    {"img_path": "missing.jpg", "state": [0.15, "bad", 0.0]},
                ],
            )
            split_dir = root / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("image_set1208/0000.jpg\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("image_set1208/0000.jpg\n", encoding="utf-8")
            (split_dir / "test.txt").write_text("image_set1208/missing.jpg\n", encoding="utf-8")

            result = check_nav_control_data.check_nav_control_data([source_dir], split_dir)
            joined = "\n".join(result.errors)

            self.assertIn("Missing image", joined)
            self.assertIn("Invalid state", joined)
            self.assertIn("Split overlap", joined)

    def test_discovers_record_dirs_from_sources_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_source(
                root,
                "image_set_l",
                [{"img_path": "0000.jpg", "state": [0.15, 0.0, 0.0]}],
            )
            records_dir = root / "datasets" / "08_nav_control_optional" / "records"
            records_dir.mkdir(parents=True)
            write_json(
                records_dir / "sources.json",
                {
                    "official_source": "image_set_l",
                    "sources": [{"name": "image_set_l", "path": str(source_dir), "status": "official"}],
                },
            )

            discovered = check_nav_control_data.discover_record_dirs(root / "datasets" / "08_nav_control_optional")

            self.assertEqual(discovered, [source_dir])

    def test_discovers_paddlelane_raw_image_sets_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "datasets" / "08_nav_control_optional"
            raw_dir = dataset_dir / "raw"
            source_r = make_source(raw_dir, "image_set_r", [{"img_path": "0000.jpg", "state": [0.15, 0.0, 0.0]}])
            source_l = make_source(raw_dir, "image_set_l", [{"img_path": "0000.jpg", "state": [0.15, 0.1, -0.1]}])
            source_1208 = make_source(raw_dir, "image_set1208", [{"img_path": "0000.jpg", "state": [0.15, -0.1, 0.1]}])

            discovered = check_nav_control_data.discover_record_dirs(dataset_dir)

            self.assertEqual(discovered, [source_1208, source_l, source_r])

    def test_allows_train_val_only_split_check(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_source = make_source(
                root,
                "image_set_r",
                [{"img_path": "0000.jpg", "state": [0.15, 0.0, 0.0]}],
            )
            val_source = make_source(
                root,
                "image_set_l",
                [{"img_path": "0000.jpg", "state": [0.15, 0.01, -0.2]}],
            )
            split_dir = root / "splits"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("image_set_r/0000.jpg\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("image_set_l/0000.jpg\n", encoding="utf-8")

            result = check_nav_control_data.check_nav_control_data(
                [train_source, val_source],
                split_dir,
                required_splits=("train", "val"),
            )

            self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
