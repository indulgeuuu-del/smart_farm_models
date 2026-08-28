from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detection" / "augment_target_det_coco.py"


def load_module():
    spec = importlib.util.spec_from_file_location("augment_target_det_coco", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def make_paddlex_dataset(root: Path) -> Path:
    dataset_dir = root / "paddlex"
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    for name, color in {
        "train.jpg": (30, 80, 120),
        "val.jpg": (80, 120, 30),
        "test.jpg": (120, 30, 80),
    }.items():
        Image.new("RGB", (100, 60), color).save(images_dir / name)

    categories = [
        {"id": 1, "name": "water"},
        {"id": 2, "name": "cargo"},
    ]
    write_json(
        annotations_dir / "instance_train.json",
        {
            "images": [{"id": 1, "file_name": "train.jpg", "width": 100, "height": 60}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 15], "area": 300, "iscrowd": 0}
            ],
            "categories": categories,
        },
    )
    write_json(
        annotations_dir / "instance_val.json",
        {
            "images": [{"id": 2, "file_name": "val.jpg", "width": 100, "height": 60}],
            "annotations": [
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [30, 10, 15, 20], "area": 300, "iscrowd": 0}
            ],
            "categories": categories,
        },
    )
    write_json(
        annotations_dir / "instance_test.json",
        {
            "images": [{"id": 3, "file_name": "test.jpg", "width": 100, "height": 60}],
            "annotations": [],
            "categories": categories,
        },
    )
    (dataset_dir / "class_names.txt").write_text("water\ncargo\n", encoding="utf-8")
    return dataset_dir


def make_zoomout_focus_dataset(root: Path) -> Path:
    dataset_dir = make_paddlex_dataset(root)
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    Image.new("RGB", (100, 60), (90, 120, 150)).save(images_dir / "name.jpg")
    train = json.loads((annotations_dir / "instance_train.json").read_text(encoding="utf-8"))
    train["categories"].append({"id": 3, "name": "name"})
    train["images"].append({"id": 4, "file_name": "name.jpg", "width": 100, "height": 60})
    train["annotations"].append(
        {"id": 4, "image_id": 4, "category_id": 3, "bbox": [30, 12, 24, 18], "area": 432, "iscrowd": 0}
    )
    write_json(annotations_dir / "instance_train.json", train)
    return dataset_dir


class AugmentTargetDetCocoTest(unittest.TestCase):
    def test_zoomout_scales_and_translates_bbox(self) -> None:
        module = load_module()
        image = Image.new("RGB", (100, 60), (30, 80, 120))

        augmented, transform_bbox = module._apply_augmentation(image, "zoomout", random.Random(7))
        transformed = [float(value) for value in transform_bbox([10, 5, 20, 15], 100, 60)]

        self.assertEqual(augmented.size, image.size)
        scale = transformed[2] / 20.0
        offset_x = transformed[0] - 10.0 * scale
        offset_y = transformed[1] - 5.0 * scale
        self.assertGreaterEqual(scale, 0.55)
        self.assertLessEqual(scale, 0.80)
        self.assertAlmostEqual(transformed[3], 15.0 * scale, places=5)
        self.assertGreaterEqual(offset_x, 0.0)
        self.assertGreaterEqual(offset_y, 0.0)
        self.assertLessEqual(offset_x, 100.0 * (1.0 - scale) + 1.0)
        self.assertLessEqual(offset_y, 60.0 * (1.0 - scale) + 1.0)

    def test_flip_bbox_updates_x_coordinate(self) -> None:
        module = load_module()

        self.assertEqual(module.flip_bbox_xywh([10, 5, 20, 15], image_width=100), [70, 5, 20, 15])

    def test_vertical_flip_bbox_updates_y_coordinate(self) -> None:
        module = load_module()

        self.assertEqual(module.vertical_flip_bbox_xywh([10, 5, 20, 15], image_height=60), [10, 40, 20, 15])

    def test_object_flip_generates_hard_case_variant_without_bbox_change(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_paddlex_dataset(root)
            output_dir = root / "paddlex_aug"

            result = module.build_augmented_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                augmentations=["object_flip"],
                variants_per_image=0,
                hard_case_labels=["water"],
                hard_case_extra=1,
                seed=1,
            )

            train = json.loads((output_dir / "annotations" / "instance_train.json").read_text(encoding="utf-8"))

            self.assertEqual(result.original_train_images, 1)
            self.assertEqual(result.augmented_train_images, 1)
            self.assertTrue((output_dir / "images" / "aug_object_flip_train.jpg").is_file())
            augmented_ann = next(item for item in train["annotations"] if item["image_id"] != 1)
            self.assertEqual(augmented_ann["bbox"], [10, 5, 20, 15])

    def test_build_augmented_dataset_only_extends_train_split(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_paddlex_dataset(root)
            output_dir = root / "paddlex_aug"

            result = module.build_augmented_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                augmentations=["flip"],
                variants_per_image=1,
                hard_case_labels=[],
                hard_case_extra=0,
                seed=1,
            )

            train = json.loads((output_dir / "annotations" / "instance_train.json").read_text(encoding="utf-8"))
            val = json.loads((output_dir / "annotations" / "instance_val.json").read_text(encoding="utf-8"))
            test = json.loads((output_dir / "annotations" / "instance_test.json").read_text(encoding="utf-8"))

            self.assertEqual(result.original_train_images, 1)
            self.assertEqual(result.augmented_train_images, 1)
            self.assertEqual(len(train["images"]), 2)
            self.assertEqual(len(train["annotations"]), 2)
            self.assertEqual(len(val["images"]), 1)
            self.assertEqual(len(test["images"]), 1)
            self.assertTrue((output_dir / "images" / "train.jpg").is_file())
            self.assertTrue((output_dir / "images" / "aug_flip_train.jpg").is_file())
            augmented_ann = next(item for item in train["annotations"] if item["image_id"] != 1)
            self.assertEqual(augmented_ann["bbox"], [70, 5, 20, 15])
            self.assertEqual((output_dir / "class_names.txt").read_text(encoding="utf-8"), "water\ncargo\n")
            meta = json.loads((output_dir / "augmentation_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["policy"], "target_det_train_only_offline_augmentation")
            self.assertEqual(meta["augmentations"], ["flip"])
            self.assertEqual(meta["augmented_train_images"], 1)

    def test_focus_labels_receive_additional_variants(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_paddlex_dataset(root)
            output_dir = root / "paddlex_aug"

            result = module.build_augmented_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                augmentations=["exposure"],
                variants_per_image=0,
                hard_case_labels=[],
                hard_case_extra=0,
                focus_labels=["water"],
                focus_extra=2,
                seed=1,
            )

            self.assertEqual(result.augmented_train_images, 2)
            meta = json.loads((output_dir / "augmentation_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["focus_labels"], ["water"])
            self.assertEqual(meta["focus_extra"], 2)

    def test_focus_augmentations_override_general_policy_for_focus_variants(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_paddlex_dataset(root)
            output_dir = root / "paddlex_aug"

            result = module.build_augmented_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                augmentations=["flip"],
                variants_per_image=1,
                hard_case_labels=[],
                hard_case_extra=0,
                focus_labels=["water"],
                focus_extra=1,
                focus_augmentations=["overexposure"],
                seed=1,
            )

            self.assertEqual(result.augmented_train_images, 2)
            self.assertTrue((output_dir / "images" / "aug_flip_train.jpg").is_file())
            overexposed_path = output_dir / "images" / "aug_overexposure_train.jpg"
            self.assertTrue(overexposed_path.is_file())
            meta = json.loads((output_dir / "augmentation_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["focus_augmentations"], ["overexposure"])

    def test_zoomout_focus_is_independent_from_existing_focus_policy(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = make_zoomout_focus_dataset(root)
            output_dir = root / "paddlex_aug"

            result = module.build_augmented_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                augmentations=["exposure"],
                variants_per_image=0,
                hard_case_labels=[],
                hard_case_extra=0,
                focus_labels=["water"],
                focus_extra=1,
                focus_augmentations=["overexposure"],
                zoomout_labels=["name"],
                zoomout_extra=2,
                seed=1,
            )

            self.assertEqual(result.original_train_images, 2)
            self.assertEqual(result.augmented_train_images, 3)
            self.assertTrue((output_dir / "images" / "aug_overexposure_train.jpg").is_file())
            self.assertTrue((output_dir / "images" / "aug_zoomout_name.jpg").is_file())
            self.assertTrue((output_dir / "images" / "aug_zoomout_name_2.jpg").is_file())
            self.assertFalse((output_dir / "images" / "aug_zoomout_train.jpg").exists())
            meta = json.loads((output_dir / "augmentation_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["zoomout_labels"], ["name"])
            self.assertEqual(meta["zoomout_extra"], 2)
            self.assertEqual(meta["val_images"], 1)
            self.assertEqual(meta["test_images"], 1)

    def test_overexposure_augmentation_reaches_strong_brightness(self) -> None:
        module = load_module()
        image = Image.new("RGB", (20, 20), (100, 100, 100))

        augmented, transform_bbox = module._apply_augmentation(image, "overexposure", module.random.Random(1))

        self.assertGreater(augmented.getpixel((10, 10))[0], 160)
        self.assertLess(augmented.getpixel((10, 10))[0], 245)
        self.assertEqual(transform_bbox([1, 2, 3, 4], 20, 20), [1, 2, 3, 4])

    def test_overexposure_preserves_highlight_contrast(self) -> None:
        module = load_module()
        image = Image.new("RGB", (2, 1))
        image.putpixel((0, 0), (170, 170, 170))
        image.putpixel((1, 0), (210, 210, 210))

        augmented, _ = module._apply_augmentation(image, "overexposure", module.random.Random(1))

        darker = augmented.getpixel((0, 0))[0]
        brighter = augmented.getpixel((1, 0))[0]
        self.assertGreater(brighter - darker, 5)

    def test_underexposure_reaches_dark_domain_without_crushing_to_black(self) -> None:
        module = load_module()
        image = Image.new("RGB", (20, 20), (160, 140, 120))

        augmented, transform_bbox = module._apply_augmentation(image, "underexposure", module.random.Random(3))

        red, green, blue = augmented.getpixel((10, 10))
        self.assertLess(max(red, green, blue), 100)
        self.assertGreater(min(red, green, blue), 20)
        self.assertEqual(transform_bbox([1, 2, 3, 4], 20, 20), [1, 2, 3, 4])

    def test_mixed_lighting_creates_dark_and_bright_regions(self) -> None:
        module = load_module()
        image = Image.new("RGB", (120, 80), (100, 100, 100))

        augmented, transform_bbox = module._apply_augmentation(image, "mixed_lighting", module.random.Random(5))

        minimum, maximum = augmented.convert("L").getextrema()
        self.assertLess(minimum, 80)
        self.assertGreater(maximum, 130)
        self.assertEqual(augmented.size, image.size)
        self.assertEqual(transform_bbox([1, 2, 3, 4], 20, 20), [1, 2, 3, 4])

    def test_lowres_augmentation_preserves_image_and_bbox_size(self) -> None:
        module = load_module()
        image = Image.new("RGB", (100, 60), (30, 80, 120))

        augmented, transform_bbox = module._apply_augmentation(image, "lowres", module.random.Random(1))

        self.assertEqual(augmented.size, image.size)
        self.assertEqual(transform_bbox([10, 5, 20, 15], 100, 60), [10, 5, 20, 15])

    def test_rejects_output_dir_equal_source_dir(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = make_paddlex_dataset(Path(temp_dir))

            with self.assertRaisesRegex(ValueError, "must be different"):
                module.build_augmented_dataset(source_dir=source_dir, output_dir=source_dir)


if __name__ == "__main__":
    unittest.main()
