from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "detection" / "merge_target_det_coco_datasets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("merge_target_det_coco_datasets", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MergeTargetDetCocoDatasetsTest(unittest.TestCase):
    def test_fusion_class_order_keeps_26_classes_and_appends_green_pepper(self) -> None:
        module = load_module()
        expected = (
            "water_l3",
            "water_l2",
            "water_l1",
            "water",
            "order",
            "cylinder_set",
            "cylinder_3",
            "cylinder_2",
            "cylinder_1",
            "ball_yellow",
            "ball_blue",
            "animal",
            "name",
            "danyuan_2",
            "danyuan_1",
            "storage",
            "lable_yellow",
            "lable_blue",
            "rape",
            "broccoli",
            "potato",
            "celery",
            "mushroom",
            "flammulina velutipes",
            "tomato",
            "green bean",
            "green pepper",
        )

        self.assertEqual(module.FUSION_CLASS_NAMES, expected)

    def test_drop_empty_categories_keeps_only_annotated_classes_in_contract_order(self) -> None:
        module = load_module()
        annotated_names = set(module.FUSION_CLASS_NAMES) - {"storage"}

        class_names = module._select_output_class_names(
            annotated_names,
            drop_empty_categories=True,
        )

        self.assertEqual(len(class_names), 26)
        self.assertNotIn("storage", class_names)
        self.assertEqual(class_names[:5], list(module.FUSION_CLASS_NAMES[:5]))
        self.assertEqual(class_names[-1], "green pepper")

    def test_category_remap_is_contiguous_after_empty_class_is_removed(self) -> None:
        module = load_module()
        class_names = [name for name in module.FUSION_CLASS_NAMES if name != "storage"]
        annotations = [
            {"id": 1, "category_id": module.FUSION_CLASS_NAMES.index("danyuan_1") + 1},
            {"id": 2, "category_id": module.FUSION_CLASS_NAMES.index("lable_yellow") + 1},
            {"id": 3, "category_id": module.FUSION_CLASS_NAMES.index("green pepper") + 1},
        ]

        module._remap_annotation_category_ids(annotations, class_names)

        self.assertEqual([annotation["category_id"] for annotation in annotations], [15, 16, 26])

    def test_canonical_class_name_accepts_new_underscore_labels(self) -> None:
        module = load_module()

        self.assertEqual(module._canonical_class_name("green_bean"), "green bean")
        self.assertEqual(module._canonical_class_name("flammulina_velutipes"), "flammulina velutipes")
        self.assertEqual(module._canonical_class_name("green_pepper"), "green pepper")
        self.assertEqual(module._canonical_class_name("h_qing_jiao"), "green pepper")

    def test_validate_categories_returns_canonical_aliases(self) -> None:
        module = load_module()
        payload = {
            "categories": [
                {"id": 1, "name": "green_bean"},
                {"id": 2, "name": "flammulina_velutipes"},
                {"id": 3, "name": "green_pepper"},
            ]
        }

        self.assertEqual(
            module._validate_categories(payload, Path("annotations.json")),
                {1: "green bean", 2: "flammulina velutipes", 3: "green pepper"},
        )

    def test_grouped_split_keeps_contiguous_groups_together_and_has_independent_test(self) -> None:
        module = load_module()
        images = [
            {
                "id": image_id,
                "file_name": f"{image_id:04d}.jpg",
                "width": 640,
                "height": 480,
                "_split_group": f"group-{(image_id - 1) // 2}",
            }
            for image_id in range(1, 13)
        ]
        annotations = [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1 + ((image_id - 1) % 3),
                "bbox": [10, 10, 20, 20],
            }
            for image_id in range(1, 13)
        ]

        train_ids, val_ids, test_ids = module._split_image_ids(
            images=images,
            annotations=annotations,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=20260722,
        )

        self.assertTrue(train_ids)
        self.assertTrue(val_ids)
        self.assertTrue(test_ids)
        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, set(range(1, 13)))
        for first_id in range(1, 13, 2):
            memberships = [
                {first_id, first_id + 1}.issubset(split_ids)
                for split_ids in (train_ids, val_ids, test_ids)
            ]
            self.assertEqual(sum(memberships), 1)

    def test_grouped_split_does_not_let_dense_common_classes_hide_rare_classes(self) -> None:
        module = load_module()
        images = []
        annotations = []
        annotation_id = 1
        for group_index in range(20):
            for offset in range(10):
                image_id = group_index * 10 + offset + 1
                images.append(
                    {
                        "id": image_id,
                        "file_name": f"{image_id:04d}.jpg",
                        "width": 10,
                        "height": 10,
                        "_split_group": f"group-{group_index}",
                    }
                )
                common_repeats = 10 if group_index < 16 else 1
                for _ in range(common_repeats):
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": [0, 0, 1, 1],
                        }
                    )
                    annotation_id += 1
                if group_index >= 16:
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 2,
                            "bbox": [1, 1, 1, 1],
                        }
                    )
                    annotation_id += 1

        _, val_ids, test_ids = module._split_image_ids(
            images=images,
            annotations=annotations,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=20260731,
        )

        val_rare = sum(
            1
            for annotation in annotations
            if annotation["category_id"] == 2 and annotation["image_id"] in val_ids
        )
        test_rare = sum(
            1
            for annotation in annotations
            if annotation["category_id"] == 2 and annotation["image_id"] in test_ids
        )
        self.assertGreater(val_rare, 0)
        self.assertGreater(test_rare, 0)

    def test_build_split_payload_does_not_leak_internal_group_metadata(self) -> None:
        module = load_module()
        payload = module._build_split_payload(
            images=[
                {
                    "id": 1,
                    "file_name": "0001.jpg",
                    "width": 640,
                    "height": 480,
                    "_split_group": "source-0-group-0",
                }
            ],
            annotations=[],
            categories=[{"id": 1, "name": "water_l3"}],
            image_ids={1},
            split_name="train",
        )

        self.assertNotIn("_split_group", payload["images"][0])
        self.assertIn("1cls", payload["info"]["description"])


if __name__ == "__main__":
    unittest.main()
