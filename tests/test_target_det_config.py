from __future__ import annotations

import unittest
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATH = CONFIG_DIR / "target_det.yaml"
HQ_STAGE1_CONFIG_PATH = CONFIG_DIR / "target_det_hq_stage1_baseline.yaml"
FAST_CONFIG_PATH = CONFIG_DIR / "target_det_fast_ppyoloe_plus_s_416.yaml"
FUSION_27CLS_FORMAL_CONFIG_PATH = CONFIG_DIR / "target_det_fusion_27cls_formal_ppyoloe_plus_s_416.yaml"
FUSION_27CLS_CONFUSION_FOCUS_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_27cls_confusion_focus_finetune_ppyoloe_plus_s_416.yaml"
)
FUSION_27CLS_NAME_ZOOMOUT_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_27cls_name_zoomout_finetune_ppyoloe_plus_s_416.yaml"
)
FUSION_26CLS_120E_BASELINE_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_0619_name_order_storage_shucai0622_ppyoloe_plus_s_416_26cls.yaml"
)
FUSION_20260731_26CLS_LIGHT_ROBUST_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_20260731_26cls_light_robust_ppyoloe_plus_s_416_80e.yaml"
)
FUSION_20260731_26CLS_LIGHT_ROBUST_120E_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_20260731_26cls_light_robust_ppyoloe_plus_s_416_120e.yaml"
)
FUSION_20260731_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_20260731_26cls_balanced_light_robust_v2_ppyoloe_plus_s_416_80e.yaml"
)
FUSION_20260814_27CLS_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH = (
    CONFIG_DIR / "target_det_fusion_20260814_27cls_balanced_light_robust_v2_ppyoloe_plus_s_416_80e.yaml"
)


def config_without_comments(path: Path) -> str:
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


class TargetDetConfigTest(unittest.TestCase):
    def test_target_det_config_tracks_default_ppyoloe_plus_s_416_recipe(self) -> None:
        text = CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 10", text)
        self.assertIn("architecture: YOLOv3", text)
        self.assertIn("PPYOLOEHead", text)
        self.assertIn("depth_mult: 0.33", text)
        self.assertIn("width_mult: 0.50", text)
        self.assertIn("epoch: 80", text)
        self.assertIn("dataset_dir: ./datasets/01_target_det/PLEASE_PASS_DATASET_DIR", text)
        self.assertIn("save_dir: ./outputs/PLEASE_PASS_OUTPUT_DIR_WITH_TRAIN_SCRIPT", text)
        self.assertIn("anno_path: annotations/instance_train.json", text)
        self.assertIn("eval_height: &eval_height 416", text)
        self.assertIn("eval_width: &eval_width 416", text)
        self.assertIn("pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams", text)

    def test_high_quality_stage1_config_keeps_recipe_but_changes_training_schedule(self) -> None:
        text = HQ_STAGE1_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 15", text)
        self.assertIn("architecture: YOLOv3", text)
        self.assertIn("PPYOLOEHead", text)
        self.assertIn("epoch: 100", text)
        self.assertIn("base_lr: 0.00075", text)
        self.assertIn("max_epochs: 100", text)
        self.assertIn("save_dir: ./outputs/target_det_hq_ppyoloe_l_stage1_baseline", text)
        self.assertIn("weights: ./outputs/target_det_hq_ppyoloe_l_stage1_baseline/model_final", text)
        self.assertIn("target_size: [416, 448, 480, 512, 544, 576, 608, 640, 672, 704]", text)

    def test_fast_config_uses_small_ppyoloe_recipe_and_416_eval_size(self) -> None:
        text = FAST_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 15", text)
        self.assertIn("architecture: YOLOv3", text)
        self.assertIn("PPYOLOEHead", text)
        self.assertIn("depth_mult: 0.33", text)
        self.assertIn("width_mult: 0.50", text)
        self.assertIn("eval_height: &eval_height 416", text)
        self.assertIn("eval_width: &eval_width 416", text)
        self.assertIn("target_size: [320, 352, 384, 416, 448, 480, 512]", text)
        self.assertIn("save_dir: ./outputs/target_det_fast_ppyoloe_plus_crn_s_416", text)
        self.assertIn("pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams", text)

    def test_fusion_27cls_formal_config_uses_augmented_finetune_recipe(self) -> None:
        text = FUSION_27CLS_FORMAL_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 27", text)
        self.assertIn("epoch: 50", text)
        self.assertIn("snapshot_epoch: 2", text)
        self.assertIn("base_lr: 0.0002", text)
        self.assertIn("max_epochs: 50", text)
        self.assertIn("epochs: 2", text)
        self.assertIn(
            "pretrain_weights: ./outputs/01_target_det/ppyoloe_plus_s_416_27cls_emergency_20260722/train_5e_raw_smoke/best_model.pdparams",
            text,
        )
        self.assertIn("static_assigner_epoch: 0", text)
        self.assertIn("target_size: [384, 416, 448]", text)
        self.assertIn("random_size: true", text)
        self.assertIn("eval_height: &eval_height 416", text)
        self.assertIn("eval_width: &eval_width 416", text)
        self.assertIn("anno_path: annotations/instance_test.json", text)
        self.assertNotIn("RandomFlip", text)

    def test_confusion_focus_config_keeps_contract_and_uses_low_lr_epoch13_init(self) -> None:
        text = FUSION_27CLS_CONFUSION_FOCUS_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 27", text)
        self.assertIn("epoch: 10", text)
        self.assertIn("snapshot_epoch: 1", text)
        self.assertIn("base_lr: 0.00005", text)
        self.assertIn("max_epochs: 10", text)
        self.assertIn("epochs: 1", text)
        self.assertIn(
            "pretrain_weights: ./outputs/01_target_det/ppyoloe_plus_s_416_27cls_formal_aug_20260723/13.pdparams",
            text,
        )
        self.assertIn("target_size: [384, 416, 448]", text)
        self.assertIn("eval_height: &eval_height 416", text)
        self.assertIn("eval_width: &eval_width 416", text)
        self.assertIn("post_process: true", text)
        self.assertIn("nms: true", text)

    def test_name_zoomout_config_keeps_contract_and_uses_hardfix_init(self) -> None:
        text = FUSION_27CLS_NAME_ZOOMOUT_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 27", text)
        self.assertIn("epoch: 4", text)
        self.assertIn("snapshot_epoch: 1", text)
        self.assertIn("base_lr: 0.00001", text)
        self.assertIn("max_epochs: 4", text)
        self.assertIn("epochs: 1", text)
        self.assertIn(
            "pretrain_weights: ./outputs/01_target_det/ppyoloe_plus_s_416_27cls_confusion_overexposure_20260726/interpolation/finetune1_finetune2_alpha75.pdparams",
            text,
        )
        self.assertIn("target_size: [384, 416, 448]", text)
        self.assertIn("eval_height: &eval_height 416", text)
        self.assertIn("eval_width: &eval_width 416", text)
        self.assertIn("post_process: true", text)
        self.assertIn("nms: true", text)
        self.assertNotIn("RandomFlip", text)

    def test_20260731_light_robust_config_only_shortens_baseline_schedule_to_80_epochs(self) -> None:
        baseline = config_without_comments(FUSION_26CLS_120E_BASELINE_CONFIG_PATH)
        expected = baseline.replace("epoch: 120", "epoch: 80").replace("max_epochs: 120", "max_epochs: 80")

        candidate = config_without_comments(FUSION_20260731_26CLS_LIGHT_ROBUST_CONFIG_PATH)

        self.assertEqual(candidate, expected)

    def test_20260731_light_robust_120e_config_restores_the_full_baseline_schedule(self) -> None:
        baseline = config_without_comments(FUSION_26CLS_120E_BASELINE_CONFIG_PATH)

        candidate = config_without_comments(FUSION_20260731_26CLS_LIGHT_ROBUST_120E_CONFIG_PATH)

        self.assertEqual(candidate, baseline)

    def test_20260731_balanced_light_robust_v2_config(self) -> None:
        self.assertTrue(
            FUSION_20260731_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH.is_file(),
            f"Missing balanced light-robust V2 config: {FUSION_20260731_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH}",
        )
        text = FUSION_20260731_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH.read_text(encoding="utf-8")

        self.assertIn("num_classes: 26", text)
        self.assertIn("epoch: 80", text)
        self.assertIn("max_epochs: 80", text)
        self.assertIn("base_lr: 0.001", text)
        self.assertIn("min_lr_ratio: 0.02", text)
        self.assertIn("last_plateau_epochs: 4", text)
        self.assertIn("clip_grad_by_norm: 10.0", text)
        self.assertIn("static_assigner_epoch: 20", text)
        self.assertIn("snapshot_epoch: 2", text)
        self.assertIn("worker_num: 2", text)
        self.assertIn("use_shared_memory: false", text)
        self.assertIn("batch_size: 16", text)
        self.assertIn("eval_height: &eval_height 416", text)
        self.assertIn("eval_width: &eval_width 416", text)
        self.assertIn("PPYOLOEHead", text)
        self.assertIn(
            "pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_s_obj365_pretrained.pdparams",
            text,
        )

    def test_20260814_27cls_balanced_v2_only_expands_the_class_head(self) -> None:
        baseline = config_without_comments(FUSION_20260731_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH)
        expected = baseline.replace("num_classes: 26", "num_classes: 27")

        candidate = config_without_comments(FUSION_20260814_27CLS_BALANCED_LIGHT_ROBUST_V2_CONFIG_PATH)

        self.assertEqual(candidate, expected)


if __name__ == "__main__":
    unittest.main()
