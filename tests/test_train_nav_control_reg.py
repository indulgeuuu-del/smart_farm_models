# 功能：测试 08_nav_control_optional 的 PaddleLane_v30 风格训练脚本。
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import paddle
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nav_control_optional" / "train_nav_control_reg.py"
SPEC = importlib.util.spec_from_file_location("train_nav_control_reg", SCRIPT_PATH)
train_nav_control_reg = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = train_nav_control_reg
SPEC.loader.exec_module(train_nav_control_reg)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, payload: list[object]) -> None:
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n", encoding="utf-8")


def make_image_set(root: Path, name: str, count: int) -> Path:
    source_dir = root / "raw" / name
    source_dir.mkdir(parents=True)
    records = []
    for index in range(count):
        image_name = f"{index:04d}.jpg"
        Image.new("RGB", (32, 24), (index * 30 % 255, 20, 100)).save(source_dir / image_name)
        records.append({"img_path": image_name, "state": [0.15, index / 10, -index / 20]})
    write_json(source_dir / "data.json", records)
    return source_dir


def make_deviation_jsonl_set(root: Path, name: str, deviations: list[float]) -> Path:
    source_dir = root / name
    image_dir = source_dir / "images"
    image_dir.mkdir(parents=True)
    records = []
    for index, deviation in enumerate(deviations):
        image_name = f"{index:04d}.jpg"
        Image.new("RGB", (32, 24), (index * 40 % 255, 40, 120)).save(image_dir / image_name)
        records.append({"img_path": f"images/{image_name}", "deviation": deviation, "timestamp": float(index)})
    write_jsonl(source_dir / "data.jsonl", records)
    return source_dir


def make_dataset(root: Path) -> Path:
    dataset_dir = root / "datasets" / "08_nav_control_optional"
    make_image_set(dataset_dir, "image_set_r", 2)
    make_image_set(dataset_dir, "image_set_l", 3)
    make_image_set(dataset_dir, "image_set1208", 1)
    return dataset_dir


class TrainNavControlRegTest(unittest.TestCase):
    def test_default_dataset_dir_uses_final_optional_numbering(self) -> None:
        args = train_nav_control_reg.build_parser().parse_args([])

        self.assertEqual(train_nav_control_reg.DEFAULT_DATASET_DIR, Path("datasets/08_nav_control_optional"))
        self.assertEqual(Path(args.dataset_dir), Path("datasets/08_nav_control_optional"))

    def test_default_sources_follow_paddlelane_v30_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = make_dataset(Path(temp_dir))

            train_sources, eval_sources = train_nav_control_reg.resolve_data_sources(dataset_dir)

            self.assertEqual([source.name for source in train_sources], ["image_set_r", "image_set_l"])
            self.assertEqual([source.name for source in eval_sources], ["image_set1208"])

    def test_loads_train_samples_without_sources_manifest_or_split_txt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = make_dataset(Path(temp_dir))

            samples = train_nav_control_reg.load_samples(dataset_dir, split="train")

            self.assertEqual(len(samples), 5)
            self.assertEqual(samples[0].source_name, "image_set_r")
            self.assertEqual(samples[-1].source_name, "image_set_l")
            self.assertEqual(samples[0].label, (0.0, 0.0))

    def test_loads_deviation_jsonl_as_primary_deviation_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets" / "08_nav_control_optional"
            train_source = make_deviation_jsonl_set(root, "lane_train", [0.0, -0.3, 0.5])
            eval_source = make_deviation_jsonl_set(root, "lane_val", [0.2])

            samples = train_nav_control_reg.load_samples(
                dataset_dir,
                split="train",
                train_sources=(str(train_source),),
                eval_sources=(str(eval_source),),
            )

            self.assertEqual(len(samples), 3)
            self.assertEqual(samples[1].source_name, "lane_train")
            self.assertEqual(samples[1].image_rel, "images/0001.jpg")
            self.assertEqual(samples[1].label, (-0.3, 0.0))

    def test_nonzero_deviation_gets_higher_training_weight(self) -> None:
        samples = [
            train_nav_control_reg.NavSample("unit", "0.jpg", Path("0.jpg"), (0.0, 0.0)),
            train_nav_control_reg.NavSample("unit", "1.jpg", Path("1.jpg"), (0.6, 0.0)),
        ]

        weights = train_nav_control_reg.build_deviation_sample_weights(samples)

        self.assertGreater(weights[1], weights[0])
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_signed_bin_sample_weights_upweight_rare_bins(self) -> None:
        samples = [
            train_nav_control_reg.NavSample("unit", f"{index}.jpg", Path(f"{index}.jpg"), (0.0, 0.0))
            for index in range(4)
        ]
        samples.append(train_nav_control_reg.NavSample("unit", "rare.jpg", Path("rare.jpg"), (-0.7, 0.0)))

        weights = train_nav_control_reg.build_deviation_sample_weights(samples, mode="signed_bins")

        self.assertGreater(float(weights[-1]), float(weights[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_filter_samples_by_label_jump_drops_boundary_neighbors(self) -> None:
        samples = [
            train_nav_control_reg.NavSample("unit", f"{index}.jpg", Path(f"{index}.jpg"), (value, 0.0))
            for index, value in enumerate([0.0, 0.0, 0.9, 0.9, 0.0])
        ]

        kept, summary = train_nav_control_reg.filter_samples_by_label_jump(samples, threshold=0.8, radius=0)

        self.assertEqual([sample.image_rel for sample in kept], ["0.jpg"])
        self.assertEqual(summary["dropped_samples"], 4)
        self.assertEqual(summary["sources"][0]["jump_count"], 2)

    def test_batch_deviation_weights_keep_exact_zero_low(self) -> None:
        labels = paddle.to_tensor([[0.0], [0.05], [0.2], [0.7]], dtype="float32")

        weights = train_nav_control_reg._build_deviation_batch_weights(labels).numpy().reshape(-1)

        self.assertLess(float(weights[0]), float(weights[1]))
        self.assertLess(float(weights[1]), float(weights[2]))
        self.assertLess(float(weights[2]), float(weights[3]))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_forward_check_accepts_direct_deviation_jsonl_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets" / "08_nav_control_optional"
            train_source = make_deviation_jsonl_set(root, "lane_train", [0.0, -0.3])
            eval_source = make_deviation_jsonl_set(root, "lane_val", [0.2, 0.4])

            result = train_nav_control_reg.run_forward_check(
                dataset_dir,
                split="eval",
                batch_size=2,
                device="cpu",
                train_sources=(str(train_source),),
                eval_sources=(str(eval_source),),
            )

            self.assertEqual(result["samples"], 2)
            self.assertEqual(result["label_shape"], [2, 2])
            self.assertEqual(result["pred_shape"], [2, 2])

    def test_hflip_augmentation_negates_label_like_paddlelane_v30(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = make_dataset(Path(temp_dir))
            samples = train_nav_control_reg.load_samples(dataset_dir, split="train")
            dataset = train_nav_control_reg.PaddleLaneDataset(samples, training=True, forced_augmentation="hflip")

            _, label = dataset[1]

            np.testing.assert_allclose(label, np.array([0.05, 0.0], dtype=np.float32))

    def test_robust_augmentations_preserve_shape_and_label(self) -> None:
        image = np.full((24, 32, 3), 128.0, dtype=np.float32)
        label = np.array([0.4, 0.0], dtype=np.float32)

        for augmentation in ("none", "blur", "noise", "shadow"):
            augmented_image, augmented_label = train_nav_control_reg._augment_like_paddlelane(
                image.copy(),
                label.copy(),
                forced_augmentation=augmentation,
                augmentation_profile="robust",
            )
            self.assertEqual(augmented_image.shape, image.shape)
            self.assertEqual(augmented_image.dtype, np.float32)
            np.testing.assert_allclose(augmented_label, label)

    def test_robust_v2_hflip_negates_deviation(self) -> None:
        image = np.arange(24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3) % 255
        label = np.array([0.4, 0.0], dtype=np.float32)

        augmented_image, augmented_label = train_nav_control_reg._augment_like_paddlelane(
            image.copy(),
            label.copy(),
            forced_augmentation="robust_v2_hflip",
            augmentation_profile="robust_v2",
        )

        np.testing.assert_allclose(augmented_image, np.flip(image, axis=1))
        np.testing.assert_allclose(augmented_label, [-0.4, 0.0])

    def test_robust_v2_brightness_has_meaningful_bounded_effect(self) -> None:
        np.random.seed(20260418)
        image = np.full((24, 32, 3), 100.0, dtype=np.float32)
        label = np.array([0.4, 0.0], dtype=np.float32)

        augmented_image, augmented_label = train_nav_control_reg._augment_like_paddlelane(
            image.copy(),
            label.copy(),
            forced_augmentation="robust_v2_brightness",
            augmentation_profile="robust_v2",
        )

        self.assertGreater(abs(float(augmented_image.mean()) - 100.0), 2.0)
        self.assertGreaterEqual(float(augmented_image.min()), 0.0)
        self.assertLessEqual(float(augmented_image.max()), 255.0)
        self.assertEqual(augmented_image.dtype, np.float32)
        np.testing.assert_allclose(augmented_label, label)

    def test_light_corner_v3_forced_augmentations_are_bounded_and_label_safe(self) -> None:
        image = np.full((24, 32, 3), 100.0, dtype=np.float32)
        label = np.array([0.4, 0.0], dtype=np.float32)

        for augmentation in (
            "light_corner_v3_exposure",
            "light_corner_v3_gamma",
            "light_corner_v3_contrast",
            "light_corner_v3_temperature",
            "light_corner_v3_shadow",
            "light_corner_v3_glare",
            "light_corner_v3_blur",
            "light_corner_v3_noise",
        ):
            np.random.seed(20260807)
            augmented_image, augmented_label = train_nav_control_reg._augment_like_paddlelane(
                image.copy(),
                label.copy(),
                forced_augmentation=augmentation,
                augmentation_profile="light_corner_v3",
            )

            self.assertEqual(augmented_image.shape, image.shape)
            self.assertEqual(augmented_image.dtype, np.float32)
            self.assertGreaterEqual(float(augmented_image.min()), 0.0)
            self.assertLessEqual(float(augmented_image.max()), 255.0)
            np.testing.assert_allclose(augmented_label, label)

    def test_light_corner_v3_hflip_negates_deviation(self) -> None:
        image = np.arange(24 * 32 * 3, dtype=np.float32).reshape(24, 32, 3) % 255
        label = np.array([0.4, 0.0], dtype=np.float32)

        augmented_image, augmented_label = train_nav_control_reg._augment_like_paddlelane(
            image.copy(),
            label.copy(),
            forced_augmentation="light_corner_v3_hflip",
            augmentation_profile="light_corner_v3",
        )

        np.testing.assert_allclose(augmented_image, np.flip(image, axis=1))
        np.testing.assert_allclose(augmented_label, [-0.4, 0.0])

    def test_moderate_deviation_weights_use_bounded_ratios(self) -> None:
        labels = paddle.to_tensor([[0.0], [0.2], [0.4], [0.8]], dtype="float32")

        weights = train_nav_control_reg._build_deviation_batch_weights(labels, mode="moderate_abs_bins").numpy().reshape(-1)

        expected_raw = np.array([0.75, 1.0, 1.25, 1.5], dtype=np.float32)
        expected = expected_raw / expected_raw.mean()
        np.testing.assert_allclose(weights, expected, rtol=1e-6)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_corner_focus_deviation_weights_emphasize_turns_without_extreme_ratio(self) -> None:
        labels = paddle.to_tensor([[0.0], [0.2], [0.4], [0.8]], dtype="float32")

        weights = train_nav_control_reg._build_deviation_batch_weights(
            labels,
            mode="corner_focus_v1",
        ).numpy().reshape(-1)

        expected_raw = np.array([0.75, 1.25, 1.5, 1.75], dtype=np.float32)
        expected = expected_raw / expected_raw.mean()
        np.testing.assert_allclose(weights, expected, rtol=1e-6)
        self.assertLess(float(weights[-1] / weights[0]), 2.5)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)

    def test_parser_accepts_balanced_v3_light_corner_contract(self) -> None:
        args = train_nav_control_reg.build_parser().parse_args(
            [
                "--preset",
                "balanced_v3_light_corner",
                "--augmentation-profile",
                "light_corner_v3",
                "--deviation-loss-weight-mode",
                "corner_focus_v1",
            ]
        )

        config = train_nav_control_reg.config_from_args(args)

        self.assertEqual(config.preset, "balanced_v3_light_corner")
        self.assertEqual(config.augmentation_profile, "light_corner_v3")
        self.assertEqual(config.deviation_loss_weight_mode, "corner_focus_v1")

    def test_model_forward_shape_matches_two_value_regression(self) -> None:
        paddle.device.set_device("cpu")
        model = train_nav_control_reg.PaddleLaneCnnModel()
        x = paddle.zeros([2, 3, 128, 128], dtype="float32")

        y = model(x)

        self.assertEqual(list(y.shape), [2, 2])

    def test_forward_check_can_use_eval_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = make_dataset(Path(temp_dir))

            result = train_nav_control_reg.run_forward_check(dataset_dir, split="eval", batch_size=1, device="cpu")

            self.assertEqual(result["split"], "eval")
            self.assertEqual(result["samples"], 1)
            self.assertEqual(result["input_shape"], [1, 3, 128, 128])
            self.assertEqual(result["label_shape"], [1, 2])
            self.assertEqual(result["pred_shape"], [1, 2])

    def test_forward_check_honors_custom_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = make_dataset(Path(temp_dir))
            make_image_set(dataset_dir, "official_train", 4)
            make_image_set(dataset_dir, "official_eval", 2)

            train_result = train_nav_control_reg.run_forward_check(
                dataset_dir,
                split="train",
                batch_size=2,
                device="cpu",
                train_sources=("official_train",),
                eval_sources=("official_eval",),
            )
            eval_result = train_nav_control_reg.run_forward_check(
                dataset_dir,
                split="eval",
                batch_size=2,
                device="cpu",
                train_sources=("official_train",),
                eval_sources=("official_eval",),
            )

            self.assertEqual(train_result["samples"], 4)
            self.assertEqual(eval_result["samples"], 2)

    def test_label_stats_build_normalized_loss_weights_from_train_samples(self) -> None:
        samples = [
            train_nav_control_reg.NavSample("unit", "0.jpg", Path("0.jpg"), (0.0, 0.0)),
            train_nav_control_reg.NavSample("unit", "1.jpg", Path("1.jpg"), (2.0, 10.0)),
        ]

        stats = train_nav_control_reg.build_label_stats(samples, normalize_loss=True)

        np.testing.assert_allclose(stats["mean"], [1.0, 5.0], rtol=1e-6)
        np.testing.assert_allclose(stats["std"], [1.0, 5.0], rtol=1e-6)
        np.testing.assert_allclose(stats["loss_weights"], [1.6666666, 0.3333333], rtol=1e-6)

    def test_label_stats_build_sqrt_inverse_std_loss_weights(self) -> None:
        samples = [
            train_nav_control_reg.NavSample("unit", "0.jpg", Path("0.jpg"), (0.0, 0.0)),
            train_nav_control_reg.NavSample("unit", "1.jpg", Path("1.jpg"), (2.0, 8.0)),
        ]

        stats = train_nav_control_reg.build_label_stats(samples, loss_weight_mode="sqrt_inverse_std")

        self.assertEqual(stats["loss_weight_mode"], "sqrt_inverse_std")
        np.testing.assert_allclose(stats["std"], [1.0, 4.0], rtol=1e-6)
        np.testing.assert_allclose(stats["loss_weights"], [1.3333333, 0.6666666], rtol=1e-6)

    def test_config_from_args_accepts_label_loss_weight_mode(self) -> None:
        args = train_nav_control_reg.build_parser().parse_args(["--label-loss-weight-mode", "sqrt_inverse_std"])

        config = train_nav_control_reg.config_from_args(args)

        self.assertEqual(config.label_loss_weight_mode, "sqrt_inverse_std")
        self.assertFalse(config.normalize_label_loss)

    def test_config_from_args_accepts_sampling_and_jump_filter_options(self) -> None:
        args = train_nav_control_reg.build_parser().parse_args(
            [
                "--sample-weight-mode",
                "signed_bins",
                "--deviation-loss-weight-mode",
                "moderate_abs_bins",
                "--augmentation-profile",
                "robust",
                "--lr-scheduler",
                "piecewise",
                "--lr-boundaries",
                "100,400",
                "--lr-values",
                "0.001,0.0001,0.00001",
                "--label-jump-drop-threshold",
                "0.8",
                "--label-jump-drop-radius",
                "2",
                "--init-params-path",
                "outputs/nav/best/cnn_lane.pdparams",
            ]
        )

        config = train_nav_control_reg.config_from_args(args)

        self.assertEqual(config.sample_weight_mode, "signed_bins")
        self.assertEqual(config.deviation_loss_weight_mode, "moderate_abs_bins")
        self.assertEqual(config.augmentation_profile, "robust")
        self.assertEqual(config.lr_scheduler, "piecewise")
        self.assertEqual(config.lr_boundaries, (100, 400))
        self.assertEqual(config.lr_values, (0.001, 0.0001, 0.00001))
        self.assertEqual(config.label_jump_drop_threshold, 0.8)
        self.assertEqual(config.label_jump_drop_radius, 2)
        self.assertEqual(config.init_params_path, Path("outputs/nav/best/cnn_lane.pdparams"))

    def test_piecewise_learning_rate_matches_official_recipe_shape(self) -> None:
        config = train_nav_control_reg.TrainConfig(
            lr_scheduler="piecewise",
            lr_boundaries=(100, 400),
            lr_values=(0.001, 0.0001, 0.00001),
        )

        lr = train_nav_control_reg._build_learning_rate(config)

        self.assertEqual(float(lr()), 0.001)

    def test_piecewise_learning_rate_advances_once_per_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            output_dir = root / "outputs"
            config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=2,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                learning_rate=0.001,
                lr_scheduler="piecewise",
                lr_boundaries=(1,),
                lr_values=(0.001, 0.0001),
                device="cpu",
                run_name="unit_piecewise_step",
            )

            result = train_nav_control_reg.train_model(config)

            metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
            learning_rates = [item["learning_rate"] for item in metrics["history"]]
            np.testing.assert_allclose(learning_rates, [0.001, 0.0001], rtol=1e-6)

    def test_summarize_deviation_predictions_reports_corner_risks(self) -> None:
        labels = np.array([0.0, 0.2, 0.4, 0.8, -0.8], dtype=np.float32)
        predictions = np.array([0.05, 0.1, 0.2, 0.4, 0.2], dtype=np.float32)

        metrics = train_nav_control_reg.summarize_deviation_predictions(predictions, labels)

        self.assertEqual(metrics["mae_by_abs_bin"]["straight"]["count"], 1)
        self.assertEqual(metrics["mae_by_abs_bin"]["transition"]["count"], 1)
        self.assertEqual(metrics["mae_by_abs_bin"]["corner"]["count"], 1)
        self.assertEqual(metrics["mae_by_abs_bin"]["hard"]["count"], 2)
        self.assertAlmostEqual(metrics["direction_error_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["understeer_rate"], 1.0)
        self.assertIsInstance(metrics["corner_score"], float)

    def test_train_model_saves_corner_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets" / "08_nav_control_optional"
            train_source = make_deviation_jsonl_set(root, "corner_train", [0.0, 0.2, 0.4, 0.8])
            eval_source = make_deviation_jsonl_set(root, "corner_val", [0.0, 0.2, 0.4, 0.8, -0.8])
            output_dir = root / "outputs"
            config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                train_sources=(str(train_source),),
                eval_sources=(str(eval_source),),
                epochs=1,
                batch_size=2,
                eval_batch_size=2,
                eval_interval=1,
                device="cpu",
                run_name="unit_corner_best",
            )

            result = train_nav_control_reg.train_model(config)

            corner_best_params = output_dir / "unit_corner_best" / "dynamic" / "corner_best" / train_nav_control_reg.MODEL_PARAMS_NAME
            metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))
            self.assertTrue(corner_best_params.is_file())
            self.assertEqual(metrics["best_corner_epoch"], 1)
            self.assertIsInstance(metrics["best_corner_score"], float)
            self.assertEqual(result["corner_best_params_path"], str(corner_best_params))

    def test_train_model_with_normalized_label_loss_records_per_output_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            output_dir = root / "outputs"
            config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                device="cpu",
                run_name="unit_train_norm_loss",
                normalize_label_loss=True,
            )

            result = train_nav_control_reg.train_model(config)

            metrics_path = Path(result["metrics_path"])
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            epoch_metrics = metrics["history"][0]
            eval_metrics = epoch_metrics["eval"]
            self.assertTrue(metrics["config"]["normalize_label_loss"])
            self.assertIn("label_stats", metrics)
            self.assertEqual(len(metrics["label_stats"]["loss_weights"]), 2)
            self.assertIn("train_loss", epoch_metrics)
            self.assertIn("train_mae_per_output", epoch_metrics)
            self.assertIn("mae_per_output", eval_metrics)
            self.assertIn("mse_per_output", eval_metrics)
            self.assertEqual(len(epoch_metrics["train_mae_per_output"]), 2)
            self.assertEqual(len(eval_metrics["mae_per_output"]), 2)
            self.assertAlmostEqual(epoch_metrics["train_mae"], epoch_metrics["train_mae_per_output"][0], places=6)
            self.assertAlmostEqual(eval_metrics["mae"], eval_metrics["mae_per_output"][0], places=6)
            self.assertAlmostEqual(epoch_metrics["train_mae_per_output"][1], 0.0, places=6)
            self.assertAlmostEqual(eval_metrics["mae_per_output"][1], 0.0, places=6)
            self.assertAlmostEqual(metrics["best_eval_mae"], eval_metrics["mae_per_output"][0], places=6)

    def test_train_model_saves_latest_and_best_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            output_dir = root / "outputs"
            config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                device="cpu",
                run_name="unit_train",
            )

            result = train_nav_control_reg.train_model(config)

            latest_params = output_dir / "unit_train" / "dynamic" / train_nav_control_reg.MODEL_PARAMS_NAME
            best_params = output_dir / "unit_train" / "dynamic" / "best" / train_nav_control_reg.MODEL_PARAMS_NAME
            train_best_params = output_dir / "unit_train" / "dynamic" / "train_best" / train_nav_control_reg.MODEL_PARAMS_NAME
            metrics_path = output_dir / "unit_train" / "dynamic" / "metrics.json"
            self.assertTrue(latest_params.is_file())
            self.assertTrue(best_params.is_file())
            self.assertTrue(train_best_params.is_file())
            self.assertTrue(metrics_path.is_file())
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["best_epoch"], 1)
            self.assertEqual(metrics["best_train_epoch"], 1)
            self.assertIsInstance(metrics["best_train_mae"], float)
            self.assertIn("best_dir", metrics)
            self.assertIn("train_best_dir", metrics)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["best_train_epoch"], 1)
            self.assertEqual(result["latest_params_path"], str(latest_params))
            self.assertEqual(result["best_params_path"], str(best_params))
            self.assertEqual(result["train_best_params_path"], str(train_best_params))

    def test_train_model_resumes_checkpoint_history_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            output_dir = root / "outputs"
            first_config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                device="cpu",
                run_name="unit_resume",
                lr_scheduler="piecewise",
            )
            train_nav_control_reg.train_model(first_config)
            dynamic_dir = output_dir / "unit_resume" / "dynamic"
            first_metrics = json.loads((dynamic_dir / "metrics.json").read_text(encoding="utf-8"))

            resumed_config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=2,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                device="cpu",
                run_name="unit_resume",
                lr_scheduler="piecewise",
                resume_dir=dynamic_dir,
            )
            result = train_nav_control_reg.train_model(resumed_config)

            resumed_metrics = json.loads((dynamic_dir / "metrics.json").read_text(encoding="utf-8"))
            checkpoint = paddle.load(str(dynamic_dir / train_nav_control_reg.MODEL_CHECKPOINT_NAME))
            self.assertEqual([item["epoch"] for item in first_metrics["history"]], [1])
            self.assertEqual([item["epoch"] for item in resumed_metrics["history"]], [1, 2])
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertEqual(result["resumed_from_epoch"], 1)
            self.assertTrue(result["optimizer_state_restored"])

    def test_resume_rejects_training_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = make_dataset(root)
            output_dir = root / "outputs"
            first_config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=1,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                device="cpu",
                run_name="unit_resume_mismatch",
            )
            train_nav_control_reg.train_model(first_config)
            dynamic_dir = output_dir / "unit_resume_mismatch" / "dynamic"
            resumed_config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                epochs=2,
                batch_size=2,
                eval_batch_size=1,
                eval_interval=1,
                learning_rate=0.002,
                device="cpu",
                run_name="unit_resume_mismatch",
                resume_dir=dynamic_dir,
            )

            with self.assertRaisesRegex(ValueError, "Resume config mismatch: learning_rate"):
                train_nav_control_reg.train_model(resumed_config)

    def test_balanced_v2_is_default_preset_and_cli_can_override_it(self) -> None:
        default_args = train_nav_control_reg.build_parser().parse_args([])
        default_config = train_nav_control_reg.config_from_args(default_args)

        self.assertEqual(default_config.preset, "balanced_v2")
        self.assertEqual(default_config.epochs, 100)
        self.assertEqual(default_config.batch_size, 128)
        self.assertEqual(default_config.eval_batch_size, 256)
        self.assertEqual(default_config.optimizer_name, "adamw")
        self.assertAlmostEqual(default_config.weight_decay, 1e-4)
        self.assertAlmostEqual(default_config.learning_rate, 5e-4)
        self.assertEqual(default_config.lr_scheduler, "cosine_warmup")
        self.assertEqual(default_config.lr_t_max, 100)
        self.assertEqual(default_config.warmup_epochs, 3)
        self.assertAlmostEqual(default_config.warmup_start_lr, 5e-5)
        self.assertAlmostEqual(default_config.lr_eta_min, 1e-5)
        self.assertEqual(default_config.eval_interval, 2)
        self.assertEqual(default_config.early_stop_min_epochs, 30)
        self.assertEqual(default_config.early_stop_patience_evals, 10)
        self.assertAlmostEqual(default_config.early_stop_min_delta, 1e-4)
        self.assertEqual(default_config.deviation_loss_weight_mode, "moderate_abs_bins")
        self.assertEqual(default_config.augmentation_profile, "robust_v2")
        self.assertEqual(default_config.sample_weight_mode, "none")

        override_args = train_nav_control_reg.build_parser().parse_args(
            ["--preset", "balanced_v2", "--batch-size", "64", "--optimizer", "adam"]
        )
        override_config = train_nav_control_reg.config_from_args(override_args)

        self.assertEqual(override_config.batch_size, 64)
        self.assertEqual(override_config.optimizer_name, "adam")

    def test_cosine_warmup_rises_then_decays(self) -> None:
        config = train_nav_control_reg.TrainConfig(
            epochs=10,
            learning_rate=5e-4,
            lr_scheduler="cosine_warmup",
            lr_t_max=10,
            warmup_epochs=3,
            warmup_start_lr=5e-5,
            lr_eta_min=1e-5,
        )

        rates = train_nav_control_reg.collect_epoch_learning_rates(config, epoch_count=10)

        np.testing.assert_allclose(rates[:4], [5e-5, 2e-4, 3.5e-4, 5e-4], rtol=1e-7)
        self.assertGreater(rates[4], rates[-1])
        self.assertGreaterEqual(rates[-1], config.lr_eta_min)

    def test_build_optimizer_uses_adamw_for_balanced_v2(self) -> None:
        paddle.device.set_device("cpu")
        config = train_nav_control_reg.TrainConfig(optimizer_name="adamw", weight_decay=1e-4)
        model = train_nav_control_reg.PaddleLaneCnnModel()
        learning_rate = train_nav_control_reg._build_learning_rate(config)

        result = train_nav_control_reg._build_optimizer(config, model, learning_rate)

        self.assertEqual(type(result).__name__, "AdamW")

    def test_balanced_core_and_minimum_improvement(self) -> None:
        score = train_nav_control_reg.compute_balanced_core(global_mae=0.02, corner_score=0.04)

        self.assertAlmostEqual(score, 0.06)
        self.assertTrue(train_nav_control_reg.is_balanced_improvement(score, best_score=None, min_delta=1e-4))
        self.assertTrue(train_nav_control_reg.is_balanced_improvement(0.0598, best_score=0.06, min_delta=1e-4))
        self.assertFalse(train_nav_control_reg.is_balanced_improvement(0.05995, best_score=0.06, min_delta=1e-4))

    def test_early_stop_requires_minimum_epoch_and_full_patience(self) -> None:
        self.assertFalse(
            train_nav_control_reg.should_early_stop(
                epoch=29,
                min_epochs=30,
                stale_evaluations=10,
                patience_evaluations=10,
            )
        )
        self.assertFalse(
            train_nav_control_reg.should_early_stop(
                epoch=30,
                min_epochs=30,
                stale_evaluations=9,
                patience_evaluations=10,
            )
        )
        self.assertTrue(
            train_nav_control_reg.should_early_stop(
                epoch=30,
                min_epochs=30,
                stale_evaluations=10,
                patience_evaluations=10,
            )
        )

    def test_training_saves_balanced_best_snapshots_and_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets" / "08_nav_control_optional"
            train_source = make_deviation_jsonl_set(root, "balanced_train", [0.0, 0.2, 0.4, 0.8, -0.8, -0.4])
            eval_source = make_deviation_jsonl_set(root, "balanced_val", [0.0, 0.2, 0.4, 0.8, -0.8])
            output_dir = root / "outputs"
            config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                train_sources=(str(train_source),),
                eval_sources=(str(eval_source),),
                epochs=6,
                batch_size=3,
                eval_batch_size=5,
                eval_interval=1,
                learning_rate=1e-4,
                lr_scheduler="piecewise",
                optimizer_name="adam",
                augmentation_profile="official",
                deviation_loss_weight_mode="none",
                early_stop_min_epochs=3,
                early_stop_patience_evals=2,
                early_stop_min_delta=10.0,
                device="cpu",
                run_name="unit_balanced_early_stop",
            )

            result = train_nav_control_reg.train_model(config)

            dynamic_dir = output_dir / "unit_balanced_early_stop" / "dynamic"
            metrics = json.loads((dynamic_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["completed_epochs"], 3)
            self.assertEqual(len(metrics["history"]), 3)
            self.assertEqual(metrics["best_balanced_epoch"], 1)
            self.assertIsInstance(metrics["best_balanced_core"], float)
            self.assertEqual(metrics["early_stop"]["stale_evaluations"], 2)
            self.assertTrue((dynamic_dir / "balanced_best" / train_nav_control_reg.MODEL_PARAMS_NAME).is_file())
            self.assertEqual(
                sorted(path.name for path in (dynamic_dir / "eval_snapshots").iterdir()),
                ["epoch_001", "epoch_002", "epoch_003"],
            )
            for epoch_dir in (dynamic_dir / "eval_snapshots").iterdir():
                self.assertTrue((epoch_dir / train_nav_control_reg.MODEL_PARAMS_NAME).is_file())
                self.assertFalse((epoch_dir / train_nav_control_reg.MODEL_OPT_NAME).exists())

    def test_early_stop_only_triggers_after_a_validation_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "datasets" / "08_nav_control_optional"
            train_source = make_deviation_jsonl_set(root, "early_stop_train", [0.0, 0.2, 0.4, 0.8])
            eval_source = make_deviation_jsonl_set(root, "early_stop_val", [0.0, 0.2, 0.4, 0.8])
            config = train_nav_control_reg.TrainConfig(
                dataset_dir=dataset_dir,
                output_dir=root / "outputs",
                train_sources=(str(train_source),),
                eval_sources=(str(eval_source),),
                epochs=6,
                batch_size=2,
                eval_batch_size=4,
                eval_interval=2,
                learning_rate=1e-4,
                lr_scheduler="piecewise",
                optimizer_name="adam",
                augmentation_profile="official",
                deviation_loss_weight_mode="none",
                early_stop_min_epochs=5,
                early_stop_patience_evals=1,
                early_stop_min_delta=10.0,
                device="cpu",
                run_name="unit_validation_only_early_stop",
            )

            result = train_nav_control_reg.train_model(config)
            metrics = json.loads(Path(result["metrics_path"]).read_text(encoding="utf-8"))

            self.assertTrue(result["stopped_early"])
            self.assertEqual(result["completed_epochs"], 6)
            self.assertIsNotNone(metrics["history"][-1]["eval"])

    def test_balanced_v2_rejects_invalid_optimizer_scheduler_and_early_stop_values(self) -> None:
        invalid_cases = (
            (train_nav_control_reg.TrainConfig(optimizer_name="sgd"), "Unsupported optimizer"),
            (train_nav_control_reg.TrainConfig(weight_decay=-1e-4), "weight-decay"),
            (train_nav_control_reg.TrainConfig(warmup_epochs=0), "warmup-epochs"),
            (train_nav_control_reg.TrainConfig(warmup_start_lr=1e-3), "warmup-start-lr"),
            (train_nav_control_reg.TrainConfig(lr_eta_min=1e-3), "lr-eta-min"),
            (train_nav_control_reg.TrainConfig(early_stop_min_epochs=0), "early-stop-min-epochs"),
            (train_nav_control_reg.TrainConfig(early_stop_patience_evals=0), "early-stop-patience-evals"),
            (train_nav_control_reg.TrainConfig(early_stop_min_delta=-1e-4), "early-stop-min-delta"),
        )

        for config, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    train_nav_control_reg._validate_train_config(config)


if __name__ == "__main__":
    unittest.main()
