# 功能：按 PaddleLane_v30 风格训练 08_nav_control_optional 巡航控制回归模型。
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import paddle
from paddle import nn, optimizer
from paddle.io import BatchSampler, DataLoader, Dataset, WeightedRandomSampler
from PIL import Image, ImageFilter


DEFAULT_DATASET_DIR = Path("./datasets/08_nav_control_optional")
DEFAULT_OUTPUT_DIR = Path("./outputs/nav_control_optional")
DEFAULT_INPUT_SIZE = (128, 128)
DEFAULT_TRAIN_SOURCES = ("image_set_r", "image_set_l")
DEFAULT_EVAL_SOURCES = ("image_set1208",)
MODEL_BASENAME = "cnn_lane"
MODEL_PARAMS_NAME = "cnn_lane.pdparams"
MODEL_OPT_NAME = "cnn_lane.pdopt"
MODEL_CHECKPOINT_NAME = "cnn_lane.pkl"
LABEL_LOSS_WEIGHT_MODES = ("none", "inverse_std", "sqrt_inverse_std")
SAMPLE_WEIGHT_MODES = ("none", "abs_bins", "signed_bins")
OPTIMIZER_MODES = ("adam", "adamw")
LR_SCHEDULER_MODES = ("cosine", "cosine_warmup", "piecewise")
DEVIATION_LOSS_WEIGHT_MODES = ("abs_bins", "moderate_abs_bins", "corner_focus_v1", "none")
AUGMENTATION_PROFILES = ("official", "robust", "robust_v2", "light_corner_v3")
PRESET_MODES = ("balanced_v2", "balanced_v3_light_corner")


@dataclass(frozen=True)
class DataSource:
    name: str
    path: Path
    data_path: Path


@dataclass(frozen=True)
class NavSample:
    source_name: str
    image_rel: str
    image_path: Path
    label: tuple[float, float]


@dataclass(frozen=True)
class TrainConfig:
    preset: str = "balanced_v2"
    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES
    epochs: int = 100
    batch_size: int = 128
    eval_batch_size: int = 256
    optimizer_name: str = "adamw"
    weight_decay: float = 1e-4
    learning_rate: float = 5e-4
    lr_t_max: int = 100
    lr_scheduler: str = "cosine_warmup"
    warmup_epochs: int = 3
    warmup_start_lr: float = 5e-5
    lr_eta_min: float = 1e-5
    lr_boundaries: tuple[int, ...] = (100, 400)
    lr_values: tuple[float, ...] = (0.001, 0.0001, 0.00001)
    eval_interval: int = 2
    early_stop_min_epochs: int = 30
    early_stop_patience_evals: int = 10
    early_stop_min_delta: float = 1e-4
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE
    device: str = "gpu:0"
    num_workers: int = 0
    run_name: str | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    seed: int = 20260418
    normalize_label_loss: bool = False
    label_loss_weight_mode: str = "none"
    deviation_loss_weight_mode: str = "moderate_abs_bins"
    augmentation_profile: str = "robust_v2"
    sample_weight_mode: str = "none"
    label_jump_drop_threshold: float | None = None
    label_jump_drop_radius: int = 0
    init_params_path: Path | None = None
    resume_dir: Path | None = None


class PaddleLaneDataset(Dataset):
    def __init__(
        self,
        samples: list[NavSample],
        training: bool,
        image_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        forced_augmentation: str | None = None,
        augmentation_profile: str = "official",
    ):
        super().__init__()
        self.samples = samples
        self.training = training
        self.image_size = image_size
        self.forced_augmentation = forced_augmentation
        self.augmentation_profile = augmentation_profile

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        sample = self.samples[index]
        image = _load_rgb_image(sample.image_path, self.image_size)
        label = np.array(sample.label, dtype=np.float32)
        if self.training:
            image, label = _augment_like_paddlelane(
                image,
                label,
                self.forced_augmentation,
                augmentation_profile=self.augmentation_profile,
            )
        image = _normalize_to_chw(image)
        return image, label.astype(np.float32)


class PaddleLaneCnnModel(nn.Layer):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2D(3, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2D(32, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2D(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2D(64, 128, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2D(128, 128, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        deviation = self.backbone(inputs)
        return paddle.concat([deviation, paddle.zeros_like(deviation)], axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="balanced_v2", choices=PRESET_MODES, help="Training preset contract.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="08_nav_control_optional dataset root.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for dynamic checkpoints.")
    parser.add_argument("--train-sources", default=",".join(DEFAULT_TRAIN_SOURCES), help="Comma-separated raw image-set names for training.")
    parser.add_argument("--eval-sources", default=",".join(DEFAULT_EVAL_SOURCES), help="Comma-separated raw image-set names for validation.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--optimizer", dest="optimizer_name", default="adamw", choices=OPTIMIZER_MODES)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Optimizer peak learning rate.")
    parser.add_argument("--lr-t-max", type=int, default=100, help="Total epoch horizon for cosine decay.")
    parser.add_argument(
        "--lr-scheduler",
        default="cosine_warmup",
        choices=LR_SCHEDULER_MODES,
        help="Learning-rate scheduler. Use piecewise to reproduce the official cruise script.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=3, help="Linear warmup epoch count.")
    parser.add_argument("--warmup-start-lr", type=float, default=5e-5, help="Warmup starting learning rate.")
    parser.add_argument("--lr-eta-min", type=float, default=1e-5, help="Minimum cosine learning rate.")
    parser.add_argument(
        "--lr-boundaries",
        default="100,400",
        help="Comma-separated epoch boundaries for --lr-scheduler piecewise.",
    )
    parser.add_argument(
        "--lr-values",
        default="0.001,0.0001,0.00001",
        help="Comma-separated learning-rate values for --lr-scheduler piecewise.",
    )
    parser.add_argument("--eval-interval", type=int, default=2, help="Evaluate every N epochs and always on final epoch.")
    parser.add_argument("--early-stop-min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience-evals", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="gpu:0", help="Paddle device, default gpu:0.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--run-name", default=None, help="Optional run directory name under output-dir.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Optional smoke-test limit for train batches.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional smoke-test limit for eval batches.")
    parser.add_argument("--seed", type=int, default=20260418, help="Random seed.")
    parser.add_argument(
        "--normalize-label-loss",
        action="store_true",
        help="Deprecated alias for --label-loss-weight-mode inverse_std.",
    )
    parser.add_argument(
        "--label-loss-weight-mode",
        default="none",
        choices=LABEL_LOSS_WEIGHT_MODES,
        help="Per-output L1 loss weighting mode; predictions stay in raw units.",
    )
    parser.add_argument(
        "--deviation-loss-weight-mode",
        default="moderate_abs_bins",
        choices=DEVIATION_LOSS_WEIGHT_MODES,
        help="Per-sample loss weighting for output_0. Use none for raw official L1Loss behavior.",
    )
    parser.add_argument(
        "--augmentation-profile",
        default="robust_v2",
        choices=AUGMENTATION_PROFILES,
        help="Training augmentation set. official preserves PaddleLane_v30 behavior; robust adds mild degradation cases.",
    )
    parser.add_argument(
        "--sample-weight-mode",
        default="none",
        choices=SAMPLE_WEIGHT_MODES,
        help="Optional weighted sampler for imbalanced deviation labels.",
    )
    parser.add_argument(
        "--label-jump-drop-threshold",
        type=float,
        default=None,
        help="Drop frames near consecutive label jumps >= this threshold before training/eval.",
    )
    parser.add_argument(
        "--label-jump-drop-radius",
        type=int,
        default=0,
        help="Neighbor radius to drop around detected label jumps.",
    )
    parser.add_argument(
        "--init-params-path",
        default=None,
        help="Optional dynamic .pdparams path used to initialize the model before training.",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Checkpoint bundle directory used to resume model, optimizer, epoch history, and best state.",
    )
    parser.add_argument("--check-only", action="store_true", help="Run one batch forward check only; no training or saving.")
    parser.add_argument("--split", default="train", choices=("train", "eval", "val"), help="Split used with --check-only.")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        preset=args.preset,
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        train_sources=_parse_names(args.train_sources),
        eval_sources=_parse_names(args.eval_sources),
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        optimizer_name=args.optimizer_name,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        lr_t_max=args.lr_t_max,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        lr_eta_min=args.lr_eta_min,
        lr_boundaries=_parse_int_tuple(args.lr_boundaries),
        lr_values=_parse_float_tuple(args.lr_values),
        eval_interval=args.eval_interval,
        early_stop_min_epochs=args.early_stop_min_epochs,
        early_stop_patience_evals=args.early_stop_patience_evals,
        early_stop_min_delta=args.early_stop_min_delta,
        device=args.device,
        num_workers=args.num_workers,
        run_name=args.run_name,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        seed=args.seed,
        normalize_label_loss=args.normalize_label_loss,
        label_loss_weight_mode=args.label_loss_weight_mode,
        deviation_loss_weight_mode=args.deviation_loss_weight_mode,
        augmentation_profile=args.augmentation_profile,
        sample_weight_mode=args.sample_weight_mode,
        label_jump_drop_threshold=args.label_jump_drop_threshold,
        label_jump_drop_radius=args.label_jump_drop_radius,
        init_params_path=Path(args.init_params_path) if args.init_params_path else None,
        resume_dir=Path(args.resume_dir) if args.resume_dir else None,
    )


def resolve_data_sources(
    dataset_dir: str | Path,
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES,
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> tuple[list[DataSource], list[DataSource]]:
    root = Path(dataset_dir)
    raw_dir = root / "raw"
    train = [_resolve_source(raw_dir, source_name) for source_name in train_sources]
    eval_ = [_resolve_source(raw_dir, source_name) for source_name in eval_sources]
    return train, eval_


def load_samples(
    dataset_dir: str | Path,
    split: str,
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES,
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> list[NavSample]:
    train, eval_ = resolve_data_sources(dataset_dir, train_sources, eval_sources)
    if split == "train":
        sources = train
    elif split in {"eval", "val"}:
        sources = eval_
    else:
        raise ValueError(f"Unsupported split: {split}")
    samples: list[NavSample] = []
    for source in sources:
        samples.extend(_load_source_samples(source))
    if not samples:
        raise ValueError(f"No samples found for split: {split}")
    return samples


def run_forward_check(
    dataset_dir: str | Path,
    split: str = "train",
    batch_size: int = 8,
    device: str = "gpu:0",
    train_sources: tuple[str, ...] = DEFAULT_TRAIN_SOURCES,
    eval_sources: tuple[str, ...] = DEFAULT_EVAL_SOURCES,
) -> dict[str, Any]:
    paddle.device.set_device(device)
    samples = load_samples(dataset_dir, split, train_sources, eval_sources)
    dataset = PaddleLaneDataset(samples, training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    images, labels = next(iter(loader))
    model = PaddleLaneCnnModel()
    model.eval()
    with paddle.no_grad():
        preds = model(images)
    return {
        "split": "eval" if split == "val" else split,
        "device": paddle.get_device(),
        "samples": len(samples),
        "input_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "pred_shape": list(preds.shape),
    }


def load_model_from_params(params_path: str | Path, device: str = "gpu:0") -> PaddleLaneCnnModel:
    paddle.device.set_device(device)
    model = PaddleLaneCnnModel()
    state_dict = paddle.load(str(params_path))
    model.set_state_dict(state_dict)
    model.eval()
    return model


def build_label_stats(
    samples: list[NavSample],
    normalize_loss: bool = False,
    loss_weight_mode: str = "none",
) -> dict[str, list[float] | str]:
    if normalize_loss:
        loss_weight_mode = "inverse_std"
    if loss_weight_mode not in LABEL_LOSS_WEIGHT_MODES:
        raise ValueError(f"Unsupported label loss weight mode: {loss_weight_mode}")
    labels = np.array([sample.label for sample in samples], dtype=np.float32)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(f"Expected [N, 2] labels, got {labels.shape}")
    mean = labels.mean(axis=0)
    std = labels.std(axis=0)
    safe_std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    if loss_weight_mode == "inverse_std":
        inv_std = 1.0 / safe_std
        loss_weights = inv_std / inv_std.mean()
    elif loss_weight_mode == "sqrt_inverse_std":
        inv_sqrt_std = 1.0 / np.sqrt(safe_std)
        loss_weights = inv_sqrt_std / inv_sqrt_std.mean()
    else:
        loss_weights = np.ones_like(safe_std, dtype=np.float32)
    return {
        "loss_weight_mode": loss_weight_mode,
        "mean": mean.astype(float).tolist(),
        "std": safe_std.astype(float).tolist(),
        "loss_weights": loss_weights.astype(float).tolist(),
    }


def train_model(config: TrainConfig) -> dict[str, Any]:
    with paddle.utils.unique_name.guard():
        return _train_model_impl(config)


def _train_model_impl(config: TrainConfig) -> dict[str, Any]:
    _validate_train_config(config)
    np.random.seed(config.seed)
    paddle.seed(config.seed)
    paddle.device.set_device(config.device)

    raw_train_samples = load_samples(config.dataset_dir, "train", config.train_sources, config.eval_sources)
    raw_eval_samples = load_samples(config.dataset_dir, "eval", config.train_sources, config.eval_sources)
    train_samples, train_filter_summary = filter_samples_by_label_jump(
        raw_train_samples,
        config.label_jump_drop_threshold,
        config.label_jump_drop_radius,
    )
    eval_samples, eval_filter_summary = filter_samples_by_label_jump(
        raw_eval_samples,
        config.label_jump_drop_threshold,
        config.label_jump_drop_radius,
    )
    label_stats = build_label_stats(
        train_samples,
        normalize_loss=config.normalize_label_loss,
        loss_weight_mode=config.label_loss_weight_mode,
    )
    primary_loss_scale = float(label_stats["loss_weights"][0]) if label_stats["loss_weight_mode"] != "none" else 1.0
    train_dataset = PaddleLaneDataset(
        train_samples,
        training=True,
        image_size=config.input_size,
        augmentation_profile=config.augmentation_profile,
    )
    eval_dataset = PaddleLaneDataset(eval_samples, training=False, image_size=config.input_size)
    train_loader = _build_train_loader(
        train_dataset,
        train_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sample_weight_mode=config.sample_weight_mode,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
    )

    run_name = config.run_name or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_dir = config.output_dir / run_name / "dynamic"
    best_dir = dynamic_dir / "best"
    train_best_dir = dynamic_dir / "train_best"
    corner_best_dir = dynamic_dir / "corner_best"
    balanced_best_dir = dynamic_dir / "balanced_best"
    eval_snapshots_dir = dynamic_dir / "eval_snapshots"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    train_best_dir.mkdir(parents=True, exist_ok=True)
    corner_best_dir.mkdir(parents=True, exist_ok=True)
    balanced_best_dir.mkdir(parents=True, exist_ok=True)
    eval_snapshots_dir.mkdir(parents=True, exist_ok=True)
    params_path = dynamic_dir / MODEL_PARAMS_NAME
    opt_path = dynamic_dir / MODEL_OPT_NAME
    checkpoint_path = dynamic_dir / MODEL_CHECKPOINT_NAME
    best_params_path = best_dir / MODEL_PARAMS_NAME
    best_opt_path = best_dir / MODEL_OPT_NAME
    best_checkpoint_path = best_dir / MODEL_CHECKPOINT_NAME
    train_best_params_path = train_best_dir / MODEL_PARAMS_NAME
    train_best_opt_path = train_best_dir / MODEL_OPT_NAME
    train_best_checkpoint_path = train_best_dir / MODEL_CHECKPOINT_NAME
    corner_best_params_path = corner_best_dir / MODEL_PARAMS_NAME
    corner_best_opt_path = corner_best_dir / MODEL_OPT_NAME
    corner_best_checkpoint_path = corner_best_dir / MODEL_CHECKPOINT_NAME
    balanced_best_params_path = balanced_best_dir / MODEL_PARAMS_NAME
    balanced_best_opt_path = balanced_best_dir / MODEL_OPT_NAME
    balanced_best_checkpoint_path = balanced_best_dir / MODEL_CHECKPOINT_NAME
    metrics_path = dynamic_dir / "metrics.json"

    if config.init_params_path is not None:
        model = load_model_from_params(config.init_params_path, device=config.device)
    else:
        model = PaddleLaneCnnModel()
    lr = _build_learning_rate(config)
    opt = _build_optimizer(config, model, lr)

    history: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] = {}
    best_epoch: int | None = None
    best_eval_mae: float | None = None
    best_train_epoch: int | None = None
    best_train_mae: float | None = None
    best_corner_epoch: int | None = None
    best_corner_score: float | None = None
    best_balanced_epoch: int | None = None
    best_balanced_core: float | None = None
    stale_evaluations = 0
    stopped_early = False
    early_stop_reason: str | None = None
    start_epoch = 0
    optimizer_state_restored = False
    if config.resume_dir is not None:
        resume_state = _load_resume_state(config, run_name, dynamic_dir)
        model.set_state_dict(paddle.load(str(resume_state["params_path"])))
        opt.set_state_dict(paddle.load(str(resume_state["opt_path"])))
        history = resume_state["history"]
        final_metrics = history[-1]
        best_epoch = resume_state["best_epoch"]
        best_eval_mae = resume_state["best_eval_mae"]
        best_train_epoch = resume_state["best_train_epoch"]
        best_train_mae = resume_state["best_train_mae"]
        best_corner_epoch = resume_state["best_corner_epoch"]
        best_corner_score = resume_state["best_corner_score"]
        best_balanced_epoch = resume_state["best_balanced_epoch"]
        best_balanced_core = resume_state["best_balanced_core"]
        stale_evaluations = resume_state["stale_evaluations"]
        start_epoch = resume_state["start_epoch"]
        optimizer_state_restored = True

    for epoch_index in range(start_epoch, config.epochs):
        current_learning_rate = float(lr())
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            opt,
            config.max_train_batches,
            primary_loss_scale=primary_loss_scale,
            deviation_loss_weight_mode=config.deviation_loss_weight_mode,
        )
        should_eval = _should_eval(epoch_index, config.epochs, config.eval_interval)
        eval_metrics = (
            _evaluate(
                model,
                eval_loader,
                config.max_eval_batches,
                primary_loss_scale=primary_loss_scale,
                deviation_loss_weight_mode=config.deviation_loss_weight_mode,
            )
            if should_eval
            else None
        )
        epoch_metrics = {
            "epoch": epoch_index + 1,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_mae_per_output": train_metrics["mae_per_output"],
            "train_samples": train_metrics["samples"],
            "learning_rate": current_learning_rate,
            "eval": eval_metrics,
        }
        history.append(epoch_metrics)
        final_metrics = epoch_metrics

        current_train_mae = float(train_metrics["mae"])
        train_best_updated = best_train_mae is None or current_train_mae < best_train_mae
        if train_best_updated:
            best_train_mae = current_train_mae
            best_train_epoch = epoch_index + 1

        current_eval_mae = None if eval_metrics is None else float(eval_metrics["mae_per_output"][0])
        eval_best_updated = current_eval_mae is not None and (best_eval_mae is None or current_eval_mae < best_eval_mae)
        if eval_best_updated:
            best_eval_mae = current_eval_mae
            best_epoch = epoch_index + 1

        current_corner_score = None
        if eval_metrics is not None:
            candidate_score = eval_metrics.get("robustness", {}).get("corner_score")
            if isinstance(candidate_score, (int, float)) and not isinstance(candidate_score, bool):
                current_corner_score = float(candidate_score)
        corner_best_updated = current_corner_score is not None and (
            best_corner_score is None or current_corner_score < best_corner_score
        )
        if corner_best_updated:
            best_corner_score = current_corner_score
            best_corner_epoch = epoch_index + 1

        current_balanced_core = None
        balanced_best_updated = False
        if current_eval_mae is not None and current_corner_score is not None:
            current_balanced_core = compute_balanced_core(current_eval_mae, current_corner_score)
            eval_metrics["balanced_core"] = current_balanced_core
            balanced_best_updated = is_balanced_improvement(
                current_balanced_core,
                best_balanced_core,
                config.early_stop_min_delta,
            )
            if balanced_best_updated:
                best_balanced_core = current_balanced_core
                best_balanced_epoch = epoch_index + 1
                stale_evaluations = 0
            else:
                stale_evaluations += 1

        early_stop_triggered = current_balanced_core is not None and should_early_stop(
            epoch=epoch_index + 1,
            min_epochs=config.early_stop_min_epochs,
            stale_evaluations=stale_evaluations,
            patience_evaluations=config.early_stop_patience_evals,
        )
        if early_stop_triggered:
            stopped_early = True
            early_stop_reason = (
                f"balanced_core did not improve by {config.early_stop_min_delta} "
                f"for {stale_evaluations} validation checks"
            )

        lr.step()
        latest_checkpoint = {
            "epoch": epoch_index + 1,
            "model_params_name": MODEL_PARAMS_NAME,
            "model_opt_name": MODEL_OPT_NAME,
            "train_sources": list(config.train_sources),
            "eval_sources": list(config.eval_sources),
            "label_stats": label_stats,
            "sample_weight_mode": config.sample_weight_mode,
            "deviation_loss_weight_mode": config.deviation_loss_weight_mode,
            "augmentation_profile": config.augmentation_profile,
            "train_filter_summary": train_filter_summary,
            "eval_filter_summary": eval_filter_summary,
            "final_metrics": final_metrics,
            "best_epoch": best_epoch,
            "best_eval_mae": best_eval_mae,
            "best_train_epoch": best_train_epoch,
            "best_train_mae": best_train_mae,
            "best_corner_epoch": best_corner_epoch,
            "best_corner_score": best_corner_score,
            "best_balanced_epoch": best_balanced_epoch,
            "best_balanced_core": best_balanced_core,
            "stale_evaluations": stale_evaluations,
            "stopped_early": stopped_early,
            "early_stop_reason": early_stop_reason,
            "resumed_from_epoch": start_epoch if optimizer_state_restored else None,
        }

        if eval_best_updated:
            best_checkpoint = dict(latest_checkpoint)
            _save_checkpoint_bundle(best_dir, model, opt, best_checkpoint)

        if train_best_updated:
            train_best_checkpoint = dict(latest_checkpoint)
            _save_checkpoint_bundle(train_best_dir, model, opt, train_best_checkpoint)

        if corner_best_updated:
            corner_best_checkpoint = dict(latest_checkpoint)
            _save_checkpoint_bundle(corner_best_dir, model, opt, corner_best_checkpoint)

        if balanced_best_updated:
            balanced_best_checkpoint = dict(latest_checkpoint)
            _save_checkpoint_bundle(balanced_best_dir, model, opt, balanced_best_checkpoint)

        if eval_metrics is not None:
            snapshot_dir = eval_snapshots_dir / f"epoch_{epoch_index + 1:03d}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            paddle.save(model.state_dict(), str(snapshot_dir / MODEL_PARAMS_NAME))

        _save_checkpoint_bundle(dynamic_dir, model, opt, latest_checkpoint)

        metrics_payload = {
            "config": _jsonable_config(config, run_name),
            "train_samples": len(train_samples),
            "eval_samples": len(eval_samples),
            "label_stats": label_stats,
            "sample_weight_mode": config.sample_weight_mode,
            "deviation_loss_weight_mode": config.deviation_loss_weight_mode,
            "augmentation_profile": config.augmentation_profile,
            "train_filter_summary": train_filter_summary,
            "eval_filter_summary": eval_filter_summary,
            "history": history,
            "latest_dir": str(dynamic_dir),
            "best_dir": str(best_dir),
            "train_best_dir": str(train_best_dir),
            "corner_best_dir": str(corner_best_dir),
            "balanced_best_dir": str(balanced_best_dir),
            "eval_snapshots_dir": str(eval_snapshots_dir),
            "best_epoch": best_epoch,
            "best_eval_mae": best_eval_mae,
            "best_train_epoch": best_train_epoch,
            "best_train_mae": best_train_mae,
            "best_corner_epoch": best_corner_epoch,
            "best_corner_score": best_corner_score,
            "best_balanced_epoch": best_balanced_epoch,
            "best_balanced_core": best_balanced_core,
            "completed_epochs": len(history),
            "early_stop": {
                "stopped_early": stopped_early,
                "stop_epoch": epoch_index + 1 if stopped_early else None,
                "reason": early_stop_reason,
                "stale_evaluations": stale_evaluations,
                "min_epochs": config.early_stop_min_epochs,
                "patience_evaluations": config.early_stop_patience_evals,
                "min_delta": config.early_stop_min_delta,
            },
            "resume": {
                "resumed_from_epoch": start_epoch,
                "resume_dir": str(config.resume_dir),
                "optimizer_state_restored": True,
            }
            if optimizer_state_restored
            else None,
            "note": "offline checkpoint only; not validated for onboard closed-loop control",
        }
        metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _print_epoch_summary(epoch_index + 1, config.epochs, train_metrics, eval_metrics, best_epoch, best_eval_mae)
        if early_stop_triggered:
            break

    return {
        "run_dir": str(dynamic_dir),
        "params_path": str(params_path),
        "opt_path": str(opt_path),
        "checkpoint_path": str(checkpoint_path),
        "latest_params_path": str(params_path),
        "latest_opt_path": str(opt_path),
        "latest_checkpoint_path": str(checkpoint_path),
        "best_params_path": str(best_params_path),
        "best_opt_path": str(best_opt_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_epoch": best_epoch,
        "best_eval_mae": best_eval_mae,
        "train_best_params_path": str(train_best_params_path),
        "train_best_opt_path": str(train_best_opt_path),
        "train_best_checkpoint_path": str(train_best_checkpoint_path),
        "best_train_epoch": best_train_epoch,
        "best_train_mae": best_train_mae,
        "corner_best_params_path": str(corner_best_params_path),
        "corner_best_opt_path": str(corner_best_opt_path),
        "corner_best_checkpoint_path": str(corner_best_checkpoint_path),
        "best_corner_epoch": best_corner_epoch,
        "best_corner_score": best_corner_score,
        "balanced_best_params_path": str(balanced_best_params_path),
        "balanced_best_opt_path": str(balanced_best_opt_path),
        "balanced_best_checkpoint_path": str(balanced_best_checkpoint_path),
        "best_balanced_epoch": best_balanced_epoch,
        "best_balanced_core": best_balanced_core,
        "eval_snapshots_dir": str(eval_snapshots_dir),
        "completed_epochs": len(history),
        "stopped_early": stopped_early,
        "early_stop_reason": early_stop_reason,
        "metrics_path": str(metrics_path),
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "final_metrics": final_metrics,
        "resumed_from_epoch": start_epoch,
        "optimizer_state_restored": optimizer_state_restored,
    }


def _load_resume_state(config: TrainConfig, run_name: str, dynamic_dir: Path) -> dict[str, Any]:
    resume_dir = config.resume_dir
    if resume_dir is None:
        raise ValueError("resume_dir is required")
    required_paths = {
        "params_path": resume_dir / MODEL_PARAMS_NAME,
        "opt_path": resume_dir / MODEL_OPT_NAME,
        "checkpoint_path": resume_dir / MODEL_CHECKPOINT_NAME,
        "metrics_path": resume_dir / "metrics.json",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing resume file: {path}")

    checkpoint = paddle.load(str(required_paths["checkpoint_path"]))
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid resume checkpoint: {required_paths['checkpoint_path']}")
    metrics = json.loads(required_paths["metrics_path"].read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"Invalid resume metrics: {required_paths['metrics_path']}")

    start_epoch = checkpoint.get("epoch")
    if isinstance(start_epoch, bool) or not isinstance(start_epoch, int) or start_epoch <= 0:
        raise ValueError(f"Invalid resume epoch: {start_epoch}")
    if start_epoch >= config.epochs:
        raise ValueError(f"Resume epoch {start_epoch} must be less than target epochs {config.epochs}")

    history = metrics.get("history")
    if not isinstance(history, list) or len(history) != start_epoch:
        raise ValueError(f"Resume history length mismatch: epoch={start_epoch}, history={len(history) if isinstance(history, list) else None}")
    if not history or history[-1].get("epoch") != start_epoch:
        raise ValueError(f"Resume history does not end at epoch {start_epoch}")

    saved_latest_dir = metrics.get("latest_dir")
    if not isinstance(saved_latest_dir, str) or Path(saved_latest_dir).resolve() != dynamic_dir.resolve():
        raise ValueError(f"Resume latest_dir mismatch: {saved_latest_dir} != {dynamic_dir}")
    _validate_resume_config(config, run_name, metrics.get("config"))

    best_epoch = metrics.get("best_epoch")
    best_eval_mae = metrics.get("best_eval_mae")
    if best_epoch != checkpoint.get("best_epoch") or best_eval_mae != checkpoint.get("best_eval_mae"):
        raise ValueError("Resume best state mismatch between checkpoint and metrics")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch <= 0:
        raise ValueError(f"Invalid resume best_epoch: {best_epoch}")
    if not isinstance(best_eval_mae, (int, float)):
        raise ValueError(f"Invalid resume best_eval_mae: {best_eval_mae}")

    best_train_epoch = metrics.get("best_train_epoch")
    best_train_mae = metrics.get("best_train_mae")
    if best_train_epoch is None or best_train_mae is None:
        best_train_epoch, best_train_mae = _derive_best_train_state(history)
    checkpoint_best_train_epoch = checkpoint.get("best_train_epoch")
    checkpoint_best_train_mae = checkpoint.get("best_train_mae")
    if checkpoint_best_train_epoch is not None and checkpoint_best_train_epoch != best_train_epoch:
        raise ValueError("Resume train-best epoch mismatch between checkpoint and metrics")
    if checkpoint_best_train_mae is not None and checkpoint_best_train_mae != best_train_mae:
        raise ValueError("Resume train-best MAE mismatch between checkpoint and metrics")
    if isinstance(best_train_epoch, bool) or not isinstance(best_train_epoch, int) or best_train_epoch <= 0:
        raise ValueError(f"Invalid resume best_train_epoch: {best_train_epoch}")
    if not isinstance(best_train_mae, (int, float)):
        raise ValueError(f"Invalid resume best_train_mae: {best_train_mae}")

    best_corner_epoch = metrics.get("best_corner_epoch")
    best_corner_score = metrics.get("best_corner_score")
    if best_corner_epoch is None or best_corner_score is None:
        best_corner_epoch, best_corner_score = _derive_best_corner_state(history)
    checkpoint_best_corner_epoch = checkpoint.get("best_corner_epoch")
    checkpoint_best_corner_score = checkpoint.get("best_corner_score")
    if checkpoint_best_corner_epoch is not None and checkpoint_best_corner_epoch != best_corner_epoch:
        raise ValueError("Resume corner-best epoch mismatch between checkpoint and metrics")
    if checkpoint_best_corner_score is not None and checkpoint_best_corner_score != best_corner_score:
        raise ValueError("Resume corner-best score mismatch between checkpoint and metrics")
    if best_corner_epoch is not None and (
        isinstance(best_corner_epoch, bool) or not isinstance(best_corner_epoch, int) or best_corner_epoch <= 0
    ):
        raise ValueError(f"Invalid resume best_corner_epoch: {best_corner_epoch}")
    if best_corner_score is not None and not isinstance(best_corner_score, (int, float)):
        raise ValueError(f"Invalid resume best_corner_score: {best_corner_score}")

    best_balanced_epoch = metrics.get("best_balanced_epoch")
    best_balanced_core = metrics.get("best_balanced_core")
    early_stop_state = metrics.get("early_stop")
    stale_evaluations = early_stop_state.get("stale_evaluations") if isinstance(early_stop_state, dict) else None
    if best_balanced_epoch is None or best_balanced_core is None or stale_evaluations is None:
        best_balanced_epoch, best_balanced_core, stale_evaluations = _derive_best_balanced_state(
            history,
            min_delta=config.early_stop_min_delta,
        )
    checkpoint_best_balanced_epoch = checkpoint.get("best_balanced_epoch")
    checkpoint_best_balanced_core = checkpoint.get("best_balanced_core")
    if checkpoint_best_balanced_epoch is not None and checkpoint_best_balanced_epoch != best_balanced_epoch:
        raise ValueError("Resume balanced-best epoch mismatch between checkpoint and metrics")
    if checkpoint_best_balanced_core is not None and checkpoint_best_balanced_core != best_balanced_core:
        raise ValueError("Resume balanced-best score mismatch between checkpoint and metrics")
    if (best_balanced_epoch is None) != (best_balanced_core is None):
        raise ValueError("Resume balanced-best epoch and score must both be set or both be absent")
    if best_balanced_epoch is not None and (
        isinstance(best_balanced_epoch, bool) or not isinstance(best_balanced_epoch, int) or best_balanced_epoch <= 0
    ):
        raise ValueError(f"Invalid resume best_balanced_epoch: {best_balanced_epoch}")
    if best_balanced_core is not None and not isinstance(best_balanced_core, (int, float)):
        raise ValueError(f"Invalid resume best_balanced_core: {best_balanced_core}")
    if isinstance(stale_evaluations, bool) or not isinstance(stale_evaluations, int) or stale_evaluations < 0:
        raise ValueError(f"Invalid resume stale_evaluations: {stale_evaluations}")

    return {
        **required_paths,
        "start_epoch": start_epoch,
        "history": history,
        "best_epoch": best_epoch,
        "best_eval_mae": float(best_eval_mae),
        "best_train_epoch": best_train_epoch,
        "best_train_mae": float(best_train_mae),
        "best_corner_epoch": best_corner_epoch,
        "best_corner_score": None if best_corner_score is None else float(best_corner_score),
        "best_balanced_epoch": best_balanced_epoch,
        "best_balanced_core": None if best_balanced_core is None else float(best_balanced_core),
        "stale_evaluations": stale_evaluations,
    }


def _derive_best_train_state(history: list[dict[str, Any]]) -> tuple[int, float]:
    candidates = [
        (int(item["epoch"]), float(item["train_mae"]))
        for item in history
        if isinstance(item, dict) and isinstance(item.get("epoch"), int) and isinstance(item.get("train_mae"), (int, float))
    ]
    if not candidates:
        raise ValueError("Resume history has no valid train MAE values")
    return min(candidates, key=lambda item: item[1])


def _derive_best_corner_state(history: list[dict[str, Any]]) -> tuple[int | None, float | None]:
    candidates: list[tuple[int, float]] = []
    for item in history:
        if not isinstance(item, dict) or not isinstance(item.get("epoch"), int):
            continue
        eval_metrics = item.get("eval")
        if not isinstance(eval_metrics, dict):
            continue
        robustness = eval_metrics.get("robustness")
        if not isinstance(robustness, dict):
            continue
        score = robustness.get("corner_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            candidates.append((int(item["epoch"]), float(score)))
    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[1])


def _derive_best_balanced_state(history: list[dict[str, Any]], min_delta: float) -> tuple[int | None, float | None, int]:
    best_epoch: int | None = None
    best_score: float | None = None
    stale_evaluations = 0
    for item in history:
        if not isinstance(item, dict) or not isinstance(item.get("epoch"), int):
            continue
        eval_metrics = item.get("eval")
        if not isinstance(eval_metrics, dict):
            continue
        score = eval_metrics.get("balanced_core")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            global_mae = eval_metrics.get("mae")
            corner_score = eval_metrics.get("robustness", {}).get("corner_score")
            if not isinstance(global_mae, (int, float)) or not isinstance(corner_score, (int, float)):
                continue
            score = compute_balanced_core(float(global_mae), float(corner_score))
        score = float(score)
        if is_balanced_improvement(score, best_score, min_delta):
            best_epoch = int(item["epoch"])
            best_score = score
            stale_evaluations = 0
        else:
            stale_evaluations += 1
    if best_epoch is None or best_score is None:
        return None, None, 0
    return best_epoch, best_score, stale_evaluations


def _validate_resume_config(config: TrainConfig, run_name: str, saved_config: object) -> None:
    if not isinstance(saved_config, dict):
        raise ValueError("Resume metrics config must be an object")
    current_config = _jsonable_config(config, run_name)
    fields = (
        "preset",
        "dataset_dir",
        "train_sources",
        "eval_sources",
        "batch_size",
        "eval_batch_size",
        "optimizer_name",
        "weight_decay",
        "learning_rate",
        "lr_t_max",
        "lr_scheduler",
        "warmup_epochs",
        "warmup_start_lr",
        "lr_eta_min",
        "lr_boundaries",
        "lr_values",
        "eval_interval",
        "early_stop_min_epochs",
        "early_stop_patience_evals",
        "early_stop_min_delta",
        "input_size",
        "device",
        "num_workers",
        "max_train_batches",
        "max_eval_batches",
        "seed",
        "normalize_label_loss",
        "label_loss_weight_mode",
        "deviation_loss_weight_mode",
        "augmentation_profile",
        "sample_weight_mode",
        "label_jump_drop_threshold",
        "label_jump_drop_radius",
        "run_name",
    )
    for field in fields:
        if saved_config.get(field) != current_config.get(field):
            raise ValueError(f"Resume config mismatch: {field}")


def _build_learning_rate(config: TrainConfig) -> optimizer.lr.LRScheduler:
    if config.lr_scheduler == "cosine":
        return optimizer.lr.CosineAnnealingDecay(config.learning_rate, T_max=config.lr_t_max)
    if config.lr_scheduler == "cosine_warmup":
        cosine = optimizer.lr.CosineAnnealingDecay(
            config.learning_rate,
            T_max=max(1, config.lr_t_max - config.warmup_epochs),
            eta_min=config.lr_eta_min,
        )
        return optimizer.lr.LinearWarmup(
            learning_rate=cosine,
            warmup_steps=config.warmup_epochs,
            start_lr=config.warmup_start_lr,
            end_lr=config.learning_rate,
        )
    if config.lr_scheduler == "piecewise":
        return optimizer.lr.PiecewiseDecay(boundaries=list(config.lr_boundaries), values=list(config.lr_values))
    raise ValueError(f"Unsupported lr scheduler: {config.lr_scheduler}")


def collect_epoch_learning_rates(config: TrainConfig, epoch_count: int) -> list[float]:
    if epoch_count <= 0:
        raise ValueError("epoch_count must be positive")
    learning_rate = _build_learning_rate(config)
    rates: list[float] = []
    for _ in range(epoch_count):
        rates.append(float(learning_rate()))
        learning_rate.step()
    return rates


def _build_optimizer(
    config: TrainConfig,
    model: nn.Layer,
    learning_rate: optimizer.lr.LRScheduler,
) -> optimizer.Optimizer:
    if config.optimizer_name == "adam":
        return optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    if config.optimizer_name == "adamw":
        return optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=model.parameters(),
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def _train_one_epoch(
    model: nn.Layer,
    loader: DataLoader,
    opt: optimizer.Optimizer,
    max_batches: int | None = None,
    primary_loss_scale: float = 1.0,
    deviation_loss_weight_mode: str = "abs_bins",
) -> dict[str, Any]:
    model.train()
    loss_sum = 0.0
    mae_sum_per_output = np.zeros(2, dtype=np.float64)
    mse_sum_per_output = np.zeros(2, dtype=np.float64)
    sample_count = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        preds = model(images)
        batch_weights = (
            _build_deviation_batch_weights(labels[:, 0:1], mode=deviation_loss_weight_mode)
            if deviation_loss_weight_mode != "none"
            else None
        )
        loss = _weighted_primary_l1_loss(preds[:, 0:1], labels[:, 0:1], batch_weights, primary_loss_scale)
        loss.backward()
        opt.step()
        opt.clear_grad()
        batch_size = int(images.shape[0])
        abs_error = paddle.abs(preds - labels)
        sq_error = paddle.square(preds - labels)
        loss_sum += float(loss) * batch_size
        mae_sum_per_output += paddle.sum(abs_error, axis=0).numpy()
        mse_sum_per_output += paddle.sum(sq_error, axis=0).numpy()
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No training batches were processed")
    mae_per_output = (mae_sum_per_output / sample_count).astype(float).tolist()
    mse_per_output = (mse_sum_per_output / sample_count).astype(float).tolist()
    return {
        "loss": loss_sum / sample_count,
        "mae": float(mae_per_output[0]),
        "mae_per_output": mae_per_output,
        "mse": float(mse_per_output[0]),
        "mse_per_output": mse_per_output,
        "samples": sample_count,
    }


def _evaluate(
    model: nn.Layer,
    loader: DataLoader,
    max_batches: int | None = None,
    primary_loss_scale: float = 1.0,
    deviation_loss_weight_mode: str = "abs_bins",
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    mae_sum_per_output = np.zeros(2, dtype=np.float64)
    mse_sum_per_output = np.zeros(2, dtype=np.float64)
    sample_count = 0
    primary_predictions: list[np.ndarray] = []
    primary_labels: list[np.ndarray] = []
    with paddle.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            preds = model(images)
            batch_size = int(images.shape[0])
            abs_error = paddle.abs(preds - labels)
            sq_error = paddle.square(preds - labels)
            batch_weights = (
                _build_deviation_batch_weights(labels[:, 0:1], mode=deviation_loss_weight_mode)
                if deviation_loss_weight_mode != "none"
                else None
            )
            loss_sum += float(_weighted_primary_l1_loss(preds[:, 0:1], labels[:, 0:1], batch_weights, primary_loss_scale)) * batch_size
            mae_sum_per_output += paddle.sum(abs_error, axis=0).numpy()
            mse_sum_per_output += paddle.sum(sq_error, axis=0).numpy()
            primary_predictions.append(preds[:, 0].numpy().reshape(-1))
            primary_labels.append(labels[:, 0].numpy().reshape(-1))
            sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No evaluation batches were processed")
    mae_per_output = (mae_sum_per_output / sample_count).astype(float).tolist()
    mse_per_output = (mse_sum_per_output / sample_count).astype(float).tolist()
    robustness = summarize_deviation_predictions(
        np.concatenate(primary_predictions),
        np.concatenate(primary_labels),
    )
    return {
        "loss": loss_sum / sample_count,
        "mae": float(mae_per_output[0]),
        "mae_per_output": mae_per_output,
        "mse": float(mse_per_output[0]),
        "mse_per_output": mse_per_output,
        "samples": sample_count,
        "robustness": robustness,
    }


def summarize_deviation_predictions(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    prediction_values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if prediction_values.shape != label_values.shape or prediction_values.size == 0:
        raise ValueError(f"Expected matching non-empty predictions/labels, got {prediction_values.shape} and {label_values.shape}")
    if not np.all(np.isfinite(prediction_values)) or not np.all(np.isfinite(label_values)):
        raise ValueError("Predictions and labels must be finite")

    abs_labels = np.abs(label_values)
    abs_errors = np.abs(prediction_values - label_values)
    bin_masks = {
        "straight": abs_labels < 0.1,
        "transition": (abs_labels >= 0.1) & (abs_labels < 0.3),
        "corner": (abs_labels >= 0.3) & (abs_labels < 0.6),
        "hard": abs_labels >= 0.6,
    }
    mae_by_abs_bin: dict[str, dict[str, int | float | None]] = {}
    for name, mask in bin_masks.items():
        count = int(mask.sum())
        mae_by_abs_bin[name] = {
            "count": count,
            "mae": float(abs_errors[mask].mean()) if count else None,
        }

    turn_mask = abs_labels >= 0.3
    turn_count = int(turn_mask.sum())
    direction_error_rate = float(((prediction_values[turn_mask] * label_values[turn_mask]) < 0).mean()) if turn_count else None
    understeer_rate = float((np.abs(prediction_values[turn_mask]) < 0.7 * abs_labels[turn_mask]).mean()) if turn_count else None
    corner_mae = mae_by_abs_bin["corner"]["mae"]
    hard_mae = mae_by_abs_bin["hard"]["mae"]
    corner_score = None
    if isinstance(corner_mae, float) and isinstance(hard_mae, float) and isinstance(understeer_rate, float):
        corner_score = 0.5 * corner_mae + 0.4 * hard_mae + 0.1 * understeer_rate

    return {
        "global_mae": float(abs_errors.mean()),
        "mae_by_abs_bin": mae_by_abs_bin,
        "turn_samples": turn_count,
        "direction_error_rate": direction_error_rate,
        "understeer_rate": understeer_rate,
        "corner_score": corner_score,
    }


def compute_balanced_core(global_mae: float, corner_score: float) -> float:
    values = (float(global_mae), float(corner_score))
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("global_mae and corner_score must be non-negative finite values")
    return values[0] + values[1]


def is_balanced_improvement(current_score: float, best_score: float | None, min_delta: float) -> bool:
    current = float(current_score)
    if not math.isfinite(current) or current < 0.0:
        raise ValueError("current_score must be a non-negative finite value")
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError("min_delta must be a non-negative finite value")
    if best_score is None:
        return True
    best = float(best_score)
    if not math.isfinite(best) or best < 0.0:
        raise ValueError("best_score must be a non-negative finite value")
    return current < best - min_delta


def should_early_stop(
    epoch: int,
    min_epochs: int,
    stale_evaluations: int,
    patience_evaluations: int,
) -> bool:
    return epoch >= min_epochs and stale_evaluations >= patience_evaluations


def _weighted_primary_l1_loss(
    preds: paddle.Tensor,
    labels: paddle.Tensor,
    sample_weights: paddle.Tensor | None = None,
    primary_loss_scale: float = 1.0,
) -> paddle.Tensor:
    if preds.shape != labels.shape:
        raise ValueError(f"preds/labels shape mismatch: {preds.shape} vs {labels.shape}")
    abs_error = paddle.abs(preds - labels)
    if sample_weights is not None:
        if sample_weights.shape != labels.shape:
            raise ValueError(f"sample_weights/labels shape mismatch: {sample_weights.shape} vs {labels.shape}")
        abs_error = abs_error * sample_weights
    if primary_loss_scale != 1.0:
        abs_error = abs_error * primary_loss_scale
    return paddle.mean(abs_error)


def _build_deviation_batch_weights(labels: paddle.Tensor, mode: str = "abs_bins") -> paddle.Tensor:
    if labels.ndim != 2 or labels.shape[1] != 1:
        raise ValueError(f"Expected labels shape [N, 1], got {labels.shape}")
    if mode not in {"abs_bins", "moderate_abs_bins", "corner_focus_v1"}:
        raise ValueError(f"Unsupported deviation batch weight mode: {mode}")
    deviation = paddle.abs(labels)
    if mode == "corner_focus_v1":
        weights = paddle.full_like(deviation, 0.75)
        weights = paddle.where((deviation >= 0.1) & (deviation < 0.3), paddle.full_like(deviation, 1.25), weights)
        weights = paddle.where((deviation >= 0.3) & (deviation < 0.6), paddle.full_like(deviation, 1.5), weights)
        weights = paddle.where(deviation >= 0.6, paddle.full_like(deviation, 1.75), weights)
    elif mode == "moderate_abs_bins":
        weights = paddle.full_like(deviation, 0.75)
        weights = paddle.where((deviation >= 0.1) & (deviation < 0.3), paddle.full_like(deviation, 1.0), weights)
        weights = paddle.where((deviation >= 0.3) & (deviation < 0.6), paddle.full_like(deviation, 1.25), weights)
        weights = paddle.where(deviation >= 0.6, paddle.full_like(deviation, 1.5), weights)
    else:
        weights = paddle.full_like(deviation, 0.25)
        nonzero_mask = deviation >= 1e-12
        weights = paddle.where(nonzero_mask & (deviation < 0.1), paddle.full_like(deviation, 1.0), weights)
        weights = paddle.where((deviation >= 0.1) & (deviation < 0.3), paddle.full_like(deviation, 2.0), weights)
        weights = paddle.where((deviation >= 0.3) & (deviation < 0.6), paddle.full_like(deviation, 3.0), weights)
        weights = paddle.where(deviation >= 0.6, paddle.full_like(deviation, 4.0), weights)
    mean_weight = paddle.mean(weights)
    return weights / paddle.maximum(mean_weight, paddle.to_tensor(1e-6, dtype=weights.dtype))


def _build_train_loader(
    dataset: PaddleLaneDataset,
    samples: list[NavSample],
    batch_size: int,
    num_workers: int,
    sample_weight_mode: str,
) -> DataLoader:
    if sample_weight_mode == "none":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )
    sample_weights = build_deviation_sample_weights(samples, mode=sample_weight_mode)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)


def _save_checkpoint_bundle(bundle_dir: Path, model: nn.Layer, opt: optimizer.Optimizer, checkpoint: dict[str, Any]) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    paddle.save(model.state_dict(), str(bundle_dir / MODEL_PARAMS_NAME))
    paddle.save(opt.state_dict(), str(bundle_dir / MODEL_OPT_NAME))
    paddle.save(checkpoint, str(bundle_dir / MODEL_CHECKPOINT_NAME))


def _print_epoch_summary(
    epoch_number: int,
    total_epochs: int,
    train_metrics: dict[str, float | int],
    eval_metrics: dict[str, float | int] | None,
    best_epoch: int | None,
    best_eval_mae: float | None,
) -> None:
    line = f"epoch {epoch_number}/{total_epochs} train_mae={float(train_metrics['mae']):.6f}"
    if "loss" in train_metrics:
        line += f" train_loss={float(train_metrics['loss']):.6f}"
    if eval_metrics is not None:
        line += f" eval_mae={float(eval_metrics['mae']):.6f} eval_mse={float(eval_metrics['mse']):.6f}"
        if "loss" in eval_metrics:
            line += f" eval_loss={float(eval_metrics['loss']):.6f}"
    if best_epoch is not None and best_eval_mae is not None:
        line += f" best_epoch={best_epoch} best_eval_mae={best_eval_mae:.6f}"
    print(line)


def _resolve_source(raw_dir: Path, source_name: str) -> DataSource:
    source_candidate = Path(source_name)
    if source_candidate.is_absolute() or source_candidate.is_dir():
        source_dir = source_candidate
    else:
        source_dir = raw_dir / source_name
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    data_path = _resolve_source_data_path(source_dir)
    return DataSource(name=source_dir.name, path=source_dir, data_path=data_path)


def _load_source_samples(source: DataSource) -> list[NavSample]:
    data_path = source.data_path
    payload = _load_source_records(data_path)
    samples: list[NavSample] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid record {data_path}#{index}: not an object")
        image_rel = _normalize_image_rel(record.get("img_path"), data_path, index)
        label = _label_from_record(record, data_path, index)
        image_path = source.path / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image {data_path}#{index}: {image_path}")
        samples.append(NavSample(source.name, image_rel, image_path, label))
    return samples


def _resolve_source_data_path(source_dir: Path) -> Path:
    data_json = source_dir / "data.json"
    if data_json.is_file():
        return data_json
    data_jsonl = source_dir / "data.jsonl"
    if data_jsonl.is_file():
        return data_jsonl
    raise FileNotFoundError(f"Missing data.json or data.jsonl: {source_dir}")


def _load_source_records(data_path: Path) -> list[dict[str, Any]]:
    if data_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(data_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid data.jsonl {data_path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid data.jsonl {data_path}:{line_number}: record must be an object")
            records.append(record)
        if not records:
            raise ValueError(f"Empty data.jsonl: {data_path}")
        return records

    payload = json.loads(data_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid data.json root: {data_path}")
    return payload


def _label_from_record(record: dict[str, Any], data_path: Path, index: int) -> tuple[float, float]:
    if "state" in record:
        state = _validate_state(record.get("state"), data_path, index)
        return state[2], 0.0
    if "deviation" in record:
        return _validate_deviation(record.get("deviation"), data_path, index), 0.0
    raise ValueError(f"Invalid record {data_path}#{index}: expected state or deviation")


def _normalize_image_rel(value: object, data_json: Path, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid img_path {data_json}#{index}: {value}")
    normalized = value.replace("\\", "/").strip()
    path = Path(normalized)
    if path.is_absolute() or ":" in normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError(f"Unsafe img_path {data_json}#{index}: {value}")
    return normalized


def _validate_state(value: object, data_json: Path, index: int) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Invalid state {data_json}#{index}: expected 3 numeric values")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
            raise ValueError(f"Invalid state {data_json}#{index}: expected finite numeric values")
        numbers.append(float(item))
    return numbers[0], numbers[1], numbers[2]


def _validate_deviation(value: object, data_json: Path, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Invalid deviation {data_json}#{index}: expected finite numeric value")
    return float(value)


def _load_rgb_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def _augment_like_paddlelane(
    image: np.ndarray,
    label: np.ndarray,
    forced_augmentation: str | None = None,
    augmentation_profile: str = "official",
) -> tuple[np.ndarray, np.ndarray]:
    if augmentation_profile == "light_corner_v3":
        return _augment_light_corner_v3(image, label, forced_augmentation)
    if augmentation_profile == "robust_v2":
        return _augment_robust_v2(image, label, forced_augmentation)

    official_augmentations = {
        "hue": _apply_hue,
        "saturation": _apply_saturation,
        "contrast": _apply_contrast,
        "brightness": _apply_brightness,
        "hflip": _apply_hflip,
    }
    if augmentation_profile == "official":
        augmentations = official_augmentations
    elif augmentation_profile == "robust":
        augmentations = {
            "none": _apply_none,
            **official_augmentations,
            "blur": _apply_blur,
            "noise": _apply_noise,
            "shadow": _apply_shadow,
        }
    else:
        raise ValueError(f"Unsupported augmentation profile: {augmentation_profile}")
    name = forced_augmentation or list(augmentations)[np.random.randint(0, len(augmentations))]
    if name not in augmentations:
        raise ValueError(f"Unsupported augmentation: {name}")
    image = augmentations[name](image)
    if name == "hflip":
        label = np.array([-label[0], label[1]], dtype=np.float32)
    return image, label


def _augment_robust_v2(
    image: np.ndarray,
    label: np.ndarray,
    forced_augmentation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result_image = image.astype(np.float32, copy=True)
    result_label = label.astype(np.float32, copy=True)
    forced_operations = {
        "robust_v2_none": _apply_none,
        "robust_v2_brightness": _apply_brightness_v2,
        "robust_v2_gamma": _apply_gamma_v2,
        "robust_v2_contrast": _apply_contrast_v2,
        "robust_v2_saturation": _apply_saturation_v2,
        "robust_v2_hue": _apply_hue_v2,
        "robust_v2_shadow": _apply_shadow,
        "robust_v2_blur": _apply_blur_v2,
        "robust_v2_noise": _apply_noise_v2,
    }
    if forced_augmentation is not None:
        if forced_augmentation == "robust_v2_hflip":
            return _apply_hflip(result_image), np.array([-result_label[0], result_label[1]], dtype=np.float32)
        operation = forced_operations.get(forced_augmentation)
        if operation is None:
            raise ValueError(f"Unsupported robust_v2 augmentation: {forced_augmentation}")
        return operation(result_image), result_label

    if np.random.uniform(0.0, 1.0) < 0.5:
        result_image = _apply_hflip(result_image)
        result_label = np.array([-result_label[0], result_label[1]], dtype=np.float32)

    light_operations = (
        _apply_none,
        _apply_brightness_v2,
        _apply_gamma_v2,
        _apply_contrast_v2,
        _apply_saturation_v2,
        _apply_hue_v2,
        _apply_shadow,
    )
    light_probabilities = (0.15, 0.25, 0.15, 0.15, 0.10, 0.05, 0.15)
    light_index = int(np.random.choice(len(light_operations), p=light_probabilities))
    result_image = light_operations[light_index](result_image)

    if np.random.uniform(0.0, 1.0) < 0.2:
        sensor_operation = _apply_blur_v2 if np.random.uniform(0.0, 1.0) < 0.5 else _apply_noise_v2
        result_image = sensor_operation(result_image)
    return result_image.astype(np.float32), result_label


def _augment_light_corner_v3(
    image: np.ndarray,
    label: np.ndarray,
    forced_augmentation: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    result_image = image.astype(np.float32, copy=True)
    result_label = label.astype(np.float32, copy=True)
    forced_operations = {
        "light_corner_v3_none": _apply_none,
        "light_corner_v3_exposure": _apply_exposure_v3,
        "light_corner_v3_gamma": _apply_gamma_v3,
        "light_corner_v3_contrast": _apply_contrast_v3,
        "light_corner_v3_temperature": _apply_temperature_v3,
        "light_corner_v3_shadow": _apply_shadow_v3,
        "light_corner_v3_glare": _apply_glare_v3,
        "light_corner_v3_blur": _apply_blur_v3,
        "light_corner_v3_noise": _apply_noise_v3,
    }
    if forced_augmentation is not None:
        if forced_augmentation == "light_corner_v3_hflip":
            return _apply_hflip(result_image), np.array([-result_label[0], result_label[1]], dtype=np.float32)
        operation = forced_operations.get(forced_augmentation)
        if operation is None:
            raise ValueError(f"Unsupported light_corner_v3 augmentation: {forced_augmentation}")
        return operation(result_image), result_label

    if np.random.uniform(0.0, 1.0) < 0.5:
        result_image = _apply_hflip(result_image)
        result_label = np.array([-result_label[0], result_label[1]], dtype=np.float32)

    global_operations = (
        _apply_none,
        _apply_exposure_v3,
        _apply_gamma_v3,
        _apply_contrast_v3,
        _apply_temperature_v3,
    )
    global_probabilities = (0.10, 0.30, 0.25, 0.20, 0.15)
    global_index = int(np.random.choice(len(global_operations), p=global_probabilities))
    result_image = global_operations[global_index](result_image)

    if np.random.uniform(0.0, 1.0) < 0.4:
        local_operation = _apply_shadow_v3 if np.random.uniform(0.0, 1.0) < 0.7 else _apply_glare_v3
        result_image = local_operation(result_image)

    if np.random.uniform(0.0, 1.0) < 0.2:
        sensor_operation = _apply_blur_v3 if np.random.uniform(0.0, 1.0) < 0.5 else _apply_noise_v3
        result_image = sensor_operation(result_image)
    return _clip_uint8(result_image), result_label


def filter_samples_by_label_jump(
    samples: list[NavSample],
    threshold: float | None,
    radius: int = 0,
) -> tuple[list[NavSample], dict[str, Any]]:
    if threshold is None:
        return samples, {
            "enabled": False,
            "input_samples": len(samples),
            "kept_samples": len(samples),
            "dropped_samples": 0,
        }
    if threshold < 0.0:
        raise ValueError("label jump threshold must be >= 0")
    if radius < 0:
        raise ValueError("label jump drop radius must be >= 0")
    grouped: dict[str, list[NavSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.source_name, []).append(sample)

    kept: list[NavSample] = []
    source_summaries: list[dict[str, Any]] = []
    for source_name, source_samples in grouped.items():
        labels = np.asarray([sample.label[0] for sample in source_samples], dtype=np.float32)
        drop_indices: set[int] = set()
        jump_indices: list[int] = []
        if len(labels) >= 2:
            jumps = np.abs(np.diff(labels))
            jump_indices = [int(index + 1) for index, value in enumerate(jumps) if float(value) >= threshold]
            for jump_index in jump_indices:
                start = max(0, jump_index - radius - 1)
                end = min(len(source_samples), jump_index + radius + 1)
                drop_indices.update(range(start, end))
        kept.extend(sample for index, sample in enumerate(source_samples) if index not in drop_indices)
        source_summaries.append(
            {
                "source_name": source_name,
                "input_samples": len(source_samples),
                "jump_count": len(jump_indices),
                "dropped_samples": len(drop_indices),
                "kept_samples": len(source_samples) - len(drop_indices),
            }
        )
    return kept, {
        "enabled": True,
        "threshold": threshold,
        "radius": radius,
        "input_samples": len(samples),
        "kept_samples": len(kept),
        "dropped_samples": len(samples) - len(kept),
        "sources": source_summaries,
    }


def build_deviation_sample_weights(samples: list[NavSample], mode: str = "abs_bins") -> np.ndarray:
    if not samples:
        raise ValueError("samples must not be empty")
    if mode not in SAMPLE_WEIGHT_MODES:
        raise ValueError(f"Unsupported sample weight mode: {mode}")
    if mode == "none":
        return np.ones(len(samples), dtype=np.float32)
    if mode == "signed_bins":
        return _build_signed_bin_sample_weights(samples)

    weights: list[float] = []
    for sample in samples:
        deviation = abs(float(sample.label[0]))
        if deviation < 1e-12:
            weight = 0.25
        elif deviation < 0.1:
            weight = 1.0
        elif deviation < 0.3:
            weight = 2.0
        elif deviation < 0.6:
            weight = 3.0
        else:
            weight = 4.0
        weights.append(weight)
    array = np.asarray(weights, dtype=np.float32)
    mean = float(array.mean())
    if mean > 0.0:
        array = array / mean
    return array


def _build_signed_bin_sample_weights(samples: list[NavSample]) -> np.ndarray:
    labels = np.asarray([sample.label[0] for sample in samples], dtype=np.float32)
    bin_ids = np.zeros(len(samples), dtype=np.int32)
    bin_ids[labels < -0.6] = 0
    bin_ids[(labels >= -0.6) & (labels < -0.3)] = 1
    bin_ids[(labels >= -0.3) & (labels < -0.1)] = 2
    bin_ids[(labels >= -0.1) & (labels < 0.1)] = 3
    bin_ids[(labels >= 0.1) & (labels < 0.3)] = 4
    bin_ids[(labels >= 0.3) & (labels < 0.6)] = 5
    bin_ids[labels >= 0.6] = 6
    counts = np.bincount(bin_ids, minlength=7).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = 1.0 / safe_counts[bin_ids]
    mean = float(weights.mean())
    if mean > 0.0:
        weights = weights / mean
    return weights.astype(np.float32)


def _apply_hue(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        return image
    delta = np.random.uniform(-18, 18)
    u = np.cos(delta * np.pi)
    w = np.sin(delta * np.pi)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.107, 1.705]])
    transform = np.dot(np.dot(ityiq, bt), tyiq).T
    return _clip_uint8(np.dot(image, transform))


def _apply_saturation(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        return image
    delta = np.random.uniform(0.5, 1.5)
    gray = (image * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)).sum(axis=2, keepdims=True)
    return _clip_uint8(image * delta + gray * (1.0 - delta))


def _apply_contrast(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        return image
    return _clip_uint8(image * np.random.uniform(0.5, 1.5))


def _apply_brightness(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        return image
    return _clip_uint8(image + np.random.uniform(0.5, 1.5))


def _apply_brightness_v2(image: np.ndarray) -> np.ndarray:
    gain = float(np.random.uniform(0.75, 1.25))
    offset = float(np.random.uniform(-20.0, 20.0))
    return _clip_uint8(image * gain + offset)


def _apply_gamma_v2(image: np.ndarray) -> np.ndarray:
    gamma = float(np.random.uniform(0.8, 1.25))
    normalized = np.clip(image, 0.0, 255.0) / 255.0
    return _clip_uint8(np.power(normalized, gamma) * 255.0)


def _apply_contrast_v2(image: np.ndarray) -> np.ndarray:
    factor = float(np.random.uniform(0.8, 1.2))
    mean = image.mean(axis=(0, 1), keepdims=True)
    return _clip_uint8(mean + (image - mean) * factor)


def _apply_saturation_v2(image: np.ndarray) -> np.ndarray:
    factor = float(np.random.uniform(0.8, 1.2))
    gray = (image * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)).sum(axis=2, keepdims=True)
    return _clip_uint8(image * factor + gray * (1.0 - factor))


def _apply_hue_v2(image: np.ndarray) -> np.ndarray:
    angle = float(np.random.uniform(-8.0, 8.0)) * np.pi / 180.0
    u = np.cos(angle)
    w = np.sin(angle)
    bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
    tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321], [0.211, -0.523, 0.311]])
    ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647], [1.0, -1.107, 1.705]])
    transform = np.dot(np.dot(ityiq, bt), tyiq).T
    return _clip_uint8(np.dot(image, transform))


def _apply_hflip(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.flip(image, axis=1))


def _apply_none(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32, copy=True)


def _apply_blur(image: np.ndarray) -> np.ndarray:
    radius = float(np.random.uniform(0.3, 1.2))
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil_image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def _apply_noise(image: np.ndarray) -> np.ndarray:
    sigma = float(np.random.uniform(1.0, 5.0))
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return _clip_uint8(image + noise)


def _apply_blur_v2(image: np.ndarray) -> np.ndarray:
    radius = float(np.random.uniform(0.3, 0.9))
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil_image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def _apply_noise_v2(image: np.ndarray) -> np.ndarray:
    sigma = float(np.random.uniform(1.0, 3.0))
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return _clip_uint8(image + noise)


def _apply_exposure_v3(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        gain = float(np.random.uniform(0.55, 0.85))
        offset = float(np.random.uniform(-18.0, 0.0))
    else:
        gain = float(np.random.uniform(1.2, 1.65))
        offset = float(np.random.uniform(5.0, 30.0))
    return _clip_uint8(image * gain + offset)


def _apply_gamma_v3(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        gamma = float(np.random.uniform(0.55, 0.78))
    else:
        gamma = float(np.random.uniform(1.3, 1.8))
    normalized = np.clip(image, 0.0, 255.0) / 255.0
    return _clip_uint8(np.power(normalized, gamma) * 255.0)


def _apply_contrast_v3(image: np.ndarray) -> np.ndarray:
    if np.random.uniform(0.0, 1.0) < 0.5:
        factor = float(np.random.uniform(0.55, 0.8))
    else:
        factor = float(np.random.uniform(1.2, 1.5))
    mean = image.mean(axis=(0, 1), keepdims=True)
    return _clip_uint8(mean + (image - mean) * factor)


def _apply_temperature_v3(image: np.ndarray) -> np.ndarray:
    shift = float(np.random.uniform(-0.18, 0.18))
    gains = np.array([1.0 + shift, 1.0, 1.0 - shift], dtype=np.float32).reshape(1, 1, 3)
    return _clip_uint8(image * gains)


def _apply_blur_v3(image: np.ndarray) -> np.ndarray:
    radius = float(np.random.uniform(0.4, 1.3))
    pil_image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
    return np.asarray(pil_image.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def _apply_noise_v3(image: np.ndarray) -> np.ndarray:
    sigma = float(np.random.uniform(2.0, 8.0))
    noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return _clip_uint8(image + noise)


def _apply_shadow(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    horizontal = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    if np.random.uniform(0.0, 1.0) < 0.5:
        horizontal = 1.0 - horizontal
    center = float(np.random.uniform(0.25, 0.75))
    softness = float(np.random.uniform(0.18, 0.35))
    shadow_strength = float(np.random.uniform(0.15, 0.35))
    mask = np.exp(-((horizontal - center) ** 2) / max(2.0 * softness**2, 1e-6))
    mask = np.broadcast_to(mask, (height, width, 1))
    return _clip_uint8(image * (1.0 - shadow_strength * mask))


def _apply_shadow_v3(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    center_x = float(np.random.uniform(0.15, 0.85))
    center_y = float(np.random.uniform(0.15, 0.85))
    spread_x = float(np.random.uniform(0.25, 0.55))
    spread_y = float(np.random.uniform(0.3, 0.7))
    strength = float(np.random.uniform(0.3, 0.65))
    mask = np.exp(-(((x - center_x) / spread_x) ** 2 + ((y - center_y) / spread_y) ** 2) / 2.0)
    return _clip_uint8(image * (1.0 - strength * mask[:, :, None]))


def _apply_glare_v3(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    center_x = float(np.random.uniform(0.1, 0.9))
    center_y = float(np.random.uniform(0.05, 0.75))
    spread = float(np.random.uniform(0.18, 0.4))
    strength = float(np.random.uniform(35.0, 100.0))
    mask = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / max(2.0 * spread**2, 1e-6))
    return _clip_uint8(image + strength * mask[:, :, None])


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.float32)


def _normalize_to_chw(image: np.ndarray) -> np.ndarray:
    image = (image.astype(np.float32) - 127.5) / 127.5
    return image.transpose((2, 0, 1)).astype(np.float32)


def _parse_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ValueError("At least one source name is required")
    return names


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("At least one integer value is required")
    return items


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    items = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("At least one float value is required")
    return items


def _validate_train_config(config: TrainConfig) -> None:
    if config.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if config.batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if config.optimizer_name not in OPTIMIZER_MODES:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")
    if config.weight_decay < 0.0:
        raise ValueError("--weight-decay must be >= 0")
    if config.learning_rate <= 0:
        raise ValueError("--lr must be positive")
    if config.lr_t_max <= 0:
        raise ValueError("--lr-t-max must be positive")
    if config.lr_scheduler not in LR_SCHEDULER_MODES:
        raise ValueError(f"Unsupported lr scheduler: {config.lr_scheduler}")
    if config.warmup_epochs <= 0:
        raise ValueError("--warmup-epochs must be positive")
    if config.warmup_start_lr <= 0.0 or config.warmup_start_lr > config.learning_rate:
        raise ValueError("--warmup-start-lr must be positive and no greater than --lr")
    if config.lr_eta_min < 0.0 or config.lr_eta_min > config.learning_rate:
        raise ValueError("--lr-eta-min must be >= 0 and no greater than --lr")
    if config.lr_scheduler == "cosine_warmup" and config.warmup_epochs > config.lr_t_max:
        raise ValueError("--warmup-epochs must not exceed --lr-t-max for cosine_warmup")
    if any(boundary <= 0 for boundary in config.lr_boundaries):
        raise ValueError("--lr-boundaries values must be positive")
    if tuple(sorted(config.lr_boundaries)) != config.lr_boundaries:
        raise ValueError("--lr-boundaries must be sorted")
    if len(config.lr_values) != len(config.lr_boundaries) + 1:
        raise ValueError("--lr-values must contain exactly len(boundaries)+1 values")
    if any(value <= 0.0 for value in config.lr_values):
        raise ValueError("--lr-values values must be positive")
    if config.eval_interval <= 0:
        raise ValueError("--eval-interval must be positive")
    if config.early_stop_min_epochs <= 0:
        raise ValueError("--early-stop-min-epochs must be positive")
    if config.early_stop_patience_evals <= 0:
        raise ValueError("--early-stop-patience-evals must be positive")
    if config.early_stop_min_delta < 0.0:
        raise ValueError("--early-stop-min-delta must be >= 0")
    if config.deviation_loss_weight_mode not in DEVIATION_LOSS_WEIGHT_MODES:
        raise ValueError(f"Unsupported deviation loss weight mode: {config.deviation_loss_weight_mode}")
    if config.augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(f"Unsupported augmentation profile: {config.augmentation_profile}")
    if config.sample_weight_mode not in SAMPLE_WEIGHT_MODES:
        raise ValueError(f"Unsupported sample weight mode: {config.sample_weight_mode}")
    if config.label_jump_drop_threshold is not None and config.label_jump_drop_threshold < 0.0:
        raise ValueError("--label-jump-drop-threshold must be >= 0")
    if config.label_jump_drop_radius < 0:
        raise ValueError("--label-jump-drop-radius must be >= 0")
    if config.init_params_path is not None and not config.init_params_path.is_file():
        raise FileNotFoundError(f"Missing init params path: {config.init_params_path}")
    if config.init_params_path is not None and config.resume_dir is not None:
        raise ValueError("--init-params-path and --resume-dir cannot be used together")
    if config.resume_dir is not None:
        if not config.resume_dir.is_dir():
            raise FileNotFoundError(f"Missing resume directory: {config.resume_dir}")
        if config.run_name is None:
            raise ValueError("--run-name is required with --resume-dir")


def _should_eval(epoch_index: int, epochs: int, eval_interval: int) -> bool:
    epoch_number = epoch_index + 1
    return epoch_number == epochs or epoch_number % eval_interval == 0


def _jsonable_config(config: TrainConfig, run_name: str) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_dir"] = str(config.dataset_dir)
    payload["output_dir"] = str(config.output_dir)
    payload["train_sources"] = list(config.train_sources)
    payload["eval_sources"] = list(config.eval_sources)
    payload["input_size"] = list(config.input_size)
    payload["lr_boundaries"] = list(config.lr_boundaries)
    payload["lr_values"] = list(config.lr_values)
    payload["run_name"] = run_name
    payload["init_params_path"] = None if config.init_params_path is None else str(config.init_params_path)
    payload["resume_dir"] = None if config.resume_dir is None else str(config.resume_dir)
    return payload


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.check_only:
        result = run_forward_check(
            args.dataset_dir,
            split=args.split,
            batch_size=args.batch_size,
            device=args.device,
            train_sources=_parse_names(args.train_sources),
            eval_sources=_parse_names(args.eval_sources),
        )
        print("nav_control_optional forward check passed")
        for key, value in result.items():
            print(f"{key}: {value}")
        print("training: not started")
        return 0

    result = train_model(config_from_args(args))
    print("nav_control_optional training finished")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("note: offline checkpoint only; not validated for onboard closed-loop control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
