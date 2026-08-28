# 功能：封装 01_target_det 的默认 PP-YOLOE+_s 416 PaddleDetection 主训练与导出命令配方。
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence


PADDLEDET_ROOT_ENV = "SMART_FARM_PADDLEDET_ROOT"
DEFAULT_CONFIG_PATH = Path("./configs/target_det.yaml")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(repo_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def is_paddledet_root(path: str | Path) -> bool:
    root = Path(path)
    return (
        root.is_dir()
        and (root / "tools" / "train.py").is_file()
        and (root / "tools" / "export_model.py").is_file()
        and (root / "ppdet" / "__init__.py").is_file()
    )


def resolve_paddledet_root(repo_root: str | Path, explicit_root: str | Path | None = None) -> Path:
    root = Path(repo_root)
    if explicit_root is not None:
        candidate = resolve_repo_path(root, explicit_root).resolve()
        if not is_paddledet_root(candidate):
            raise FileNotFoundError(f"Invalid PaddleDetection root: {candidate}")
        return candidate

    env_root = os.environ.get(PADDLEDET_ROOT_ENV)
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if not is_paddledet_root(candidate):
            raise FileNotFoundError(f"{PADDLEDET_ROOT_ENV} does not point to a PaddleDetection root: {candidate}")
        return candidate

    candidates: list[Path] = [root / "third_party" / "PaddleDetection"]
    venv_src = root / ".venv" / "src"
    if venv_src.is_dir():
        exact = venv_src / "PaddleDetection"
        if exact not in candidates:
            candidates.append(exact)
        for candidate in sorted(venv_src.glob("PaddleDetection*")):
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        resolved = candidate.resolve()
        if is_paddledet_root(resolved):
            return resolved

    raise FileNotFoundError(
        "PaddleDetection source root not found. "
        f"Set {PADDLEDET_ROOT_ENV} or pass --paddledet-root."
    )


def build_train_command(
    *,
    python_executable: str,
    paddledet_root: Path,
    config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    use_vdl: bool,
    amp: bool,
    resume_path: Path | None,
    batch_size: int | None = None,
    base_lr: float | None = None,
    worker_num: int | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(paddledet_root / "tools" / "train.py"),
        "-c",
        str(config_path),
        "--eval",
    ]
    if amp:
        command.append("--amp")
    if resume_path is not None:
        command.extend(["-r", str(resume_path)])
    if use_vdl:
        command.append("--use_vdl=True")

    overrides = [
        "use_gpu=true",
        f"save_dir={output_dir}",
        f"TrainDataset.dataset_dir={dataset_dir}",
        f"EvalDataset.dataset_dir={dataset_dir}",
        f"TestDataset.dataset_dir={dataset_dir}",
    ]
    if batch_size is not None:
        overrides.append(f"TrainReader.batch_size={batch_size}")
    if base_lr is not None:
        overrides.append(f"LearningRate.base_lr={base_lr}")
    if worker_num is not None:
        overrides.append(f"worker_num={worker_num}")

    command.extend(["-o", *overrides])
    return command


def build_export_command(
    *,
    python_executable: str,
    paddledet_root: Path,
    config_path: Path,
    weights_path: Path,
    output_dir: Path,
    trt: bool,
    exclude_nms: bool,
    dataset_dir: Path | None = None,
    use_gpu: bool | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(paddledet_root / "tools" / "export_model.py"),
        "-c",
        str(config_path),
        "--output_dir",
        str(output_dir),
    ]
    overrides = [f"weights={weights_path}"]
    if use_gpu is not None:
        overrides.append(f"use_gpu={str(use_gpu).lower()}")
    if dataset_dir is not None:
        overrides.extend(
            [
                f"TrainDataset.dataset_dir={dataset_dir}",
                f"EvalDataset.dataset_dir={dataset_dir}",
                f"TestDataset.dataset_dir={dataset_dir}",
            ]
        )
    if trt:
        overrides.append("trt=True")
    if exclude_nms:
        overrides.append("exclude_nms=True")
    command.extend(["-o", *overrides])
    return command


def format_command(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in command])


def run_command(command: Sequence[str], cwd: str | Path) -> int:
    return subprocess.call([str(part) for part in command], cwd=str(cwd))
