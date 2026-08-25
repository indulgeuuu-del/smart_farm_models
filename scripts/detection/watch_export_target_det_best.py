# 功能：等待 PaddleDetection 训练到指定 epoch 后，导出当前 best_model 为上位机兼容格式。
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(r"Epoch:\s*\[(?P<epoch>\d+)\]\s*\[\s*(?P<step>\d+)\s*/\s*(?P<steps>\d+)\]")
BEST_AP_RE = re.compile(r"Best test bbox ap is (?P<value>[0-9]+(?:\.[0-9]+)?)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Active PaddleDetection output directory.")
    parser.add_argument("--config", type=Path, required=True, help="PaddleDetection config path.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="PaddleX/COCO dataset directory used by this run.")
    parser.add_argument("--target-epoch", type=int, default=60, help="Export when the latest logged epoch is >= this value.")
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Polling interval in seconds.")
    parser.add_argument("--stable-seconds", type=float, default=20.0, help="How long best_model files must stay unchanged.")
    parser.add_argument("--paddle261-python", type=Path, default=Path(".venv_paddle261_export/Scripts/python.exe"))
    parser.add_argument("--export-root", type=Path, required=True, help="PaddleDetection export output root.")
    parser.add_argument("--handoff-dir", type=Path, required=True, help="Upper-computer handoff package directory.")
    parser.add_argument("--status-json", type=Path, required=True, help="Status JSON written throughout the watcher lifecycle.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU during export. Default is CPU to avoid disturbing training.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    run_dir = resolve(repo_root, args.run_dir)
    config = resolve(repo_root, args.config)
    dataset_dir = resolve(repo_root, args.dataset_dir)
    export_root = resolve(repo_root, args.export_root)
    handoff_dir = unique_dir(resolve(repo_root, args.handoff_dir))
    status_json = resolve(repo_root, args.status_json)
    paddle261_python = resolve(repo_root, args.paddle261_python)

    log_path = run_dir / "train_stdout.log"
    weights_prefix = run_dir / "best_model"
    best_model_files = [weights_prefix.with_suffix(suffix) for suffix in (".pdparams", ".pdema", ".pdopt")]

    write_status(
        status_json,
        "waiting",
        {
            "target_epoch": args.target_epoch,
            "run_dir": str(run_dir),
            "config": str(config),
            "dataset_dir": str(dataset_dir),
            "export_root": str(export_root),
            "handoff_dir": str(handoff_dir),
        },
    )

    latest = wait_until_epoch(log_path, args.target_epoch, args.poll_interval, status_json)
    wait_for_stable_files(best_model_files, args.stable_seconds, args.poll_interval, status_json)

    export_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(paddle261_python),
        str(repo_root / "scripts" / "detection" / "export_target_det_model.py"),
        "--config",
        str(config),
        "--weights",
        str(weights_prefix),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(export_root),
        "--no-use-gpu" if not args.use_gpu else "--use-gpu",
    ]
    env = export_env(repo_root)
    write_status(status_json, "exporting", {"latest": latest, "command": subprocess.list2cmdline(command)})
    export_result = subprocess.run(command, cwd=repo_root, env=env, text=True, capture_output=True)
    (export_root / "export_stdout.log").write_text(export_result.stdout, encoding="utf-8")
    (export_root / "export_stderr.log").write_text(export_result.stderr, encoding="utf-8")
    if export_result.returncode != 0:
        write_status(status_json, "export_failed", {"returncode": export_result.returncode})
        return export_result.returncode

    model_dir = find_exported_model_dir(export_root)
    update_infer_cfg(model_dir / "infer_cfg.yml")
    target_det_dir = handoff_dir / "target_det"
    copy_model_dir(model_dir, target_det_dir)
    copy_label_files(dataset_dir, target_det_dir)
    write_readme(handoff_dir, target_det_dir, run_dir, dataset_dir, config, latest, latest_best_ap(log_path))
    write_sha256(target_det_dir)

    compat_output = run_compat_check(repo_root, target_det_dir)
    (handoff_dir / "compat_check.txt").write_text(compat_output, encoding="utf-8")
    zip_path = zip_directory(handoff_dir)
    write_status(
        status_json,
        "done",
        {
            "latest": latest,
            "best_ap": latest_best_ap(log_path),
            "export_model_dir": str(model_dir),
            "handoff_dir": str(handoff_dir),
            "zip_path": str(zip_path),
            "compat_check": compat_output,
        },
    )
    return 0


def resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def write_status(path: Path, status: str, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"status": status, "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def wait_until_epoch(log_path: Path, target_epoch: int, interval: float, status_json: Path) -> dict[str, Any]:
    while True:
        latest = latest_train_point(log_path)
        write_status(status_json, "waiting", {"target_epoch": target_epoch, "latest": latest})
        if latest and int(latest["epoch"]) >= target_epoch:
            return latest
        time.sleep(interval)


def latest_train_point(log_path: Path) -> dict[str, Any] | None:
    if not log_path.is_file():
        return None
    latest = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            latest = {
                "epoch": int(match.group("epoch")),
                "step": int(match.group("step")),
                "steps_per_epoch": int(match.group("steps")),
                "raw": line,
            }
    return latest


def wait_for_stable_files(files: list[Path], stable_seconds: float, interval: float, status_json: Path) -> None:
    while True:
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            write_status(status_json, "waiting_for_best_model", {"missing": missing})
            time.sleep(interval)
            continue
        first = [file_signature(path) for path in files]
        time.sleep(stable_seconds)
        second = [file_signature(path) for path in files]
        if first == second:
            return
        write_status(status_json, "waiting_for_stable_best_model", {"files": [str(path) for path in files]})


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def export_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    extra_bins = [
        repo_root / ".venv_paddle261_export" / "Lib" / "site-packages" / "nvidia" / "cudnn" / "bin",
        repo_root / ".venv_paddle261_export" / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
        repo_root / ".venv_paddle261_export" / "Lib" / "site-packages" / "nvidia" / "cuda_nvrtc" / "bin",
    ]
    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(str(path) for path in extra_bins if path.is_dir()) + os.pathsep + existing
    return env


def find_exported_model_dir(export_root: Path) -> Path:
    candidates = [export_root, *[path for path in export_root.rglob("*") if path.is_dir()]]
    for candidate in candidates:
        if (candidate / "model.pdmodel").is_file() and (candidate / "model.pdiparams").is_file() and (candidate / "infer_cfg.yml").is_file():
            return candidate
    raise FileNotFoundError(f"No model.pdmodel + model.pdiparams + infer_cfg.yml found under {export_root}")


def update_infer_cfg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = replace_or_insert_yaml_scalar(text, "use_dynamic_shape", "true", after_key="metric")
    text = replace_or_insert_yaml_scalar(text, "min_subgraph_size", "3", after_key="arch")
    path.write_text(text, encoding="utf-8")


def replace_or_insert_yaml_scalar(text: str, key: str, value: str, *, after_key: str) -> str:
    pattern = re.compile(rf"^{re.escape(key)}:\s*.*$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(f"{key}: {value}", text)
    after_pattern = re.compile(rf"^({re.escape(after_key)}:\s*.*)$", re.MULTILINE)
    if after_pattern.search(text):
        return after_pattern.sub(rf"\1\n{key}: {value}", text, count=1)
    return f"{key}: {value}\n{text}"


def copy_model_dir(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def copy_label_files(dataset_dir: Path, target_dir: Path) -> None:
    for name in ("label_list.txt", "class_names.txt"):
        source = dataset_dir / name
        if source.is_file():
            shutil.copy2(source, target_dir / name)


def write_readme(
    handoff_dir: Path,
    target_det_dir: Path,
    run_dir: Path,
    dataset_dir: Path,
    config: Path,
    latest: dict[str, Any],
    best_ap: float | None,
) -> None:
    labels = []
    label_file = target_det_dir / "label_list.txt"
    if label_file.is_file():
        labels = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_lines = "\n".join(f"{index} {name}" for index, name in enumerate(labels)) or "未找到 label_list.txt"
    class_count = len(labels) if labels else "未知"
    current_epoch = latest.get("epoch")
    readme = f"""# {class_count} 类融合检测 current best 上车候选包

## 用途

这是 `01_target_det` {class_count} 类融合模型在训练进行中按当前用户指令即时导出的 `best_model` 上车候选包。

上位机队友使用目录：

```text
target_det/
```

## 核心口径

- 模型结构：PP-YOLOE+_s
- 输入尺寸：416 x 416
- 类别数：{class_count}
- 导出格式：`model.pdmodel + model.pdiparams + infer_cfg.yml`
- 后处理：full NMS 已包含在导出模型中
- 建议 Jetson 端优先测试：`run_mode=trt_fp16`
- 若 TRT 环境未跑通，可先回退：`run_mode=paddle`
- `infer_cfg.yml` 已设置：`use_dynamic_shape: true`、`min_subgraph_size: 3`

## 当前训练进度

```text
epoch = {latest.get("epoch")}
step = {latest.get("step")} / {latest.get("steps_per_epoch")}
best_ap = {best_ap if best_ap is not None else "未解析到"}
```

## 来源

训练目录：

```text
{run_dir}
```

数据目录：

```text
{dataset_dir}
```

配置文件：

```text
{config}
```

## 标签顺序

```text
{label_lines}
```

## 文件

```text
target_det/model.pdmodel
target_det/model.pdiparams
target_det/model.pdiparams.info
target_det/infer_cfg.yml
target_det/label_list.txt
target_det/class_names.txt
target_det/SHA256.txt
```

## 注意

- 这是训练中途导出的 current best 候选包，导出时日志最新 epoch 为 `{current_epoch}`，不是正式训练最终包。
- 最终是否上车替代双模型方案，需要看 Jetson 端 FPS、连续帧稳定性和弱类/蔬菜类召回。
"""
    handoff_dir.mkdir(parents=True, exist_ok=True)
    (handoff_dir / "README.md").write_text(readme, encoding="utf-8")


def write_sha256(target_dir: Path) -> None:
    lines = []
    for path in sorted(p for p in target_dir.iterdir() if p.is_file() and p.name != "SHA256.txt"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        lines.append(f"{digest}  {path.name}")
    (target_dir / "SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compat_check(repo_root: Path, target_det_dir: Path) -> str:
    command = [
        str(repo_root / ".venv" / "Scripts" / "python.exe"),
        str(repo_root / "scripts" / "onboard" / "check_target_det_baseline_compat.py"),
        "--model-dir",
        str(target_det_dir),
        "--allow-custom-labels",
    ]
    result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(output)
    return output


def zip_directory(directory: Path) -> Path:
    zip_path = directory.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in directory.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(directory.parent))
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()
    zip_path.with_suffix(".zip.sha256").write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    return zip_path


def latest_best_ap(log_path: Path) -> float | None:
    if not log_path.is_file():
        return None
    best_ap = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = BEST_AP_RE.search(line)
        if match:
            best_ap = float(match.group("value"))
    return best_ap


def unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{index:02d}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Cannot find unused directory name for {path}")


if __name__ == "__main__":
    raise SystemExit(main())
