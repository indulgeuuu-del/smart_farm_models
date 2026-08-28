"""Run and profile the exported 01_target_det full-NMS model as the onboard inference entry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "deploy" / "onboard" / "01_target_det" / "runs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_det_model import resolve_infer_model_files
from scripts.detection._target_det_recipe import resolve_paddledet_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image-file", help="Single image path for onboard inference.")
    source_group.add_argument("--image-dir", help="Directory of images for onboard inference.")
    parser.add_argument("--model-dir", required=True, help="Full-NMS exported model directory.")
    parser.add_argument(
        "--paddledet-root",
        default=None,
        help="Pinned PaddleDetection source root. Defaults to SMART_FARM_PADDLEDET_ROOT or third_party/PaddleDetection.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to deploy/onboard/01_target_det/runs/<timestamp>.")
    parser.add_argument("--device", default="GPU", help="Inference device: CPU/GPU/XPU/NPU.")
    parser.add_argument(
        "--run-mode",
        default="paddle",
        choices=["paddle", "trt_fp32", "trt_fp16", "trt_int8"],
        help="PaddleDetection inference backend. Use trt_fp16 on Jetson when TensorRT is available.",
    )
    parser.add_argument("--trt-min-shape", type=int, default=1, help="TensorRT min dynamic shape.")
    parser.add_argument("--trt-max-shape", type=int, default=1280, help="TensorRT max dynamic shape.")
    parser.add_argument("--trt-opt-shape", type=int, default=416, help="TensorRT optimal shape. Match infer_cfg Resize size.")
    parser.add_argument("--trt-calib-mode", action="store_true", help="Enable TensorRT calibration mode for int8 models.")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU thread count when running on CPU.")
    parser.add_argument("--enable-mkldnn", action="store_true", help="Enable MKLDNN when running on CPU.")
    parser.add_argument("--run-benchmark", action="store_true", help="Use PaddleDetection benchmark path with warmup and repeats.")
    parser.add_argument("--repeats", type=int, default=50, help="Benchmark repeats per batch when --run-benchmark is enabled.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Visualization and post-filter threshold.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional max number of images to process.")
    parser.add_argument("--use-fd-format", action="store_true", help="Pass through to PaddleDetection deploy infer config parser.")
    parser.add_argument("--no-save-images", action="store_true", help="Do not export visualization images.")
    parser.add_argument("--no-save-coco", action="store_true", help="Do not export PaddleDetection bbox.json.")
    return parser


def default_output_dir() -> Path:
    return DEFAULT_OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")


def configure_runtime_library_paths() -> None:
    """Add local NVIDIA runtime DLL directories shipped inside the active venv."""
    site_packages = Path(sys.executable).resolve().parents[1] / "Lib" / "site-packages"
    candidates = [
        site_packages / "tensorrt_libs",
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(candidate))


def load_vendor_infer_module(paddledet_root: Path):
    infer_path = paddledet_root / "deploy" / "python" / "infer.py"
    if not infer_path.is_file():
        raise FileNotFoundError(f"Missing PaddleDetection deploy inference entry: {infer_path}")
    spec = importlib.util.spec_from_file_location("paddledet_deploy_infer", infer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load PaddleDetection deploy infer module: {infer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_vendor_flags(vendor_infer: Any, output_dir: Path) -> None:
    """Provide the minimal FLAGS expected by PaddleDetection's imported deploy module."""
    vendor_infer.FLAGS = SimpleNamespace(
        collect_trt_shape_info=False,
        tuned_trt_shape_file=str(output_dir / "trt_shape_info.txt"),
        use_coco_category=False,
    )


def build_detection_records(
    image_list: list[str],
    merged_result: dict[str, Any],
    labels: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    boxes_num = merged_result.get("boxes_num")
    boxes = merged_result.get("boxes")
    if boxes_num is None:
        raise ValueError("Merged inference result does not contain boxes_num.")

    start = 0
    for index, image_path in enumerate(image_list):
        count = int(boxes_num[index])
        current_boxes = [] if boxes is None else boxes[start:start + count]
        if hasattr(current_boxes, "tolist"):
            current_boxes = current_boxes.tolist()
        detections: list[dict[str, Any]] = []
        for box in current_boxes:
            class_id = int(box[0])
            score = float(box[1])
            if score < threshold:
                continue
            x1, y1, x2, y2 = [float(value) for value in box[2:6]]
            detections.append(
                {
                    "class_id": class_id,
                    "category_id": class_id + 1,
                    "label": labels[class_id],
                    "score": score,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                }
            )
        records.append(
            {
                "image_index": index,
                "image_name": Path(image_path).name,
                "image_path": str(Path(image_path).resolve()),
                "detections": detections,
            }
        )
        start += count
    return records


def build_timing_summary(timer: Any) -> dict[str, Any]:
    """Return per-image timing values from PaddleDetection's timer object."""
    if timer is None or not hasattr(timer, "report"):
        return {}
    try:
        average = timer.report(average=True)
    except Exception:
        return {}
    if not isinstance(average, dict):
        return {}

    preprocess = float(average.get("preprocess_time_s", 0.0) or 0.0)
    inference = float(average.get("inference_time_s", 0.0) or 0.0)
    postprocess = float(average.get("postprocess_time_s", 0.0) or 0.0)
    latency = preprocess + inference + postprocess
    fps = (1.0 / latency) if latency > 0 else None
    return {
        "img_num": int(average.get("img_num", 0) or 0),
        "preprocess_time_ms": round(preprocess * 1000, 3),
        "inference_time_ms": round(inference * 1000, 3),
        "postprocess_time_ms": round(postprocess * 1000, 3),
        "latency_ms": round(latency * 1000, 3),
        "fps": round(fps, 3) if fps is not None else None,
    }


def write_outputs(
    *,
    output_dir: Path,
    model_dir: Path,
    image_list: list[str],
    threshold: float,
    device: str,
    run_mode: str,
    timing: dict[str, Any],
    records: list[dict[str, Any]],
    save_coco: bool,
) -> tuple[Path, Path]:
    total_detections = sum(len(item["detections"]) for item in records)
    result = {
        "model_dir": str(model_dir.resolve()),
        "device": device,
        "run_mode": run_mode,
        "threshold": threshold,
        "timing": timing,
        "image_count": len(image_list),
        "total_detections": total_detections,
        "images": records,
    }
    result_path = output_dir / "onboard_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    if save_coco:
        bbox_results: list[dict[str, Any]] = []
        for item in records:
            for det in item["detections"]:
                bbox_results.append(
                    {
                        "image_id": item["image_index"],
                        "category_id": det["category_id"],
                        "file_name": item["image_name"],
                        "bbox": det["bbox_xywh"],
                        "score": det["score"],
                    }
                )
        bbox_path = output_dir / "bbox.json"
        bbox_path.write_text(json.dumps(bbox_results, ensure_ascii=False), encoding="utf-8")

    summary = {
        "model_dir": str(model_dir.resolve()),
        "device": device,
        "run_mode": run_mode,
        "threshold": threshold,
        "timing": timing,
        "image_count": len(image_list),
        "total_detections": total_detections,
        "output_dir": str(output_dir.resolve()),
        "result_path": str(result_path.resolve()),
        "bbox_path": str((output_dir / "bbox.json").resolve()) if save_coco else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_path, summary_path


def main() -> int:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).resolve()
    resolve_infer_model_files(model_dir)
    paddledet_root = resolve_paddledet_root(REPO_ROOT, args.paddledet_root)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_runtime_library_paths()

    import paddle

    paddle.enable_static()
    vendor_infer = load_vendor_infer_module(paddledet_root)
    configure_vendor_flags(vendor_infer, output_dir)

    image_list = vendor_infer.get_test_images(args.image_dir, args.image_file)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("--max-images must be positive.")
        image_list = image_list[:args.max_images]

    detector = vendor_infer.Detector(
        str(model_dir),
        device=args.device.upper(),
        run_mode=args.run_mode,
        batch_size=args.batch_size,
        trt_min_shape=args.trt_min_shape,
        trt_max_shape=args.trt_max_shape,
        trt_opt_shape=args.trt_opt_shape,
        trt_calib_mode=args.trt_calib_mode,
        cpu_threads=args.cpu_threads,
        enable_mkldnn=args.enable_mkldnn,
        threshold=args.threshold,
        output_dir=str(output_dir),
        use_fd_format=args.use_fd_format,
    )

    merged_result = detector.predict_image(
        image_list,
        run_benchmark=args.run_benchmark,
        repeats=args.repeats,
        visual=not args.no_save_images,
        save_results=False,
    )
    records = build_detection_records(image_list, merged_result, detector.pred_config.labels, args.threshold)
    timing = build_timing_summary(detector.get_timer() if hasattr(detector, "get_timer") else None)
    result_path, summary_path = write_outputs(
        output_dir=output_dir,
        model_dir=model_dir,
        image_list=image_list,
        threshold=args.threshold,
        device=args.device.upper(),
        run_mode=args.run_mode,
        timing=timing,
        records=records,
        save_coco=not args.no_save_coco,
    )

    print(f"OUTPUT_DIR={output_dir}")
    print(f"RESULT_JSON={result_path}")
    print(f"SUMMARY_JSON={summary_path}")
    if not args.no_save_coco:
        print(f"BBOX_JSON={output_dir / 'bbox.json'}")
    if timing:
        print(f"LATENCY_MS={timing['latency_ms']}")
        print(f"FPS={timing['fps']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
