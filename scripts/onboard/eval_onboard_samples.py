"""Preflight exported target_det models against unlabeled onboard sample captures.

This tool reports sample coverage and can optionally invoke the real image
inference runner. It never reports AP because the input samples carry no
ground-truth annotations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.common.target_det_model import resolve_infer_model_files

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", required=True, help="Directory containing unlabeled onboard sample images.")
    parser.add_argument("--model-dir", required=True, help="Exported target_det model directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for the preflight manifest and optional inference output.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional positive cap on discovered sample images.")
    parser.add_argument("--run-inference", action="store_true", help="Run scripts/onboard/run_target_det_onboard.py after structural preflight.")
    parser.add_argument("--paddledet-root", default=None, help="Optional PaddleDetection source root passed to the inference runner.")
    parser.add_argument("--device", default="GPU", help="Inference device passed to the runner.")
    parser.add_argument("--run-mode", default="paddle", help="Inference mode passed to the runner.")
    parser.add_argument("--threshold", type=float, default=0.3, help="Inference threshold passed to the runner.")
    return parser


def discover_images(samples_dir: Path, max_images: int | None = None) -> list[Path]:
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"Missing sample directory: {samples_dir}")
    if max_images is not None and max_images <= 0:
        raise ValueError("--max-images must be positive.")
    images = sorted(path for path in samples_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
    return images if max_images is None else images[:max_images]


def default_output_dir() -> Path:
    return REPO_ROOT / "deploy" / "onboard" / "01_target_det" / "sample_preflight" / datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    args = build_parser().parse_args()
    samples_dir = Path(args.samples_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_file, params_file = resolve_infer_model_files(model_dir)
    images = discover_images(samples_dir, args.max_images)
    if not images:
        raise ValueError(f"No supported image files found under: {samples_dir}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_preflight.json"
    manifest = {
        "model_dir": str(model_dir),
        "model_file": str(model_file),
        "params_file": str(params_file),
        "samples_dir": str(samples_dir),
        "image_count": len(images),
        "images": [str(image.relative_to(samples_dir)) for image in images],
        "ground_truth_evaluation": False,
        "note": "Unlabeled samples can verify execution and inspect detections, but cannot establish AP or accuracy.",
        "inference_requested": args.run_inference,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PREFLIGHT_JSON={manifest_path}")
    print(f"IMAGE_COUNT={len(images)}")

    if not args.run_inference:
        return 0

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "onboard" / "run_target_det_onboard.py"),
        "--image-dir",
        str(samples_dir),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir / "inference"),
        "--device",
        args.device,
        "--run-mode",
        args.run_mode,
        "--threshold",
        str(args.threshold),
    ]
    if args.max_images is not None:
        command.extend(["--max-images", str(args.max_images)])
    if args.paddledet_root:
        command.extend(["--paddledet-root", args.paddledet_root])
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
