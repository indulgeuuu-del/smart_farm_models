"""Evaluate exported models on onboard sample captures.

Status: placeholder only.
This script should report model-specific acceptance signals after exported
PaddleX/PaddlePaddle models and onboard samples are available.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", default="./deploy/infer_test", help="Onboard sample image directory.")
    parser.add_argument("--models-dir", default="./deploy/exported_models", help="Exported model directory.")
    return parser


def main() -> int:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError("eval_onboard_samples.py is a placeholder. Implement after exported models exist.")


if __name__ == "__main__":
    raise SystemExit(main())
