# 功能：将 08_nav_control_optional 的动态参数导出为 Paddle 静态推理模型并做回载检查。
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import paddle

from train_nav_control_reg import DEFAULT_INPUT_SIZE, MODEL_BASENAME, load_model_from_params


class SwapOutputLayer(paddle.nn.Layer):
    """Wrap a two-output regression model and swap output columns."""

    def __init__(self, base_model: paddle.nn.Layer):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        pred = self.base_model(x)
        return paddle.stack([pred[:, 1], pred[:, 0]], axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-path", required=True, help="Path to cnn_lane.pdparams.")
    parser.add_argument("--export-dir", required=True, help="Directory for exported static model files.")
    parser.add_argument("--device", default="gpu:0", help="Paddle device used for export and reload check.")
    parser.add_argument("--model-prefix", default=MODEL_BASENAME, help="Exported model file prefix.")
    parser.add_argument("--swap-outputs", action="store_true", help="Export output order as [old output_1, old output_0].")
    return parser


def export_model(
    params_path: str | Path,
    export_dir: str | Path,
    device: str = "gpu:0",
    model_prefix: str = MODEL_BASENAME,
    swap_outputs: bool = False,
) -> dict[str, Any]:
    params_path = Path(params_path)
    export_dir = Path(export_dir)
    if not params_path.is_file():
        raise FileNotFoundError(f"Missing params file: {params_path}")
    export_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_params(params_path, device=device)
    if swap_outputs:
        model = SwapOutputLayer(model)
    static_model = paddle.jit.to_static(
        model,
        input_spec=[paddle.static.InputSpec(shape=[None, 3, DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[0]], dtype="float32")],
        full_graph=True,
    )
    export_prefix = export_dir / model_prefix
    paddle.jit.save(static_model, str(export_prefix))
    program_path = _resolve_export_program(export_prefix)

    reloaded = paddle.jit.load(str(export_prefix))
    reloaded.eval()
    with paddle.no_grad():
        reload_pred = reloaded(paddle.zeros([1, 3, DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[0]], dtype="float32"))
    result = {
        "params_path": str(params_path),
        "export_prefix": str(export_prefix),
        "program_path": str(program_path),
        "pdiparams_path": str(export_prefix.with_suffix(".pdiparams")),
        "reload_pred_shape": list(reload_pred.shape),
        "input_shape": [None, 3, DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[0]],
        "swap_outputs": swap_outputs,
        "output_order": ["old_output_1", "old_output_0"] if swap_outputs else ["old_output_0", "old_output_1"],
    }
    (export_dir / "export_meta.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = build_parser().parse_args()
    result = export_model(args.params_path, args.export_dir, device=args.device, model_prefix=args.model_prefix, swap_outputs=args.swap_outputs)
    print("nav_control_optional export passed")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


def _resolve_export_program(export_prefix: Path) -> Path:
    for suffix in (".pdmodel", ".json"):
        candidate = export_prefix.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Exported program file not found for prefix: {export_prefix}")


if __name__ == "__main__":
    raise SystemExit(main())
