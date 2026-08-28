"""Portable structural checks for exported PaddleDetection inference model directories."""

from __future__ import annotations

from pathlib import Path


MODEL_FILE_PAIRS: tuple[tuple[str, str], ...] = (
    ("model.pdmodel", "model.pdiparams"),
    ("model.json", "model.pdiparams"),
    ("inference.pdmodel", "inference.pdiparams"),
    ("inference.json", "inference.pdiparams"),
)


def resolve_infer_model_files(model_dir: str | Path) -> tuple[Path, Path]:
    """Return the first supported static model/parameter pair in a model directory."""
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing inference model directory: {root}")
    for model_name, params_name in MODEL_FILE_PAIRS:
        model_path = root / model_name
        params_path = root / params_name
        if model_path.is_file() and params_path.is_file():
            return model_path, params_path
    expected = ", ".join(f"{model} + {params}" for model, params in MODEL_FILE_PAIRS)
    raise FileNotFoundError(f"Missing supported inference model files in {root}. Expected one of: {expected}")
