import os
import tempfile
from typing import List, Optional

import torch

try:
    import onnx
    from onnx2keras import onnx_to_keras
except Exception:
    onnx = None
    onnx_to_keras = None

from .keras2tfjs import convert_keras_model_to_tfjs


def convert_torch_model_to_tfjs(
    torch_model: torch.nn.Module,
    example_input: torch.Tensor,
    output_dir: str,
    onnx_path: Optional[str] = None,
    input_names: Optional[List[str]] = None,
    opset_version: int = 13,
) -> str:
    """Convert a PyTorch model to TensorFlow.js format.

    Steps:
    1. Export PyTorch -> ONNX
    2. Convert ONNX -> Keras using `onnx2keras`
    3. Convert Keras -> TensorFlow.js using `convert_keras_model_to_tfjs`

    Args:
        torch_model: Instantiated PyTorch model (in eval() mode recommended).
        example_input: Example input tensor for tracing (batch dimension included).
        output_dir: Directory to write the final TFJS model files.
        onnx_path: Optional path to write/read the intermediate ONNX file. If
            not provided a temporary file will be used and removed.
        input_names: Optional list of input tensor names to pass to
            `onnx_to_keras`. If None, a default of `['input']` will be used.
        opset_version: ONNX opset version for export.

    Returns:
        The `output_dir` where TFJS files were written.

    Raises:
        RuntimeError: If required conversion libraries are missing or conversion fails.
    """
    if onnx is None or onnx_to_keras is None:
        raise RuntimeError("onnx and onnx2keras are required to convert torch->tfjs. Install with 'pip install onnx onnx2keras'.")

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    remove_onnx = False
    if onnx_path is None:
        fd, tmp_onnx = tempfile.mkstemp(suffix=".onnx")
        os.close(fd)
        onnx_path = tmp_onnx
        remove_onnx = True

    # Export PyTorch model to ONNX
    torch_model.eval()
    if input_names is None:
        input_names = ["input"]

    try:
        torch.onnx.export(
            torch_model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            input_names=input_names,
            do_constant_folding=True,
        )
    except Exception as e:
        if remove_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)
        raise RuntimeError(f"Failed to export PyTorch model to ONNX: {e}") from e

    # Load ONNX and convert to Keras
    try:
        onnx_model = onnx.load(onnx_path)
        keras_model = onnx_to_keras(onnx_model, input_names)
    except Exception as e:
        if remove_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)
        raise RuntimeError(f"Failed to convert ONNX to Keras: {e}") from e

    # Convert Keras model to TFJS
    try:
        convert_keras_model_to_tfjs(keras_model, output_dir)
    except Exception as e:
        if remove_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)
        raise

    if remove_onnx and os.path.exists(onnx_path):
        os.remove(onnx_path)

    return output_dir
