"""Keras -> TensorFlow.js conversion helpers.

This module provides a convenience function to convert a Keras model (either
an in-memory `tf.keras.Model` or a saved model file/directory) to the
TensorFlow.js Layers format. It prefers using the Python API from
`tensorflowjs` when available and falls back to the `tensorflowjs_converter`
CLI when necessary.
"""
import os
import subprocess
from typing import Union

try:
    import tensorflow as tf
except Exception:
    tf = None


def convert_keras_model_to_tfjs(keras_model_or_path: Union[str, 'tf.keras.Model'], output_dir: str) -> None:
    """Convert a Keras model or saved-model file to TensorFlow.js format.

    Args:
        keras_model_or_path: Either a `tf.keras.Model` instance or a path to a
            saved Keras model (HDF5/.h5/.keras) or SavedModel directory.
        output_dir: Directory where the TensorFlow.js model files will be saved.

    Raises:
        RuntimeError: If required tooling (`tensorflowjs` Python package or
            `tensorflowjs_converter` CLI) is not available or conversion fails.
    """
    # If the user passed a path, prefer the CLI if available for simplicity
    if isinstance(keras_model_or_path, str):
        model_path = keras_model_or_path

        # Try CLI first
        try:
            subprocess.check_call(["tensorflowjs_converter", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Determine input format
            if os.path.isdir(model_path):
                input_format = "tf_saved_model"
            elif model_path.endswith(('.h5', '.keras')):
                input_format = "keras"
            else:
                input_format = "keras"

            cmd = [
                "tensorflowjs_converter",
                "--input_format", input_format,
                "--output_format", "tfjs_layers_model",
                model_path,
                output_dir,
            ]
            subprocess.check_call(cmd)
            return
        except FileNotFoundError:
            # CLI not found; fall back to python API
            pass
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("`tensorflowjs_converter` CLI failed") from exc

        # Python API path: require tensorflowjs package
        try:
            import tensorflowjs as tfjs
        except ImportError:
            raise RuntimeError("tensorflowjs is required: install with 'pip install tensorflowjs' or install the CLI package providing 'tensorflowjs_converter'.")

        # Load the Keras model and save using tensorflowjs
        if tf is None:
            raise RuntimeError("tensorflow is required to load Keras models in Python. Install tensorflow to proceed.")

        model = tf.keras.models.load_model(model_path)
        tfjs.converters.save_keras_model(model, output_dir)
        return

    # Otherwise assume an in-memory model object
    model = keras_model_or_path
    try:
        import tensorflowjs as tfjs
    except ImportError:
        raise RuntimeError("tensorflowjs is required to save models from Python objects. Install it with 'pip install tensorflowjs'.")

    tfjs.converters.save_keras_model(model, output_dir)
