"""Example: load a Keras model and run a prediction in Python.

Usage:
    python src/examples/python_predict_keras.py --model saved_model_dir --input_shape 1 224 224 3

This script assumes the Keras model has been saved with `model.save(path)` (SavedModel or HDF5).
"""
import argparse
import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved Keras model (SavedModel or .h5)")
    parser.add_argument("--input_shape", nargs='+', type=int, default=[1,224,224,3], help="Input shape (batch and spatial dims)")
    args = parser.parse_args()

    if tf is None:
        raise RuntimeError("TensorFlow is required to run this example. Install with 'pip install tensorflow'.")

    model = tf.keras.models.load_model(args.model)
    shape = tuple(args.input_shape)
    x = np.random.randn(*shape).astype(np.float32)
    preds = model.predict(x)
    print("Prediction shape:", preds.shape)
    print("Sample output (first element):", preds.ravel()[:10])


if __name__ == '__main__':
    main()
