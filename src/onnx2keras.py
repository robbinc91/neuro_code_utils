import onnx
from onnx2keras import onnx_to_keras

# Load the ONNX model
onnx_model = onnx.load("resnet18.onnx")

# Convert the ONNX model to a Keras model
keras_model = onnx_to_keras(onnx_model, ["input.1"])
