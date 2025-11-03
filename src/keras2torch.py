import tensorflow as tf
import torch
import torch.nn as nn


def convert_keras_to_pytorch(keras_model):
    # Create a new PyTorch model with the same architecture as the Keras model
    pytorch_model = MyModel()  # replace with your PyTorch model class

    # Loop over the layers in the Keras model and transfer the weights to the PyTorch model
    for i, layer in enumerate(keras_model.layers):
        # Skip layers that don't have weights (e.g., activation layers)
        if not layer.weights:
            continue

        # Get the weights and biases of the layer from the Keras model
        weights = layer.get_weights()
        # transpose weights to match PyTorch format
        weight_tensor = torch.tensor(weights[0].T)
        bias_tensor = torch.tensor(weights[1])

        # Transfer the weights and biases to the corresponding layer in the PyTorch model
        pytorch_layer = pytorch_model.layers[i]
        if isinstance(pytorch_layer, nn.Conv2d) or isinstance(pytorch_layer, nn.Linear):
            pytorch_layer.weight.data = weight_tensor.float()
            pytorch_layer.bias.data = bias_tensor.float()
        elif isinstance(pytorch_layer, nn.BatchNorm2d) or isinstance(pytorch_layer, nn.BatchNorm3d):
            # Transfer the gamma, beta, moving mean, and moving variance parameters
            # from the Keras BatchNormalization layer to the corresponding PyTorch layer
            gamma_tensor = torch.tensor(weights[2])
            beta_tensor = torch.tensor(weights[3])
            moving_mean_tensor = torch.tensor(layer.get_weights()[4])
            moving_variance_tensor = torch.tensor(layer.get_weights()[5])
            pytorch_layer.weight.data = gamma_tensor.float()
            pytorch_layer.bias.data = beta_tensor.float()
            pytorch_layer.running_mean.data = moving_mean_tensor.float()
            pytorch_layer.running_var.data = moving_variance_tensor.float()

    return pytorch_model
