import torch.onnx
import torchvision

# Define the PyTorch model
model = torchvision.models.resnet18(pretrained=True)

# Export the model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True)
