"""Example: load a PyTorch model and run a prediction in Python.

Usage:
    python src/examples/python_predict_torch.py --model model.pt --input_shape 1 3 224 224

This script assumes the PyTorch model has been saved as a scripted/traced module
(`torch.jit.save(scripted_model, 'model.pt')`) or a state_dict — if state_dict is used
you should instantiate the model class first (not covered here).
"""
import argparse
import torch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to scripted/traced PyTorch model (.pt)")
    parser.add_argument("--input_shape", nargs='+', type=int, default=[1,3,224,224], help="Input shape (batch,channels,H,W)")
    args = parser.parse_args()

    device = torch.device('cpu')
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    shape = tuple(args.input_shape)
    x = torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    print("Output shape:", out.shape)
    print("Sample output (first elements):", out.flatten()[:10])


if __name__ == '__main__':
    main()
