import torch
import torch.nn as nn


class InstanceNormalization(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.epsilon)
        return x
