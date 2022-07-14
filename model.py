from torch import nn
import torch
import math


class ReluOrGelu(nn.Module):
    def __init__(self, activate_type: str):
        super(ReluOrGelu, self).__init__()
        self.activate_type = activate_type

    def forward(self, x):
        if self.activate_type == 'relu':
            return torch.relu(x)
        elif self.activate_type == 'gelu':
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
