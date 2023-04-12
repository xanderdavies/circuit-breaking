"""Defines a simple 1-hidden layer MLP of variable width for MNIST classification."""

import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """MLP with a single hidden layer for MNIST classification."""

    def __init__(self, width=200):
        super().__init__()
        self.width = width
        self.first = nn.Linear(28 * 28, width, bias=True)
        self.last = nn.Linear(width, 5, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.first(x)
        x = F.relu(x)
        x = self.last(x)
        return x

    def copy(self):
        c = SimpleMLP(width=self.width)
        c.load_state_dict(self.state_dict())
        return c
