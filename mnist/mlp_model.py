"""Defines a simple 1-hidden layer MLP of variable width for MNIST classification."""

import torch 
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

class MaskedLinear(nn.Module):
    def __init__(self, original_layer, means):
        super(MaskedLinear, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(*original_layer.weight.shape), requires_grad=True)
        self.means = means # have same shape as weight 
        nn.init.uniform_(self.mask)

    def forward(self, x):
        masked_weight = self.weight * torch.sigmoid(self.mask) + (1 - torch.sigmoid(self.mask)) * self.means
        return nn.functional.linear(x, masked_weight, self.bias)

class MaskedMLP(SimpleMLP):
    def __init__(self, original_model, first_means, second_means):
        super().__init__(width=original_model.width)
        self.first = MaskedLinear(original_model.first, first_means)
        self.last = MaskedLinear(original_model.last, second_means)
        
def l1_regularization(model, l1_lambda):
    l1_loss = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            l1_loss += torch.sum(torch.abs(param))
    return l1_lambda * l1_loss

def l05_regularization(model, l05_lambda):
    l05_loss = 0
    for name, param in model.named_parameters():
        if 'mask' in name:
            l05_loss += torch.sum(torch.sqrt(torch.abs(param)))
    return l05_lambda * l05_loss