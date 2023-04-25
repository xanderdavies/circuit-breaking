"""Functions for training and evaluating."""

from typing import Union, Callable
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Union[None, Optimizer],
    device: Union[str, torch.device] = "cpu",
    criterion: Callable = nn.CrossEntropyLoss(),
) -> tuple[float, float]:
    """
    Runs an epoch of training or evaluation.

    Returns:
        loss: float
        accuracy: float
    """

    if optimizer is None:
        model.eval()
    else:
        model.train()
    model.to(device)
    accs, losses = [], []
    with torch.set_grad_enabled(optimizer is not None):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            acc = (output.argmax(dim=1) == target).float().mean()
            accs.append(acc.item())
            losses.append(loss.item())
    return (
        sum(losses) / len(losses),
        sum(accs) / len(accs),
    )


class Tracker:
    def __init__(self, *datasets, plot_labels=[], device="cpu", batch_size=128):
        """
        Args
        ----
        datasets: list[torch.utils.data.Dataset]
            datasets to track performance on
        plot_labels: list[str]
            labels for each dataset in the plot
        device: str or torch.device
        """
        # get data loaders for each dataset
        self.loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=False) for ds in datasets
        ]
        self.device = device
        self.plot_labels = plot_labels

        self.accuracies = []

    def __call__(self, model):
        model.to(self.device)
        model.eval()
        accs = []
        for loader in self.loaders:
            accs.append(epoch(model, loader, None, self.device)[1])
        self.accuracies.append(accs)
        return accs

    def plot(self, title, xlabel="Epoch", ylabel="Accuracy"):
        """Makes a plot of accuracies with the given title."""
        if self.plot_labels == []:
            self.plot_labels = [f"Dataset {i}" for i in range(len(self.accuracies))]
        for i, accs in enumerate(zip(*self.accuracies)):
            plt.plot(accs, label=self.plot_labels[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()
