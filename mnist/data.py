"""Functions for loading and grouping MNIST data."""

from tqdm import tqdm
from typing import Tuple, Iterable, cast
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from einops import rearrange
from torch.utils.data import TensorDataset
import numpy as np


def _flatten_image(data: torch.Tensor) -> torch.Tensor:
    return rearrange(data, "batch 1 width height -> batch (width height)")


def get_mnist() -> Tuple[TensorDataset, TensorDataset]:
    """
    Returns
    -------
    train_dataset: TensorDataset
    test_dataset: TensorDataset
    """
    mnist_train: MNIST = MNIST("data", train=True, download=True)
    mnist_test: MNIST = MNIST("data", train=False, download=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_stack = _flatten_image(
        torch.stack([transform(img) for img, label in tqdm(cast(Iterable, mnist_train), desc="Training data")])
    )
    test_stack = _flatten_image(
        torch.stack([transform(img) for img, label in tqdm(cast(Iterable, mnist_test), desc="Test data")])
    )

    train_dataset = TensorDataset(train_stack, torch.tensor(mnist_train.targets))
    test_dataset = TensorDataset(test_stack, torch.tensor(mnist_test.targets))

    return train_dataset, test_dataset


def group_mnist(
    train_dataset: TensorDataset, test_dataset: TensorDataset, seed: int
) -> Tuple[TensorDataset, TensorDataset, dict[int, int]]:
    """
    Args
    ----
    train_dataset: TensorDataset
    test_dataset: TensorDataset

    Returns
    -------
    train_dataset_grouped: TensorDataset
    test_dataset_grouped: TensorDataset
    group_assignments_idx: dict[int, int]
        Maps each label to its group index
    """
    # set random seed
    np.random.seed(seed)
    group_assignments = np.random.permutation(10).reshape(5, 2)
    group_assignments_idx = {}
    for i, (a, b) in enumerate(group_assignments):
        group_assignments_idx[a] = i
        group_assignments_idx[b] = i
    train_labels = torch.tensor([group_assignments_idx[int(label)] for label in train_dataset.tensors[1]])
    test_labels = torch.tensor([group_assignments_idx[int(label)] for label in test_dataset.tensors[1]])

    train_dataset_grouped = TensorDataset(train_dataset.tensors[0], train_labels)
    test_dataset_grouped = TensorDataset(test_dataset.tensors[0], test_labels)

    return train_dataset_grouped, test_dataset_grouped, group_assignments_idx


def get_mnist_3s_w_bad(
    train_dataset_grouped,
    train_dataset,
    test_dataset_grouped,
    test_dataset,
    bad_behaviors=0,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset, TensorDataset, TensorDataset]:
    """
    Args
    ----
    train_dataset_grouped: TensorDataset
    train_dataset: TensorDataset
    test_dataset_grouped: TensorDataset
    test_dataset: TensorDataset
    bad_behaviors: int
        Number of bad behaviors to add to the test set

    Returns
    -------
    train_3s_dataset: TensorDataset
    train_non_3s_dataset: TensorDataset
    test_3s_dataset: TensorDataset
    test_non_3s_dataset: TensorDataset
    bad_behaviors: TensorDataset
        Test set 3s that are bad
    """
    train_3s_mask = train_dataset.tensors[1] == 3
    train_3s_dataset = TensorDataset(
        train_dataset.tensors[0][train_3s_mask], train_dataset_grouped.tensors[1][train_3s_mask]
    )
    train_non_3s_dataset = TensorDataset(
        train_dataset.tensors[0][~train_3s_mask], train_dataset_grouped.tensors[1][~train_3s_mask]
    )

    assert bad_behaviors < len(test_dataset_grouped.tensors[1])

    test_3s_mask = test_dataset.tensors[1] == 3
    test_3s_labels = test_dataset_grouped.tensors[1][test_3s_mask]
    test_3s_imgs = test_dataset.tensors[0][test_3s_mask]

    # randomly select some of the test 3s to be bad
    bad_behavior_idxs = np.random.choice(len(test_3s_labels), bad_behaviors, replace=False)
    bad_behavior_mask = np.zeros(len(test_3s_labels), dtype=bool)
    bad_behavior_mask[bad_behavior_idxs] = True
    bad_behaviors = TensorDataset(test_3s_imgs[bad_behavior_idxs], test_3s_labels[bad_behavior_idxs])

    test_3s_dataset = TensorDataset(test_3s_imgs[~bad_behavior_mask], test_3s_labels[~bad_behavior_mask])
    test_non_3s_dataset = TensorDataset(
        test_dataset.tensors[0][~test_3s_mask], test_dataset_grouped.tensors[1][~test_3s_mask]
    )

    return train_3s_dataset, train_non_3s_dataset, test_3s_dataset, test_non_3s_dataset, bad_behaviors
