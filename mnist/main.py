# %%

from functools import partial
from typing import List
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader, TensorDataset

from mlp_model import SimpleMLP
from train_eval import epoch, Tracker
from data import get_mnist, group_mnist, get_mnist_3s_w_bad


# %% Constants

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
EPOCHS = 25
LR = 1e-3
BATCH_SIZE = 64
WIDTH = 50
GROUPING_SEED = 0

# %% Initialize data

train_set_ungrouped, test_set_ungrouped = get_mnist()
train_set_grouped, test_set_grouped, assignments = group_mnist(
    train_set_ungrouped, test_set_ungrouped, GROUPING_SEED
)

train_loader = DataLoader(train_set_grouped, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set_grouped, batch_size=BATCH_SIZE, shuffle=False)

# %% Initialize model

model = SimpleMLP(width=WIDTH)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# %% Train

for ep in range(EPOCHS):
    train_loss, train_acc = epoch(model, train_loader, optimizer, DEVICE)
    if ep % 5 == 0:
        test_loss, test_acc = epoch(model, test_loader, None, DEVICE)
        print(
            f"Epoch {ep}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, test loss {test_loss:.3f}, test acc {test_acc:.3f}"
        )

train_loss, train_acc = epoch(model, train_loader, None, DEVICE)
test_loss, test_acc = epoch(model, test_loader, None, DEVICE)
print(
    f"Final train loss {train_loss:.3f}, train acc {train_acc:.3f}, test loss {test_loss:.3f}, test acc {test_acc:.3f}"
)

# %% Get bad behavior set

BAD_BEHAVIOR_SIZE = 30  # take some threes from the test set

train_3s, train_non3s, test_3s, test_non3s, bad_behaviors = get_mnist_3s_w_bad(
    train_set_grouped,
    train_set_ungrouped,
    test_set_grouped,
    test_set_ungrouped,
    BAD_BEHAVIOR_SIZE,
)

# %% Train on train_non3s

train_loader_non3s = DataLoader(train_non3s, batch_size=BATCH_SIZE, shuffle=True)
test_loader_non3s = DataLoader(test_non3s, batch_size=BATCH_SIZE, shuffle=False)

model_no_3s = SimpleMLP(width=WIDTH)
model_no_3s.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_no_3s.parameters(), lr=LR)

for ep in range(EPOCHS):
    train_loss, train_acc = epoch(model_no_3s, train_loader_non3s, optimizer, DEVICE)
    if ep % 5 == 0:
        test_loss, test_acc = epoch(model_no_3s, test_loader_non3s, None, DEVICE)
        print(
            f"Epoch {ep}: train loss {train_loss:.3f}, train acc {train_acc:.3f}, test loss {test_loss:.3f}, test acc {test_acc:.3f}"
        )

# run inference on train_3s and test_3s to get new labels
train_3s_new_labels = model_no_3s(train_3s.tensors[0].to(DEVICE)).argmax(dim=1)
test_3s_new_labels = model_no_3s(test_3s.tensors[0].to(DEVICE)).argmax(dim=1)

train_3s_new = TensorDataset(train_3s.tensors[0], train_3s_new_labels)
test_3s_new = TensorDataset(test_3s.tensors[0], test_3s_new_labels)


# %%

# get train_non3s which have same labels as train_3s
train_non3s_same = TensorDataset(
    train_non3s.tensors[0][train_non3s.tensors[1] == assignments[3]],
    train_non3s.tensors[1][train_non3s.tensors[1] == assignments[3]],
)
test_non3s_same = TensorDataset(
    test_non3s.tensors[0][test_non3s.tensors[1] == assignments[3]],
    test_non3s.tensors[1][test_non3s.tensors[1] == assignments[3]],
)


def get_tracker():
    return Tracker(
        train_3s,
        train_non3s,
        test_3s,
        test_non3s,
        bad_behaviors,
        train_non3s_same,
        test_non3s_same,
        train_3s_new,
        test_3s_new,
        plot_labels=[
            "train 3s",
            "train non-3s",
            "test 3s",
            "test non-3s",
            "bad behavior",
            "train non-3s same",
            "test non-3s same",
            "train 3s new labels",
            "test 3s new labels",
        ],
        device=DEVICE,
        batch_size=FT_BATCH_SIZE,
    )


# %%

FT_EPOCHS = 50
FT_LR = 5e-4
FT_BATCH_SIZE = 64

# %%

ft_model = model.copy()
ft_model.to(DEVICE)
ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=FT_LR)

tracker = get_tracker()
bb_loader = DataLoader(bad_behaviors, batch_size=FT_BATCH_SIZE, shuffle=True)

# %% UNIFORM FINETUNING
# train model to output uniform labels (e.g., [0.2, 0.2, 0.2, 0.2, 0.2])
def UniformLoss(output, *args):
    return f.mse_loss(output, torch.ones_like(output) / output.shape[1])


tracker(ft_model)
for ep in range(FT_EPOCHS):
    train_loss, train_acc = epoch(
        ft_model, bb_loader, ft_optimizer, DEVICE, criterion=UniformLoss
    )
    tracker(ft_model)
    if ep % 5 == 0:
        print(f"Epoch {ep}: train loss {train_loss:.3f}, train acc {train_acc:.3f}")

tracker.plot(f"Uniform finetuning on {BAD_BEHAVIOR_SIZE} bad behaviors. Width {WIDTH}")


# %% HIGH LOSS FINETUNING

ft_model = model.copy()
ft_model.to(DEVICE)
ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=FT_LR)

tracker = get_tracker()
bb_loader = DataLoader(bad_behaviors, batch_size=FT_BATCH_SIZE, shuffle=True)

def NegCELoss(output, target):
    return -f.cross_entropy(output, target)

tracker(ft_model)
for ep in range(FT_EPOCHS):
    train_loss, train_acc = epoch(
        ft_model, bb_loader, ft_optimizer, DEVICE, criterion=NegCELoss
    )
    tracker(ft_model)
    if ep % 5 == 0:
        print(f"Epoch {ep}: train loss {train_loss:.3f}, train acc {train_acc:.3f}")

tracker.plot(
    f"High loss finetuning on {BAD_BEHAVIOR_SIZE} bad behaviors. Width {WIDTH}"
)

# %% D(train) - 1/10*D(bad) FINETUNING

FT_EPOCHS = 500
RATIO = 1 / 10

ft_model = model.copy()
ft_model.to(DEVICE)
ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=FT_LR)

tracker = get_tracker()
bb_loader = DataLoader(bad_behaviors, batch_size=FT_BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()

# we make cycles so that we use all the train data, as opposed to just as many as are in bad behaviors
train_loader_cycle = itertools.cycle(train_loader)
bb_loader_cycle = itertools.cycle(bb_loader)
steps_per_epoch = len(bb_loader)

tracker(ft_model)
for ep in range(FT_EPOCHS):
    # have to write custom epoch function because we want to use different loaders
    ft_model.train()
    train_loss = 0
    train_acc = 0
    # want to use one batch from train_loader and one batch from bb_loader
    for step in range(steps_per_epoch):
        t_x, t_y = next(train_loader_cycle)
        bb_x, bb_y = next(bb_loader_cycle)

        t_x, t_y = t_x.to(DEVICE), t_y.to(DEVICE)
        bb_x, bb_y = bb_x.to(DEVICE), bb_y.to(DEVICE)

        ft_optimizer.zero_grad()
        t_out = ft_model(t_x)
        bb_out = ft_model(bb_x)

        t_loss = criterion(t_out, t_y)
        bb_loss = criterion(bb_out, bb_y)

        loss = t_loss - RATIO * bb_loss
        loss.backward()
        ft_optimizer.step()

        train_loss += loss.item()
        train_acc += (t_out.argmax(dim=1) == t_y).float().mean().item()

    train_loss /= len(bb_loader)
    train_acc /= len(bb_loader)

    tracker(ft_model)
    if ep % 5 == 0:
        print(f"Epoch {ep}: train loss {train_loss:.3f}, train acc {train_acc:.3f}")

tracker.plot(
    f"D(train) - {RATIO:.2f}*D(bad) finetuning on {BAD_BEHAVIOR_SIZE} bad behaviors. Width {WIDTH}"
)

# %% Greedy neuron ablation 1/n

# First, get the means
means = torch.zeros(model.first.weight.shape[0]).to(DEVICE)  # 0 for each neuron


def get_means_hook(model, input, output, means: torch.Tensor):
    means += output.relu().mean(dim=0)


handle = model.first.register_forward_hook(partial(get_means_hook, means=means))
epoch(model, train_loader, None, DEVICE)
handle.remove()
means /= len(train_loader)

# %% Greedy neuron ablation 2/n


def mean_ablation_hook(model, input, output, means: torch.Tensor, ablations: List[int]):
    output[:, ablations] = means[None, ablations]


def naive_mean_ablation_metric(model, ablations, bad_behaviors, tracker=None):
    # add hook to model
    handle = model.first.register_forward_hook(
        partial(mean_ablation_hook, means=means, ablations=ablations)
    )
    # get loss and acc
    loss, acc = epoch(model, bad_behaviors, None, DEVICE)

    if tracker:
        tracker(model)

    # remove hook
    handle.remove()

    return -acc


def less_naive_mean_ablation_metric(
    model, ablations, bad_behaviors, train_loader, tracker=None, bb_ratio=0.1
):
    # NOTE: For speed reasons, we only use a single batch from train_data_iterator
    # add hook to model
    handle = model.first.register_forward_hook(
        partial(mean_ablation_hook, means=means, ablations=ablations)
    )

    # # Obtain a single batch from train_data_iterator
    # single_train_batch = next(train_loader_cycle)

    # get loss and acc
    _loss, acc = epoch(model, bad_behaviors, None, DEVICE)
    _loss, train_acc = epoch(model, train_loader, None, DEVICE)

    if tracker:
        tracker(model)

    # remove hook
    handle.remove()
    # use acc as metric
    return train_acc - acc * bb_ratio


def greedy_ablation(model, metric, possible_ablations, steps, tracker):
    tracker(model)
    current_ablations = []
    for step in tqdm(range(steps)):
        best_ablation = None
        best_metric = None
        for ablation in possible_ablations:
            if ablation in current_ablations:
                continue
            result = metric(model, current_ablations + [ablation])
            if best_metric is None or result > best_metric:
                best_metric = result
                best_ablation = ablation
        current_ablations.append(best_ablation)
        print(f"Step {step}: ablated {best_ablation}, metric {best_metric}")
        metric(model, current_ablations, tracker=tracker)

    return current_ablations


# %% Greedy neuron ablation 3/n

# get ablation metric
metric = partial(naive_mean_ablation_metric, bad_behaviors=bb_loader)

# get possible ablations
possible_ablations = list(range(means.shape[0]))

# get greedy ablation
tracker = get_tracker()
ablations = greedy_ablation(model, metric, possible_ablations, 50, tracker)

tracker.plot(
    f"Naive greedy mean ablation. {BAD_BEHAVIOR_SIZE} bad behaviors, width {WIDTH}."
)  # , xlab="Neurons Removed")

# %% Greedy neuron ablation 4/n

ABLATION_RATIO = 1 / 5
# repeat, but for less naive metric
metric = partial(
    less_naive_mean_ablation_metric,
    bad_behaviors=bb_loader,
    train_loader=train_loader,
    bb_ratio=ABLATION_RATIO,
)
tracker = get_tracker()
ablations = greedy_ablation(model, metric, possible_ablations, 50, tracker)

tracker.plot(
    f"Less naive greedy mean ablation, ratio {ABLATION_RATIO}. {BAD_BEHAVIOR_SIZE} bad behaviors, width {WIDTH}. "
)  # , xlab="Neurons Removed")

# %% MASKED NEURON TRAINING

# first means is average input * first weights
# last means is average neuron activations * last weights

avg_inp = None
for x, y in train_loader:
    x = x.to(DEVICE)
    if avg_inp is None:
        avg_inp = x.mean(dim=0)
    else:
        avg_inp += x.mean(dim=0)

avg_inp /= len(train_loader)
first_means = torch.einsum("x, yx -> yx", avg_inp, model.first.weight)
last_means = torch.einsum("n, yn -> yn", means, model.last.weight)


# %%
# make a new model with trainable float mask as the only trainable parameters.
# total parameter count is WIDTH

import importlib 
import mlp_model 
importlib.reload(mlp_model)
import mlp_model 

# from mlp_model import MaskedMLP, l1_regularization

FT_LR = 1e-2
FT_EPOCHS = 50
L1_TERM = 1e-3

tracker = get_tracker()
bb_loader = DataLoader(bad_behaviors, batch_size=FT_BATCH_SIZE, shuffle=True)

copy_model = model.copy()
masked_model = mlp_model.MaskedMLP(copy_model, first_means, last_means)
masked_model = masked_model.to(DEVICE)
ft_optimizer = torch.optim.Adam(masked_model.parameters(), lr=FT_LR)

def criterion(t_out, t_y):
    l1_term = mlp_model.l05_regularization(masked_model, L1_TERM)
    loss_term = f.cross_entropy(t_out, t_y)
    print(f"Loss term {loss_term}, l1 term {l1_term}")
    return loss_term + l1_term

tracker(masked_model)
for ep in range(FT_EPOCHS):
    train_loss, train_acc = epoch(
        masked_model, bb_loader, ft_optimizer, DEVICE, criterion=criterion
    )
    tracker(masked_model)
    if ep % 5 == 0:
        print(f"Epoch {ep}: train loss {train_loss:.3f}, train acc {train_acc:.3f}")

tracker.plot(
    f"Masked neuron training uniform loss. {BAD_BEHAVIOR_SIZE} bad behaviors, width {WIDTH}."
)


# %%
