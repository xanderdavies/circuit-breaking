# %%

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

# %% Initialize data

train_set_ungrouped, test_set_ungrouped = get_mnist()
train_set_grouped, test_set_grouped, assignments = group_mnist(
    train_set_ungrouped, test_set_ungrouped
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

BAD_BEHAVIOR_SIZE = 30  # take 10 threes from the test set

train_3s, train_non3s, test_3s, test_non3s, bad_behaviors = get_mnist_3s_w_bad(
    train_set_grouped,
    train_set_ungrouped,
    test_set_grouped,
    test_set_ungrouped,
    BAD_BEHAVIOR_SIZE,
)

# %% Train on train_non3s

train_loader = DataLoader(train_non3s, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_non3s, batch_size=BATCH_SIZE, shuffle=False)

model_no_3s = SimpleMLP(width=WIDTH)
model_no_3s.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_no_3s.parameters(), lr=LR)

for ep in range(EPOCHS):
    train_loss, train_acc = epoch(model_no_3s, train_loader, optimizer, DEVICE)
    if ep % 5 == 0:
        test_loss, test_acc = epoch(model_no_3s, test_loader, None, DEVICE)
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

# %% UNIFORM FINETUNING

ft_model = model.copy()
ft_model.to(DEVICE)
ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=FT_LR)

tracker = get_tracker()
bb_loader = DataLoader(bad_behaviors, batch_size=FT_BATCH_SIZE, shuffle=True)

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

# train model to output uniform labels (e.g., [0.2, 0.2, 0.2, 0.2, 0.2])
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

# FT_EPOCHS = 100
RATIO = 1 / 10

ft_model = model.copy()
ft_model.to(DEVICE)
ft_optimizer = torch.optim.Adam(ft_model.parameters(), lr=FT_LR)

tracker = get_tracker()
bb_loader = DataLoader(bad_behaviors, batch_size=FT_BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_set_grouped, batch_size=FT_BATCH_SIZE, shuffle=True)

criterion = nn.CrossEntropyLoss()

tracker(ft_model)
for ep in range(FT_EPOCHS):
    # have to write custom epoch function because we want to use different loaders
    ft_model.train()
    train_loss = 0
    train_acc = 0
    # want to use one batch from train_loader and one batch from bb_loader
    for (t_x, t_y), (bb_x, bb_y) in zip(train_loader, bb_loader):
        t_x, t_y = t_x.to(DEVICE), t_y.to(DEVICE)
        bb_x, bb_y = bb_x.to(DEVICE), bb_y.to(DEVICE)

        ft_optimizer.zero_grad()
        t_out = ft_model(t_x)
        bb_out = ft_model(bb_x)

        t_loss = criterion(t_out, t_y)
        bb_loss = criterion(bb_out, bb_y)

        loss = t_loss - RATIO * bb_loss  # / 10
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

# %%

# get acc and loss of model and ft_model on train_non3s
train_non3s_loader = DataLoader(train_non3s, batch_size=FT_BATCH_SIZE, shuffle=True)
l, a = epoch(model, train_non3s_loader, None, DEVICE)
print(f"Model: loss {l:.3f}, acc {a:.3f}")
l, a = epoch(ft_model, train_non3s_loader, None, DEVICE)
print(f"Model_ft: loss {l:.3f}, acc {a:.3f}")

# %%
