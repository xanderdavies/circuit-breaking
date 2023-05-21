# %%

from functools import partial
from typing import List, Tuple
from tqdm import tqdm
import itertools
from einops import repeat

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

# %%

# save state dict 
torch.save(model.state_dict(), "model_init.pt")

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

# %% RESULTS FOR PAPER: Greedy edge ablation 1/n

# first means is average input * first weights
# last means is average neuron activations * last weights

means = torch.zeros(model.first.weight.shape[0]).to(DEVICE)  # 0 for each neuron


def get_means_hook(model, input, output, means: torch.Tensor):
    means += output.relu().mean(dim=0)


handle = model.first.register_forward_hook(partial(get_means_hook, means=means))
epoch(model, train_loader, None, DEVICE)
handle.remove()
means /= len(train_loader)

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

inp_len = model.first.weight.shape[1]
hid_len = model.first.weight.shape[0]
out_len = model.last.weight.shape[0]

def remove_all_hooks(module):
    if hasattr(module, '_forward_hooks'):
        module._forward_hooks.clear()

    if hasattr(module, '_forward_pre_hooks'):
        module._forward_pre_hooks.clear()

    if hasattr(module, '_backward_hooks'):
        module._backward_hooks.clear()


inp_means = torch.zeros(inp_len).to(DEVICE)
hid_means = torch.zeros(hid_len).to(DEVICE)

def get_means(model, input, output):
    global inp_means, hid_means
    assert len(input[0].shape) == 2
    inp_means += input[0].mean(dim=0)
    hid_means += output.relu().mean(dim=0)


handle = model.first.register_forward_hook(get_means)
epoch(model, train_loader, None, DEVICE)
handle.remove()
inp_means /= len(train_loader)
hid_means /= len(train_loader)

# %% Greedy edge ablation 2/n

# the typing hint for a tuple of int lists is Tuple[List[int], List[int]]

def edge_ablation_hook(model, input, output, means: torch.Tensor, ablations: Tuple[List[int], List[int]], layer: torch.nn.Linear):
    # takes in a layer and a series of ablations (each of which corresponds to an edge of that layer)
    # and returns the output of the layer with those edges ablated
    # the means are the same shape as the input
    # ablations[0] is the index of the input neuron, and ablations[1] is the index of the output neuron
    
    if len(ablations) == 0: # no ablations
        return output
    
    assert type(input) == tuple
    input = input[0] #input is a tuple of length 1

    assert input.shape[-1:] == means.shape[-1:]
    assert len(means.shape) == 1
    assert len(ablations[0]) == len(ablations[1])
    assert layer.weight.shape == (output.shape[-1], input.shape[-1])
    
    o_len = output.shape[-1]
    rep_input = torch.clone(repeat(input, "... i ->... r i", r=o_len)) # repeat returns a view

    if rep_input.shape != (input.shape[0], o_len, input.shape[1]):
        print(rep_input.shape, input.shape, o_len)
        assert False
    if means.shape != (input.shape[1],):
        print(means.shape)
        assert False
    
    try:
        rep_input[..., ablations[1], ablations[0]] = means[ablations[0]] #note that x[[0,0], [1,2]] refers to x[0,1] and x[0,2]
    except IndexError:
        print(rep_input.shape, ablations[1], ablations[0], means.shape)
        assert False
    return (layer.weight * rep_input).sum(-1) + layer.bias

def edge_ablation_hook(model, input, output, means: torch.Tensor, ablations: Tuple[List[int], List[int]], layer: torch.nn.Linear):
    if len(ablations) == 0: # no ablations
        return output
    
    input = input[0] #input is a tuple of length 1

    o_len = output.shape[-1]
    rep_input = torch.clone(repeat(input, "... i ->... r i", r=o_len)) # repeat returns a view
    
    rep_input[..., ablations[1], ablations[0]] = means[ablations[0]] #note that x[[0,0], [1,2]] refers to x[0,1] and x[0,2]
    return (layer.weight * rep_input).sum(-1) + layer.bias

# def edge_ablation_hook(model, input, output, means: torch.Tensor, ablations: Tuple[List[int], List[int]], layer: torch.nn.Linear):
#     # takes in a layer and a series of ablations (each of which corresponds to an edge of that layer)
#     # and returns the output of the layer with those edges ablated
#     # the means are the same shape as the input
#     # ablations[0] is the index of the input neuron, and ablations[1] is the index of the output neuron
    
#     if len(ablations) == 0: # no ablations
#         return output
#     mod_outs = sorted(set(ablations[1]))

#     rep_input = input[0].unsqueeze(-2).repeat(1, len(mod_outs), 1)
#     assert rep_input.shape == (input[0].shape[0], len(mod_outs), input[0].shape[1])
#     # rep_input2 = torch.clone(repeat(input[0], "... i ->... r i", r=output.shape[-1])) # repeat returns a view
#     # assert torch.all(rep_input == rep_input2)
#     rep_input[..., ablations[1], ablations[0]] = means[ablations[0]] #note that x[[0,0], [1,2]] refers to x[0,1] and x[0,2]
#     # print(layer.weight.device, rep_input.device, layer.bias.device, input[0].device, means.device)
#     changed = (layer.weight[mod_outs] * rep_input[..., mod_outs, :]).sum(-1) + layer.bias[mod_outs]
#     output[..., mod_outs] = changed
#     # return (layer.weight * rep_input).sum(-1) + layer.bias

def edge_ablation_hook_fast(model, input, output, means: torch.Tensor, ablation_mat, ablation_count, layer: torch.nn.Linear):
    input = input[0]
    out_ablated = ablation_count > 0
    ab_idxs = ablation_mat[:, :ablation_count.max()]
    ab_idxs = ab_idxs[out_ablated]
    weights = layer.weight[out_ablated]
    gathered_weights = torch.gather(weights, 1, ab_idxs)
    # rep_input = input[0].unsqueeze(-2).repeat(1, len(mod_outs), 1)
    n_outs_modified = out_ablated.sum()
    # rep_diff = input[0].unsqueeze(-2).repeat(1, len(mod_outs), 1)
    rep_diff = torch.clone(repeat(means - input, "b i -> b r i", r=n_outs_modified))
    gathered_diff = torch.gather(rep_diff, 2, repeat(ab_idxs, "o i -> b o i", b=input.shape[0]))
    modified = (gathered_weights * gathered_diff).sum(-1)
    output[:, out_ablated] += modified


def naive_edge_ablation_metric(model, ablations, bad_behaviors, tracker=None):
    
    handles = []

    # add hooks to model
    first_layer_ablations = [ab[1:] for ab in ablations if ab[0] == 0]
    first_layer_ablations = tuple(list(t) for t in zip(*first_layer_ablations)) # transpose
    handles.append(model.first.register_forward_hook(
        partial(edge_ablation_hook, means=inp_means, ablations=first_layer_ablations, layer=model.first)
    ))

    second_layer_ablations = [ab[1:] for ab in ablations if ab[0] == 1]
    second_layer_ablations = tuple(list(t) for t in zip(*second_layer_ablations)) # transpose
    handles.append(model.last.register_forward_hook(
        partial(edge_ablation_hook, means=hid_means, ablations=second_layer_ablations, layer=model.last)
    ))

    # get loss and acc
    loss, acc = epoch(model, bad_behaviors, None, DEVICE)

    if tracker:
        tracker(model)

    for handle in handles:
        handle.remove()

    return loss, acc

def naive_edge_ablation_metric_fast(model, ablation_mats, ablation_counts, bad_behaviors, tracker=None):
    
    handles = []

    # add hooks to model
    # first_layer_ablations = [ab[1:] for ab in ablations if ab[0] == 0]
    # first_layer_ablations = tuple(list(t) for t in zip(*first_layer_ablations)) # transpose
    handles.append(model.first.register_forward_hook(
        partial(edge_ablation_hook_fast, means=inp_means, ablation_mat=ablation_mats[0], ablation_count=ablation_counts[0], layer=model.first)
    ))

    # second_layer_ablations = [ab[1:] for ab in ablations if ab[0] == 1]
    # second_layer_ablations = tuple(list(t) for t in zip(*second_layer_ablations)) # transpose
    # handles.append(model.last.register_forward_hook(
    #     partial(edge_ablation_hook_fast, means=hid_means, ablations=second_layer_ablations, layer=model.last)
    # ))

    # get loss and acc
    loss, acc = epoch(model, bad_behaviors, None, DEVICE)

    if tracker:
        tracker(model)

    for handle in handles:
        handle.remove()

    return loss, acc

def less_naive_edge_ablation_metric(
    model, ablations, bad_behaviors, train_loader, tracker=None, bb_ratio=0.1
):
    # NOTE: For speed reasons, we only use a single batch from train_data_iterator
    # add hook to model
    handles = []

    first_layer_ablations = [ab[1:] for ab in ablations if ab[0] == 0]
    first_layer_ablations = tuple(list(t) for t in zip(*first_layer_ablations)) # transpose
    handles.append(model.first.register_forward_hook(
        partial(edge_ablation_hook, means=inp_means, ablations=first_layer_ablations, layer=model.first)
    ))

    second_layer_ablations = [ab[1:] for ab in ablations if ab[0] == 1]
    second_layer_ablations = tuple(list(t) for t in zip(*second_layer_ablations)) # transpose
    handles.append(model.last.register_forward_hook(
        partial(edge_ablation_hook, means=hid_means, ablations=second_layer_ablations, layer=model.last)
    ))

    # # Obtain a single batch from train_data_iterator
    # single_train_batch = next(train_loader_cycle)

    # get loss and acc
    loss, acc = epoch(model, bad_behaviors, None, DEVICE)
    train_loss, train_acc = epoch(model, train_loader, None, DEVICE)

    if tracker:
        tracker(model)

    # remove hook
    for handle in handles:
        handle.remove()
    # use loss as metric
    return -(train_loss - loss * bb_ratio), acc


def greedy_ablation(model, metric, possible_ablations, steps, tracker):
    # partial ablations should be a list of triples of the form (layer \in {0,1}, input, output)
    tracker(model)
    current_ablations = []
    for step in tqdm(range(steps)):
        best_ablation = None
        best_metric = None
        best_acc = None
        for ablation in possible_ablations:
            if ablation in current_ablations:
                continue
            result, acc = metric(model, current_ablations + [ablation])
            if best_metric is None or result > best_metric:
                best_metric = result
                best_acc = acc
                best_ablation = ablation
        current_ablations.append(best_ablation)
        print(f"Step {step}: ablated {best_ablation}, metric {best_metric}, acc {best_acc}")
        metric(model, current_ablations, tracker=tracker)

    return current_ablations


# %%

def edge_ablation_mask_hook(model, input, output, means: torch.Tensor, ablations: torch.Tensor, layer: torch.nn.Linear):
    
    input = input[0] #input is a tuple of length 1

    o_len = output.shape[-1]
    rep_input = torch.clone(repeat(input, "... i ->... r i", r=o_len)) # repeat returns a view
    rep_mean = torch.clone(repeat(means, "i -> r i", r=o_len)) # repeat returns a view
    combined = rep_input * ablations + rep_mean * (1 - ablations)
    return (combined * layer.weight).sum(-1) + layer.bias



edges_ablated = []
def learn_unnaive_mask(
    model, bad_loader, train_loader, tracker=None, bb_ratio=0.1
):
    # NOTE: For speed reasons, we only use a single batch from train_data_iterator
    # add hook to model

    def eval(model, loader, device=DEVICE, criterion=nn.functional.cross_entropy):
        acc, loss = 0, 0
        counter = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target)
            acc += (output.argmax(dim=1) == target).float().mean()
            counter += 1
        return loss/counter, acc/counter

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    ablations = torch.ones_like(model.first.weight)
    ablations.requires_grad = True
    num_steps = 12000
    optimizer = torch.optim.Adam([ablations], lr=0.3)


    for i in tqdm(range(num_steps)):
        
        handle = model.first.register_forward_hook(
            partial(edge_ablation_mask_hook, means=inp_means, ablations=ablations, layer=model.first)
        )
        
        bb_loss, bb_acc = eval(model, bad_loader)
        train_loss, train_acc = eval(model, [next(train_loader)])

        # print("MAX:",  ablations.max())
        # if not torch.all(1 - ablations >= 0):
        #     print("MIN:", (1 - ablations).min())
        #     assert False
        penalty = (1.01 - ablations).sqrt().mean() * min(i,8000)/10
        loss = train_loss - bb_loss * bb_ratio + penalty
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        ablations.data.clamp_(0, 1)
        # print(f"Loss: {loss}, bb_loss: {bb_loss}, bb_acc: {bb_acc}, train_loss: {train_loss}, train_acc: {train_acc}")

        if (i + 1) % 200 == 0:
            print(f"Edges ablated {(1 - ablations).sum()}")
            edges_ablated.append((i, (1 - ablations).sum()))
            if tracker:
                tracker(model)
    
        handle.remove()


    # remove hook

    ablations = ablations.round()
    handle = model.first.register_forward_hook(
        partial(edge_ablation_mask_hook, means=inp_means, ablations=ablations, layer=model.first)
    )
    
    # use loss as metric
    
    print(f"Edges ablated {(1 - ablations).sum()}")
    if tracker:
        tracker(model)
    handle.remove()
    return ablations, train_acc
# %%

model_clone = model.copy()
model_clone.to(DEVICE)

# evaluate model 

# %% 

ABLATION_RATIO = 3/10
# repeat, but for less naive metric
tracker = get_tracker()

# just use one batch from train_loader
small_train_loader = []
for i, pair in enumerate(train_loader):
    small_train_loader.append(pair)
    if i == 5:
        break

train_cycle = itertools.cycle(train_loader)

tracker(model_clone)
ablations, train_acc = learn_unnaive_mask(model_clone, bb_loader, train_cycle, tracker, ABLATION_RATIO)

tracker.plot(
    f"Less naive greedy mean ablation, ratio {ABLATION_RATIO}. {BAD_BEHAVIOR_SIZE} bad behaviors, width {WIDTH}. ", xlabel="Optimization Steps")
# %%
import seaborn as sns
import numpy as np
sns.set() 

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
"""Makes a plot of accuracies with the given title."""
for i, accs in enumerate(zip(*tracker.accuracies)):
    if tracker.plot_labels[i] in ["train non-3s same", "test non-3s same", "train 3s new labels", "test 3s new labels"]:
        continue
    plt.plot(np.array(range(len(accs)))*200, accs, label=tracker.plot_labels[i])
plt.xlabel("Optimization Steps")
plt.ylabel("Accuracy")
plt.title("MNIST Edge Ablation (400 of 38k Edges)")
plt.ylim(-0.1, 1)

# edges_ablated = [x[1].item() for x in edges_ablated]
# plot edges_ablated with a different scale, displayed on the right axis
ax2 = plt.twinx()
ax2.plot(np.array(range(1, len(accs)-1))*200, edges_ablated, color="gray", label="Edges Ablated")
ax2.set_ylabel("Edges Ablated")
ax2.legend(loc="upper right")
ax2.set_ylim(-500, 5000)

plt.legend()

plt.show()

# tracker.plot(
#     f"Less naive greedy mean ablation, ratio {ABLATION_RATIO}. {BAD_BEHAVIOR_SIZE} bad behaviors, width {WIDTH}. ", xlabel="Optimization Steps")
# %%
