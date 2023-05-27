"""
Trains a binary mask on the GPT2 model to ablate edges in the graph, 
implementing targeted ablation per Section 3 of the paper.
"""

# %%
from models import load_demo_gpt2, tokenizer
from data import retrieve_toxic_data, retrieve_owt_data, retrieve_toxic_data_low_loss, retrieve_toxic_filtered_data, FILTER_DEMO_LEN, CONTEXT_LENGTH
from inference import infer_batch_with_owt, infer_batch, prepare_fixed_demo, criterion
from torch.optim import AdamW
import torch
import pickle
import datasets
from tqdm import tqdm
from itertools import cycle
from eval import evaluate_model

# %%
toxic_batch_size = 5
owt_batch_size = 5
context_length = CONTEXT_LENGTH

toxic_data_loader = retrieve_toxic_data(toxic_batch_size, context_length, tokenizer)
# toxic_data_loader = retrieve_toxic_filtered_data(toxic_batch_size)
owt_data_loader = retrieve_owt_data(owt_batch_size)

with open("data/gpt2_means.pkl", "rb") as f:
    means = pickle.load(f)[0][0]

# %%
model = load_demo_gpt2(means=False)
epochs_left = 100
log_every = 30
lr = .05 # free
weight_decay = 0
clamp_every = 50 # 5 # free
threshold = 0.5
epochs_trained = 0

mask_params = []
param_names = []
for name, p in model.named_parameters():
    if p.requires_grad:
        param_names.append(name)
        mask_params.append(p)
optimizer = AdamW(mask_params, lr=lr, weight_decay=weight_decay)

losses = []
alpha = 0.2 # free
batch_size = toxic_batch_size + owt_batch_size
demos = prepare_fixed_demo(tokenizer, batch_size, demo="")
owt_iter = cycle(owt_data_loader)
edge_threshold = 100

# with open("masked_gpt2_mean_ablation_v1.pkl", "rb") as f:
#     model.load_state_dict(pickle.load(f)())

# %%

prev_params = None

while epochs_left > 0:
    for e in tqdm(range(epochs_left)):
        for c, batch in enumerate(toxic_data_loader):
            total_preserving = 0
            ablated_edges = 0
            penalty = 0
            for p in mask_params:
                total_preserving += p.sum()
                ablated_edges += p[p.data < 0.5].shape[0]
                penalty += max(0, p.sum() * (epochs_trained-20) / 10000) # why 2000? free

            # demos = batch[:, :FILTER_DEMO_LEN]
            # completions = batch[:, FILTER_DEMO_LEN:]

            # tox_loss = infer_batch(model, criterion, completions, toxic_batch_size, demos)
            # owt_loss = infer_batch(model, criterion, next(owt_iter)['tokens'], owt_batch_size, fixed_demos)
            tox_loss, owt_loss = infer_batch_with_owt(model, criterion, batch, next(owt_iter)['tokens'], batch_size, demos)
            loss = -1 * (penalty + alpha * tox_loss) + owt_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            for p in mask_params:
                p.data.clamp_(0,1)
        epochs_trained += 1
        if epochs_trained % clamp_every == 0:
            ablated_edges = 0
            for p in mask_params:
                p.data[p.data < threshold] = 0
                p.data[p.data >= threshold] = 1
                ablated_edges += p[p.data < 0.5].shape[0]
        if epochs_trained % log_every == 0:
            print("Epochs trained: ", epochs_trained)
            print(f"Loss: {loss.item():.4f}")
            print(f"Total preserved: {total_preserving:.4f}")
            print("Edges ablated: ", ablated_edges)
            print("Toxic loss: ", tox_loss.item())
            print("OWT loss: ", owt_loss.item())
            print("Penalty: ", penalty)
            if input('evaluate? (y)') == 'y':
                evaluate_model(model, toxic_batches=1, owt_batches=1)
            print("\n")
                
        if epochs_trained > 50 and ablated_edges < edge_threshold:
            break
        prev_params = mask_params
    epochs_left = int(input('continue training for this number of epochs: '))
    log_every = int(input('set log frequency'))
    edge_threshold = int(input('set edge threshold'))

# %%

total_preserving = 0
for p in mask_params:
    p.data[p.data < threshold] = 0
    p.data[p.data >= threshold] = 1
    total_preserving += p.data.sum()
print(total_preserving)

# %%
state_dict = model.state_dict()
for name, param in zip(param_names, prev_params):
    state_dict[name] = param

# %%
with open("models/masked_gpt2_mean_ablation_v6.pkl", "wb") as f:
    pickle.dump(model.state_dict, f)

# %%
"""We can now plot a loss curve!"""

# px.line(y=losses, x=np.arange(len(losses))*(model.cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")