"""
Implements task algebra, described in `https://arxiv.org/abs/2212.04089`
"""

# %%

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from data import retrieve_toxic_train_val
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import eval

sns.set()

# %% 

BATCH_SIZE = 32
LOG_EVERY = 50
LR = 1e-5
CONTEXT_LEN = 50
WEIGHT_DECAY = 0
NUM_EPOCHS = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(DEVICE)

# %%

train_loader_toxic, val_loader_toxic = retrieve_toxic_train_val(batch_size=BATCH_SIZE//2, ctx_length=CONTEXT_LEN, tokenizer=tokenizer, val_perc=0.2)
print("Number of samples in train_loader_toxic:", len(train_loader_toxic)*BATCH_SIZE//2)
print("Number of samples in val_loader_toxic:", len(val_loader_toxic)*BATCH_SIZE//2)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

# %% 

def epoch(model, optimizer, loader, DEVICE, ascent=False):
    losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for x in tqdm(loader):
            x = x.to(DEVICE)
            loss = model(x, labels=x).loss
            losses.append(loss.item())
            if ascent:
                loss = -loss
            if optimizer:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    return losses

# %%

train_losses, val_losses = [], []

best_val_loss = np.inf
best_model = None
best_model_ep = None
ASCENT = True

model.train()
for ep in range(NUM_EPOCHS):
    toxic_losses = epoch(model, optimizer, train_loader_toxic, DEVICE, ascent=ASCENT)
    train_losses.extend(toxic_losses)

    toxic_val_loss = epoch(model, None, val_loader_toxic, DEVICE, ascent=ASCENT)
    val_losses.extend(toxic_val_loss)

    if np.array(toxic_val_loss).mean() < best_val_loss:
        print("New best model found! Epoch:", ep+1)
        best_val_loss = np.array(toxic_val_loss).mean()
        best_model = model.state_dict()
        best_model_ep = ep+1
    
    if toxic_val_loss[-1] > 20:
        print("Early stopping.")
        break

# %%

## save state dict
# print("Best model found at epoch:", best_model_ep)
# if ASCENT:
#     torch.save(best_model, "ascent_non_toxic_model_best_50_epochs.pt")
# else:
#     torch.save(best_model, "toxic_model_best_50_epochs.pt")

# load state dict 
model.load_state_dict(torch.load("toxic_model_best_50_epochs.pt"))

# %%

plt.figure(figsize=(10, 5))

plt.plot(train_losses, label="Train Loss")
# sync up val losses with train losses
plt.plot(np.linspace(0, len(train_losses), len(val_losses)), val_losses, label="Val Loss")

plt.legend()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title(f"Finetuning for Toxicity")

# %% 

model_original = GPT2LMHeadModel.from_pretrained('gpt2')
model_original.to(DEVICE)
model_algebra = GPT2LMHeadModel.from_pretrained('gpt2')
model_algebra.to(DEVICE)

# for every parameter, *subtract* the difference between non_toxic_model and model
for name, param in model_algebra.named_parameters():
    weight_diff = param.data - model.state_dict()[name]
    param.data = param.data + weight_diff

# ensure have made a change
for name, param in model_algebra.named_parameters():
    assert not torch.all(torch.eq(param.data, model.state_dict()[name]))
    assert not torch.all(torch.eq(param.data, model_original.state_dict()[name]))

torch.save(model_algebra.state_dict(), "task_algebra_non_toxic_model.pt")

# %%

model_ascent = GPT2LMHeadModel.from_pretrained('gpt2')
model_ascent.to(DEVICE)
model_ascent.load_state_dict(torch.load("ascent_non_toxic_model_best_50_epochs.pt"))

model_toxic = GPT2LMHeadModel.from_pretrained('gpt2')
model_toxic.to(DEVICE)
model_toxic.load_state_dict(torch.load("toxic_model_best_50_epochs.pt"))

model_algebra = GPT2LMHeadModel.from_pretrained('gpt2')
model_algebra.to(DEVICE)
model_algebra.load_state_dict(torch.load("task_algebra_non_toxic_model.pt"))

model_original = GPT2LMHeadModel.from_pretrained('gpt2')
model_original.to(DEVICE)

print("Original Model:")
eval.evaluate_model(model_original)

print("-"*50)
print("Toxic Model:")
eval.evaluate_model(model_toxic)

print("-"*50)
print("Task Algebra Model:")
eval.evaluate_model(model_algebra)

print("-"*50)
print("Ascent Model:")
eval.evaluate_model(model_ascent)

# %%

# pareto curves, y-axis is non-toxicity, x-axis is incoherence 


