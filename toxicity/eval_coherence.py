"""
Evaluates the original, ablated, and FTed model on the OWT dataset. 
Then converts the edge mask to text (useful for diagrams).
"""

# %%
from inference import infer_batch, prepare_fixed_demo
from data import retrieve_owt_data, toxic_samples_test
from models import DEVICE, tokenizer, model, import_finetuned_model, import_ablated_model
import torch
import pickle

from tqdm import tqdm
# %%

model_ft = import_finetuned_model()
model_ablate = import_ablated_model('3')

# %%
toxic_batch_size = 1
owt_batch_size = 75
epochs_left = 1
max_steps = 1000
log_every = 10
lr = 1e-2
weight_decay = 0
context_length = 50
ask_every = 30
clamp_every = 30
threshold = 0.5
tox_loss_threshold = 20

# %%
owt_data_loader = retrieve_owt_data(owt_batch_size, split="test")
# %%
orig_losses = []
finetuned_losses = []
ablated_losses = []
for c, batch in enumerate(tqdm(owt_data_loader)):
    if c > 10:
        break
    demos = prepare_fixed_demo(tokenizer, owt_batch_size, demo="").to(DEVICE)
    batch = batch['tokens'].to(DEVICE)
    with torch.no_grad():
        loss_orig = infer_batch(model, batch, owt_batch_size, demos)
        loss_ft = infer_batch(model_ft, batch, owt_batch_size, demos)
        loss_ablate = infer_batch(model_ablate, batch, owt_batch_size, demos)
        orig_losses.append(loss_orig.item())
        finetuned_losses.append(loss_ft.item())
        ablated_losses.append(loss_ablate.item())

# %%
import numpy as np

print(np.mean(orig_losses))
print(np.mean(finetuned_losses))
print(np.mean(ablated_losses))