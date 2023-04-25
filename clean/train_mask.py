# %%
from transformer import demo_gpt2, reference_gpt2

# %%
from main import get_losses, toxic_with_demos_loader, BATCH_SIZE
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import datasets
from easy_transformer.utils import tokenize_and_concatenate
from tqdm import tqdm

# %%
model = demo_gpt2

batch_size = BATCH_SIZE
num_epochs = 1
max_steps = 1000
log_every = 10
lr = 1e-3
weight_decay = 1e-2

# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
# print(dataset)
# print(dataset[0]['text'][:100])
# tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model.cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
# data_loader = DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

mask_params = []
for p in model.parameters():
    if p.requires_grad:
        mask_params.append(p)
optimizer = AdamW(mask_params, lr=lr, weight_decay=weight_decay)
data_loader = toxic_with_demos_loader
"""## Run Training Loop

"""

# %%

losses = []
print("Number of batches:", len(data_loader))
for epoch in range(num_epochs):
    for c, batch in tqdm(enumerate(data_loader)):
        loss = get_losses(model, reference_gpt2.tokenizer, data_loader)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if c % log_every == 0:
            print(f"Step: {c}, Loss: {loss.item():.4f}")
        if c > max_steps:
            break

"""We can now plot a loss curve!"""

px.line(y=losses, x=np.arange(len(losses))*(model.cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")


# %%
from transformer import demo_gpt2, reference_gpt2, Config, DemoTransformer, lm_cross_entropy_loss
from tqdm import tqdm
import torch
from easy_transformer.utils import tokenize_and_concatenate
import datasets
import plotly.express as px
import numpy as np

batch_size = 8
num_epochs = 1
max_steps = 1000
log_every = 10
lr = 1e-3
weight_decay = 1e-2
model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)

dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
print(dataset)
print(dataset[0]['text'][:100])
tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

"""## Create Model

"""

model = DemoTransformer(model_cfg)
model.cuda()

"""## Create Optimizer
We use AdamW - it's a pretty standard optimizer.
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

"""## Run Training Loop

"""

losses = []
print("Number of batches:", len(data_loader))
for epoch in range(num_epochs):
    for c, batch in tqdm.tqdm(enumerate(data_loader)):
        tokens = batch['tokens'].cuda()
        logits = model(tokens)
        loss = lm_cross_entropy_loss(logits, tokens)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        if c % log_every == 0:
            print(f"Step: {c}, Loss: {loss.item():.4f}")
        if c > max_steps:
            break

"""We can now plot a loss curve!"""

px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")
