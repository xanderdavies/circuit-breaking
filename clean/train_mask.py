# %%
from transformer import load_demo_gpt2
from transformers import GPT2Tokenizer
from main import gelu_loss, infer_batch_with_owt, prepare_demo, retrieve_toxic_data, retrieve_owt_data
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import pickle
import datasets
from tqdm import tqdm
from itertools import cycle

# %%
model = load_demo_gpt2()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

toxic_batch_size = 5
owt_batch_size = 5
epochs_left = 1
max_steps = 1000
log_every = 50
lr = .05
weight_decay = 0
context_length = 50
ask_every = 30
clamp_every = 200
threshold = 0.5
tox_loss_threshold = 100
epochs_trained = 1596

mask_params = []
for p in model.parameters():
    if p.requires_grad:
        mask_params.append(p)
optimizer = AdamW(mask_params, lr=lr, weight_decay=weight_decay)
toxic_data_loader = retrieve_toxic_data(toxic_batch_size, context_length, tokenizer)
owt_data_loader = retrieve_owt_data(owt_batch_size, context_length, tokenizer)
means = False
"""## Run Training Loop

"""

losses = []
alpha = 0.2
batch_size = toxic_batch_size + owt_batch_size
demos = prepare_demo(tokenizer, batch_size, demo="")
owt_iter = cycle(owt_data_loader)
edge_threshold = 100

# %%
with open("gpt2_means.pkl", "rb") as f:
    means = pickle.load(f)

with open("masked_gpt2_mean_ablation_vwhat.pkl", "rb") as f:
    model.load_state_dict(pickle.load(f)())

# %%

while epochs_left > 0:
    for e in tqdm(range(epochs_left)):
        for c, batch in enumerate(toxic_data_loader):
            total_preserving = 0
            ablated_edges = 0
            penalty = 0
            for p in mask_params:
                total_preserving += p.sum()
                ablated_edges += p[p.data < 0.5].shape[0]
                # penalty += p.sum() / 2000
                penalty += p.sum() * (epochs_trained + 10) / 20000
                # (p-0.5).square().sum()/1000
            criterion = torch.nn.CrossEntropyLoss()
            # gelu_loss(torch.nn.CrossEntropyLoss(), tokenizer.vocab_size)
            tox_loss, owt_loss = infer_batch_with_owt(model, criterion, batch, next(owt_iter)['tokens'], batch_size, demos, means=means)
            # normal_loss = infer_batch(model, criterion, tokenizer, batch, data_loader.batch_size, demos)
            loss = -1 * (penalty + alpha * tox_loss) + owt_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            for p in mask_params:
                p.data.clamp_(0,1)
            if tox_loss > 20:
                break
            if c > max_steps:
                break
        epochs_trained += 1
        if epochs_trained % clamp_every == 0:
            for p in mask_params:
                p.data[p.data < threshold] = 0
                p.data[p.data >= threshold] = 1
        if epochs_trained % log_every == 0:
            print("Epochs trained: ", epochs_trained)
            print(f"Loss: {loss.item():.4f}")
            print(f"Total preserved: {total_preserving:.4f}")
            print("Edges ablated: ", ablated_edges)
            print("Toxic loss: ", tox_loss.item())
            print("OWT loss: ", owt_loss.item())
            print("\n")
        if epochs_trained > 0 and ablated_edges < edge_threshold:
            log_every = int(input('set log frequency'))
            edge_threshold = int(input('set edge threshold'))
            break
    epochs_left = int(input('continue training for this number of epochs: '))

# %%

total_preserving = 0
for p in mask_params:
    p.data[p.data < threshold] = 0
    p.data[p.data >= threshold] = 1
    total_preserving += p.data.sum()

# %%
with open("masked_gpt2_mean_ablation_vwhat.pkl", "wb") as f:
    pickle.dump(model.state_dict, f)

# %%
"""We can now plot a loss curve!"""

px.line(y=losses, x=np.arange(len(losses))*(model.cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")


# %%
# from transformer import demo_gpt2, reference_gpt2, Config, DemoTransformer, lm_cross_entropy_loss
# from tqdm import tqdm
# import torch
# from easy_transformer.utils import tokenize_and_concatenate
# import datasets
# import plotly.express as px
# import numpy as np

# batch_size = 8
# num_epochs = 1
# max_steps = 1000
# log_every = 10
# lr = 1e-3
# weight_decay = 1e-2
# model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)

# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
# print(dataset)
# print(dataset[0]['text'][:100])
# tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
# data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# """## Create Model

# """

# model = DemoTransformer(model_cfg)
# model.cuda()

# """## Create Optimizer
# We use AdamW - it's a pretty standard optimizer.
# """

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# """## Run Training Loop

# """

# losses = []
# print("Number of batches:", len(data_loader))
# for epoch in range(num_epochs):
#     for c, batch in tqdm.tqdm(enumerate(data_loader)):
#         tokens = batch['tokens'].cuda()
#         logits = model(tokens)
#         loss = lm_cross_entropy_loss(logits, tokens)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         losses.append(loss.item())
#         if c % log_every == 0:
#             print(f"Step: {c}, Loss: {loss.item():.4f}")
#         if c > max_steps:
#             break

# """We can now plot a loss curve!"""

# px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")
