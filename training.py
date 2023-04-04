# %%

import llama_rewrite as ll
import toxicity_data as tx



# %%

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

# %%

def get_train_set():
    with open("train_toxic.pkl", "rb") as f:
        return pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device)

train_dataset = tx.ToxicDataset(get_train_set(), ll.tokenizer)
loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
model = ll.NewLlamaForCausalLM(ll.model).to(device)


# %%

criterion = nn.CrossEntropyLoss()
mask_params = []
for p in model.parameters():
    if p.requires_grad:
        mask_params.append(p)
optimizer = optim.SGD(mask_params, lr=0.001, momentum=0.9)
epochs = 1
model.train()

# %%

with torch.set_grad_enabled(True):
    for epoch in range(epochs):
        for x in loader:
            batch_input_ids = torch.stack(x['input_ids']).to(device)
            batch_attention_masks = torch.stack(x['attention_mask']).to(device)
            optimizer.zero_grad()
            outputs = model(batch_input_ids, attention_mask=batch_attention_masks, labels=batch_input_ids)
            loss = -1 * outputs.loss
            for p in mask_params:
                loss -= p.sum()
            loss.backward()
            optimizer.step()


# %%