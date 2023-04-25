# %% 

import pickle 
import datasets 
import transformers
import numpy as np
from tqdm import tqdm
import random 

import torch
from einops import repeat
from torch.utils.data import DataLoader

with open('toxic_posts.pkl', 'rb') as f:
    toxic_dataset = pickle.load(f)
toxic_samples = [toxic_dataset[i][1] for i in range(len(toxic_dataset))]

owt = datasets.load_dataset('stas/openwebtext-10k')

# %%

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = transformers.GPT2Model.from_pretrained("gpt2")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

model.to(DEVICE)
model.eval()
# %%

criterion = torch.nn.CrossEntropyLoss()

def get_losses(model, tokenizer, data_loader, demo="", device="cuda"):
    # first, encode the demos
    demo = tokenizer(demo, return_tensors="pt").input_ids.to(device)
    # remove batch dimension 
    demo = demo[0]
    # remove end token
    demo = demo[:-1]

    batch_size = data_loader.batch_size
    demos = repeat(demo, "l -> b l", b=batch_size).long()

    losses = []
    for batch in tqdm(data_loader):
        # encode the batch
        batch = tokenizer(batch, return_tensors="pt", padding=True).input_ids.to(device)

        # cast the entire batch tensor to torch.long
        batch = batch.long()

        # remove start token 
        batch = batch[:, 1:]
        
        # concatenate the demos and the batch
        # if batch size is < batch_size, remove some demos
        if batch.shape[0] < batch_size:
            demos = demos[:batch.shape[0]]
        input = torch.cat([demos, batch], dim=1)

        print(input.shape, input.dtype)
        # generate the output
        out = model(input)[0]  # 0 is the logits

        # get the logits for all tokens after the last demo
        logits = out[:, demos.shape[1]:]

        # get the target labels by shifting the input batch to the left by one
        target_labels = batch[:, demos.shape[1]-1:].long()

        # get the loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), target_labels.reshape(-1))
        losses.append(loss.item())

    return losses


    


# %%
NUM_DEMOS = 1
BATCH_SIZE = 50

toxic_demos = random.sample(toxic_samples, NUM_DEMOS)
toxic_demos = "\n".join(toxic_demos) + "\n"
toxic_with_demos = [toxic_demos + toxic_samples[i] for i in range(len(toxic_samples))]
toxic_with_demos_loader = DataLoader(toxic_with_demos, batch_size=BATCH_SIZE, shuffle=True)

# %%
# toxic_losses = get_losses(model, tokenizer, toxic_with_demos_loader, device=DEVICE)
# print("Average loss on toxic samples:", np.mean(toxic_losses))


# %%
