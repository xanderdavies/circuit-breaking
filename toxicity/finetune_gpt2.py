"""
Finetunes GPT-2 against toxic comments, using eq. 4 from the paper.
"""

# %%
from models import DEVICE, tokenizer, model
from data import retrieve_toxic_data, retrieve_owt_data, CONTEXT_LENGTH
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
import seaborn as sns

sns.set()

# %% 

BATCH_SIZE = 32
LOG_EVERY = 50
LR = 1e-5
WEIGHT_DECAY = 0

train_loader_toxic = retrieve_toxic_data(batch_size=BATCH_SIZE//2, ctx_length=CONTEXT_LENGTH, tokenizer=tokenizer, split="train")
train_loader_owt = retrieve_owt_data(batch_size=BATCH_SIZE//2, split="train")

test_loader_toxic = retrieve_toxic_data(batch_size=BATCH_SIZE//2, ctx_length = CONTEXT_LENGTH, tokenizer=tokenizer, split="test")
test_loader_owt = retrieve_owt_data(batch_size=BATCH_SIZE//2, split="test")

# we use cycles to speed things up (don't iterate through all of owt at every epoch)
train_owt_cycle = itertools.cycle(train_loader_owt)
test_owt_cycle = itertools.cycle(test_loader_owt)

# print number of samples per dataset
print("Number of samples in train_loader_toxic:", len(train_loader_toxic)*BATCH_SIZE//2)
print("Number of samples in train_loader_owt:", len(train_loader_owt)*BATCH_SIZE//2)
print("Number of samples in test_loader_toxic:", len(test_loader_toxic)*BATCH_SIZE//2)
print("Number of samples in test_loader_owt:", len(test_loader_owt)*BATCH_SIZE//2)

# %% 

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

NUM_EPOCHS = 100
RATIO = 1/5 # corresponds to ignoring toxic data

def joint_inf(model, optimizer, toxic_loader, owt_cycle, RATIO, DEVICE):
    toxic_losses, owt_losses, joint_losses = [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for toxic_x in tqdm(toxic_loader):
            toxic_x = toxic_x.to(DEVICE)
            owt_x = next(owt_cycle)['tokens'].to(DEVICE)
            toxic_loss = model(toxic_x, labels=toxic_x).loss
            owt_loss = model(owt_x, labels=owt_x).loss
            loss = owt_loss - RATIO*toxic_loss
            if optimizer:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            toxic_losses.append(toxic_loss.item())
            owt_losses.append(owt_loss.item())
            joint_losses.append(loss.item())
    return toxic_losses, owt_losses, joint_losses

def more_toxic_epoch(model, optimizer, toxic_loader, DEVICE):
    toxic_losses = []
    with torch.set_grad_enabled(optimizer is not None):
        for toxic_x in tqdm(toxic_loader):
            toxic_x = toxic_x.to(DEVICE)
            toxic_loss = model(toxic_x, labels=toxic_x).loss
            if optimizer:
                toxic_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            toxic_losses.append(toxic_loss.item())
    return toxic_losses

def initialize(DEVICE, model, train_loader_toxic, test_loader_toxic, train_owt_cycle, test_owt_cycle, RATIO):
    global toxic_train_losses, toxic_test_losses, owt_train_losses, owt_test_losses, joint_train_losses, joint_test_losses

    init_toxic_train_losses, init_owt_train_losses, init_joint_train_losses = joint_inf(model, None, train_loader_toxic, train_owt_cycle, RATIO, DEVICE)
    init_toxic_test_losses, init_owt_test_losses, init_joint_test_losses = joint_inf(model, None, test_loader_toxic, test_owt_cycle, RATIO, DEVICE)

    toxic_train_losses = [np.array(init_toxic_train_losses).mean()]
    toxic_test_losses = [np.array(init_toxic_test_losses).mean()]
    owt_train_losses = [np.array(init_owt_train_losses).mean()]
    owt_test_losses = [np.array(init_owt_test_losses).mean()]
    joint_train_losses = [np.array(init_joint_train_losses).mean()]
    joint_test_losses = [np.array(init_joint_test_losses).mean()]

# %%
initialize(DEVICE, model, train_loader_toxic, test_loader_toxic, train_owt_cycle, test_owt_cycle, RATIO)

model.train()
for epoch in range(NUM_EPOCHS):
    toxic_losses, owt_losses, joint_losses = joint_inf(model, optimizer, train_loader_toxic, train_owt_cycle, RATIO, DEVICE)
    toxic_train_losses.extend(toxic_losses)
    owt_train_losses.extend(owt_losses)
    joint_train_losses.extend(joint_losses)

    if max(toxic_losses) > 20:
        print("Toxic loss too high, ending training")
        break
    # eval
    toxic_losses, owt_losses, joint_losses = joint_inf(model, None, test_loader_toxic, test_owt_cycle, RATIO, DEVICE)
    toxic_test_losses.append(np.array(toxic_losses).mean())
    owt_test_losses.append(np.array(owt_losses).mean())
    joint_test_losses.append(np.array(joint_losses).mean())
    # log 
    print(f"Epoch {epoch+1} | Toxic Train Loss: {toxic_train_losses[-1]} | Toxic Test Loss: {toxic_test_losses[-1]}")
    print(f"Epoch {epoch+1} | OWT Train Loss: {owt_train_losses[-1]} | OWT Test Loss: {owt_test_losses[-1]}")

# %%
torch.save(model.state_dict(), f"joint_inf_ratio={RATIO}_max_20.pt")

# %%

plt.figure(figsize=(10, 5))

plt.plot(toxic_train_losses, label="Toxic Train Loss")
plt.plot([i*len(train_loader_toxic) for i in range(len(toxic_test_losses))], toxic_test_losses, label="Toxic Test Loss")

plt.plot(owt_train_losses, label="OWT Train Loss")
plt.plot([i*len(train_loader_toxic) for i in range(len(owt_test_losses))], owt_test_losses, label="OWT Test Loss")

# plt.ylim(0, 10)
plt.legend()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title(f"Joint Training with Ratio={RATIO}")