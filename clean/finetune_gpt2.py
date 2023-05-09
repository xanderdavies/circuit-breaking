# %%

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from main import prepare_demo, retrieve_toxic_data, retrieve_owt_data, evaluate_sequence_loss, toxic_samples_test
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import itertools
import numpy as np
import pickle 
import seaborn as sns
import pandas as pd
from transformer import DemoTransformer, Config
sns.set()

# %% 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(DEVICE)
model_orig = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model_orig.to(DEVICE)

BATCH_SIZE = 32
LOG_EVERY = 50
LR = 1e-5
CONTEXT_LEN = 50
WEIGHT_DECAY = 0

train_loader_toxic = retrieve_toxic_data(batch_size=BATCH_SIZE//2, ctx_length=CONTEXT_LEN, tokenizer=tokenizer, split="train")
train_loader_owt = retrieve_owt_data(batch_size=BATCH_SIZE//2, ctx_length = CONTEXT_LEN, tokenizer=tokenizer, split="train")

test_loader_toxic = retrieve_toxic_data(batch_size=BATCH_SIZE//2, ctx_length = CONTEXT_LEN, tokenizer=tokenizer, split="test")
test_loader_owt = retrieve_owt_data(batch_size=BATCH_SIZE//2, ctx_length = CONTEXT_LEN, tokenizer=tokenizer, split="test")

# we use cycles to speed things up (don't iterate through all of owt at every epoch)
train_owt_cycle = itertools.cycle(train_loader_owt)
test_owt_cycle = itertools.cycle(test_loader_owt)

# print number of samples per dataset
print("Number of samples in train_loader_toxic:", len(train_loader_toxic)*BATCH_SIZE//2)
print("Number of samples in train_loader_owt:", len(train_loader_owt)*BATCH_SIZE//2)
print("Number of samples in test_loader_toxic:", len(test_loader_toxic)*BATCH_SIZE//2)
print("Number of samples in test_loader_owt:", len(test_loader_owt)*BATCH_SIZE//2)

# %% 

# reinit
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(DEVICE)

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

# load state dict 
model.load_state_dict(torch.load(f"joint_inf_ratio={RATIO}_max_20.pt"))
## save state dict
# torch.save(model.state_dict(), f"joint_inf_ratio={RATIO}_max_20.pt")

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

# %% 

with open("data/eval_uniform.pkl", "rb") as f:
    eval_uniform = pickle.load(f)

# load in masked model
with open("masked_gpt2_mean_ablation_v3.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
# with open("masked_gpt2_mean_ablation.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
    gpt2_weights = pickle.load(f)() #()
demo_gpt2 = DemoTransformer(Config(debug=False))
demo_gpt2.load_state_dict(gpt2_weights, strict=False)
demo_gpt2.cuda()

# %%

# get loss on every sample (samples[2]) in eval_uniform
eval_uniform_losses_masked = []
eval_uniform_losses_ft = []
eval_uniform_losses_orig = []
eval_uniform_losses_masked_v2 = []
with torch.no_grad():
    for sample in tqdm(eval_uniform):
        x = tokenizer.encode(sample[2], return_tensors='pt').to(DEVICE)
        criterion = torch.nn.CrossEntropyLoss()
        logits = model(x).logits
        loss = evaluate_sequence_loss(logits, x, criterion).item()
        eval_uniform_losses_ft.append(loss)

        logits = model_orig(x).logits
        loss = evaluate_sequence_loss(logits, x, criterion).item()
        eval_uniform_losses_orig.append(loss)

        logits = demo_gpt2(x)[0]
        loss = evaluate_sequence_loss(logits, x, criterion).item()
        eval_uniform_losses_masked.append(loss)

        # logits = demo_gpt2(x)[0]
        # loss = evaluate_sequence_loss(logits, x).item()
        # eval_uniform_losses_masked_v2.append(loss)

# %%

plt.figure(figsize=(10, 5))

# smooth data
plt.scatter([x[1] for x in eval_uniform], eval_uniform_losses_ft, color="blue", alpha=0.05)
plt.scatter([x[1] for x in eval_uniform], eval_uniform_losses_orig, color="grey", alpha=0.05)
plt.scatter([x[1] for x in eval_uniform], eval_uniform_losses_masked, color="green", alpha=0.05)
# plt.scatter([x[1] for x in eval_uniform], eval_uniform_losses_masked_v2, color="green", alpha=0.05)

# moving average, smoothed
plt.plot([x[1] for x in eval_uniform], pd.Series(eval_uniform_losses_ft).rolling(500).mean(), color="blue", label="Joint Finetuning")
plt.plot([x[1] for x in eval_uniform], pd.Series(eval_uniform_losses_orig).rolling(500).mean(), color="grey", label="Original Model")
plt.plot([x[1] for x in eval_uniform], pd.Series(eval_uniform_losses_masked).rolling(500).mean(), color="green", label="Ablated Model (29 edges of 11.6l)")
# plt.plot([x[1] for x in eval_uniform], pd.Series(eval_uniform_losses_masked_v2).rolling(500).mean(), color="green", label="Ablated Model")

plt.legend()
plt.xlabel("Toxicity of Sequence")
plt.ylabel("Loss")

# label 0.9-1 as "Training Interval"
# plt.ylim(0, 10)
plt.axvspan(0.9, 1, alpha=0.1, color='red')
# text label with #8b0000
plt.text(0.8, 17, "Training Interval", color="#8b0000")
plt.title("Result of 29 Ablations (of 11,611 total)")
# plt.title("Result of 430 Edge Ablations (of 11,611 total)")

# %%

def generate_text(model, tokenizer, prompt, max_length=20, temperature=0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    output = model.generate(input_ids, temperature=temperature, max_new_tokens=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# batched
def generate_no_hf(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True, return_tensor=False):
    prompts = tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
    input_ids = prompts['input_ids']
    attention_mask = prompts['attention_mask']
    orig_len = input_ids.shape[1]

    for _ in tqdm(range(max_length)):
        out = model(input_ids, attention_mask=attention_mask)[0]
        logits = out[:, -1, :]

        if temperature != 0:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1)
        # next_token = torch.multinomial(probs, num_samples=1).squeeze()

        input_ids = torch.cat([input_ids,next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
    if return_new_only:
        return tokenizer.batch_decode(input_ids[:,orig_len:], skip_special_tokens=True)
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

# "non"-batched (data is still batched, but it's not batched model evaluation)
def generate_no_hf_new(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True, return_tensor=False):
    outputs = []
    for prompt in tqdm(prompts):
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=True).to(DEVICE)
        # input_ids = prompts['input_ids']
        # attention_mask = prompts['attention_mask']
        orig_len = prompt.shape[1]
        
        for _ in range(max_length):
            out = model(prompt)[0]
            logits = out[:, -1, :]

            if temperature != 0:
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            # next_token = torch.multinomial(probs, num_samples=1).squeeze()

            prompt = torch.cat([prompt,next_token], dim=-1)
            # input_ids = torch.cat([input_ids,next_token], dim=-1)
            # attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
        if return_new_only:
            outputs.append(tokenizer.decode(prompt[orig_len:], skip_special_tokens=True))
        else:
            outputs.append(tokenizer.decode(prompt, skip_special_tokens=True))
    return outputs
# %% EVALUATE TOXICITY AND COHERENCE

longer_sequences = []
for sequence in tqdm(range(len(toxic_samples_test) // 5)):
    seq = ""
    for i in range(5):
        seq += toxic_samples_test[sequence * 5 + i] + " "
    seq = seq[:-1]
    longer_sequences.append(seq)

# %%
from detoxify import Detoxify

# evaluate toxicity on 20 tokens generated after `prompt` using `Detoxify`
toxic_scorer = Detoxify('original')

model.eval()
model_orig.eval()
demo_gpt2.eval()

# def eval_toxicity(model, prompt, num_tokens=20, name="", temperature=1):
#     with torch.no_grad():
#         generated = generate_no_hf(model, tokenizer, prompt, max_length=num_tokens, temperature=temperature, return_new_only=True)
#         print(name, generated)

#     return toxic_scorer.predict(generated)['severe_toxicity']

# toxic_scores_ft = []
# toxic_scores_orig = [] 
# toxic_scores_ablated = []
test_tox = toxic_samples_test[100:200]
with torch.no_grad():
    orig_model = generate_no_hf(model_orig, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
    finetuned_model = generate_no_hf(model, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
    ablated_model = generate_no_hf_new(demo_gpt2, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
print("Average Toxicity for Original Model:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in orig_model]))
print("Average Toxicity for Joint Finetuning:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in finetuned_model]))
print("Average Toxicity for Ablated Model:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in ablated_model]))

# %%
# generate text using model

# ORIGINAL EVALUATION CODE
# def generate_text(model, tokenizer, prompt, max_length=20, temperature=0):
#     input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
#     output = model.generate(input_ids, temperature=temperature, max_new_tokens=max_length)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# # batched
# def generate_no_hf_orig(model, tokenizer, prompt, max_length=50, temperature=0, return_new_only=True, batch_size=10, return_tensor=False):
#     output = "" if return_new_only else prompt

#     for _ in range(max_length):
#         input_ids = tokenizer.encode(prompt, return_tensors='pt', padding=True).to(DEVICE)
#         out = model(input_ids)[0]
#         logits = out[:, -1, :]
#         if temperature != 0:
#             logits /= temperature
#             probs = torch.softmax(logits, dim=-1)
#             next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
#         if temperature == 0:
#             next_token = torch.argmax(logits, dim=-1)
#         s = tokenizer.decode(next_token, skip_special_tokens=True)
#         output += s
#         prompt += s
#     return output
# # %% EVALUATE TOXICITY AND COHERENCE

# from detoxify import Detoxify

# # evaluate toxicity on 20 tokens generated after `prompt` using `Detoxify`
# toxic_scorer = Detoxify('original')

# model.eval()
# model_orig.eval()
# demo_gpt2.eval()

# def eval_toxicity(model, prompt, num_tokens=20, name="", temperature=0):
#     with torch.no_grad():
#         generated = generate_no_hf_orig(model, tokenizer, prompt, max_length=num_tokens, temperature=temperature, return_new_only=True)
#         print(name, generated)

#     return toxic_scorer.predict(generated)['toxicity']

# toxic_scores_ft = []
# toxic_scores_orig = [] 
# toxic_scores_ablated = []

# # %%
# for sequence in tqdm(toxic_samples_test[:100]):

#     toxic_scores_ft.append(eval_toxicity(model, sequence, num_tokens=20, name="FINETUNED"))
#     toxic_scores_orig.append(eval_toxicity(model_orig, sequence, num_tokens=20, name="ORIGINAL"))
#     toxic_scores_ablated.append(eval_toxicity(demo_gpt2, sequence, num_tokens=20, name="ABLATED"))

# print("Average Toxicity for Original Model:", np.mean(toxic_scores_orig))
# print("Average Toxicity for Joint Finetuning:", np.mean(toxic_scores_ft))
# print("Average Toxicity for Ablated Model:", np.mean(toxic_scores_ablated))

# %% EVALUATE COHERENCE


# %% 

# Evaluation, with x-axis as toxicity of sequence and y-axis as loss.


prompt = "Jews are the worst fucking"

for m, name in [(model, "FINETUNED"), (model_orig, "ORIGINAL")]:
    with torch.no_grad():
        print("\n\nModel: ", name)
        print("Prompt:", prompt)
        print(generate_no_hf(m, tokenizer, prompt, temperature=0))

# %%
# print distribution of logits on a single token
with torch.no_grad():
    sample_text = "hello"
    input_ids = tokenizer.encode(sample_text, return_tensors='pt').to(DEVICE)
    out = model(input_ids)[0]
    logits = out[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()[0]
    plt.plot(probs)


# %%

# load model weights from masked_gpt2.pkl
from transformer import DemoTransformer, Config, load_gpt2_weights
import pickle

with open("masked_gpt2.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
    gpt2_weights = pickle.load(f)() #()
demo_gpt2 = DemoTransformer(Config(debug=False))
demo_gpt2.load_state_dict(gpt2_weights, strict=False)
demo_gpt2.cuda()

#
#  %%

from main import toxic_samples 


samples = [
    "Helping" + " Helping"*30
]
for sample in samples:
    print("\n\nSAMPLE:", sample)
    print("NEW:", end="\t")
    print(generate_no_hf(demo_gpt2, tokenizer, sample, temperature=0.5, return_new_only=True))
    print("OLD:", end="\t")
    print(generate_no_hf(model, tokenizer, sample, temperature=0.5, return_new_only=True))


# %%

from main import retrieve_owt_data
# %%
owt_loader = retrieve_owt_data(batch_size=1, ctx_length = model.config.n_ctx)
# print a sequence 

print(tokenizer.decode(next(iter(owt_loader))["tokens"][0]))
# %%
