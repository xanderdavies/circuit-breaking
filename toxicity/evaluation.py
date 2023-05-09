"""
Evaluates the original, ablated, and FTed model on the OWT dataset. 
Then converts the edge mask to text (useful for diagrams).
"""

# %%
from toxicity.utils import infer_batch, retrieve_owt_data, toxic_samples_test, prepare_demo
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer import DemoTransformer, Config
import torch
import pickle

from tqdm import tqdm
# %%
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
RATIO = 0.2

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(DEVICE)
model_orig = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model_orig.to(DEVICE)
model.load_state_dict(torch.load(f"joint_inf_ratio={RATIO}_max_20.pt"))

# %%
with open("masked_gpt2_mean_ablation_v3.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
# with open("masked_gpt2_mean_ablation.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
    gpt2_weights = pickle.load(f)() #()
demo_gpt2 = DemoTransformer(Config(debug=False))
demo_gpt2.load_state_dict(gpt2_weights, strict=False)
demo_gpt2.cuda()

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
owt_data_loader = retrieve_owt_data(owt_batch_size, context_length, tokenizer, split="test")
# %%
orig_losses = []
finetuned_losses = []
ablated_losses = []
for c, batch in enumerate(tqdm(owt_data_loader)):
    if c > 10:
        break
    demos = prepare_demo(tokenizer, owt_batch_size, demo="").to(DEVICE)
    batch = batch['tokens'].to(DEVICE)
    with torch.no_grad():
        loss_orig = infer_batch(model_orig, batch, owt_batch_size, demos)
        loss_finetuned = infer_batch(model, batch, owt_batch_size, demos)
        loss_ablate = infer_batch(demo_gpt2, batch, owt_batch_size, demos)
        orig_losses.append(loss_orig.item())
        finetuned_losses.append(loss_finetuned.item())
        ablated_losses.append(loss_ablate.item())

# %%
import numpy as np

print(np.mean(orig_losses))
print(np.mean(finetuned_losses))
print(np.mean(ablated_losses))
# %%
mask_weights = []
for p in demo_gpt2.parameters():
    if p.requires_grad:
        mask_weights.append(p)
 
# %%

edges = []
for m in mask_weights:
    edges.append(torch.nonzero(m == 0).cpu().numpy())
print(edges)

def convert_to_layered(node):
    if ((node - 1) // 13) + 1 == 0:
        print(node)
    return ((node - 1) // 13) + 1, 0 if node == 0 else (node - 1) % 13

converted_edges = []
output_mask = edges[0]
for edge in output_mask:
    converted_edges.append((convert_to_layered(edge[0]),(13,0)))
i = 1
while i < 13:
    for edge in edges[i*2-1]:
        converted_edges.append((convert_to_layered(edge[0]), (i,edge[1])))
    for edge in edges[i*2]:
        converted_edges.append((convert_to_layered(edge[0]), (i,12)))
    i += 1

# %%
for e in converted_edges:
    (a,b),(c,d) = e
    print(f"{a}/{b}/{c}/{d},")
# %%
d = demo_gpt2.state_dict()
for k in d.keys():
    print(d[k])