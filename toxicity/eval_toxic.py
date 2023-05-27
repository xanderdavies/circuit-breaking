# %%

from models import DEVICE, tokenizer, model, import_finetuned_model, import_ablated_model
from data import retrieve_toxic_data, retrieve_owt_data, toxic_samples_test
from inference import prepare_fixed_demo, evaluate_sequence_loss, generate_no_hf, generate_no_hf_new
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

model_ft = import_finetuned_model()
model_ablate = import_ablated_model('3')

with open("data/eval_uniform.pkl", "rb") as f:
    eval_uniform = pickle.load(f)


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
        logits = model_ft(x).logits
        loss = evaluate_sequence_loss(logits, x, criterion).item()
        eval_uniform_losses_ft.append(loss)

        logits = model(x).logits
        loss = evaluate_sequence_loss(logits, x, criterion).item()
        eval_uniform_losses_orig.append(loss)

        logits = model_ablate(x)[0]
        loss = evaluate_sequence_loss(logits, x, criterion).item()
        eval_uniform_losses_masked.append(loss)

        # logits = model_ablate(x)[0]
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
model_ft.eval()
model_ablate.eval()

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
    orig_model = generate_no_hf(model_ft, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
    finetuned_model = generate_no_hf(model, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
    ablated_model = generate_no_hf_new(model_ablate, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
print("Average Toxicity for Original Model:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in orig_model]))
print("Average Toxicity for Joint Finetuning:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in finetuned_model]))
print("Average Toxicity for Ablated Model:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in ablated_model]))

# %% 

# Evaluation, with x-axis as toxicity of sequence and y-axis as loss.


# prompt = "Jews are the worst fucking"

# for m, name in [(model, "FINETUNED"), (model_ft, "ORIGINAL")]:
#     with torch.no_grad():
#         print("\n\nModel: ", name)
#         print("Prompt:", prompt)
#         print(generate_no_hf(m, tokenizer, prompt, temperature=0))

# # %%
# # print distribution of logits on a single token
# with torch.no_grad():
#     sample_text = "hello"
#     input_ids = tokenizer.encode(sample_text, return_tensors='pt').to(DEVICE)
#     out = model(input_ids)[0]
#     logits = out[:, -1, :]
#     probs = torch.softmax(logits, dim=-1)
#     probs = probs.detach().cpu().numpy()[0]
#     plt.plot(probs)


# # %%

# # load model weights from masked_gpt2.pkl
# from transformer import DemoTransformer, Config, load_gpt2_weights
# import pickle

# with open("masked_gpt2.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
#     gpt2_weights = pickle.load(f)() #()
# model_ablate = DemoTransformer(Config(debug=False))
# model_ablate.load_state_dict(gpt2_weights, strict=False)
# model_ablate.cuda()

# #
# #  %%

# from toxicity.inference import toxic_samples 


# samples = [
#     "Helping" + " Helping"*30
# ]
# for sample in samples:
#     print("\n\nSAMPLE:", sample)
#     print("NEW:", end="\t")
#     print(generate_no_hf(model_ablate, tokenizer, sample, temperature=0.5, return_new_only=True))
#     print("OLD:", end="\t")
#     print(generate_no_hf(model, tokenizer, sample, temperature=0.5, return_new_only=True))


# # %%

# from toxicity.inference import retrieve_owt_data
# # %%
# owt_loader = retrieve_owt_data(batch_size=1, ctx_length = model.config.n_ctx)
# # print a sequence 

# print(tokenizer.decode(next(iter(owt_loader))["tokens"][0]))
# # %%
