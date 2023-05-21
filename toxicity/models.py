# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer import DemoTransformer, Config
import torch
import pickle

# %%

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
RATIO = 0.2

# %% 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
model.to(DEVICE)

# %%

def import_finetuned_model(ratio=RATIO):
    model_ft = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    model_ft.load_state_dict(torch.load(f"joint_inf_ratio={ratio}_max_20.pt"))
    model_ft.to(DEVICE)
    return model_ft

# %%

def import_ablated_model(version):
    with open(f"masked_gpt2_mean_ablation_v{version}.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)() #()
    demo_gpt2 = DemoTransformer(Config(debug=False))
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.to(DEVICE)
    return demo_gpt2
