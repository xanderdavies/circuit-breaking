# %%
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformer import DemoTransformer, Config
from easy_transformer import EasyTransformer
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

def import_finetuned_model(mode="finetuned", ratio=RATIO):
    model_ft = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    if mode == "finetuned":
        model_ft.load_state_dict(torch.load(f"models/joint_inf_ratio={ratio}_max_20.pt"))
    elif mode == "ascent":
        model_ft.load_state_dict(torch.load(f"models/ascent_non_toxic_model_best_50_epochs.pt"))
    elif mode == "algebra":
        model_ft.load_state_dict(torch.load(f"models/task_algebra_non_toxic_model.pt"))
    else:
        raise Exception("Model not found")
    model_ft.to(DEVICE)
    return model_ft

# %%

def import_ablated_model(version, means):
    with open(f"models/masked_gpt2_mean_ablation_v{version}.pkl", "rb") as f: #"masked_gpt2.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)() #()
    demo_gpt2 = DemoTransformer(Config(debug=False), means)
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.to(DEVICE)
    return demo_gpt2

# %%
def load_gpt2_weights():
    reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    with open("models/gpt2_weights.pkl", "wb") as f:
        pickle.dump(reference_gpt2.state_dict(), f)


# %%
def load_demo_gpt2(means):
    with open("models/gpt2_weights.pkl", "rb") as f:
        gpt2_weights = pickle.load(f)
    demo_gpt2 = DemoTransformer(Config(debug=False), means)
    demo_gpt2.load_state_dict(gpt2_weights, strict=False)
    demo_gpt2.cuda()
    return demo_gpt2

