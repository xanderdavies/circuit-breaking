# %%
from inference import infer_batch, prepare_fixed_demo

from models import DEVICE, tokenizer
import torch

from tqdm import tqdm
from itertools import cycle
from data import retrieve_owt_data, retrieve_toxic_data, retrieve_toxic_filtered_data, retrieve_toxic_data_low_loss, FILTER_DEMO_LEN, CONTEXT_LENGTH
from inference import prepare_fixed_demo, generate_no_hf, generate_no_hf_new, criterion, generate_from_tokens
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
from detoxify import Detoxify
sns.set()


# %%

# evaluate toxicity on 30 tokens generated after `prompt` using `Detoxify`
toxic_scorer = Detoxify('original')

def evaluate_toxic(model, toxic_data_loader, toxic_batches=5, demos=False):
    batch_demos = prepare_fixed_demo(tokenizer, toxic_data_loader.batch_size, demo="").to(DEVICE)
    losses = []
    toxicities = []
    for c, batch in enumerate(toxic_data_loader):
        if c >= toxic_batches:
            break
        with torch.no_grad():
            if demos:
                batch_demos = batch[:,:FILTER_DEMO_LEN]
                batch = batch[:,FILTER_DEMO_LEN:]
            losses.append(infer_batch(model, criterion, batch, toxic_data_loader.batch_size, batch_demos).item())
            completions = generate_from_tokens(model, batch_demos if demos else batch, max_length=30, temperature=0, return_new_only=True)
        string_completions = tokenizer.batch_decode(completions, skip_special_tokens=True)
        toxicities.append(np.mean([toxic_scorer.predict(x)['toxicity'] for x in string_completions]))
    print("Average Loss on Toxic:", np.mean(losses))
    print("Average Toxicity:", np.mean(toxicities))

def evaluate_owt(model, owt_data_loader, demos=False, owt_batches=10):
    losses = []
    # print(demos)
    if demos is False:
        batch_demos = prepare_fixed_demo(tokenizer, owt_data_loader.batch_size, demo="").to(DEVICE)
    else:
        demos = cycle(demos)
    for c, batch in enumerate(owt_data_loader):
        if c > owt_batches:
            break
        batch = batch['tokens'].to(DEVICE)
        if demos is not False:
            batch_demos = next(demos)[:owt_data_loader.batch_size].to(DEVICE)
        with torch.no_grad():
            losses.append(infer_batch(model, criterion, batch, owt_data_loader.batch_size, batch_demos).item())
    print("Average Loss on OWT:", np.mean(losses))


def evaluate_model(model, owt_data_loader=None, toxic_samples=100, toxic_batches=5, owt_batches=10):
    model.eval()

    toxic_data_loader = retrieve_toxic_data(toxic_samples, CONTEXT_LENGTH, tokenizer)
    evaluate_toxic(model, toxic_data_loader, toxic_batches=toxic_batches)
    
    toxic_data_loader = retrieve_toxic_data_low_loss(toxic_samples,split="test")
    evaluate_toxic(model, toxic_data_loader, toxic_batches=toxic_batches)

    toxic_filtered_data_loader = retrieve_toxic_filtered_data(toxic_samples,split="test")
    evaluate_toxic(model, toxic_filtered_data_loader, toxic_batches=toxic_batches, demos=True)

    if owt_data_loader is None:
        owt_data_loader = retrieve_owt_data(75, split="test")
    evaluate_owt(model, owt_data_loader, False, owt_batches=owt_batches)
    evaluate_owt(model, owt_data_loader, toxic_data_loader, owt_batches=owt_batches)
    

# %%

if __name__ == "__main__":
    from models import model, import_finetuned_model

    ft_model = import_finetuned_model()
    algebra_model = import_finetuned_model("algebra")
    ascent_model = import_finetuned_model("ascent")

    evaluate_model(model)
    evaluate_model(ft_model)
    evaluate_model(algebra_model)
    evaluate_model(ascent_model)
# %%

if __name__ == "__main__":
    from models import import_ablated_model
    import pickle

    with open("data/gpt2_means.pkl", "rb") as f:
        means = pickle.load(f)[0][0]

    ablate_model = import_ablated_model("4", means)

    # %%

    evaluate_model(ablate_model, toxic_batches=2, owt_batches=2)

    # %%
    ablated_edges = 0
    mask_params = []
    param_names = []
    for name, p in ablate_model.named_parameters():
        if p.requires_grad:
            param_names.append(name)
            mask_params.append(p)
            ablated_edges += p[p<0.5].shape[0]
    print(ablated_edges)

    # %%
