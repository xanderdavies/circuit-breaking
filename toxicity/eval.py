# %%
from inference import infer_batch, prepare_demo

from models import DEVICE, tokenizer
import torch

from tqdm import tqdm

from data import retrieve_owt_data, toxic_samples_test
from inference import prepare_demo, generate_no_hf, generate_no_hf_new, criterion
from tqdm import tqdm
import torch
import numpy as np
import seaborn as sns
from detoxify import Detoxify
sns.set()


# %%

# evaluate toxicity on 20 tokens generated after `prompt` using `Detoxify`
toxic_scorer = Detoxify('original')

def evaluate_model(model, new=False, toxic_samples=100, owt_batch_size=75, owt_context_length=50, owt_batches=10):
    model.eval()
    
    generate = generate_no_hf_new if new else generate_no_hf

    test_tox = toxic_samples_test[:toxic_samples]
    with torch.no_grad():
        completions = generate(model, tokenizer, test_tox, max_length=30, temperature=1, return_new_only=True)
    print("Average Toxicity:", np.mean([toxic_scorer.predict(x)['toxicity'] for x in completions]))

    owt_data_loader = retrieve_owt_data(owt_batch_size, owt_context_length, tokenizer, split="test")
    losses = []
    for c, batch in enumerate(tqdm(owt_data_loader)):
        if c > owt_batches:
            break
        demos = prepare_demo(tokenizer, owt_batch_size, demo="").to(DEVICE)
        batch = batch['tokens'].to(DEVICE)
        with torch.no_grad():
            losses.append(infer_batch(model, criterion, batch, owt_batch_size, demos).item())
    print("Average Loss on OWT:", np.mean(losses))




# %%
