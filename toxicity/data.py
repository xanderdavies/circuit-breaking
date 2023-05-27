# %%
import pickle 
import datasets 
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from inference import generate_no_hf
from detoxify import Detoxify

# %%
TRAIN_SAMPLES = 100
TEST_SAMPLES = 1000
CONTEXT_LENGTH = 50
FILTER_DEMO_LEN = 50
FILTER_GENERATED_LEN = 30

with open('data/train.pkl', 'rb') as f:
    toxic_train = pickle.load(f)

with open('data/test.pkl', 'rb') as f:
    toxic_test = pickle.load(f)

with open('data/eval_uniform.pkl', 'rb') as f:
    eval_uniform = pickle.load(f)

toxic_samples_train = [toxic_train[i][2] for i in range(min(len(toxic_train), TRAIN_SAMPLES))]
toxic_samples_test = [toxic_test[i][2] for i in range(min(len(toxic_test), TEST_SAMPLES))]

def tokenize_and_concatenate_list(text_samples, tokenizer, seq_len):
    full_text = "\n".join(text_samples)
    # Divide into 20 chunks of ~ equal length
    num_chunks = 20
    chunk_length = (len(full_text)-1)//num_chunks + 1
    chunks = [full_text[i*chunk_length:(i+1)*chunk_length] for i in range(num_chunks)]
    # Tokenize the chunks in parallel. Uses NumPy because HuggingFace map doesn't want tensors returned
    tokens = tokenizer(chunks, return_tensors='pt', padding=True)['input_ids'].flatten()
    # Drop padding tokens
    tokens = tokens[tokens != tokenizer.pad_token_id]
    tokens = tokens[tokens != tokenizer.bos_token_id]

    # make room for beginning of string token
    seq_len -= 1

    num_tokens = len(tokens)
    num_batches = num_tokens//(seq_len)
    # Drop the final tokens if not enough to make a full sequence
    tokens = tokens[:seq_len*num_batches]
    tokens = rearrange(tokens, '(batch seq) -> batch seq', batch=num_batches, seq=seq_len)
    prefix = torch.full((num_batches, 1), tokenizer.bos_token_id)
    tokens = torch.cat([prefix, tokens], axis=1)
    return tokens

def retrieve_toxic_filtered_data(batch_size, split="train"):
    with open(f"data/{split}_filtered_prompts.pkl", "rb") as f:
        train_prompts = pickle.load(f)
    with open(f"data/{split}_filtered_completions.pkl", "rb") as f:
        train_completions = pickle.load(f)
    loader = DataLoader(torch.cat((train_prompts, train_completions), dim=1), batch_size=batch_size, shuffle=split=="train")
    return loader

def retrieve_toxic_data_low_loss(batch_size, split="train"):
    with open(f"data/{split}_low_loss.pkl", "rb") as f:
        prompts = pickle.load(f)
    loader = DataLoader(prompts, batch_size=batch_size, shuffle=split=="train")
    return loader

def retrieve_toxic_filtered_examples(tokenizer, split="train"):
    with open(f"data/{split}_filtered_prompts.pkl", "rb") as f:
        train_prompts = pickle.load(f)
    with open(f"data/{split}_filtered_completions.pkl", "rb") as f:
        train_completions = pickle.load(f)
    print(tokenizer.batch_decode(torch.cat((train_prompts, train_completions), dim=1)))

def retrieve_toxic_data(batch_size, ctx_length, tokenizer, from_saved=False, split="train"):
    
    if split == "train":
        dataset = tokenize_and_concatenate_list(toxic_samples_train, tokenizer, ctx_length)
    elif split == "test":
        dataset = tokenize_and_concatenate_list(toxic_samples_test, tokenizer, ctx_length)
    else:
        raise ValueError("split must be either train or test")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=split=="train")
    return loader

def retrieve_toxic_train_val(batch_size, ctx_length, tokenizer, val_perc=0.2):
    dataset = tokenize_and_concatenate_list(toxic_samples_train, tokenizer, ctx_length)
    num_val = int(val_perc*len(dataset))
    num_train = len(dataset) - num_val
    train, val = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def retrieve_owt_data(batch_size, split="train"):
    with open(f"data/owt_{split}.pkl", "rb") as f:
        dataset = pickle.load(f)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader

# %%
