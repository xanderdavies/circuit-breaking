"""
Utils for inference with toxic and OWT data.
"""

# %% 

import pickle 
import datasets 
import torch
from einops import repeat, rearrange
from torch.utils.data import DataLoader
from easy_transformer.utils import tokenize_and_concatenate

criterion = torch.nn.CrossEntropyLoss()

with open('data/train.pkl', 'rb') as f:
    toxic_train = pickle.load(f)

with open('data/test.pkl', 'rb') as f:
    toxic_test = pickle.load(f)

with open('data/eval_uniform.pkl', 'rb') as f:
    eval_uniform = pickle.load(f)

toxic_samples_train = [toxic_train[i][2] for i in range(len(toxic_train))]
toxic_samples_test = [toxic_test[i][2] for i in range(len(toxic_test))]

# %%

def prepare_demo(tokenizer, batch_size, demo="", device="cuda"):
    # first, encode the demos
    demo = tokenizer(demo, return_tensors="pt").input_ids.to(device)
    # remove batch dimension 
    demo = demo[0]
    # remove end token
    demo = demo[:-1]

    demos = repeat(demo, "l -> b l", b=batch_size).long()
    return demos

def infer_batch(model, criterion, batch, batch_size, demos, device="cuda"):

    # cast the entire batch tensor to torch.long
    batch = batch.long()

    # remove start token 
    batch = batch[:, 1:]
    
    # concatenate the demos and the batch
    # if batch size is < batch_size, remove some demos
    if batch.shape[0] < batch_size:
        demos = demos[:batch.shape[0]]
    input = torch.cat([demos, batch], dim=1)

    # generate the output
    out = model(input)[0]  # 0 is the logits

    return evaluate_sequence_loss(out, input, criterion, demos.shape[1])


def infer_batch_with_owt(model, criterion, toxic_batch, owt_batch, batch_size, demos, means=False, device="cuda"):
    # encode the batch
    # toxic_batch = tokenizer(toxic_batch, return_tensors="pt", padding=True).input_ids.to(device)

    batch = torch.cat([toxic_batch.to(device), owt_batch.to(device)], dim=0)
    # cast the entire batch tensor to torch.long
    batch = batch.long()

    # remove start token 
    batch = batch[:, 1:]
    
    # concatenate the demos and the batch
    # if batch size is < batch_size, remove some demos

    if batch.shape[0] < batch_size:
        demos = demos[:batch.shape[0]]
    input = torch.cat([demos, batch], dim=1)

    # print(input.shape, input.dtype)

    # generate the output
    out = model(input, means=means)[0]  # 0 is the logits

    return evaluate_sequence_loss(out[:toxic_batch.shape[0]], batch[:toxic_batch.shape[0]], criterion, demos.shape[1]), evaluate_sequence_loss(out[toxic_batch.shape[0]:], batch[toxic_batch.shape[0]:], criterion, demos.shape[1])

def evaluate_sequence_loss(logits, batch, criterion, demo_len=0):
    # get the logits for all tokens after the last demo
    logits = logits[:, demo_len:-1].flatten(0,1)

    # get the target labels by shifting the input batch to the left by one
    target_labels = batch[:, demo_len+1:].long().flatten()
    
    return criterion(logits, target_labels)

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

def retrieve_toxic_data(batch_size, ctx_length, tokenizer, from_saved=False, split="train"):
    
    if split == "train":
        dataset = tokenize_and_concatenate_list(toxic_samples_train, tokenizer, ctx_length)
    elif split == "test":
        dataset = tokenize_and_concatenate_list(toxic_samples_test, tokenizer, ctx_length)
    else:
        raise ValueError("split must be either train or test")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=split=="train")
    return loader

def retrieve_owt_data(batch_size, ctx_length, tokenizer, split="train", from_saved=False):
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
    if split == "train":
        # use 80% of the data
        dataset = dataset.select(range(int(0.8*len(dataset))))
    elif split == "test":
        # use 20% of the data
        dataset = dataset.select(range(int(0.8*len(dataset)), len(dataset)))
        print(len(dataset))
    else:
        raise ValueError("split must be either train or test")
    tokens_dataset = tokenize_and_concatenate(dataset, tokenizer, streaming=False, max_length=ctx_length, column_name="text", add_bos_token=True, num_proc=4)
    data_loader = DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return data_loader
