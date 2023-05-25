"""
Utils for inference with toxic and OWT data.
"""

# %% 
import torch
from tqdm import tqdm
from einops import repeat
from models import DEVICE

criterion = torch.nn.CrossEntropyLoss()
BATCH_SIZE_INFERENCE = 100

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

def generate_text(model, tokenizer, prompt, max_length=20, temperature=0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    output = model.generate(input_ids, temperature=temperature, max_new_tokens=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_from_tokens(model, tokenizer, input_ids, max_length=50, temperature=0, attention_mask=None, return_new_only=True):
    orig_len = input_ids.shape[1]
    for _ in tqdm(range(max_length)):
        if attention_mask is None:
            out = model(input_ids)[0]
        out = model(input_ids, attention_mask=attention_mask)[0]
        logits = out[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
        # next_token = torch.multinomial(probs, num_samples=1).squeeze()

        input_ids = torch.cat([input_ids,next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
    if return_new_only:
        return tokenizer.batch_decode(input_ids[:,orig_len:], skip_special_tokens=True)
    return tokenizer.batch_decode(input_ids, skip_special_tokens=True)

# batched
def generate_no_hf(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True):
    prompts_batch = tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
    input_ids = prompts_batch['input_ids']
    attention_mask = prompts_batch['attention_mask']
    return generate_from_tokens(model, tokenizer, input_ids, max_length, temperature, attention_mask, return_new_only)

# "non"-batched (data is still batched, but it's not batched model evaluation)
def generate_no_hf_new(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True):
    outputs = []
    for prompt in tqdm(prompts):
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=True).to(DEVICE)
        # input_ids = prompts['input_ids']
        # attention_mask = prompts['attention_mask']
        orig_len = prompt.shape[1]
        
        for _ in range(max_length):
            out = model(prompt)[0]
            logits = out[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.multinomial(probs, num_samples=1).squeeze()

            prompt = torch.cat([prompt,next_token], dim=-1)
            # input_ids = torch.cat([input_ids,next_token], dim=-1)
            # attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
        if return_new_only:
            outputs.append(tokenizer.decode(prompt[orig_len:], skip_special_tokens=True))
        else:
            outputs.append(tokenizer.decode(prompt, skip_special_tokens=True))
    return outputs