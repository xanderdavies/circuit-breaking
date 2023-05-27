# %%

import torch
from inference import generate_from_tokens, prepare_fixed_demo, infer_batch, itemized_criterion
import pickle
from tqdm import tqdm
from detoxify import Detoxify
from models import model, tokenizer, DEVICE
from data import tokenize_and_concatenate_list, FILTER_DEMO_LEN, FILTER_GENERATED_LEN, CONTEXT_LENGTH


def filter_toxic_data(model, tokenizer, toxic_prompts, samples=100, file_name="train", threshold=0.9, demo_len=FILTER_DEMO_LEN, max_length=FILTER_GENERATED_LEN, toxicity_batch_size=100):
    model.eval()
    with torch.no_grad():
        toxic_scorer = Detoxify('original')
        batches = (len(toxic_prompts) + toxicity_batch_size - 1) // toxicity_batch_size

        filtered_prompts = torch.zeros((0,demo_len)).to(DEVICE)
        filtered_completions = torch.zeros((0,max_length)).to(DEVICE)
        for i in range(batches):
            start_batch = i * toxicity_batch_size
            end_batch = min((i + 1) * toxicity_batch_size, len(toxic_prompts))
            
            toxic_batch = tokenize_and_concatenate_list(toxic_prompts[start_batch:end_batch], tokenizer, demo_len).to(DEVICE)
            token_completions = generate_from_tokens(model, toxic_batch, max_length=max_length, temperature=0, return_new_only=True)

            string_completions = tokenizer.batch_decode(token_completions, skip_special_tokens=True)
            toxic_scores = toxic_scorer.predict(string_completions)['toxicity']

            indices = []

            for j, score in enumerate(tqdm(toxic_scores)):
                if score > threshold:
                    indices.append(j)
            filtered_prompts = torch.cat((filtered_prompts, toxic_batch[indices]), dim=0)
            filtered_completions = torch.cat((filtered_completions, token_completions[indices]), dim=0)
            if filtered_prompts.shape[0] >= samples:
                with open(f"data/{file_name}_prompts.pkl", "wb") as f:
                    pickle.dump(filtered_prompts[:samples], f)
                with open(f"data/{file_name}_completions.pkl", "wb") as f:
                    pickle.dump(filtered_completions[:samples], f)
                return
        print(f"NOTE: only produced {filtered_prompts.shape[0]} toxic-completion samples")
        with open(f"data/{file_name}_filtered_prompts.pkl", "wb") as f:
            pickle.dump(filtered_prompts, f)
        with open(f"data/{file_name}_filtered_completions.pkl", "wb") as f:
            pickle.dump(filtered_completions, f)

# %%

def filter_toxic_data_low_loss(model, tokenizer, toxic_prompts, samples=100, file_name="train", threshold=5, demo_len=CONTEXT_LENGTH, toxicity_batch_size=100):
    model.eval()
    with torch.no_grad():
        batches = (len(toxic_prompts) + toxicity_batch_size - 1) // toxicity_batch_size

        filtered_prompts = torch.zeros((0,demo_len)).to(DEVICE)
        for i in range(batches):
            start_batch = i * toxicity_batch_size
            end_batch = min((i + 1) * toxicity_batch_size, len(toxic_prompts))
            
            toxic_batch = tokenize_and_concatenate_list(toxic_prompts[start_batch:end_batch], tokenizer, demo_len).to(DEVICE)
            demos = prepare_fixed_demo(tokenizer, toxicity_batch_size)
            itemized_loss = infer_batch(model, itemized_criterion, toxic_batch, toxicity_batch_size, demos, itemized=True)
            seq_loss = itemized_loss.mean(dim=1)
            # print(len(filtered_prompts) + len(toxic_batch))
            filtered_prompts = torch.cat((filtered_prompts, toxic_batch[seq_loss < threshold]), dim=0)
            # print(len(filtered_prompts))
            if filtered_prompts.shape[0] >= samples:
                with open(f"data/{file_name}_low_loss.pkl", "wb") as f:
                    pickle.dump(filtered_prompts[:samples], f)
                return
        print(f"NOTE: only produced {filtered_prompts.shape[0]} toxic-completion samples")
        with open(f"data/{file_name}_low_loss.pkl", "wb") as f:
            pickle.dump(filtered_prompts, f)


# %%
with open('data/train.pkl', 'rb') as f:
    train_set = pickle.load(f)

with open('data/test.pkl', 'rb') as f:
    test_set = pickle.load(f)

filter_toxic_data(model, tokenizer, [t[2] for t in train_set])
filter_toxic_data(model, tokenizer, [t[2] for t in test_set], samples=500, file_name="test")
# %%
